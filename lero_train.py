from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import pytorch_lightning as pl
import pytorch_lightning.callbacks as plc
import torch.nn.functional as F
import torch.nn as nn
import os
from ray.util import ActorPool
from util import postgres
from util.pg_executor import ActorThatQueries, actor_call_leon
from util import envs
from util.envs import load_sql, load_training_query
from util import plans_lib
import pytorch_lightning.loggers as pl_loggers
import pickle
import math
import json
from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers
from util.dataset import LeonDataset, prepare_dataset, BucketBatchSampler, BucketDataset
from leon_experience import Experience, TIME_OUT
from util.model import PL_Leon
import numpy as np
from ray.util.iter import from_items
import ray
import time
from argparse import ArgumentParser
import copy
import wandb
import time
from tqdm import tqdm
from config import read_config
import random
from util import treeconv
from util import encoding
from utils import *
import gc
from traning_test import get_vecs, get_channels_dfs, DNN, PL_DNN
from lero_feature import *
from lero_model import LeroModel, LeroModelPairWise, LeroNet, PL_Lero
pl.seed_everything(42)
conf = read_config()
model_type = conf['leon']['model_type']

def load_model(model_path: str, prev_optimizer_state_dict=None):
    if not os.path.exists(model_path):
        if model_type == "Transformer":
            print("load transformer model")
            model = SeqFormer(
                            input_dim=configs['node_length'],
                            hidden_dim=128,
                            output_dim=1,
                            mlp_activation="ReLU",
                            transformer_activation="gelu",
                            mlp_dropout=0.2,
                            transformer_dropout=0.2,
                            query_dim=configs['query_dim'],
                            padding_size=configs['pad_length']
                        ).to(DEVICE)
        elif model_type == "TreeConv":
            print("load treeconv model")
            # model = treeconv.TreeConvolution(666, 50, 1).to(DEVICE)
            model = treeconv.ResNet(666, 50, 1, treeconv.ResidualBlock, [1, 1, 1, 1]).to(DEVICE)
        dnn_model = DNN(77, [512, 256, 128], 2)
        lero_model = LeroNet(29)
        torch.save(lero_model, "./log/lero_model.pth")
        torch.save(dnn_model, "./log/dnn_model.pth")
        torch.save(model, model_path)
    else:
        dnn_model = torch.load("./log/dnn_model.pth", map_location=DEVICE).to(DEVICE)
        lero_model = torch.load("./log/lero_model.pth", map_location=DEVICE).to(DEVICE)
        model = torch.load(model_path, map_location=DEVICE).to(DEVICE)
    model = PL_Leon(model, prev_optimizer_state_dict)
    dnn_model = PL_DNN(dnn_model)
    lero_model = PL_Lero(lero_model)
    
    return model, dnn_model, lero_model

def getPG_latency(query, hint=None, ENABLE_LEON=False, timeout_limit=None, curr_file=None):
    """
    input. a loaded query
    output. the average latency of a query get from pg
    """
    latency_sum = 0
    if timeout_limit is None:
        timeout_limit = TIME_OUT # TIME_OUT
    cnt = 1 # 3 1 
    for c in range(cnt):
        latency, json_dict = postgres.GetLatencyFromPg(query, hint, ENABLE_LEON, verbose=False, check_hint_used=False, timeout=timeout_limit, dropbuffer=False, curr_file=curr_file)
        latency_sum = latency_sum + latency
    pg_latency = latency_sum / cnt
    if pg_latency == timeout_limit:
        pg_latency = TIME_OUT
    if ENABLE_LEON and json_dict == []:
        json_dict = postgres.getPlans(query, hint, check_hint_used=False, ENABLE_LEON=ENABLE_LEON, curr_file=curr_file)[0][0][0]
        
    return pg_latency, json_dict

# UNFINISHED hint sql -> pgcosts
def plans_encoding(plans):
    '''
    input. a list of plans in type of json
    output. (seq_encoding, run_times, attention_mask, loss_mask)
        - seq_encoding torch.Size([1, 760])
    '''
    seqs = []
    attns = []
    for x in plans:        
        seq_encoding, run_times, attention_mask, loss_mask, database_id = get_plan_encoding(
            x, configs, op_name_to_one_hot, plan_parameters, feature_statistics
        ) # run_times 无法获取
        seqs.append(seq_encoding) 
        attns.append(attention_mask)

    seqs = torch.stack(seqs, dim=0)
    attns = torch.stack(attns, dim=0)

    return seqs, run_times, attns, loss_mask

def get_calibrations(model, seqs, attns, queryfeature):
    seqs = seqs.to(DEVICE)
    attns = attns.to(DEVICE)
    queryfeature = queryfeature.to(DEVICE)
    cost_iter = 10 # training 用模型推理 10 次 获得模型不确定性
    model.model.eval()
    with torch.no_grad():
        for i in range(cost_iter):
            if model_type == "Transformer":
                cali = model.model(seqs, attns, queryfeature)[:, 0] # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
            elif model_type == "TreeConv":
                cali = torch.tanh(model.model(seqs, attns, queryfeature)).add(1).squeeze(1)
            if i == 0:
                cali_all = cali.unsqueeze(1) # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
            else:
                cali_all = torch.cat((cali_all, cali.unsqueeze(1)), dim=1)
    return cali_all


def get_ucb_idx(cali_all, pgcosts):    
    cali_mean = cali_all.mean(dim = 1) # 每个 plan 的 cali 均值 [# of plan] | 值为 1 左右
    cali_var = cali_all.var(dim = 1) # [# of plan] | 值为 0 左右
    # costs = pgcosts # [# of plan]
    cost_t = torch.mul(cali_mean, costs) # 计算 calibration mean * pg返回的cost [# of plan]

    # cali_min, _ = cost_t.min(dim = 0) # plan 的 cali 最小值 [# of plan] 【cali_min只有一个值？】
    # print("cost_min.shape(): ", cali_min.shape)
    # print(cali_min)            
    # ucb = cali_var / cali_var.max() - cali_min / cali_min.max()
    # ucb = cali_var / cali_var.max()
    ucb = cost_t
    # ucb = cali_var / cali_var.max() - cost_t / cost_t.max() # [# of plan]
    # print("len(ucb)", len(ucb))
    ucb_sort_idx = torch.argsort(ucb, descending=True) # 张量排序索引 | ucb_sort[0] 是uncertainty高的plan在plan_info中的索引
    ucb_sort_idx = ucb_sort_idx.tolist()

    return ucb_sort_idx

def PlanToNode(workload, plans):
    """
    input. plans 一个 message 等价类包括多条 plans
    output. nodes
        sql 信息在 node.info['sql_str'], cost 信息在 node.cost, hint 信息在 node.hint_str(), join_tables 信息在 node.info['join_tables']
    """
    nodes = []
    for i in range(len(plans)):
        # print("plans[{}]\n".format(i))
        node = postgres.ParsePostgresPlanJson_1(plans[i], workload.workload_info.alias_to_names)
        if i == 0:
            if node.info['join_cond'] == ['']:
                return None
            temp = node.to_sql(node.info['join_cond'], with_select_exprs=True)
            node.info['sql_str'] = temp
        node.info['sql_str'] = temp
        nodes.append(node)
    
    return nodes

def getNodesCost(nodes):
    pgcosts = [] # 存所有 plan 的 pg cost
    for node in nodes:
        pgcost = node.cost
        pgcost_log = math.log(pgcost)
        pgcosts.append(pgcost_log)
    
    return pgcosts

def initEqSet():
    train_file, training_query = envs.load_train_files(conf['leon']['workload_type'])
    equ_tem = envs.find_alias(training_query)
    equ_set = set() # 用集合 方便 eq keys 中去重
    for i in equ_tem:
        e_tem = i.split(',')
        e_tem = ','.join(sorted(e_tem))
        equ_set.add(e_tem)

    return equ_set
def collects(finnode: plans_lib.Node, actor, exp: Experience, timeout, currTotalLatency, sql, query_id, model):

    join_ids_to_append = []
    allPlans = [finnode]
    while (allPlans):
        currentNode = allPlans.pop(0)
        allPlans.extend(currentNode.children)
        if currentNode.IsJoin():
            cur_join_ids = ','.join(
                sorted([i.split(" AS ")[-1] for i in currentNode.leaf_ids()]))
            join_ids_to_append.append(cur_join_ids)
    # join_ids_to_append = join_ids_to_append[:1]
    for join_id in join_ids_to_append:
        exp_key = Exp.GetExpKeys()
        temp = join_id.split(',') # sort
        if len(temp) <= 3:
            return
        join_id = ','.join(sorted(temp))
        if join_id not in exp_key:
            print('degradation collect:', join_id)
            exp.AddEqSet(join_id, query_id)
            ray.get(actor.reload_model.remote(exp.GetEqSetKeys(), model.eq_summary))
            return


def load_callbacks(logger):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='val_acc',
        mode='max',
        patience=2,
        min_delta=0.001,
        check_on_train_epoch_end=False
    ))
    if logger:
        callbacks.append(plc.ModelCheckpoint(
            dirpath= logger.experiment.dir,
            monitor='val_scan',
            filename='best-{epoch:02d}-{val_scan:.3f}',
            save_top_k=1,
            mode='min',
            save_last=False
        ))

    # if args.lr_scheduler:
    #     callbacks.append(plc.LearningRateMonitor(
    #         logging_interval='epoch'))
    return callbacks

def _batch(trees, indexes, padding_size=200):
    # 获取 batchsize
    batch_size = len(trees)
    tree_embedding_size = trees[0].size(1)
    # 初始化填充后的张量
    padded_trees = torch.zeros((batch_size, tree_embedding_size, padding_size))
    padded_indexes = torch.zeros((batch_size, padding_size, 1))

    for i in range(batch_size):
        # 获取当前样本的原始树和索引张量
        tree = trees[i]
        index = indexes[i]

        # 计算需要填充的列数
        padding_cols_tree = max(0, padding_size - tree.size(2))
        padding_cols_index = max(0, padding_size - index.size(1))

        # 使用 F.pad 进行填充
        padded_tree = F.pad(tree, (0, padding_cols_tree), value=0)
        padded_index = F.pad(index, (0, 0, 0 , padding_cols_index), value=0)

        # 将填充后的张量放入结果中
        padded_trees[i, :, :] = padded_tree
        padded_indexes[i, :, :] = padded_index

    return padded_trees, padded_indexes.long()
    

if __name__ == '__main__':

    file_path = ["./log/messages.pkl", './log/model.pth', './log/dnn_model.pth', './log/lero_model.pth']
    for file_path1 in file_path:
        # 检查文件是否存在
        if os.path.exists(file_path1):
            # 删除文件
            os.remove(file_path1)
            print(f"File {file_path1} has been successfully deleted.")
        else:
            print(f"File {file_path1} does not exist.")
    
    
    pretrain = True
    iscache = False
    if pretrain:
        # checkpoint = torch.load("./log/SimModel3.pth", map_location=DEVICE)
        checkpoint = treeconv.ResNet(820, 54, 1, treeconv.ResidualBlock, [1, 1, 1, 1]).to(DEVICE)
        torch.save(checkpoint, "./log/model.pth")
        checkpoint = LeroNet(29)
        torch.save(checkpoint, "./log/lero_model.pth")
        print("load SimModel success")
        checkpoint = torch.load("./log/DnnModel0223_job.pth", map_location=DEVICE)
        torch.save(checkpoint, "./log/dnn_model.pth")
    if iscache:
        with open("./log/exp_v5.pkl", "rb") as file:
            exp_cache = pickle.load(file)

    with open ("./conf/namespace.txt", "r") as file:
        namespace = file.read().replace('\n', '')
    with open ("./conf/ray_address.txt", "r") as file:
        ray_address = file.read().replace('\n', '')
    context = ray.init(address=ray_address, namespace=namespace, _temp_dir=conf['leon']['ray_path'] + "/log/ray") # init only once
    print(context.address_info)
    # dict_actor = ray.get_actor('querydict')
    our_ports = eval(conf['leon']['other_leon_port'])
    our_ports.append(int(conf['leon']['Port']))
    ports = eval(conf['leon']['other_db_port'])
    ports.append(int(conf['PostgreSQL']['port']))
    actors = [ActorThatQueries.options(name=f"actor{port}").remote(port, our_port) for port, our_port in zip(ports, our_ports)]
    pool = ActorPool(actors)
    
    train_files, training_query, test_files, test_query = envs.load_train_files1(conf['leon']['workload_type'])
    # ray.get(dict_actor.write_sql_id.remote(train_files))
    chunk_size = 3 # the # of sqls in a chunk
    min_batch_size = 256
    TIME_OUT_Ratio = 2
    model_path = "./log/model.pth" 
    message_path = "./log/messages.pkl"
    prev_optimizer_state_dict = None
    dnn_prev_optimizer_state_dict = None
    model, dnn_model, lero_model = load_model(model_path)
    
    Exp = Experience(eq_set=initEqSet())
    eqset = Exp.GetEqSet()
    model.eq_summary = {key: 0 for key in eqset}
    print("Init workload and equal set keys")
    channels = ['EstNodeCost', 'EstRows', 'EstBytes', 'EstRowsProcessed', 'EstBytesProcessed',
                'LeafWeightEstRowsWeightedSum', 'LeafWeightEstBytesWeightedSum']
    workload = envs.wordload_init(conf['leon']['workload_type'])
    plan_channels_init = dict()
    ops = workload.workload_info.all_ops
    ops = np.array([entry.replace(' ', '') for entry in ops])
    ops = np.where(ops == 'NestedLoop', 'NestLoop', ops)
    ops = np.where(ops == 'Materialize', 'Material', ops)
    for c in channels:
        plan_channels_init[c] = dict()
        for node_type in ops:
            plan_channels_init[c][node_type] = 0
    queryFeaturizer = plans_lib.QueryFeaturizer(workload.workload_info)
    feature_generator = FeatureGenerator()
    feature_generator.normalizer = Normalizer(
                {"Startup Cost": workload.workload_info.mins['cost'],
                 "Total Cost": workload.workload_info.mins['cost'], "Plan Rows": min(workload.workload_info.table_num_rows.values())},
                {"Startup Cost": workload.workload_info.maxs['cost'],
                 "Total Cost": workload.workload_info.maxs['cost'], "Plan Rows": max(workload.workload_info.table_num_rows.values())})
    feature_generator.feature_parser = AnalyzeJsonParser(feature_generator.normalizer, list(ops))
    if model_type == "Transformer":
        statistics_file_path = "./statistics.json"
        feature_statistics = load_json(statistics_file_path)
        add_numerical_scalers(feature_statistics)
        op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    elif model_type == "TreeConv":
        nodeFeaturizer = plans_lib.TreeNodeFeaturizer_V2(workload.workload_info)
    first_time = dict()
    all_train_time = 0
    last_train_pair = 0
    min_leon_time = dict()
    logger =  pl_loggers.WandbLogger(save_dir=os.getcwd() + '/logs', name="lero job in 202", project=conf['leon']['wandb_project'])
    for key in conf:
        logger.log_hyperparams(conf[key])
    my_step = 0
    same_actor = ray.get_actor('leon_server')
    task_counter = ray.get_actor('counter')
    runtime_pg = 0
    runtime_leon = 0
    min_exec_num = 2
    max_exec_num = 30
    encoding_dict = dict() # 用来存trees和indexes
    index_encoding = 0 # 用来记录索引值
    train_gpu = int(conf['leon']['train_gpu'])
    random_tensor = torch.rand((90000, 1000, 27)).to(f'cuda:{train_gpu}')
    
    remote = bool(conf['leon']['remote'])
    pct = float(conf['leon']['pct']) # 执行 percent 比例的 plan
    planning_time = 10000 # pg timout会考虑planning时间
    sql_id = [] # 达到局部最优解的query集合
    pg_time = []
    leon_time_list = []
    for pg_i in range(len(test_files)):
        print(f"------------- sending query {pg_i} ------------")
        query_latency, _ = getPG_latency(test_query[pg_i], ENABLE_LEON=False, timeout_limit=0)
        print("latency pg ", query_latency)
        pg_time.append(query_latency)
    print("pg_time", pg_time)
    print("pg_time sum", sum(pg_time))
    logger.log_metrics({f"Runtime/PG_Time": sum(pg_time)}, step=my_step)
    # ===== ITERATION OF CHUNKS ====
    ch_start_idx = 0 # the start idx of the current chunk in train_files
    while ch_start_idx + chunk_size <= len(train_files):
        print(f"\n+++++++++ a chunk of sql from {ch_start_idx}  ++++++++")
        sqls_chunk = load_sql(list(range(ch_start_idx, ch_start_idx + chunk_size)), training_query=training_query)
        curr_file = train_files[ch_start_idx : ch_start_idx + chunk_size]
        print(train_files[ch_start_idx : ch_start_idx + chunk_size])
        # tf_time = []
        pg_time1 = []
        Nodes = []
        # ++++ PHASE 1. ++++ send a chunk of queries with ENABLE_LEON=True
        # ENABLE_LEON = bool
        
        for q_send_cnt in range(chunk_size):
            print(f"------------- sending query {q_send_cnt} starting from idx {ch_start_idx} ------------")
            query_latency1, _ = getPG_latency(sqls_chunk[q_send_cnt], ENABLE_LEON=False, timeout_limit=0)
            print("latency pg ", query_latency1)
            postgres.getPlans(sqls_chunk[q_send_cnt], None, check_hint_used=False, ENABLE_LEON=True, curr_file=curr_file[q_send_cnt])
            # query_latency2, json_dict = getPG_latency(sqls_chunk[q_send_cnt], ENABLE_LEON=True, timeout_limit=int(conf['leon']['leon_timeout']), curr_file=curr_file[q_send_cnt])
            json_dict = postgres.getPlans(sqls_chunk[q_send_cnt], None, check_hint_used=False, ENABLE_LEON=True, curr_file=curr_file[q_send_cnt])[0][0][0]
            # print("latency leon ", query_latency2)
            node = postgres.ParsePostgresPlanJson(json_dict)
            Nodes.append(node)
            # tf_time.append(query_latency2)
            pg_time1.append(query_latency1)
            # logger.log_metrics({f"Query/{curr_file[q_send_cnt]}pg_latency": query_latency1}, step=my_step)
            # logger.log_metrics({f"Query/{curr_file[q_send_cnt]}leon_latency": query_latency2}, step=my_step)
            # if curr_file[q_send_cnt] not in first_time:
            #     # first_time[curr_file[q_send_cnt]] = query_latency2 # 第一次leontime
            #     first_time[curr_file[q_send_cnt]] = query_latency1 # 第一次pgtime
            # if curr_file[q_send_cnt] not in min_leon_time:
            #     min_leon_time[curr_file[q_send_cnt]] = query_latency2
            # else:
            #     min_leon_time[curr_file[q_send_cnt]] = min(min_leon_time[curr_file[q_send_cnt]], query_latency2)
            # if min_leon_time[curr_file[q_send_cnt]] / first_time[curr_file[q_send_cnt]] < 0.8 and curr_file[q_send_cnt] not in sql_id:
            #     sql_id.append(curr_file[q_send_cnt])
        # logger.log_metrics({f"Runtime/pg_latency": sum(pg_time1)}, step=my_step)
        # logger.log_metrics({f"Runtime/leon_latency": sum(tf_time)}, step=my_step)
        # runtime_pg += sum(pg_time1)
        # runtime_leon += sum(tf_time)
        # logger.log_metrics({f"Runtime/all_pg": runtime_pg}, step=my_step)
        # logger.log_metrics({f"Runtime/all_leon": runtime_leon}, step=my_step)
        
        ###############收集新的等价类##############################  
        # for q_send_cnt in range(chunk_size):
        #     if retrain_count >= 3 and curr_file[q_send_cnt] not in sql_id: # 最好情况好于0.8
        #     # if retrain_count >= 5:
        #         curNode = Nodes[q_send_cnt]
        #         if curNode:
        #             collects(curNode, task_counter, Exp, None, None, sqls_chunk[q_send_cnt], curr_file[q_send_cnt], model)
        exp_key = Exp.GetExpKeys()
        
        for q_send_cnt in range(chunk_size):
            # postgres.getPlans(sqls_chunk[q_send_cnt], None, check_hint_used=False, ENABLE_LEON=True, curr_file=curr_file[q_send_cnt])
            curNode = Nodes[q_send_cnt]
            if curNode:
                curNode.info['sql_str'] = sqls_chunk[q_send_cnt]
                curNode.GetOrParseSql()
                allPlans = [curNode]
                while (allPlans):
                    currentNode = allPlans.pop(0)
                    allPlans.extend(currentNode.children)
                    if currentNode.IsJoin():
                        cur_join_ids = ','.join(
                            sorted([i.split(" AS ")[-1] for i in currentNode.leaf_ids()]))
                        if cur_join_ids in exp_key:
                            currentNode.info['sql_str'] = currentNode.to_sql(curNode.info['parsed_join_conds'], with_select_exprs=True)
                            postgres.getPlans(currentNode.info['sql_str'], None, check_hint_used=False, ENABLE_LEON=True, curr_file=curr_file[q_send_cnt])
                            ray.get(task_counter.WriteOnline.remote(True))
                            postgres.getPlans(currentNode.info['sql_str'], None, check_hint_used=False, ENABLE_LEON=True, curr_file=curr_file[q_send_cnt])
                            ray.get(task_counter.WriteOnline.remote(False))

        
        ##########################################################
        leon_node = []
        for node in Nodes:
            # print(node)
            all_node = []
            all_plan = [node]
            while (all_plan):
                currentNode = all_plan.pop(0)
                all_plan.extend(currentNode.children)
                if currentNode.IsJoin():
                    all_node.append(currentNode)
            for child_node in all_node:
                tbls = [table.split(' ')[-1] for table in child_node.leaf_ids(with_alias=True)]
                eq_temp = ','.join(sorted(tbls))
                # print(eq_temp)
                if eq_temp in Exp.GetEqSet():
                    print(','.join(sorted(tbls)))
                    child_node.info['join_tables'] = ','.join(sorted(tbls))
                    # print(child_node)
                    leon_node.append(child_node)
        

        # ==== ITERATION OF RECIEVED QUERIES IN A CHUNK ====
        q_recieved_cnt = 0 # the # of recieved queries; value in [1, chunk_size]
        curr_QueryId = "" # to imply the difference of QueryId between two messages

        # to ensure that all sent sqls are processed as messages before loading .pkl file
        
        # PKL_READY = ray.get(same_actor.complete_all_tasks.remote())
        completed_tasks = ray.get(same_actor.GetCompletedTasks.remote())
        recieved_task = ray.get(task_counter.GetRecievedTask.remote())
        if completed_tasks == recieved_task:
            PKL_READY = True
        else:
            PKL_READY = False
        while(not PKL_READY):
            time.sleep(0.1)
            print("waiting for PKL_READY ...")
            completed_tasks = ray.get(same_actor.GetCompletedTasks.remote())
            recieved_task = ray.get(task_counter.GetRecievedTask.remote())
            if completed_tasks == recieved_task:
                PKL_READY = True
            else:
                PKL_READY = False
            # PKL_READY = ray.get(same_actor.complete_all_tasks.remote())
        exec_plan = [] # 总的需要执行的计划
        start_time = time.time()
        # ++++ PHASE 2. ++++ get messages of a chunk, nodes, and experience
        PKL_exist = True
        if os.path.exists(message_path):
            with open(message_path, "rb") as file:
                while(PKL_exist):
                    try:
                        message = pickle.load(file)
                    except:
                        PKL_exist = False # the last message is already loaded
                        break

                    # curr_dict = ray.get(dict_actor.get_dict.remote())
                    # curr_sql_id = curr_dict[message[0]['QueryId']]
                    curr_sql_id = message[0]['QueryId']
                    q_recieved_cnt = curr_file.index(curr_sql_id) # start to recieve equal sets from a new sql
                    if curr_QueryId != message[0]['QueryId']: # the QueryId of the first plan in the message
                        print(f"------------- recieving query {q_recieved_cnt} starting from idx {ch_start_idx} ------------")
                        
                        curr_QueryId = message[0]['QueryId']
                    print(f">>> message with {len(message)} plans")
                    # for m in message:
                    #     print(m['Plan']['Total Cost'])
                    
                    # STEP 1) get node
                    if model_type == "Transformer":
                        encoded_plans, attns, queryfeature, nodes = envs.leon_encoding(model_type, message, 
                                                                                       require_nodes=True, workload=workload, 
                                                                                       configs=configs, op_name_to_one_hot=op_name_to_one_hot,
                                                                                       plan_parameters=plan_parameters, feature_statistics=feature_statistics, sql=sqls_chunk[q_recieved_cnt])
                    elif model_type == "TreeConv":
                        encoded_plans, attns, queryfeature, nodes = envs.leon_encoding(model_type, message, 
                                                                                       require_nodes=True, workload=workload, 
                                                                                       queryFeaturizer=queryFeaturizer, nodeFeaturizer=nodeFeaturizer, sql=None)
                    # nodes = PlanToNode(workload, message)
                    lero_feature, _ = feature_generator.transform(message)


                    if nodes is None:
                        continue
                    
                    # plans_lib.GatherUnaryFiltersInfo(nodes)
                    # postgres.EstimateFilterRows(nodes) 
                    # null_nodes = plans_lib.Binarize(nodes)
                    # query_vecs = torch.from_numpy(queryFeaturizer(nodes[0]))
                    # for node in nodes:
                    #     node.info['query_feature'] = query_vecs
                    # if model_type == "Transformer":
                    #     encoded_plans, _, attns, _ = plans_encoding(message) # encoding. plan -> plan_encoding / seqs torch.Size([26, 1, 760])
                    # elif model_type == "TreeConv":
                    #     encoded_plans, attns = encoding.TreeConvFeaturize(nodeFeaturizer, null_nodes)
                    
                    # STEP 2) pick node to execute
                    
                    
                    ##############################拿leon所选的执行计划#################
                    for node2 in leon_node:
                        for i, node1 in enumerate(nodes):
                            temp = node1.info['join_tables'].split(',') # sort
                            temp = ','.join(sorted(temp))
                            if temp == node2.info['join_tables']:
                                if round(node1.cost * 100) / 100.0 == node2.cost:
                                    # Exp.collectRate(node2.info['join_tables'], first_time[curr_file[q_recieved_cnt]], tf_time[q_recieved_cnt], curr_file[q_recieved_cnt])
                                    c_node = node1
                                    
                                    # c_plan = [c_node,
                                    #           encoded_plans[i],
                                    #           attns[i]]
                                    c_plan = [c_node,
                                              None,
                                              None]
                                    if node2.actual_time_ms is not None:
                                        c_plan[0].info['latency'] = node2.actual_time_ms
                                    else:
                                        if c_plan[0].info.get('latency') is None:
                                            if not Exp.isCache(c_node.info['join_tables'], c_plan) and not envs.CurrCache(exec_plan, c_plan):
                                                c_plan[0].info['index'] = index_encoding
                                                plan_channels = copy.deepcopy(plan_channels_init)
                                                get_channels_dfs(c_plan[0], plan_channels)
                                                needed = torch.from_numpy(get_vecs([plan_channels]))
                                                encoding_dict[index_encoding] = (encoded_plans[i], attns[i], needed, lero_feature[i])
                                                index_encoding += 1
                                                exec_plan.append((c_plan,(pg_time1[q_recieved_cnt] * TIME_OUT_Ratio + planning_time), temp, c_node.cost))
                                                
                                            break
                                            # hint_node = plans_lib.FilterScansOrJoins(c_node.Copy())
                                            # c_plan[0].info['latency'], _ = getPG_latency(hint_node.info['sql_str'], hint_node.hint_str(), ENABLE_LEON=False, timeout_limit=(pg_time1[q_recieved_cnt] * 3 + planning_time)) # timeout 10s
                                    if not Exp.isCache(c_node.info['join_tables'], c_plan):
                                        c_plan[0].info['index'] = index_encoding
                                        plan_channels = copy.deepcopy(plan_channels_init)
                                        get_channels_dfs(c_plan[0], plan_channels)
                                        needed = torch.from_numpy(get_vecs([plan_channels]))
                                        encoding_dict[index_encoding] = (encoded_plans[i], attns[i], needed, lero_feature[i])
                                        index_encoding += 1
                                        Exp.AppendExp(c_node.info['join_tables'], c_plan)
                                    else:
                                        Exp.ChangeTime(c_node.info['join_tables'], c_plan)
                                    break
                            else:
                                break
                    ##################################################################
                    
                    costs = torch.tensor(getNodesCost(nodes)).to(DEVICE)
                    # OneNode = nodes[0]
                    # plans_lib.GatherUnaryFiltersInfo(OneNode)
                    # postgres.EstimateFilterRows(OneNode)  
                    # OneQueryFeature = queryFeaturizer(OneNode)
                    # OneQueryFeature = torch.from_numpy(OneQueryFeature).unsqueeze(0)
                    # OneQueryFeature = nodes[0].info['query_feature'].unsqueeze(0)
                    # queryfeature = OneQueryFeature.repeat(encoded_plans.shape[0], 1) 
                    model.model.to(DEVICE)
                    e_pad, a_pad = _batch(encoded_plans, attns)
                    cali_all = get_calibrations(model, queryfeature, e_pad, a_pad)
                    
                    gc.collect()
                    torch.cuda.empty_cache()

                    ucb_idx = get_ucb_idx(cali_all, costs)
                    min_exec_num = min(len(ucb_idx), min_exec_num)
                    num_to_exe = max(min(math.ceil(pct * len(ucb_idx)), max_exec_num), min_exec_num)
                    # TODO: 选择cost 差异大于1.2的plan
                    # costs_index = torch.argsort(costs, descending=False)
                    costs = costs.cpu().numpy()
                    sorted_indices = np.argsort(costs)
                    # Start with the minimum cost index
                    selected_indices = [sorted_indices[0]]
                    # Iterate over the sorted costs
                    for i in range(1, len(sorted_indices)):
                        # If the difference with the last selected cost is greater than 1.2, add it to the selected indices
                        if (costs[sorted_indices[i]] / costs[selected_indices[-1]]) > 1.02:
                            selected_indices.append(sorted_indices[i])
                        if len(selected_indices) == num_to_exe:
                            break
                    if len(selected_indices) < num_to_exe:
                        for i in range(1, len(sorted_indices)):
                            if sorted_indices[i] not in selected_indices:
                                selected_indices.append(sorted_indices[i])
                                if len(selected_indices) == num_to_exe:
                                    break
                    costs_index = selected_indices

                    # print("ucb_idx to exe", ucb_idx[:num_to_exe])
                    # print("num_to_exe", num_to_exe)
                    # STEP 3) execute with ENABLE_LEON=False and add exp
                    # 经验 [[logcost, sql, hint, latency, [query_vector, node], join, joinids], ...]
                    # for i in tqdm(range(num_to_exe)):
                    for i in range(num_to_exe):
                        # if i < 3: # 后续删掉！！这里只是为了节省代码运行时间！！！！
                        node_idx = ucb_idx[i] # 第 i 大ucb 的 node idx
                        cost_index = costs_index[i]
                        a_node = nodes[node_idx]
                        b_node = nodes[cost_index]
                        # (1) add new EqSet key in exp
                        
                        if i == 0: 
                            eqKey = a_node.info['join_tables']
                            Exp.AddEqSet(eqKey, curr_file[q_recieved_cnt])
                            # if Exp.GetEqSet()[eqKey].first_latency == TIME_OUT: # 不pick node后,直接删
                            #     Exp.DeleteOneEqset(eqKey)
                            #     break
                          
                        # (2) add experience of certain EqSet key
                        # a_plan = [a_node, # with sql, hint, latency, cost
                        #           encoded_plans[node_idx],
                        #           attns[node_idx]]
                        # b_plan = [b_node,
                        #           encoded_plans[cost_index],
                        #           attns[cost_index]]
                        a_plan = [a_node, # with sql, hint, latency, cost
                                  None,
                                  None]
                        b_plan = [b_node,
                                  None,
                                  None]
                        # print(i)
                        if not Exp.isCache(eqKey, a_plan) and not envs.CurrCache(exec_plan, a_plan):
                            a_plan[0].info['index'] = index_encoding
                            plan_channels = copy.deepcopy(plan_channels_init)
                            get_channels_dfs(a_plan[0], plan_channels)
                            needed = torch.from_numpy(get_vecs([plan_channels]))
                            encoding_dict[index_encoding] = (encoded_plans[node_idx], attns[node_idx], needed, lero_feature[i])
                            index_encoding += 1
                            exec_plan.append((a_plan, (pg_time1[q_recieved_cnt] * TIME_OUT_Ratio + planning_time), eqKey, a_node.cost))
                            

                        if not Exp.isCache(eqKey, b_plan) and not envs.CurrCache(exec_plan, b_plan):
                            b_plan[0].info['index'] = index_encoding
                            plan_channels = copy.deepcopy(plan_channels_init)
                            get_channels_dfs(b_plan[0], plan_channels)
                            needed = torch.from_numpy(get_vecs([plan_channels]))
                            encoding_dict[index_encoding] = (encoded_plans[cost_index], attns[cost_index], needed, lero_feature[i])
                            index_encoding += 1
                            exec_plan.append((b_plan, (pg_time1[q_recieved_cnt] * TIME_OUT_Ratio + planning_time), eqKey, b_node.cost))
                            
                            
                        # if not Exp.isCache(eqKey, a_plan): # 该行放 get latency 前面 ！！！
                        #     hint_node = plans_lib.FilterScansOrJoins(a_node.Copy())
                        #     # print(hint_node.hint_str(), hint_node.info['sql_str'])
                        # a_plan[0].info['latency'], _ = getPG_latency(hint_node.info['sql_str'], hint_node.hint_str(), ENABLE_LEON=False, timeout_limit=(pg_time1[q_recieved_cnt] * 3 + planning_time)) # timeout 10s(pg_time1[q_recieved_cnt] * 3)
                        #     Exp.AppendExp(eqKey, a_plan)
                        # if not Exp.isCache(eqKey, b_plan): # 该行放 get latency 前面 ！！！
                        #     hint_node = plans_lib.FilterScansOrJoins(b_node.Copy())
                        #     b_plan[0].info['latency'], _ = getPG_latency(hint_node.info['sql_str'], hint_node.hint_str(), ENABLE_LEON=False, timeout_limit=(pg_time1[q_recieved_cnt] * 3 + planning_time)) # timeout 10s
                        #     Exp.AppendExp(eqKey, b_plan)
                    # print("len(Exp.GetExp(eqKey))", len(Exp.GetExp(eqKey)))
        print("Curr_Plan_Len: ", len(exec_plan))
        exec_plan = sorted(exec_plan, key=lambda x: x[0][0].cost, reverse=True)
        if iscache:
            new_exec_plan = []
            for exec_one_plan in exec_plan:
                should_add = True
                cache_list = exp_cache.get(exec_one_plan[2])
                if cache_list:
                    for one_cache in cache_list:
                        if one_cache[0].cost == exec_one_plan[0][0].cost and \
                        one_cache[0].info['sql_str'] == exec_one_plan[0][0].info['sql_str'] and one_cache[0].hint_str() == exec_one_plan[0][0].hint_str():
                            exec_one_plan[0][0].info['latency'] = one_cache[0].info['latency']
                            Exp.AppendExp(exec_one_plan[2], exec_one_plan[0])
                            should_add = False
                            break
                if should_add:
                    new_exec_plan.append(exec_one_plan)
            ## 给索引
            results = pool.map_unordered(actor_call_leon, new_exec_plan)
        else:
            results = pool.map_unordered(actor_call_leon, exec_plan)
        loss_node = 0
        for result in results:
            if result == None:
                loss_node += 1
                # print("*" * 50)
                # print("result is None")
                continue
            Exp.AppendExp(result[1], result[0])
        if loss_node > 0:
            print("loss_node", loss_node)
        end_time = time.time()
        pick_nodes_time = end_time - start_time
        logger.log_metrics({"Time/pick_nodes_time": end_time - start_time}, step=my_step)
        # Pick Node 结束
        del queryfeature, encoded_plans, attns
        gc.collect()
        torch.cuda.empty_cache()

        ##########删除等价类#############
        Exp.DeleteEqSet(sql_id)
        eqset = Exp.GetEqSet()
        print("len_eqset", Exp._getEqNum())
        logger.log_metrics({"len_eqset": Exp._getEqNum()}, step=my_step)
        for eq in eqset:
            print(f"{Exp.GetQueryId(eq)}Eq:{eq},len:{len(Exp.GetExp(eq))},opt_time:{round(Exp.GetEqSet()[eq].opt_time, 2)},eqset_latency:{round(Exp.GetEqSet()[eq].eqset_latency, 2)}")
        # print(eqset)
        
        
        # ++++ PHASE 3. ++++ model training
        # PHASE 3 还有小问题
        
        train_pairs = Exp.Getpair()
        # dnn_pairs = Exp.PreGetpair()
        # TODO: 每个batch只用一个等价类，等价类中的每个plan只推理一次

        logger.log_metrics({"train_pairs": len(train_pairs)}, step=my_step)
        print("len(train_pairs)" ,len(train_pairs))
        # print("len(dnn_pairs)" ,len(dnn_pairs))
        # if len(train_pairs) > 0 and last_train_pair > 0:
        #     if max(len(train_pairs), last_train_pair) / min(len(train_pairs), last_train_pair) < 1.1:
        #         retrain_count += 1
        #     else:
        #         retrain_count = 0
        # last_train_pair = len(train_pairs)
        # print(retrain_count)
        del random_tensor
        torch.cuda.empty_cache()
        if len(train_pairs) > min_batch_size:
            leon_dataset = prepare_dataset(train_pairs, True, nodeFeaturizer, encoding_dict)
            del train_pairs
            gc.collect()
            dataset_size = len(leon_dataset)
            train_size = int(0.8 * dataset_size)
            val_size = dataset_size - train_size
            train_ds, val_ds = torch.utils.data.random_split(leon_dataset, [train_size, val_size])
            def collate_fn(batch):
                # 获取每个batch中的最大长度
                
                # 在collate_fn中对每个batch进行padding
                padded_batch = {}
                for key in batch[0].keys():
                    if key.startswith('lero_feature1'):
                        padded_batch[key] = [item[key] for item in batch]
                    elif key.startswith('lero_feature2'):
                        padded_batch[key] = [item[key] for item in batch]
                    else:
                        padded_batch[key] = torch.stack([item[key] for item in batch])

                return padded_batch
            dataloader_train = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=7, collate_fn=collate_fn)
            dataloader_val = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=7, collate_fn=collate_fn)
            lero_model.optimizer_state_dict = dnn_prev_optimizer_state_dict

            trainer = pl.Trainer(accelerator="gpu",
                                devices=[train_gpu],
                                enable_progress_bar=False,
                                max_epochs=100,
                                callbacks=[plc.EarlyStopping(
                                            monitor='lero_val_acc',
                                            mode='max',
                                            patience=3,
                                            min_delta=0.001,
                                            check_on_train_epoch_end=False,
                                            verbose=True
                                        )],
                                logger=logger)
            start_time = time.time()
            trainer.fit(lero_model, dataloader_train, dataloader_val)
            end_time = time.time()
            dnn_train_time = end_time - start_time
            del leon_dataset, train_ds, val_ds, dataloader_train, dataloader_val

        # if len(train_pairs) > min_batch_size:
        #     leon_dataset = prepare_dataset(train_pairs, True, nodeFeaturizer, encoding_dict)
        #     del train_pairs
        #     gc.collect()
        #     dataset_size = len(leon_dataset)
        #     train_size = int(0.8 * dataset_size)
        #     val_size = dataset_size - train_size
        #     train_ds, val_ds = torch.utils.data.random_split(leon_dataset, [train_size, val_size])
        #     dataloader_train = DataLoader(train_ds, batch_size=1024, shuffle=True, num_workers=7)
        #     dataloader_val = DataLoader(val_ds, batch_size=1024, shuffle=False, num_workers=7)
        #     # dataset_test = BucketDataset(Exp.OnlyGetExp(), keys=Exp.GetExpKeys(), nodeFeaturizer=nodeFeaturizer, dict=encoding_dict)
        #     # batch_sampler = BucketBatchSampler(dataset_test.buckets, batch_size=1)
        #     # dataloader_test = DataLoader(dataset_test, batch_sampler=batch_sampler, num_workers=7)
        #     # model = load_model(model_path, prev_optimizer_state_dict).to(DEVICE)
        #     model.optimizer_state_dict = prev_optimizer_state_dict
        #     callbacks = load_callbacks(logger=None)
        #     # del random_tensor
        #     torch.cuda.empty_cache()
        #     trainer = pl.Trainer(accelerator="gpu",
        #                         devices=[train_gpu],
        #                         enable_progress_bar=False,
        #                         max_epochs=100,
        #                         callbacks=callbacks,
        #                         logger=logger)
        #     start_time = time.time()
        #     trainer.fit(model, dataloader_train, dataloader_val)
        #     end_time = time.time()
        #     leon_train_time = end_time - start_time
        #     # trainer.test(model, dataloader_test)
        #     model.eq_summary = {key: 0 for key in eqset}
        #     prev_optimizer_state_dict = trainer.optimizers[0].state_dict()
        #     del leon_dataset, train_ds, val_ds, dataloader_train, dataloader_val #, dataset_test, batch_sampler, dataloader_test
        gc.collect()
        torch.cuda.empty_cache()
        random_tensor = torch.rand((90000, 1000, 27)).to(f'cuda:{train_gpu}')

        print("*"*20)
        print("Current Accuracy For Each EqSet: ", model.eq_summary)
        print("*"*20)
        ch_start_idx = ch_start_idx + chunk_size
        # save model
        torch.save(dnn_model.model, "./log/dnn_model.pth")
        torch.save(model.model, model_path)
        torch.save(lero_model.model, "./log/lero_model.pth")
        ray.get(task_counter.reload_model.remote(eqset.keys(), model.eq_summary))
        # 模型更新需要通知到server.py
        # clear pkl
        if os.path.exists(message_path):
            os.remove(message_path)
            print(f"Successfully remove {message_path}")
        else:
            print(f"Fail to remove {message_path}")

        try:
            train_time = dnn_train_time
        except:
            train_time = 0
        all_train_time = all_train_time + train_time + pick_nodes_time
        logger.log_metrics({"Time/train_time": train_time}, step=my_step)
        logger.log_metrics({"Time/all_train_time": all_train_time}, step=my_step)
        leon_time = []
        for pg_i in range(len(test_files)):
            print(f"------------- sending query {pg_i} ------------")
            postgres.getPlans(test_query[pg_i], None, check_hint_used=False, ENABLE_LEON=True, curr_file=test_files[pg_i])
            query_latency, _ = getPG_latency(test_query[pg_i], ENABLE_LEON=True, timeout_limit=int(conf['leon']['leon_timeout']), curr_file=test_files[pg_i])
            print("latency pg ", query_latency)
            leon_time.append(query_latency)
        leon_time_list.append(leon_time)
        print("leon_time", sum(leon_time))
        logger.log_metrics({f"Time/Leon_Time": sum(leon_time)}, step=my_step)
        my_step += 1
        # with open("./log/exp_v5.pkl", 'wb') as f:
        #     pickle.dump(Exp.OnlyGetExp(), f) 
    min_time = 900000
    min_gmrl = 900000
    min_leon_time = []
    for leon_time in leon_time_list:
        if sum(leon_time) < min_time:
            min_time = sum(leon_time)
            min_leon_time = leon_time
        temp = np.power(np.prod(np.array(leon_time) / np.array(pg_time)), 1/len(leon_time))
        if temp < min_gmrl:
            min_gmrl = temp
    print("min_gmrl", min_gmrl)
    print("min_leon_time", min_leon_time)
    min_values = np.min(leon_time_list, axis=0)
    print("min_time", min_time)
        
        



    
