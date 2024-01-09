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
            model = treeconv.TreeConvolution(820, 54, 1).to(DEVICE)
        torch.save(model, model_path)
    else:
        model = torch.load(model_path, map_location=DEVICE).to(DEVICE)
    model = PL_Leon(model, prev_optimizer_state_dict)
    
    return model

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
    # cost_t = torch.mul(cali_mean, costs) # 计算 calibration mean * pg返回的cost [# of plan]

    # cali_min, _ = cost_t.min(dim = 0) # plan 的 cali 最小值 [# of plan] 【cali_min只有一个值？】
    # print("cost_min.shape(): ", cali_min.shape)
    # print(cali_min)            
    # ucb = cali_var / cali_var.max() - cali_min / cali_min.max()
    ucb = cali_var / cali_var.max()
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
    join_ids_to_append = join_ids_to_append[:1]
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
        patience=3,
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

    file_path1 = "./log/messages.pkl"

    # 检查文件是否存在
    if os.path.exists(file_path1):
        # 删除文件
        os.remove(file_path1)
        print(f"File {file_path1} has been successfully deleted.")
    else:
        print(f"File {file_path1} does not exist.")
    pretrain = False
    ports = [1120, 1125, 1130, 1135]
    if pretrain:
        checkpoint = torch.load("./log/SimModel.pth", map_location=DEVICE)
        torch.save(checkpoint, "./log/model.pth")
        print("load SimModel success")

    with open ("./conf/namespace.txt", "r") as file:
        namespace = file.read().replace('\n', '')
    with open ("./conf/ray_address.txt", "r") as file:
        ray_address = file.read().replace('\n', '')

    
    train_files, training_query = envs.load_train_files(conf['leon']['workload_type'])
    # ray.get(dict_actor.write_sql_id.remote(train_files))
    chunk_size = 1 # the # of sqls in a chunk
    min_batch_size = 1
    model_path = "./log/model.pth" 
    message_path = "./log/messages.pkl"
    prev_optimizer_state_dict = None
    model = load_model(model_path)
    
    Exp = Experience(eq_set=initEqSet())
    print("Init workload and equal set keys")
    
    workload = envs.wordload_init(conf['leon']['workload_type'])
    queryFeaturizer = plans_lib.QueryFeaturizer(workload.workload_info)
    if model_type == "Transformer":
        statistics_file_path = "./statistics.json"
        feature_statistics = load_json(statistics_file_path)
        add_numerical_scalers(feature_statistics)
        op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    elif model_type == "TreeConv":
        nodeFeaturizer = plans_lib.TreeNodeFeaturizer_V2(workload.workload_info)
    first_time = dict()
    last_train_pair = 0
    retrain_count = 3
    min_leon_time = dict()
    max_query_latency1 = 0
    my_step = 0
    runtime_pg = 0
    runtime_leon = 0
    max_exec_num = 50
    encoding_dict = dict() # 用来存trees和indexes
    index_encoding = 0 # 用来记录索引值
    train_gpu = int(conf['leon']['train_gpu'])
    random_tensor = torch.rand((90000, 1000, 27)).to(f'cuda:{train_gpu}')
    
    with open ("./conf/namespace.txt", "r") as file:
        namespace = file.read().replace('\n', '')
    with open ("./conf/ray_address.txt", "r") as file:
        ray_address = file.read().replace('\n', '')
    context = ray.init(address=ray_address, namespace=namespace, _temp_dir=conf['leon']['ray_path'] + "/log/ray") # init only once
    same_actor = ray.get_actor('leon_server')
    task_counter = ray.get_actor('counter')
    remote = bool(conf['leon']['remote'])
    pct = float(conf['leon']['pct']) # 执行 percent 比例的 plan
    planning_time = 3000 # pg timout会考虑planning时间
    sql_id = [] # 达到局部最优解的query集合
    # ===== ITERATION OF CHUNKS ====
    ch_start_idx = 0 # the start idx of the current chunk in train_files
    while ch_start_idx + chunk_size <= len(train_files):
        print(f"\n+++++++++ a chunk of sql from {ch_start_idx}  ++++++++")
        sqls_chunk = load_sql(list(range(ch_start_idx, ch_start_idx + chunk_size)), training_query=training_query)
        curr_file = train_files[ch_start_idx : ch_start_idx + chunk_size]
        print(train_files[ch_start_idx : ch_start_idx + chunk_size])
        time_ratio = []
        tf_time = []
        pg_time1 = []
        Nodes = []
        # ++++ PHASE 1. ++++ send a chunk of queries with ENABLE_LEON=True
        # ENABLE_LEON = bool
        
        for q_send_cnt in range(chunk_size):
            print(f"------------- sending query {q_send_cnt} starting from idx {ch_start_idx} ------------")
            ray.get(task_counter.WriteOnline.remote(True))
            json_dict = postgres.getPlans(sqls_chunk[q_send_cnt], None, check_hint_used=False, ENABLE_LEON=True, curr_file=curr_file[q_send_cnt])[0][0][0]
            ray.get(task_counter.WriteOnline.remote(False))
            

            
            

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


                    node = PlanToNode(workload, [message[0]])
                    if node is None:
                        continue

                    # 定义文件夹名称和路径
                    folder_name = "my_job"
                    folder_path = os.path.join(os.getcwd(), folder_name)

                    # 创建文件夹
                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)

                    # 模拟一些数据，你需要替换这部分为你的实际数据


                    # 构造文件名和文件内容
                    file_name = f"{curr_file[q_send_cnt]}.sql"
                    file_content = node[0].info['sql_str']

                    # 完整文件路径
                    file_path = os.path.join(folder_path, file_name)

                    # 创建并写入文件
                    with open(file_path, 'w') as file:
                        file.write(file_content)

                    print(f"文件已创建：{file_path}")
        ch_start_idx += chunk_size
        if os.path.exists(message_path):
            os.remove(message_path)
            print(f"Successfully remove {message_path}")
        else:
            print(f"Fail to remove {message_path}")

                    