from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc
import torch.nn.functional as F
import torch.nn as nn
import os
from util import postgres
from util import pg_executor
from util import postgres
from util import envs
from util import plans_lib
import pytorch_lightning.loggers as pl_loggers
import pickle
import math
import json
from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers
from leon_experience import Experience
import numpy as np
import ray
import time
from argparse import ArgumentParser
import copy
import wandb
DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
Transformer_model = SeqFormer(
                        input_dim=configs['node_length'],
                        hidden_dim=128,
                        output_dim=1,
                        mlp_activation="ReLU",
                        transformer_activation="gelu",
                        mlp_dropout=0.3,
                        transformer_dropout=0.2,
                    )

def load_sql(file_list: list):
    """
    :param file_list: list of query file in str
    :return: list of sql query string
    """
    sqls = []
    for file_str in file_list:
        sqlFile = '/data1/wyz/online/LEONForPostgres/join-order-benchmark/' + file_str + '.sql'
        if not os.path.exists(sqlFile):
            raise IOError("File Not Exists!")
        with open(sqlFile, 'r') as f:
            data = f.read().splitlines()
            sql = ' '.join(data)
        sqls.append(sql)
        f.close()
    return sqls

def load_model(model_path: str, prev_optimizer_state_dict=None):
    if not os.path.exists(model_path):
        model = Transformer_model.to(DEVICE)
        model = PL_Leon(model, prev_optimizer_state_dict)
    else:
        model = torch.load(model_path).to(DEVICE)
        model = PL_Leon(model, prev_optimizer_state_dict)
    
    return model

def getPG_latency(query, hint=None, ENABLE_LEON=False, timeout_limit=None):
    """
    input. a loaded query
    output. the average latency of a query get from pg
    """
    latency_sum = 0
    if timeout_limit is None:
        timeout_limit = 90000 # 90000
    cnt = 1 # 3 1 
    for c in range(cnt):
        latency, json_dict = postgres.GetLatencyFromPg(query, hint, ENABLE_LEON, verbose=False, check_hint_used=False, timeout=timeout_limit, dropbuffer=False)
        latency_sum = latency_sum + latency
    if latency_sum == timeout_limit:
        latency_sum = 90000
    pg_latency = latency_sum / cnt
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

def get_calibrations(model, seqs, attns, IF_TRAIN):
    seqs = seqs.to(DEVICE)
    attns = attns.to(DEVICE)
    if IF_TRAIN:
        cost_iter = 10 # training 用模型推理 10 次 获得模型不确定性
        model.model.eval()
        with torch.no_grad():
            for i in range(cost_iter):
                cali = model.model(seqs, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
                if i == 0:
                    cali_all = cali[:, 0].unsqueeze(1) # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
                else:
                    cali_all = torch.cat((cali_all, cali[:, 0].unsqueeze(1)), dim=1)
        return cali_all

    else: 
        cali = model.model(seqs, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
        cali_all = cali[:, 0]
        return cali_all

def get_ucb_idx(cali_all, pgcosts):    
    cali_mean = cali_all.mean(dim = 1) # 每个 plan 的 cali 均值 [# of plan] | 值为 1 左右
    cali_var = cali_all.var(dim = 1) # [# of plan] | 值为 0 左右
    costs = pgcosts # [# of plan]
    cost_t = torch.mul(cali_mean, costs) # 计算 calibration mean * pg返回的cost [# of plan]

    # cali_min, _ = cost_t.min(dim = 0) # plan 的 cali 最小值 [# of plan] 【cali_min只有一个值？】
    # print("cost_min.shape(): ", cali_min.shape)
    # print(cali_min)            
    # ucb = cali_var / cali_var.max() - cali_min / cali_min.max()

    ucb = cali_var / cali_var.max() - cost_t / cost_t.max() # [# of plan]
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
    # equ_tem = ['cn,mc', 'ct,mc', 'ct,mc,t', 'ci,cn,mc,t', 'cn,mc,t']
    # equ_tem = ['t,mk,k', 'k,t,cct2,cc,mc', 'k,t,cct1,cc,mc', 'kt,t,cct2,cc,mc,cn', 'kt,t,cct1,cc,mc,cn',
    #            'mc,cn', 'mc,cn,t','mc,cn,t,at', 'cn,mc,t,ci', 'n,an', 'n,an,ci', 'it,mi', 'ct,mc',
    #            'ct,mc,t', 'ct,mc,t,mi', 'mc,cn', 'k,mk', 'k,mk,mi_idx', 'k,mk,miidx', 'a1,n1', 'a1,n1,ci']
    equ_tem = ['an,chn', 'an,ci', 'an,cn', 'an,mc', 'an,n', 'an,rt', 'an,t', 'chn,ci', 'chn,cn', 'chn,mc',
               'chn,n', 'chn,rt', 'chn,t', 'ci,cn', 'ci,mc', 'ci,n', 'ci,rt', 'ci,t', 'cn,mc', 'cn,n',
               'cn,rt', 'cn,t', 'mc,n', 'mc,rt', 'mc,t', 'n,rt', 'n,t', 'rt,t']
    equ_set = set() # 用集合 方便 eq keys 中去重
    # 'title,movie_keyword,keyword' -> 'keyword,movie_keyword,title'
    for i in equ_tem:
        e_tem = i.split(',')
        e_tem = ','.join(sorted(e_tem))
        equ_set.add(e_tem)

    return equ_set

# def collects(finnode: plans_lib.Node, workload, exp: Experience, timeout, currTotalLatency, sql):
#     join_ids_to_append = []
#     allPlans = [finnode]
#     while (allPlans):
#         currentNode = allPlans.pop(0)
#         allPlans.extend(currentNode.children)
#         if currentNode.IsJoin() and (currentNode.actual_time_ms > 0.6 * currTotalLatency):
#             cur_join_ids = ','.join(
#                 sorted([i.split(" AS ")[-1] for i in currentNode.leaf_ids()]))
#             join_ids_to_append.append(cur_join_ids)

#     first_join_ids = join_ids_to_append.pop(0)
#     print('degradation collect:', first_join_ids)
#     exp.AddEqSet(first_join_ids)
#     if join_ids_to_append:
#         last_join_ids = join_ids_to_append.pop(-1)
#         print('degradation collect:', last_join_ids)
#         exp.AddEqSet(last_join_ids)

def collects(finnode: plans_lib.Node, workload, exp: Experience, timeout, currTotalLatency, sql):
    join_ids_to_append = []
    allPlans = [finnode]
    while (allPlans):
        currentNode = allPlans.pop(0)
        allPlans.extend(currentNode.children)
        if currentNode.IsJoin():
            cur_join_ids = ','.join(
                sorted([i.split(" AS ")[-1] for i in currentNode.leaf_ids()]))
            join_ids_to_append.append(cur_join_ids)

    for join_id in reversed(join_ids_to_append):
        exp_key = Exp.GetExpKeys()
        temp = join_id.split(',') # sort
        join_id = ','.join(sorted(temp))
        if join_id not in exp_key:
            print('degradation collect:', join_id)
            exp.AddEqSet(join_id)
            break


# create dataset
class LeonDataset(Dataset):
    def __init__(self, labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2):
        self.labels = labels
        self.costs1 = costs1
        self.costs2 = costs2
        self.encoded_plans1 = encoded_plans1
        self.encoded_plans2 = encoded_plans2
        self.attns1 = attns1
        self.attns2 = attns2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.labels[idx],
                self.costs1[idx],
                self.costs2[idx],
                self.encoded_plans1[idx],
                self.encoded_plans2[idx],
                self.attns1[idx],
                self.attns2[idx])
    
def prepare_dataset(pairs):
    labels = []
    costs1 = []
    costs2 = []
    encoded_plans1 = []
    encoded_plans2 = []
    attns1 = []
    attns2 = []
    for pair in pairs:
        if pair[0][0].info['latency'] > pair[1][0].info['latency']:
            label = 0
        else:
            label = 1
        labels.append(label)
        costs1.append(pair[0][0].cost)
        costs2.append(pair[1][0].cost)
        encoded_plans1.append(pair[0][1])
        encoded_plans2.append(pair[1][1])
        attns1.append(pair[0][2])
        attns2.append(pair[1][2])
    labels = torch.tensor(labels)
    costs1 = torch.tensor(costs1)
    costs2 = torch.tensor(costs2)
    encoded_plans1 = torch.stack(encoded_plans1)
    encoded_plans2 = torch.stack(encoded_plans2)
    attns1 = torch.stack(attns1)
    attns2 = torch.stack(attns2)
    dataset = LeonDataset(labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2)
    return dataset



def load_callbacks(logger):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='v_acc',
        mode='max',
        patience=3,
        min_delta=0.001
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

class PL_Leon(pl.LightningModule):
    def __init__(self, model, optimizer_state_dict=None, learning_rate=0.001):
        super(PL_Leon, self).__init__()
        self.model = model
        self.optimizer_state_dict = optimizer_state_dict
        self.learning_rate = 0.001


    def forward(self, batch_pairs):
        pass

    def getBatchPairsLoss(self, labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2):
        """
        batch_pairs: a batch of train pairs
        return. a batch of loss
        """
        loss_fn = nn.BCELoss()
        # step 1. retrieve encoded_plans and attns from pairs


        # step 2. calculate batch_cali and calied_cost
        # 0是前比后大 1是后比前大
        batsize = costs1.shape[0]
        encoded_plans = torch.cat((encoded_plans1, encoded_plans2), dim=0)
        attns = torch.cat((attns1, attns2), dim=0)
        cali = self.model(encoded_plans, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
        cali = cali[:, 0]
        costs = torch.cat((costs1, costs2), dim=0)
        # print(costs1)
        # print(costs2)
        # print(labels)
        # print(cali)
        calied_cost = torch.log(costs) * cali
        try:
            sigmoid = F.sigmoid(-(calied_cost[:batsize] - calied_cost[batsize:]))
            loss = loss_fn(sigmoid, labels.float())
        except:
            print(calied_cost, sigmoid)
        # print(loss)
        with torch.no_grad():
            prediction = torch.round(sigmoid)
            # print(prediction)
            accuracy = torch.sum(prediction == labels).item() / len(labels)
        # print(softm[:, 1].shape, labels.shape)
        
        
        return loss, accuracy


    def training_step(self, batch):
        labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2 = batch
        loss, acc  = self.getBatchPairsLoss(labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2)
        self.log_dict({'t_loss': loss, 't_acc': acc}, on_epoch=True)
        return loss

    def validation_step(self, batch):
        labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2 = batch
        loss, acc  = self.getBatchPairsLoss(labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2)
        self.log_dict({'v_loss': loss, 'v_acc': acc}, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        if self.optimizer_state_dict is not None:
            # Checks the params are the same.
            # 'params': [139581476513104, ...]
            curr = optimizer.state_dict()['param_groups'][0]['params']
            prev = self.optimizer_state_dict['param_groups'][0]['params']
            assert curr == prev, (curr, prev)
            # print('Loading last iter\'s optimizer state.')
            # Prev optimizer state's LR may be stale.
            optimizer.load_state_dict(self.optimizer_state_dict)
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            assert optimizer.state_dict(
            )['param_groups'][0]['lr'] == self.learning_rate
            # print('LR', self.learning_rate)
        return optimizer

if __name__ == '__main__':
    # train_files = ['1a', '2a', '3a', '4a']
    train_files = ['9c'] * 50
    chunk_size = 1 # the # of sqls in a chunk
    IF_TRAIN = True
    model_path = "./log/model.pth"
    message_path = "./log/messages.pkl"

    ray.init(address='auto', namespace='server_namespace', _temp_dir="/data1/wyz/online/LEONForPostgres/log/ray") # init only once
    Exp = Experience(eq_set=initEqSet())
    print("Init workload and equal set keys")
    workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
    workload.workload_info.alias_to_names = postgres.GetAllAliasToNames(workload.workload_info.rel_ids)
    statistics_file_path = "/data1/wyz/online/LEONForPostgres/statistics.json"
    feature_statistics = load_json(statistics_file_path)
    add_numerical_scalers(feature_statistics)
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    pg_time_flag = True
    pg_time = 0
    last_train_pair = 0
    retrain_count = 0
    max_query_latency1 = 0
    logger =  pl_loggers.WandbLogger(save_dir=os.getcwd() + '/logs', name="10a", project='leon')
    
    prev_optimizer_state_dict = None
    # ===== ITERATION OF CHUNKS ====
    ch_start_idx = 0 # the start idx of the current chunk in train_files
    while ch_start_idx + chunk_size <= len(train_files):
        print(f"\n+++++++++ a chunk of sql from {ch_start_idx}  ++++++++")
        sqls_chunk = load_sql(train_files[ch_start_idx : ch_start_idx + chunk_size])
        print(train_files[ch_start_idx : ch_start_idx + chunk_size])
        time_ratio = []
        tf_time = []
        pg_time1 = []
        Nodes = []

        # ++++ PHASE 1. ++++ send a chunk of queries with ENABLE_LEON=True
        # ENABLE_LEON = bool
        for q_send_cnt in range(chunk_size):
            print(f"------------- sending query {q_send_cnt} starting from idx {ch_start_idx} ------------")
            query_latency1, _ = getPG_latency(sqls_chunk[q_send_cnt], ENABLE_LEON=False, timeout_limit=0)
            print("latency pg ", query_latency1)
            query_latency2, json_dict = getPG_latency(sqls_chunk[q_send_cnt], ENABLE_LEON=True, timeout_limit=0)
            # todo : 如果timeout执行explain拿json
            print("latency leon ", query_latency2)
            node = postgres.ParsePostgresPlanJson(json_dict)
            max_query_latency1 = max(max_query_latency1, query_latency1)
            Nodes.append(node)
            time_ratio.append(query_latency2 / query_latency1)
            tf_time.append(query_latency2)
            pg_time1.append(query_latency1)
            logger.log_metrics({f"{train_files[q_send_cnt]}pg_latency": query_latency1})
            logger.log_metrics({f"{train_files[q_send_cnt]}leon_latency": query_latency2})
        if pg_time_flag:
            pg_time_flag = False
            pg_time = max_query_latency1 * 3
        ###############收集新的等价类##############################
        # for q_send_cnt in range(chunk_size):
        #     if time_ratio[q_send_cnt] > 0.75 and retrain_count >= 3: # and tf_time[q_send_cnt] > 1000
        #     # if retrain_count >= 5:
        #         curNode = Nodes[q_send_cnt]
        #         if curNode:
        #             collects(curNode, workload, Exp, None, tf_time[q_send_cnt], sqls_chunk[q_send_cnt])
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
                    print(child_node)
                    leon_node.append(child_node)
        

        # ==== ITERATION OF RECIEVED QUERIES IN A CHUNK ====
        q_recieved_cnt = 0 # the # of recieved queries; value in [1, chunk_size]
        curr_QueryId = "" # to imply the difference of QueryId between two messages
        model = load_model(model_path)
        


        # to ensure that all sent sqls are processed as messages before loading .pkl file
        same_actor = ray.get_actor('leon_server')
        PKL_READY = ray.get(same_actor.complete_all_tasks.remote())
        while(not PKL_READY):
            time.sleep(0.1)
            print("waiting for PKL_READY ...")
            PKL_READY = ray.get(same_actor.complete_all_tasks.remote())
        
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

                    if curr_QueryId != message[0]['QueryId']: # the QueryId of the first plan in the message
                        print(f"------------- recieving query {q_recieved_cnt} starting from idx {ch_start_idx} ------------")
                        q_recieved_cnt = q_recieved_cnt + 1 # start to recieve equal sets from a new sql
                        curr_QueryId = message[0]['QueryId']
                    print(f">>> message with {len(message)} plans")

                    # STEP 1) get node
                    nodes = PlanToNode(workload, message)
                    encoded_plans, _, attns, _ = plans_encoding(message) # encoding. plan -> plan_encoding / seqs torch.Size([26, 1, 760])

                    # STEP 2) pick node to execute
                    
                    
                    ##############################拿leon所选的执行计划#################
                    for node2 in leon_node:
                        for i, node1 in enumerate(nodes):
                            temp = node1.info['join_tables'].split(',') # sort
                            temp = ','.join(sorted(temp))
                            if temp == node2.info['join_tables']:
                                if round(node1.cost,2) == node2.cost:
                                    c_node = node1
                                    Exp.AddEqSet(c_node.info['join_tables'])
                                    c_plan = [c_node,
                                              encoded_plans[i],
                                              attns[i]]
                                    c_plan[0].info['latency'] = node2.actual_time_ms
                                    # print('actual_time_ms', node2.actual_time_ms)
                                    # hint_node = plans_lib.FilterScansOrJoins(c_node.Copy())
                                    # hint_node.info['latency'], _ = getPG_latency(hint_node.info['sql_str'], hint_node.hint_str(), ENABLE_LEON=False, timeout_limit=pg_time) # timeout 10s
                                    # print('hint_time', hint_node.info['latency'])
                                    if not Exp.isCache(c_node.info['join_tables'], c_plan):
                                        Exp.AppendExp(c_node.info['join_tables'], c_plan)
                                    else:
                                        Exp.ChangeTime(c_node.info['join_tables'], c_plan)
                                    break
                            else:
                                break
                    ##################################################################
                    pct = 0.1 # 执行 percent 比例的 plan
                    costs = torch.tensor(getNodesCost(nodes)).to(DEVICE)
                    cali_all = get_calibrations(model, encoded_plans, attns, IF_TRAIN)
                    ucb_idx = get_ucb_idx(cali_all, costs)
                    costs_index = torch.argsort(costs, descending=False)
                    num_to_exe = math.ceil(pct * len(ucb_idx))

                    # print("ucb_idx to exe", ucb_idx[:num_to_exe])
                    # print("num_to_exe", num_to_exe)
                    # STEP 3) execute with ENABLE_LEON=False and add exp
                    # 经验 [[logcost, sql, hint, latency, [query_vector, node], join, joinids], ...]
                    for i in range(num_to_exe):
                        # if i < 3: # 后续删掉！！这里只是为了节省代码运行时间！！！！
                        node_idx = ucb_idx[i] # 第 i 大ucb 的 node idx
                        cost_index = costs_index[i]
                        a_node = nodes[node_idx]
                        b_node = nodes[cost_index]
                        # (1) add new EqSet key in exp
                        if i == 0:
                            eqKey = a_node.info['join_tables']
                            Exp.collectRate(eqKey, pg_time1[q_recieved_cnt - 1], tf_time[q_recieved_cnt - 1])
                            # Exp.AddEqSet(eqKey)

                                                        
                        # (2) add experience of certain EqSet key
                        a_plan = [a_node, # with sql, hint, latency, cost
                                  encoded_plans[node_idx],
                                  attns[node_idx]]
                        b_plan = [b_node,
                                  encoded_plans[cost_index],
                                  attns[cost_index]]
                        
                        if not Exp.isCache(eqKey, a_plan): # 该行放 get latency 前面 ！！！
                            hint_node = plans_lib.FilterScansOrJoins(a_node.Copy())
                            # print(hint_node.info['sql_str'], hint_node.hint_str())
                            a_plan[0].info['latency'], _ = getPG_latency(hint_node.info['sql_str'], hint_node.hint_str(), ENABLE_LEON=False, timeout_limit=(pg_time1[q_recieved_cnt - 1] * 3)) # timeout 10s
                            Exp.AppendExp(eqKey, a_plan)
                        if not Exp.isCache(eqKey, b_plan): # 该行放 get latency 前面 ！！！
                            hint_node = plans_lib.FilterScansOrJoins(b_node.Copy())
                            b_plan[0].info['latency'], _ = getPG_latency(hint_node.info['sql_str'], hint_node.hint_str(), ENABLE_LEON=False, timeout_limit=(pg_time1[q_recieved_cnt - 1] * 3)) # timeout 10s
                            Exp.AppendExp(eqKey, b_plan)
                        

                    # print("len(Exp.GetExp(eqKey))", len(Exp.GetExp(eqKey)))
        
        ##########删除等价类#############
        # Exp.DeleteEqSet()
        eqset = Exp.GetEqSet()
        print("len_eqset", Exp._getEqNum())
        logger.log_metrics({"len_eqset": Exp._getEqNum()})
        for eq in eqset:
            print(f"{eq}: {len(Exp.GetExp(eq))}")
        # print(eqset)
        
        # ++++ PHASE 3. ++++ model training
        # PHASE 3 还有小问题
        train_pairs = Exp.Getpair()
        logger.log_metrics({"train_pairs": len(train_pairs)})
        print("len(train_pairs)" ,len(train_pairs))

        if len(train_pairs) > 0 and last_train_pair > 0:
            if max(len(train_pairs), last_train_pair) / min(len(train_pairs), last_train_pair) < 1.1:
                retrain_count += 1
            else:
                retrain_count = 0
        last_train_pair = len(train_pairs)
        print(retrain_count)
        if len(train_pairs) > 64:
            leon_dataset = prepare_dataset(train_pairs)
            dataloader_train = DataLoader(leon_dataset, batch_size=64, shuffle=True, num_workers=0)
            dataloader_val = DataLoader(leon_dataset, batch_size=64, shuffle=False, num_workers=0)
            model = load_model(model_path, prev_optimizer_state_dict).to(DEVICE)
            callbacks = load_callbacks(logger=None)
            trainer = pl.Trainer(accelerator="gpu",
                                devices=[3],
                                max_epochs=100,
                                callbacks=callbacks,
                                logger=logger)
            trainer.fit(model, dataloader_train, dataloader_val)
            prev_optimizer_state_dict = trainer.optimizers[0].state_dict()

        ch_start_idx = ch_start_idx + chunk_size
        # save model
        torch.save(model.model, model_path)
        ray.get(same_actor.reload_model.remote(eqset))
        # 模型更新需要通知到server.py
        # clear pkl
        if os.path.exists(message_path):
            os.remove(message_path)
            print(f"Successfully remove {message_path}")
        else:
            print(f"Fail to remove {message_path}")
        



    
