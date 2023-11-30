import os
from util import postgres
from util import pg_executor
from util import postgres
from util import envs
from util import plans_lib
import pickle
import torch
from torch import nn
import math
import json
from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers
from leon_experience import Experience
import numpy as np
import ray
import time

"""
nodes 是一条query 的所有 plans, 需要用到 server.py 中的解析

数据路径
leon train 提交一条 query 
-> PG 
-> leon server 做校验
-> PG 根据校验选择 dp 中最优的 path 执行
-> 获得执行 feedback (pg 返回的一条query的执行时间)

train pair
[query_encoding, latency, cost]

"""

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
        sqlFile = '/data1/zengximu/LEON-research/LEON/join-order-benchmark/' + file_str + '.sql'
        if not os.path.exists(sqlFile):
            raise IOError("File Not Exists!")
        with open(sqlFile, 'r') as f:
            data = f.read().splitlines()
            sql = ' '.join(data)
        sqls.append(sql)
        f.close()
    return sqls

def load_model(model_path: str):
    if not os.path.exists(model_path):
        model = Transformer_model
    else:
        model = torch.load(model_path)
    
    return model

def getPG_latency(query, hint=None, ENABLE_LEON=False):
    """
    input. a loaded query
    output. the average latency of a query get from pg
    """
    latency_sum = 0
    timeout_limit = 1000 # 90000
    cnt = 1 # 3 1 
    for c in range(cnt):
        latency_sum = latency_sum + postgres.GetLatencyFromPg(query, hint, ENABLE_LEON, verbose=False, check_hint_used=False, timeout=timeout_limit,
                                                dropbuffer=False)
    pg_latency = latency_sum / cnt
    return pg_latency

# UNFINISHED hint sql -> pgcosts
def plans_encoding(plans, IF_TRAIN):
    '''
    input. a list of plans in type of json
    output. (seq_encoding, run_times, attention_mask, loss_mask)
        - seq_encoding torch.Size([1, 760])
    '''
    statistics_file_path = "/data1/zengximu/LEON-research/LEONForPostgres/statistics.json"
    feature_statistics = load_json(statistics_file_path)
    add_numerical_scalers(feature_statistics)
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)

    seqs = []
    attns = []
    for x in plans:        
        seq_encoding, run_times, attention_mask, loss_mask, database_id = get_plan_encoding(
            x, configs, op_name_to_one_hot, plan_parameters, feature_statistics, IF_TRAIN
        ) # run_times 无法获取
        seqs.append(seq_encoding) 
        attns.append(attention_mask)

    seqs = torch.stack(seqs, dim=0)
    attns = torch.stack(attns, dim=0)

    if IF_TRAIN:
        return seqs, run_times, attns, loss_mask
    else:
        return seqs, None, attns, None, None

def get_calibrations(model, seqs, attns, IF_TRAIN):
    if IF_TRAIN:
        cost_iter = 10 # training 用模型推理 10 次 获得模型不确定性
    else:
        model.eval() # 关闭 drop out，否则模型波动大
        cost_iter = 1 # testing 推理 1 次
    
    with torch.no_grad():    
        for i in range(cost_iter):
            cali = model(seqs, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
            if i == 0:
                cali_all = cali[:, 0].unsqueeze(1) # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
            else:
                cali_all = torch.cat((cali_all, cali[:, 0].unsqueeze(1)), dim=1)
            cali = cali[:, 0].cpu().detach().numpy() # 选择每一行的第一个元素

    return cali_all

def get_ucb_idx(cali_all, pgcosts):    
    cali_mean = cali_all.mean(dim = 1) # 每个 plan 的 cali 均值 [# of plan] | 值为 1 左右
    cali_var = cali_all.var(dim = 1) # [# of plan] | 值为 0 左右
    costs = torch.tensor(pgcosts) # [# of plan]
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
        if node.info['join_cond'] == ['']:
            break
        node = plans_lib.FilterScansOrJoins(node)
        plans_lib.GatherUnaryFiltersInfo(node)
        postgres.EstimateFilterRows(node)
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
    equ_tem = ['title,movie_keyword,keyword', 'kind_type,title,comp_cast_type,complete_cast,movie_companies', 'kind_type,title,comp_cast_type,complete_cast,movie_companies,company_name', 'movie_companies,company_name', 'movie_companies,company_name,title',
            'movie_companies,company_name,title,aka_title', 'company_name,movie_companies,title,cast_info', 'name,aka_name', 'name,aka_name,cast_info', 'info_type,movie_info_idx', 'company_type,movie_companies',
            'company_type,movie_companies,title', 'company_type,movie_companies,title,movie_info', 'movie_companies,company_name', 'keyword,movie_keyword', 'keyword,movie_keyword,movie_info_idx']
    equ_set = set() # 用集合 方便 eq keys 中去重
    # 'title,movie_keyword,keyword' -> 'keyword,movie_keyword,title'
    for i in equ_tem:
        e_tem = i.split(',')
        e_tem = ','.join(sorted(e_tem))
        equ_set.add(e_tem)

    return equ_set

def getBatchPairsLoss(model, batch_pairs):
    """
    batch_pairs: a batch of train pairs
    return. a batch of loss
    """
    loss_fn = nn.MarginRankingLoss(margin=15)

    # step 1. retrieve encoded_plans and attns from pairs
    encoded_plans = []
    attns = []
    latencies = []
    costs = []
    for pair in batch_pairs:
        latencies.append(pair[0][0].info['latency'])
        latencies.append(pair[1][0].info['latency'])
        costs.append(pair[0][0].cost)
        costs.append(pair[1][0].cost)
        encoded_plans.append(pair[0][1])
        encoded_plans.append(pair[1][1])
        attns.append(pair[0][2])
        attns.append(pair[1][2])

    encoded_plans = torch.stack(encoded_plans, dim=0)
    attns = torch.stack(attns, dim=0)

    # step 2. calculate batch_cali and calied_cost
    batch_cali = get_calibrations(model, encoded_plans, attns, IF_TRAIN=False) # pair0的cali1, pair0的cali2, ...
    # print("len(cali_all)", len(batch_cali))
    batch_cali = batch_cali.view(-1, 2)
    costs = torch.tensor(costs).view(-1, 2) # torch.Size([3, 2])
    # print("costs.shape", costs.shape)
    calied_cost = batch_cali * costs # torch.Size([3, 2])
    print("calied_Cost.shape", calied_cost.shape)
    c1, c2 = torch.chunk(calied_cost, 2, dim=1) # torch.Size([3, 1]) c1 取 0 2 4 ...
    # print("c1.shape", c1.shape)
    c1 = torch.squeeze(c1)
    c2 = torch.squeeze(c2)

    assert (2 * len(calied_cost) == len(latencies)) and (len(latencies) % 2 == 0)
    res = []
    for i in range(0, len(latencies), 2):
        if latencies[i] > latencies[i + 1]:
            res.append(1)
        else:
            res.append(-1)
    res = torch.tensor(res)

    return loss_fn(c1, c2, res)

if __name__ == '__main__':
    # train_files = ['1a', '2a', '3a', '4a']
    train_files = ['1a', '2a']
    chunk_size = 2 # the # of sqls in a chunk
    IF_TRAIN = True
    model_path = "model.pth"
    message_path = "messages.pkl"

    ray.init(address='auto', namespace='server_namespace', _temp_dir="/data1/zengximu/LEON-research/ray") # init only once
    Exp = Experience(eq_set=initEqSet())
    print("Init workload and equal set keys")
    workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
    # workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(workload.workload_info.rel_names)
    workload.workload_info.alias_to_names = postgres.GetAllAliasToNames(workload.workload_info.rel_ids)

    # ===== ITERATION OF CHUNKS ====
    ch_start_idx = 0 # the start idx of the current chunk in train_files
    while ch_start_idx + chunk_size <= len(train_files):
        print(f"\n+++++++++ a chunk of sql from {ch_start_idx}  ++++++++")
        sqls_chunk = load_sql(train_files[ch_start_idx : ch_start_idx + chunk_size])
        print(train_files[ch_start_idx : ch_start_idx + chunk_size])

        # ++++ PHASE 1. ++++ send a chunk of queries with ENABLE_LEON=True
        # ENABLE_LEON = bool
        for q_send_cnt in range(chunk_size):
            print(f"------------- sending query {q_send_cnt} starting from idx {ch_start_idx} ------------")
            query_latency = getPG_latency(sqls_chunk[q_send_cnt], ENABLE_LEON=False)
            print("latency pg ", query_latency)
            query_latency = getPG_latency(sqls_chunk[q_send_cnt], ENABLE_LEON=True)
            print("latency leon ", query_latency)

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
                encoded_plans, _, attns, _ = plans_encoding(message, IF_TRAIN) # encoding. plan -> plan_encoding / seqs torch.Size([26, 1, 760])

                # STEP 2) pick node to execute
                pct = 0.1 # 执行 percent 比例的 plan
                costs = getNodesCost(nodes)
                cali_all = get_calibrations(model, encoded_plans, attns, IF_TRAIN)
                ucb_idx = get_ucb_idx(cali_all, costs)
                num_to_exe = math.ceil(pct * len(ucb_idx))
                # print("ucb_idx to exe", ucb_idx[:num_to_exe])
                print("num_to_exe", num_to_exe)

                # STEP 3) execute with ENABLE_LEON=False and add exp
                # 经验 [[logcost, sql, hint, latency, [query_vector, node], join, joinids], ...]
                for i in range(num_to_exe):
                    if i < 3: # 后续删掉！！这里只是为了节省代码运行时间！！！！
                        node_idx = ucb_idx[i] # 第 i 大ucb 的 node idx
                        a_node = nodes[node_idx]
                        a_node.info['latency'] = getPG_latency(a_node.info['sql_str'], a_node.hint_str(), ENABLE_LEON=False)

                        # (1) add new EqSet key in exp
                        if i == 0:
                            eqKey = a_node.info['join_tables']
                            Exp.AddEqSet(eqKey)
                        # (2) add experience of certain EqSet key
                        a_plan = [a_node, # with sql, hint, latency, cost
                                  encoded_plans[node_idx],
                                  attns[node_idx]]
                        if not Exp.isCache(eqKey, a_plan): # 该行放 get latency 前面 ！！！
                            Exp.AppendExp(eqKey, a_plan)

                print("len(Exp.GetExp(eqKey))", len(Exp.GetExp(eqKey)))
        
        Exp.DeleteEqSet()
        
        # ++++ PHASE 3. ++++ model training
        # PHASE 3 还有小问题
        batchsize = 3 # 128
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        train_pairs = Exp.Getpair()
        if len(train_pairs) < batchsize:
            pass # !!! 加上 iter 后 改为 continue; 一个 chunk 的pairs不足时, 与下一个 chunk 的pairs合并 !!!

        # STEP 1) get train pairs of a batch, calculate the cali of the batch
        shuffled_idx = np.random.permutation(len(train_pairs))
        # print(shuffled_idx)
        curr_idx = 0
        while curr_idx + batchsize <= len(shuffled_idx):
            print(f"------- start to process train pairs from {curr_idx} with {len(train_pairs)} total pairs -------")
            batch_pairs = [train_pairs[i] for i in shuffled_idx[curr_idx: curr_idx + batchsize]]
            
            batch_loss  = getBatchPairsLoss(model, batch_pairs)
            loss = torch.mean(batch_loss, 0)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            curr_idx = curr_idx + batchsize
        
        ch_start_idx = ch_start_idx + chunk_size
        # save model
        torch.save(model, model_path)
        # 模型更新需要通知到server.py
        # clear pkl
        if os.path.exists(message_path):
            os.remove(message_path)
            print(f"Successfully remove {message_path}")
        else:
            print(f"Fail to remove {message_path}")


