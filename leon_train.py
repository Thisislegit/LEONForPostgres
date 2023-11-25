import os
from util import postgres
from util import pg_executor
from util import postgres
from util import envs
from util import plans_lib
import pickle
import torch
import math
import json
from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers

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

def getPG_latency(query, ENABLE_LEON=False):
    """
    input. a loaded query
    output. the average latency of a query get from pg
    """
    latency_sum = 0
    cnt = 3
    for c in range(cnt):
        latency_sum = latency_sum + postgres.GetLatencyFromPg(query, None, ENABLE_LEON, verbose=False, check_hint_used=False, timeout=90000,
                                                dropbuffer=False)
    pg_latency = latency_sum / cnt
    return pg_latency

# UNFINISHED hint sql -> pgcosts
def plans_encoding(plans, IF_TRAIN):
    '''
    input. a list of plans in type of json
    output. (seq_encoding, run_times, attention_mask, loss_mask)
        - run_times 是归一化之后的
    '''
    statistics_file_path = "/data1/zengximu/LEON-research/LEONForPostgres/statistics.json"
    feature_statistics = load_json(statistics_file_path)
    add_numerical_scalers(feature_statistics)
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)

    seqs = []
    attns = []
    for x in plans:
        # print(x)
        # run_times 无法获取
        seq_encoding, run_times, attention_mask, loss_mask, database_id = get_plan_encoding(
            x, configs, op_name_to_one_hot, plan_parameters, feature_statistics, IF_TRAIN
        )
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

    print("cali_all.shape ", cali_all.shape)            
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
    print("len(ucb)", len(ucb))
    ucb_sort_idx = torch.argsort(ucb, descending=True) # 张量排序索引 | ucb_sort[0] 是uncertainty高的plan在plan_info中的索引
    ucb_sort_idx = ucb_sort_idx.tolist()

    return ucb_sort_idx

def PlanToNode(workload, plans):
    '''
    input. plans 
        一个 message 等价类包括多条 plans
    output. nodes
        一个 node 的 sql 信息在 node.info['sql_str'], cost 信息在 node.cost, hint 信息在 node.hint_str()
    '''
    nodes = []
    for i in range(len(plans)):
        # 一个 message 等价类包括多条 plans
        # print("================================")
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
        # print("node\n", node)
    
    return nodes

def getNodesCost(nodes):
    pgcosts = [] # 存所有 plan 的 pg cost
    for node in nodes:
        pgcost = node.cost
        pgcost_log = math.log(pgcost)
        pgcosts.append(pgcost_log)
    
    return pgcosts

if __name__ == '__main__':
    # train_files = ['1a', '2a', '3a']
    train_files = ['1a']
    train_sqls = load_sql(train_files)
    # print(train_sqls[0])

    # PHASE 1. send query with ENABLE_LEON=True
    ENABLE_LEON = bool
    for i in range(len(train_sqls)):
        print("------------- query {} ------------".format(i))
        query_latency = getPG_latency(train_sqls[i], ENABLE_LEON=False)
        print("-- query_latency pg --", query_latency)
        query_latency = getPG_latency(train_sqls[i], ENABLE_LEON=True)
        print("-- query_latency leon --", query_latency)

    # PHASE 2. get messages and train model
    workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
    workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(workload.workload_info.rel_names)
    workload.workload_info.alias_to_names = postgres.GetAllAliasToNames(workload.workload_info.rel_ids)
    IF_TRAIN = True
    model_path = "model.pth"
    if not os.path.exists(model_path):
        model = Transformer_model
    else:
        model = torch.load(model_path)
    
    # pkl_cnt = 42
    pkl_cnt = 1
    with open("messages.pkl", "rb") as file:
        for i in range(pkl_cnt):
            pkl_cnt = pkl_cnt -1
            message = pickle.load(file) # message = plans 是一个等价类 / 子查询
            print("\nlen(message)", len(message))
            # STEP 1) get node, [query_encoding, latency, cost]
            nodes = PlanToNode(workload, message)
            pgcosts = getNodesCost(nodes)
            
            seqs, un_times, attns, loss_mask = plans_encoding(message, IF_TRAIN) # encoding. plan -> plan encoding
            cali_all = get_calibrations(model, seqs, attns, IF_TRAIN)

            # STEP 2) pick node to execut with ENABLE_LEON=False
            pct = 0.1 # 执行 percent 比例的 plan
            ucb_idx = get_ucb_idx(cali_all, pgcosts)
            print(ucb_idx)
            n = math.ceil(pct * len(ucb_idx))
            # for i in range(n):
            #     plan_info_pct.append(plan_info[ucb_idx[i]]) # 存要执行 plan_info_pct


