import os
from util import postgres
from util import pg_executor

"""
nodes 是一条query 的所有 plans 需要用到 server.py 中的解析

数据路径
leon train 提交一条 query 
-> PG 
-> leon server 做校验
-> PG 根据校验选择 dp 中最优的 path 执行
-> 获得执行 feedback (pg 返回的一条query的执行时间)

"""

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

if __name__ == '__main__':
    train_files = ['1a', '2a', '3a']
    train_sqls = load_sql(train_files)
    # print(train_sqls[0])

    ENABLE_LEON = bool

    for i in range(len(train_sqls)):
        print("------------- query {} ------------".format(i))
        query_latency = getPG_latency(train_sqls[i], ENABLE_LEON=False)
        print("-- query_latency pg --", query_latency)
        query_latency = getPG_latency(train_sqls[i], ENABLE_LEON=True)
        print("-- query_latency leon --", query_latency)



