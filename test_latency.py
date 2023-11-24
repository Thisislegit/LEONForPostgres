import os
from util import pg_executor
import subprocess
import json
"""
nodes 是一条query 的所有 plans 需要用到 server.py 中的解析

数据路径
leon train 提交一条 query 
-> PG 
-> leon server 做校验
-> PG 根据校验选择 dp 中最优的 path 执行
-> 获得执行 feedback (pg 返回的一条query的执行时间)

"""

def _run_explain(explain_str,
                 sql,
                 comment,
                 verbose,
                 geqo_off=False,
                 timeout_ms=10000,
                 cursor=None,
                 is_test=False,
                 remote=False):
    """
    Run the given SQL statement with appropriate EXPLAIN commands.

    timeout_ms is for both setting the timeout for PG execution and for the PG
    cluster manager, which will release the server after timeout expires.
    """
    # if is_test:
    #     assert remote, "testing queries must run on remote Postgres servers"
    if cursor is None and not remote:
        with pg_executor.Cursor() as cursor:
            return _run_explain(explain_str, sql, comment, verbose, geqo_off,
                                timeout_ms, cursor, remote)

    end_of_comment_idx = sql.find('*/')
    if end_of_comment_idx == -1:
        existing_comment = None
    else:
        split_idx = end_of_comment_idx + len('*/\n')
        existing_comment = sql[:split_idx]
        sql = sql[split_idx:]

    # Fuse hint comments.
    if comment:
        assert comment.startswith('/*+') and comment.endswith('*/'), (
            'Don\'t know what to do with these', sql, existing_comment, comment)
        if existing_comment is None:
            fused_comment = comment
        else:
            comment_body = comment[len('/*+ '):-len(' */')].rstrip()
            existing_comment_body_and_tail = existing_comment[len('/*+'):]
            fused_comment = '/*+\n' + comment_body + '\n' + existing_comment_body_and_tail
    else:
        fused_comment = existing_comment

    if fused_comment:
        s = fused_comment + '\n' + str(explain_str).rstrip() + '\n' + sql
    else:
        s = str(explain_str).rstrip() + '\n' + sql
    if remote:
        assert cursor is None
        return pg_executor.ExecuteRemote(s, verbose, geqo_off, timeout_ms)
    else:
        return pg_executor.Execute(s, verbose, geqo_off, timeout_ms, cursor)


def DropBufferCache():
    # WARNING: no effect if PG is running on another machine
    subprocess.check_output(['free', '&&', 'sync'])
    subprocess.check_output(
        ['sudo', 'sh', '-c', 'echo 3 > /proc/sys/vm/drop_caches'])
    subprocess.check_output(['free'])
    #  print('drop cache')
    with pg_executor.Cursor() as cursor:
        cursor.execute('create extension pg_dropcache;')
        cursor.execute('select pg_dropcache();')
        cursor.execute('DISCARD ALL;')

def GetLatencyFromPg(sql, hint, ENABLE_LEON, verbose=False, check_hint_used=False, timeout=10000, dropbuffer=False):
    if dropbuffer:
        DropBufferCache()
    with pg_executor.Cursor() as cursor:
        # GEQO must be disabled for hinting larger joins to work.
        # Why 'verbose': makes ParsePostgresPlanJson() able to access required
        # fields, e.g., 'Output' and 'Alias'.  Also see SqlToPlanNode() comment.
        if ENABLE_LEON:
            cursor.execute('SET geqo=off')
            cursor.execute('SET enable_leon=on')
        else:
            cursor.execute('SET geqo=off')
            cursor.execute('SET enable_leon=off')
        geqo_off = hint is not None and len(hint) > 0
        result = _run_explain('explain(verbose, format json)',
                              sql,
                              hint,
                              verbose=True,
                              geqo_off=geqo_off,
                              cursor=cursor, timeout_ms=timeout * 1.5).result

    if (result == []):
        return 10000
    json_dict = result[0][0][0]
    latency = float(json_dict['Execution Time'])
    print("latency", latency)

    return latency


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

def get_latency(query, ENABLE_LEON=False):
    """
    input. a loaded query
    output. the average latency of a query get from pg
    """
    latency = GetLatencyFromPg(query, None, ENABLE_LEON, verbose=False, check_hint_used=False, timeout=20000, dropbuffer=False)
    return latency

def save_to_json(data):
    # 打开文件的模式: 常用的有’r’（读取模式，缺省值）、‘w’（写入模式）、‘a’（追加模式）等
    with open('./data.json', 'w') as f:
        # 使用json.dump()函数将序列化后的JSON格式的数据写入到文件中
        json.dump(data, f, indent=4)

def read_to_json():
    # 打开文件的模式: 常用的有’r’（读取模式，缺省值）、‘w’（写入模式）、‘a’（追加模式）等
    with open('./data.json', 'r') as f:
        # 使用json.dump()函数将序列化后的JSON格式的数据写入到文件中
        data = json.load(f)
    return data

if __name__ == '__main__':
    files = ['1a', '1b', '1c', '1d', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '4a',
             '4b', '4c', '5a', '5b', '5c', '6a', '6b', '6c', '6d', '6e', '6f', '7a', 
             '7b', '7c', '8a', '8b', '8c', '8d', '9a', '9b', '9c', '9d', '10a', '10b', 
             '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', 
             '13d', '14a', '14b', '14c', '15a', '15b', '15c', '15d', '16a', '16b', '16c',
             '16d', '17a', '17b', '17c', '17d', '17e', '17f', '18a', '18b', '18c', '19a',
             '19b', '19c', '19d', '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b',
             '22c', '22d', '23a', '23b', '23c', '24a', '24b', '25a', '25b', '25c', '26a', 
             '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c', '29a', '29b', '29c',
             '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c', '1a']
    sqls = load_sql(files)
    # print(train_sqls[0])
    label_1a = False
    ENABLE_LEON = bool

    for i in range(len(files)):
        value = files[i]
        if label_1a and value == '1a':
            print("------------- query {} ------------".format(value))
            query_latency = get_latency(sqls[i], ENABLE_LEON=True)
            print("-- query_latency leon --", query_latency)
            continue
        if value == '1a' and label_1a == False:
            label_1a = True
        print("------------- query {} ------------".format(value))
        data = read_to_json()
        data[value] = []
        save_to_json(data)
        # query_latency = get_latency(sqls[i], ENABLE_LEON=False)
        # print("-- query_latency pg --", query_latency)
        query_latency = get_latency(sqls[i], ENABLE_LEON=True)
        print("-- query_latency leon --", query_latency)



