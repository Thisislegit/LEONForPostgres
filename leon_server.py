import json
import struct
import socketserver
from utils import *
import util.envs as envs
import util.treeconv as treeconv
import util.postgres as postgres
import util.plans_lib as plans_lib
import torch
import os
import util.DP as DP
import copy
import re
import math
import time
from test_case import SeqFormer, SeqFormer_tree
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers
import wandb

import json

files = ['1a', '1b', '1c', '1d', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '4a',
             '4b', '4c', '5a', '5b', '5c', '6a', '6b', '6c', '6d', '6e', '6f', '7a', 
             '7b', '7c', '8a', '8b', '8c', '8d', '9a', '9b', '9c', '9d', '10a', '10b', 
             '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', 
             '13d', '14a', '14b', '14c', '15a', '15b', '15c', '15d', '16a', '16b', '16c',
             '16d', '17a', '17b', '17c', '17d', '17e', '17f', '18a', '18b', '18c', '19a',
             '19b', '19c', '19d', '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b',
             '22c', '22d', '23a', '23b', '23c', '24a', '24b', '25a', '25b', '25c', '26a', 
             '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c', '29a', '29b', '29c',
             '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c', 'end']
def save_to_json(data):
    # 打开文件的模式: 常用的有’r’（读取模式，缺省值）、‘w’（写入模式）、‘a’（追加模式）等
    with open('./data2.json', 'w') as f:
        # 使用json.dump()函数将序列化后的JSON格式的数据写入到文件中
        json.dump(data, f, indent=4)

def read_to_json():
    # 打开文件的模式: 常用的有’r’（读取模式，缺省值）、‘w’（写入模式）、‘a’（追加模式）等
    with open('./data2.json', 'r') as f:
        # 使用json.dump()函数将序列化后的JSON格式的数据写入到文件中
        data = json.load(f)
    return data


class LeonModel:

    def __init__(self):
        self.eval_time = True
        run = wandb.init(
                    project="transformer",
                    )
        self.__model = None
        self.encoding_model = 'transformer'
        self.inference_model = 'transformer' 
        self.analysis_all_time = 0
        self.encoding_all_time = 0
        self.inference_all_time = 0
        self.n = 0
        self.curr_file = '1a'
        # 初始化
        if self.encoding_model == 'tree':
            self.workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
            self.workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(self.workload.workload_info.rel_names)
            self.workload.workload_info.alias_to_names = postgres.GetAllAliasToNames(self.workload.workload_info.rel_ids)
            self.queryFeaturizer = plans_lib.QueryFeaturizer(self.workload.workload_info)
            self.nodeFeaturizer = plans_lib.PhysicalTreeNodeFeaturizer(self.workload.workload_info)
            if self.inference_model == 'tree_conv':
                self.tree_model = treeconv.TreeConvolution(820, 123, 1)
                pass
            else:
                self.Transformer_model = SeqFormer_tree(
                    query_dim=820,
                    input_dim= 155,
                    hidden_dim=128,
                    output_dim=1,
                    padding_dim=128,
                    mlp_activation="ReLU",
                    transformer_activation="gelu",
                    mlp_dropout=0.3,
                    transformer_dropout=0.2,
                    )
        else:
            self.Transformer_model = SeqFormer(
                input_dim=configs['node_length'],
                hidden_dim=128,
                output_dim=1,
                mlp_activation="ReLU",
                transformer_activation="gelu",
                mlp_dropout=0.3,
                transformer_dropout=0.2,
                )
            statistics_file_path = "/data1/wyz/online/LEONForPostgres/statistics.json"
            self.feature_statistics = load_json(statistics_file_path)
            add_numerical_scalers(self.feature_statistics)
            self.op_name_to_one_hot = get_op_name_to_one_hot(self.feature_statistics)
            
    def plans_encoding(self, plans):
        '''
        input. a list of plans in type of json
        output. (seq_encoding, run_times, attention_mask, loss_mask)
            - run_times 是归一化之后的
        '''
        seqs = []
        attns = []
        pgcosts = [] # 存所有 plan 的 pg cost
        for x in plans:
            seq_encoding, run_times, attention_mask, loss_mask, database_id = get_plan_encoding(
                x, configs, self.op_name_to_one_hot, plan_parameters, self.feature_statistics)
            seqs.append(seq_encoding) 
            attns.append(attention_mask)
            pgcost = 100 # pg cost model 返回的 cost 【plan变成 hint + sql 再使用 pg 获得 cost】 看【pg_train.py】
            pgcost_log = math.log(pgcost) # 4.605
            pgcosts.append(pgcost_log)
        seqs = torch.stack(seqs, dim=0)
        attns = torch.stack(attns, dim=0)

        return seqs, run_times, attns, loss_mask, pgcosts
    
    def get_calibrations(self, a, b, c=None):
        with torch.no_grad():
            # cost_iter
            if self.inference_model == 'tree_conv' and self.encoding_model == 'tree':
                self.tree_model.eval() # tree_conv
                query_feats = a
                trees = b
                indexes = c
                cali_all = torch.tanh(self.tree_model(query_feats, trees, indexes)).add(1).squeeze(1)
            else:
                self.Transformer_model.eval() # 关闭 drop out，否则模型波动大    
                if c is None: # transformer
                    seqs = a
                    attns = b
                    cali = self.Transformer_model(seqs, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
                else:
                    query_feats = a # tree_transformer
                    trees = b
                    indexes = c
                    cali = self.Transformer_model(query_feats, trees, indexes) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
                cali_all = cali[:, 0] # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
        # print(cali_all)
        return cali_all
    
    def encoding(self, X): 
        if self.encoding_model == "tree":
            nodes = []
            queryencoding = []
            for i in range(0, len(X)):
                # print(X[i])
                # print(node)
                node = postgres.ParsePostgresPlanJson_1(X[i], self.workload.workload_info.alias_to_names)
                if node.info['join_cond'] == ['']: # return 1.00
                    return None, None, None
                node = plans_lib.FilterScansOrJoins(node)
                plans_lib.GatherUnaryFiltersInfo(node)
                postgres.EstimateFilterRows(node)   
                if i == 0:
                    temp = node.to_sql(node.info['join_cond'], with_select_exprs=True)
                    node.info['sql_str'] = temp
                    query_vecs = torch.from_numpy(self.queryFeaturizer(node)).unsqueeze(0)
                node.info['sql_str'] = temp
                node.info['query_encoding'] = copy.deepcopy(query_vecs)
                queryencoding.append(query_vecs)
                nodes.append(node)
            if nodes:
                tensor_query_encoding = (torch.cat(queryencoding, dim=0))
                trees, indexes = encoding.TreeConvFeaturize(self.nodeFeaturizer, nodes)
                # print("tensor_query_encoding", tensor_query_encoding.shape,
                #     "trees", trees.shape,
                #     "indexes", indexes.shape)
            return tensor_query_encoding, trees, indexes
        
        else:
            seqs, _, attns, _, _ = self.plans_encoding(X)
            return seqs, attns, None
    
    def inference(self, a, b, c):
        cali_all = self.get_calibrations(a, b, c)
        cali_str = ['{:.2f}'.format(i) for i in cali_all.tolist()] # 最后一次 cali
        # print("cali_str len", len(cali_str))
        cali_strs = ','.join(cali_str)
        return cali_strs
    
    def load_model(self, path):
        pass
    
    def predict_plan(self, messages):
        if self.eval_time:
            data = read_to_json()
            for i in files:
                if i not in data:
                    break
                curr_file = i
            if curr_file != self.curr_file:
                all_time = {
                "analysis_time": self.analysis_all_time / self.n,
                "encoding_time": self.encoding_all_time / self.n,
                "inference_time": self.inference_all_time / self.n
                }
                data[self.curr_file].append(all_time)
                save_to_json(data)
                wandb.log({"analysis_time": self.analysis_all_time / self.n, "encoding_time": self.encoding_all_time / self.n, "inference_time": self.inference_all_time / self.n, "sql_id": self.curr_file})
                self.analysis_all_time = 0
                self.encoding_all_time = 0
                self.inference_all_time = 0
                self.n = 0
                self.curr_file = curr_file
            # print(self.curr_file)
            
        # json解析
        start_time = time.time()
        # print("Predicting plan for ", len(messages))
        X = messages
        if not isinstance(X, list):
            X = [X]
        for x in X:
            if not x:
                return ','.join(['1.00' for _ in X])
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        end_time = time.time()
        # 平均执行时间
        analysis_time = end_time - start_time
        # 编码
        start_time = time.time()
        a, b, c = self.encoding(X)
        end_time = time.time()
        # 平均执行时间
        encoding_time = end_time - start_time
        # 推理
        start_time = time.time()
        if a is None: # join_condition = ['']
            return ','.join(['1.00' for _ in X])
        cali_strs = self.inference(a, b, c)
        end_time = time.time()
        # 平均执行时间
        if self.eval_time:
            inference_time = end_time - start_time
            self.analysis_all_time += analysis_time
            self.encoding_all_time += encoding_time
            self.inference_all_time += inference_time
            self.n += 1
        
        return cali_strs


        # seqs = [get_plan_seq_adj(x['Plan']) for x in X]
        # print(seqs[0])
        # print(op_names)
        # seqs_encoding = [generate_seqs_encoding(x) for x in seqs]
        # print(seqs_encoding[0])
        return ';'.join(['1.00,1,0' for _ in X])

class JSONTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        str_buf = ""
        while True:
            # 这里只有断连才会退出
            str_buf += self.request.recv(1024).decode("UTF-8")
            if not str_buf:
                # no more data, connection is finished.
                return
            
            if (null_loc := str_buf.find("\n")) != -1:
                json_msg = str_buf[:null_loc].strip()
                str_buf = str_buf[null_loc + 1:]
                if json_msg:
                    try:    
                        def fix_json_msg(json):
                            pattern = r'ANY \((.*?):text\[\]\)'
                            matches = re.findall(pattern, json)
                            for match in matches:
                                extracted_string = match
                                cleaned_string = extracted_string.replace('"', '')
                                json = json.replace(extracted_string, cleaned_string)
                            return json
                        json_msg = fix_json_msg(json_msg)
                        # json_msg = json_msg.replace('"(', '(').replace(')"', ')').replace('[]),"', '[])","')
                        # print(repr(json_msg))
                        if self.handle_json(json.loads(json_msg)):
                            break
                    except json.decoder.JSONDecodeError:
                        print("Error decoding JSON:", repr(json_msg))
                        self.handle_json([])
                        break

class LeonJSONHandler(JSONTCPHandler):
    def setup(self):
        self.__messages = []
    
    def handle_json(self, data):
        if "final" in data:
            message_type = self.__messages[0]["type"]
            self.__messages = self.__messages[1:]
            if message_type == "query":
                result = self.server.leon_model.predict_plan(self.__messages)
                response = str(result).encode()
                # self.request.sendall(struct.pack("I", result))
                self.request.sendall(response)
                self.request.close()
            elif message_type == "should_opt":
                print(self.__messages)
                response = str("1").encode()
                self.request.sendall(response)
                self.request.close()
            else:
                print("Unknown message type:", message_type)
            return True
        
        self.__messages.append(data)
        return False

def start_server(listen_on, port):
    model = LeonModel()

    # if os.path.exists(DEFAULT_MODEL_PATH):
    #     print("Loading existing model")
    #     model.load_model(DEFAULT_MODEL_PATH)
    
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((listen_on, port), LeonJSONHandler) as server:
        server.leon_model = model
        server.serve_forever()


if __name__ == "__main__":
    from multiprocessing import Process
    from config import read_config


    config = read_config()
    port = int(config["Port"])
    listen_on = config["ListenOn"]

    print(f"Listening on {listen_on} port {port}")
    
    server = Process(target=start_server, args=[listen_on, port])
    
    print("Spawning server process...")
    server.start()