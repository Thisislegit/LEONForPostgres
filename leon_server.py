import json
import struct
import socketserver
from utils import *
import torch
import math

from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers

import ray
import pickle
import os
import time


ray.init()

# @ray.remote
# class actor:
#     def __init__(self, ):
#         self.filename = 1

#     def newfile()
#         opem 

#         file+ =1

#     def clear 

@ray.remote
def append_node(file_name, nodes: list):
    with open(file_name, 'ab') as file:  # 'ab' 模式用于追加
            pickle.dump(nodes, file)
    return 1

Transformer_model = SeqFormer(
                        input_dim=configs['node_length'],
                        hidden_dim=128,
                        output_dim=1,
                        mlp_activation="ReLU",
                        transformer_activation="gelu",
                        mlp_dropout=0.3,
                        transformer_dropout=0.2,
                    )
import torch
import math

from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers

Transformer_model = SeqFormer(
                        input_dim=configs['node_length'],
                        hidden_dim=128,
                        output_dim=1,
                        mlp_activation="ReLU",
                        transformer_activation="gelu",
                        mlp_dropout=0.3,
                        transformer_dropout=0.2,
                    )

class LeonModel:

    def __init__(self):
        self.__model = None
    
    def load_model(self, path):
        pass
    
    # -- UNFINISHED pgcost --
    def plans_encoding(self, plans, IF_TRAIN):
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
        pgcosts = [] # 存所有 plan 的 pg cost
        for x in plans:
            seq_encoding, run_times, attention_mask, loss_mask, database_id = get_plan_encoding(
                x, configs, op_name_to_one_hot, plan_parameters, feature_statistics, IF_TRAIN
            )
            seqs.append(seq_encoding) 
            attns.append(attention_mask)
            pgcost = 100 # pg cost model 返回的 cost 【plan变成 hint + sql 再使用 pg 获得 cost】 看【pg_train.py】
            pgcost_log = math.log(pgcost) # 4.605
            pgcosts.append(pgcost_log)
        seqs = torch.stack(seqs, dim=0)
        attns = torch.stack(attns, dim=0)

        return seqs, run_times, attns, loss_mask, pgcosts
    
    def get_calibrations(self, seqs, attns, IF_TRAIN):
        if IF_TRAIN:
            cost_iter = 10 # training 用模型推理 10 次 获得模型不确定性
        else:
            Transformer_model.eval() # 关闭 drop out，否则模型波动大
            cost_iter = 1 # testing 推理 1 次
        
        with torch.no_grad():    
            for i in range(cost_iter):
                cali = Transformer_model(seqs, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
                if i == 0:
                    cali_all = cali[:, 0].unsqueeze(1) # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
                else:
                    cali_all = torch.cat((cali_all, cali[:, 0].unsqueeze(1)), dim=1)
                cali = cali[:, 0].cpu().detach().numpy() # 选择每一行的第一个元素
            
            # print("cali_all shape: ", cali_all.shape)
        
        return cali_all
    
    def get_ucb_idx(self, cali_all, pgcosts):    
        cali_mean = cali_all.mean(dim = 1) # 每个 plan 的 cali 均值 [# of plan] | 值为 1 左右
        cali_var = cali_all.var(dim = 1) # [# of plan] | 值为 0 左右
        costs = torch.tensor(pgcosts) # [# of plan]
        cost_t = torch.mul(cali_mean, costs) # 计算 calibration mean * pg返回的cost [# of plan]

        # cali_min, _ = cost_t.min(dim = 0) # plan 的 cali 最小值 [# of plan] 【cali_min只有一个值？】
        # print("cost_min.shape(): ", cali_min.shape)
        # print(cali_min)            
        # ucb = cali_var / cali_var.max() - cali_min / cali_min.max()

        ucb = cali_var / cali_var.max() - cost_t / cost_t.max() # [# of plan]
        # print("--- ucb ---", ucb)

        ucb_sort_idx = torch.argsort(ucb, descending=True) # 张量排序索引 | ucb_sort[0] 是uncertainty高的plan在plan_info中的索引
        ucb_sort_idx = ucb_sort_idx.tolist()

        return ucb_sort_idx
    
    def predict_plan(self, messages):
        '''
        input. messages 一个等价类的所有 plan
        output. 所有 plan 的修正值
        '''
        '''
        input. messages 一个等价类的所有 plan
        output. 所有 plan 的修正值
        '''
        print("Predicting plan for ", len(messages))
        X = messages
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        t1 = time.time()
        future = append_node.remote("Nodes.pkl", X)
        # print(ray.get(future))
        print("Time to write nodes: ", time.time() - t1)

        # print(X[0])
        # for x in X:
        #     print(x)
        # return ','.join(['1.00' for _ in X])
        
        # 1. encoding. plan -> plan encoding
        IF_TRAIN = False
        if IF_TRAIN:
            seqs, un_times, attns, loss_mask, pgcosts = self.plans_encoding(X, IF_TRAIN)
        else:
            seqs, _, attns, _, pgcosts = self.plans_encoding(X, IF_TRAIN)

        # 2. calculate calibration
        IF_TRAIN = False
        cali_all = self.get_calibrations(seqs, attns, IF_TRAIN)

        # 3. 计算 ucb_idx, 存要执行 plan_info_pct 传给 exp_add
        if IF_TRAIN:
            plan_info = [] # 存所有 plan 的 cost, sql, hint 等信息
            plan_info_pct = [] # 存需要执行的 plan 的 cost, sql, hint 等信息
            pct = 0.1 # 执行 percent 比例的 plan
            ucb_idx = self.get_ucb_idx(cali_all, pgcosts)
            n = math.ceil(pct * len(ucb_idx))
            # for i in range(n):
            #     plan_info_pct.append(plan_info[ucb_idx[i]])

        cali_str = ['{:.2f}'.format(i) for i in cali_all[:, -1].tolist()] # 最后一次 cali
        # print("cali_str len", len(cali_str))
        cali_strs = ','.join(cali_str)
        # return cali_strs, seqs

        return ",".join(['1.00' for _ in X]), None

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
                        if self.handle_json(json.loads(json_msg)):
                            break
                    except json.decoder.JSONDecodeError:
                        print("Error decoding JSON:", json_msg)
                        break

class LeonJSONHandler(JSONTCPHandler):
    def setup(self):
        self.__messages = []
    
    def handle_json(self, data):
        if "final" in data:
            message_type = self.__messages[0]["type"]
            self.__messages = self.__messages[1:]
            if message_type == "query":
                print("\n=== leon_server get a query ===")
                cali_strs, seqs = self.server.leon_model.predict_plan(self.__messages)
                response = str(cali_strs).encode()
                print("\n=== leon_server get a query ===")
                cali_strs, seqs = self.server.leon_model.predict_plan(self.__messages)
                response = str(cali_strs).encode()
                # self.request.sendall(struct.pack("I", result))
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