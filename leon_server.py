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


@ray.remote
class FileWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.completed_tasks = 0

    def write_file(self, nodes):
        try:
            with open(self.file_path, 'ab') as file:
                pickle.dump(nodes, file)
                print("write one message")
            self.completed_tasks += 1
        except Exception as e:
            print("write_file() fail to open file and to write message", e)
    
    def complete_all_tasks(self, task_num):
        print(self.completed_tasks)
        if self.completed_tasks == task_num:
            return True
        else:
            return False


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
        self.__model = self.load_model("model.pth")
        ray.init(_temp_dir="/data1/zengximu/LEON-research/ray") # ray should be init in sub process
        node_path = "messages.pkl"
        self.writer_hander = FileWriter.remote(node_path)

    def load_model(self, path):
        if not os.path.exists(path):
            model = Transformer_model
        else:
            model = torch.load(path)
        return model
    
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
        for x in plans:
            seq_encoding, run_times, attention_mask, loss_mask, database_id = get_plan_encoding(
                x, configs, op_name_to_one_hot, plan_parameters, feature_statistics, IF_TRAIN
            )
            seqs.append(seq_encoding) 
            attns.append(attention_mask)
        seqs = torch.stack(seqs, dim=0)
        attns = torch.stack(attns, dim=0)

        return seqs, None, attns, None, None
    
    def get_calibrations(self, seqs, attns, IF_TRAIN):
        if IF_TRAIN:
            cost_iter = 10 # training 用模型推理 10 次 获得模型不确定性
        else:
            self.__model.eval() # 关闭 drop out，否则模型波动大
        cost_iter = 1 # testing 推理 1 次
        
        with torch.no_grad():    
            for i in range(cost_iter):
                cali = self.__model(seqs, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
                if i == 0:
                    cali_all = cali[:, 0].unsqueeze(1) # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
                else:
                    cali_all = torch.cat((cali_all, cali[:, 0].unsqueeze(1)), dim=1)
                cali = cali[:, 0].cpu().detach().numpy() # 选择每一行的第一个元素
                    
        return cali_all
    
    def predict_plan(self, messages):
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
        try:
            self.writer_hander.write_file.remote(X)
        except:
            print("The ray writer_hander cannot write file.")
        print("Time to write nodes: ", time.time() - t1)
        
        # 1. encoding. plan -> plan encoding
        IF_TRAIN = False
        seqs, _, attns, _, _ = self.plans_encoding(X, IF_TRAIN)

        # 2. calculate calibration
        cali_all = self.get_calibrations(seqs, attns, IF_TRAIN)
        print("cali_all.shape", cali_all.shape)

        cali_str = ['{:.2f}'.format(i) for i in cali_all[:, -1].tolist()] # 最后一次 cali
        cali_strs = ','.join(cali_str)
        return cali_strs, seqs

        # return ",".join(['1.00' for _ in X])

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
                cali_strs = self.server.leon_model.predict_plan(self.__messages)
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