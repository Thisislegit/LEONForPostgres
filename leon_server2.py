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
from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers
import wandb
from leon_experience import *
import json
import pickle
import ray


@ray.remote
class FileWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.completed_tasks = 0
        self.recieved_task = 0
        self.RELOAD = True
        # self.eqset = ['cast_info,company_name,movie_companies,title', 'company_name,movie_companies,title']
        # self.eqset = ['company_name,movie_companies', 'company_type,movie_companies', 'company_type,movie_companies,title']
        self.eqset = ['cn,mc', 'ct,mc', 'ct,mc,t', 'ci,cn,mc,t', 'cn,mc,t']
        # self.eqset = ['title,movie_keyword,keyword', 'kind_type,title,comp_cast_type,complete_cast,movie_companies', 'kind_type,title,comp_cast_type,complete_cast,movie_companies,company_name', 'movie_companies,company_name', 'movie_companies,company_name,title',
                # 'movie_companies,company_name,title,aka_title', 'company_name,movie_companies,title,cast_info', 'name,aka_name', 'name,aka_name,cast_info', 'info_type,movie_info_idx', 'company_type,movie_companies',
                # 'company_type,movie_companies,title', 'company_type,movie_companies,title,movie_info', 'movie_companies,company_name', 'keyword,movie_keyword', 'keyword,movie_keyword,movie_info_idx']

    def write_file(self, nodes):
        try:
            with open(self.file_path, 'ab') as file:
                pickle.dump(nodes, file)
                print("write one message")
            self.completed_tasks += 1
        except Exception as e:
            print("write_file() fail to open file and to write message", e)

    def Add_task(self):
        self.recieved_task += 1
    
    def complete_all_tasks(self):
        print(self.completed_tasks)
        if self.completed_tasks == self.recieved_task:
            return True
        else:
            return False
    
    def load_model(self):
        temp = self.RELOAD
        self.RELOAD = False
        return temp , self.eqset
    
    def reload_model(self, eqset):
        self.RELOAD = True
        self.eqset = eqset

class LeonModel:

    def __init__(self):
        # 初始化
        self.__model = None
        ray.init(namespace='server_namespace', _temp_dir="/data1/wyz/online/LEONForPostgres/log/ray") # ray should be init in sub process
        node_path = "./log/messages.pkl"
        self.writer_hander = FileWriter.options(name="leon_server").remote(node_path)
        self.eqset = None
        self.workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
        self.workload.workload_info.alias_to_names = postgres.GetAllAliasToNames(self.workload.workload_info.rel_ids)
        
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
        for x in plans:
            seq_encoding, run_times, attention_mask, loss_mask, database_id = get_plan_encoding(
                x, configs, self.op_name_to_one_hot, plan_parameters, self.feature_statistics)
            seqs.append(seq_encoding) 
            attns.append(attention_mask)
        seqs = torch.stack(seqs, dim=0)
        attns = torch.stack(attns, dim=0)

        return seqs, None, attns, None, None
    
    def get_calibrations(self, seqs, attns):
        with torch.no_grad():
            # cost_iter
            self.__model.eval() # 关闭 drop out，否则模型波动大    
            seqs = seqs.to(DEVICE)
            attns = attns.to(DEVICE)
            cali = self.__model(seqs, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
            cali_all = cali[:, 0] # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
        # print(cali_all)
        return cali_all
    
    def encoding(self, X): 
        seqs, _, attns, _, _ = self.plans_encoding(X)
        return seqs, attns
    
    def inference(self, seqs, attns):
        cali_all = self.get_calibrations(seqs, attns)
        
        print(cali_all)
        # cali_str = ['{:.2f}'.format(i) for i in cali_all.tolist()] # 最后一次 cali
        cali_str = ['{:.2f}'.format(9.99 if i * 10 >= 10 else i * 10) for i in cali_all.tolist()] # 最后一次 cali
        # print("cali_str len", len(cali_str))
        cali_strs = ','.join(cali_str)
        return cali_strs
    
    def load_model(self, path):
        if not os.path.exists(path):
            model = SeqFormer(
                input_dim=configs['node_length'],
                hidden_dim=128,
                output_dim=1,
                mlp_activation="ReLU",
                transformer_activation="gelu",
                mlp_dropout=0.3,
                transformer_dropout=0.2,
                ).to(DEVICE) # server.py 和 train.py 中的模型初始化也需要相同, 这里还没加上！！！
            torch.save(model, path)
        else:
            model = torch.load(path, map_location='cuda:3')
            # ckpt = ckpt["state_dict"]
            # new_state_dict = OrderedDict()
            # for key, value in ckpt.items():
            # new_key = key.replace("model.", "")
            # new_state_dict[new_key] = value
            # self.net.load_state_dict(new_state_dict, strict=False)
        return model
    
    def infer_equ(self, messages):
        temp, self.eqset = ray.get(self.writer_hander.load_model.remote())
        if temp:
            print(self.eqset)
            self.__model = self.load_model("./log/model.pth")
        X = messages
        if not isinstance(X, list):
            X = [X]
        Relation_IDs = X[0]['Relation IDs']
        out = ','.join(token for token in Relation_IDs.split())
        if out in self.eqset:
            print(out)
            return '1'
        else:
            return '0'


    def predict_plan(self, messages):
      
        # json解析
        print("Predicting plan for ", len(messages))
        X = messages
        if not isinstance(X, list):
            X = [X]
        for x in X:
            if not x:
                return ','.join(['1.00' for _ in X])
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        # print(X)
        try:
            ray.get(self.writer_hander.Add_task.remote()) 
            # self.writer_hander.recieved_task += 1 需要试一下能不能直接成员变量 +1?
            self.writer_hander.write_file.remote(X)
        except:
            print("The ray writer_hander cannot write file.")

        # 编码
        seqs, attns = self.encoding(X)

        # 推理
        cali_strs = self.inference(seqs, attns)

        print(cali_strs)
        if X[0]['Plan']['Relation IDs'] == 'cn mc':
            print(X[2])
            print(X[3])
        return cali_strs

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
                result = self.server.leon_model.infer_equ(self.__messages)
                # print(result)
                response = str(result).encode()
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
    torch.multiprocessing.set_start_method('spawn')


    config = read_config()
    port = int(config["Port"])
    listen_on = config["ListenOn"]

    print(f"Listening on {listen_on} port {port}")
    
    server = Process(target=start_server, args=[listen_on, port])
    
    print("Spawning server process...")
    server.start()