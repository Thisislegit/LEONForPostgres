import json
import struct
import socketserver
from utils import *
import util.envs as envs
import util.postgres as postgres
import util.plans_lib as plans_lib
import torch
import os
import util.DP as DP
import copy
import re


class LeonModel:

    def __init__(self):
        self.__model = None
        self.workload = envs.JoinOrderBenchmark(envs.JoinOrderBenchmark.Params())
        self.workload.workload_info.table_num_rows = postgres.GetAllTableNumRows(self.workload.workload_info.rel_names)
        self.workload.workload_info.alias_to_names = postgres.GetAllAliasToNames(self.workload.workload_info.rel_ids)
        print(self.workload.workload_info.scan_types)
        # print(self.workload.workload_info.alias_to_names)
        # print(self.workload.workload_info.rel_names)
        # print(self.workload.workload_info.table_num_rows)
        # print(self.workload.workload_info)
        # print(self.workload.workload_info.table_num_rows['a1'])
    
    def load_model(self, path):
        pass
    
    def predict_plan(self, messages):
        print("Predicting plan for ", len(messages))
        X = messages
        if not isinstance(X, list):
            X = [X]
        for x in X:
            if not x:
                return ','.join(['1.00' for _ in X])
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        nodes = []
        for i in range(0, len(messages)):
            node = postgres.ParsePostgresPlanJson_1(X[i], self.workload.workload_info.alias_to_names)
            if node.info['join_cond'] == ['']:
                break
            print(X[i])
            # print(node)
            node = plans_lib.FilterScansOrJoins(node)
            # print(node)
            plans_lib.GatherUnaryFiltersInfo(node)
            postgres.EstimateFilterRows(node)
            # print(node.info['all_filters_est_rows'])
            try:
                node.info['sql_str'] = node.to_sql(node.info['join_cond'], with_select_exprs=True)
                # print(node.info['sql_str'])
            except:
                print(node.info['join_cond'])
                print(X[i])
            # print(node.info['sql_str'])
            queryFeaturizer = plans_lib.QueryFeaturizer(self.workload.workload_info)
            query_vecs = torch.from_numpy(queryFeaturizer(node)).unsqueeze(0)
            node.info['query_encoding'] = copy.deepcopy(query_vecs)
            nodes.append(node)
        
            
        if nodes:
            nodeFeaturizer = plans_lib.PhysicalTreeNodeFeaturizer(self.workload.workload_info)
            trees, indexes = encoding.TreeConvFeaturize(nodeFeaturizer, nodes)
        # print(trees)
        

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