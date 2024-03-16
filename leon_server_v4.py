import gc
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
import copy
import re
import math
import time
from util.envs import PlanToNode
from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers
import wandb
from leon_experience import *
import json
import pickle
import ray
import uuid
from config import read_config
from util import treeconv
import torch.nn.functional as F
conf = read_config()

def is_subset(s1, s2):
    s1 = set(s1.split(','))
    s2 = set(s2.split(','))
    if abs(len(s1) - len(s2)) != 1:
        return False
    result1 = s1.issubset(s2)
    result2 = s2.issubset(s1)
    return result1 or result2
@ray.remote
class TaskCounter:
    def __init__(self):
        self.recieved_task = 0
        self.RELOAD = True
        self.eq_summary = dict()
        train_file, training_query = envs.load_train_files(conf['leon']['workload_type'])
        self.eqset = envs.find_alias(training_query)
        self.online_flag = False # True表示要message False表示不要
        port_list = eval(conf['leon']["other_leon_port"])
        self.port_dict = {key: True for key in port_list}

    def Add_task(self):
        self.recieved_task += 1
    
    def GetRecievedTask(self):
        return self.recieved_task
    
    def load_model(self):
        temp = self.RELOAD
        self.RELOAD = False
        return temp , self.eqset, self.eq_summary
    
    def load_port(self, port):
        temp = self.port_dict[port]
        self.port_dict[port] = False
        return temp , self.eqset, self.eq_summary
    
    def reload_model(self, eqset, eq_summary):
        self.RELOAD = True
        self.port_dict = {key: True for key in self.port_dict}
        self.eqset = eqset
        self.eq_summary = eq_summary

    def GetOnline(self):
        return self.online_flag
    
    def WriteOnline(self, flag: bool):
        self.online_flag = flag
    


@ray.remote
class FileWriter:
    def __init__(self, file_path):
        self.file_path = file_path
        self.completed_tasks = 0
        self.recieved_task = 0

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

    def GetCompletedTasks(self):
        return self.completed_tasks
    
    def complete_all_tasks(self):
        # print(self.completed_tasks)
        if self.completed_tasks == self.recieved_task:
            return True
        else:
            return False
    
def _batch(trees, indexes, padding_size=200):
    # 获取 batchsize
    batch_size = len(trees)
    tree_embedding_size = trees[0].size(1)
    # 初始化填充后的张量
    padded_trees = torch.zeros((batch_size, tree_embedding_size, padding_size))
    padded_indexes = torch.zeros((batch_size, padding_size, 1))

    for i in range(batch_size):
        # 获取当前样本的原始树和索引张量
        tree = trees[i]
        index = indexes[i]

        # 计算需要填充的列数
        padding_cols_tree = max(0, padding_size - tree.size(2))
        padding_cols_index = max(0, padding_size - index.size(1))

        # 使用 F.pad 进行填充
        padded_tree = F.pad(tree, (0, padding_cols_tree), value=0)
        padded_index = F.pad(index, (0, 0, 0 , padding_cols_index), value=0)

        # 将填充后的张量放入结果中
        padded_trees[i, :, :] = padded_tree
        padded_indexes[i, :, :] = padded_index

    return padded_trees, padded_indexes.long()    

    
# @ray.remote
# class QueryDict:
#     def __init__(self):
#         self.query_dict = dict()
#         self.sql_ids = None

#     def write_sql_id(self, sql_ids):
#         self.sql_ids = sql_ids
    
#     def write_query_id(self, query_id):
#         if self.sql_ids != [] and query_id not in self.query_dict.keys():
#             sql_id = self.sql_ids.pop(0)
#             self.query_dict[query_id] = sql_id
#             print(sql_id, query_id)
#             return True
#         elif self.sql_ids != []:
#             return True
#         else:
#             return False
    
#     def get_dict(self):
#         return self.query_dict


class LeonModel:

    def __init__(self):
        # 初始化
        self.__model = None
        GenerateUniqueNameSpace = lambda: str(uuid.uuid4())
        namespace = GenerateUniqueNameSpace()
        with open('./conf/namespace.txt', 'w') as f:
            f.write(namespace)
        context = ray.init(namespace=namespace, _temp_dir= conf['leon']['ray_path'] + "/log/ray") # ray should be init in sub process  
        print(context.address_info)
        ray_address = context.address_info['address']
        with open('./conf/ray_address.txt', 'w') as f:
            f.write(ray_address)
        node_path = "./log/messages.pkl"
        self.writer_hander = FileWriter.options(name="leon_server").remote(node_path)
        self.task_counter = TaskCounter.options(name="counter").remote()
        # self.query_dict = QueryDict.options(name="querydict").remote()
        # self.query_dict_flag = True
        self.eqset = None

        self.workload = envs.wordload_init(conf['leon']['workload_type'])
        self.queryFeaturizer = plans_lib.QueryFeaturizer(self.workload.workload_info)

        self.model_type = conf['leon']['model_type']
        if self.model_type == "TreeConv":
            self.nodeFeaturizer = plans_lib.TreeNodeFeaturizer_V2(self.workload.workload_info)
        elif self.model_type == "Transformer":
            statistics_file_path = "./statistics.json"
            self.feature_statistics = load_json(statistics_file_path)
            add_numerical_scalers(self.feature_statistics)
            self.op_name_to_one_hot = get_op_name_to_one_hot(self.feature_statistics)

        self.eq_summary = dict()
        self.current_eq_summary = None
        self.Current_Level = None
        self.Levels_Needed = None
        self.Query_Id = None
        self.Old_Query_Id = None
        self.explain_flag = False
        self.Old_Current_Level = None
        self.continuous_eqset = []
        print("finish init")
            

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
    
    def get_calibrations(self, seqs, attns, 
                         QueryFeature):
        with torch.no_grad():
            # cost_iter
            self.__model.eval() # 关闭 drop out，否则模型波动大    
            QueryFeature = QueryFeature.to(DEVICE)
            seqs = seqs.to(DEVICE)
            attns = attns.to(DEVICE)
            if self.model_type == "TreeConv":
                cali_all = torch.tanh(self.__model(QueryFeature, seqs, attns)).add(1).squeeze(1)
            elif self.model_type == "Transformer":
                cali = self.__model(seqs, attns, QueryFeature) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
                cali_all = cali[:, 0] # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
        
        return cali_all
    
    def encoding(self, X): 
        # if self.model_type == "Transformer":
        #     seqs, _, attns, _, _ = self.plans_encoding(X)
        #     OneNode = PlanToNode(self.workload, [X[0]])[0]
        #     plans_lib.GatherUnaryFiltersInfo(OneNode)
        #     postgres.EstimateFilterRows(OneNode)  
        #     OneQueryFeature = self.queryFeaturizer(OneNode)
        #     OneQueryFeature = torch.from_numpy(OneQueryFeature).unsqueeze(0)
        #     queryfeature = OneQueryFeature.repeat(seqs.shape[0], 1)
        #     return seqs, attns, queryfeature
        # elif self.model_type == "TreeConv":
        #     nodes = []
        #     null_nodes = []
        #     queryencoding = []
        #     for i in range(0, len(X)):
        #         node = postgres.ParsePostgresPlanJson_1(X[i], self.workload.workload_info.alias_to_names)
        #         if node.info['join_cond'] == ['']: # return 1.00
        #             return None, None, None
                
        #         plans_lib.GatherUnaryFiltersInfo(node)
        #         postgres.EstimateFilterRows(node)   
        #         null_node = plans_lib.Binarize(node)
        #         if i == 0:
        #             temp = node.to_sql(node.info['join_cond'], with_select_exprs=True)
        #             node.info['sql_str'] = temp
        #             query_vecs = torch.from_numpy(self.queryFeaturizer(node)).unsqueeze(0)
        #         node.info['sql_str'] = temp
        #         queryencoding.append(query_vecs)
        #         nodes.append(node)
        #         null_nodes.append(null_node)
        #     if nodes:
        #         tensor_query_encoding = (torch.cat(queryencoding, dim=0))
        #         trees, indexes = encoding.TreeConvFeaturize(self.nodeFeaturizer, null_nodes)
        #         # print("tensor_query_encoding", tensor_query_encoding.shape,
        #         #     "trees", trees.shape,
        #         #     "indexes", indexes.shape)
        #     return tensor_query_encoding, trees, indexes
        if self.model_type == "Transformer":
            encoded_plans, attns, queryfeature, _ = envs.leon_encoding(self.model_type, X, 
                                                                           require_nodes=False, workload=self.workload, 
                                                                           configs=configs, op_name_to_one_hot=self.op_name_to_one_hot,
                                                                           plan_parameters=plan_parameters, feature_statistics=self.feature_statistics)
            return encoded_plans, attns, queryfeature

        elif self.model_type == "TreeConv":
            trees, indexes, queryfeature, _ = envs.leon_encoding(self.model_type, X, 
                                                                           require_nodes=False, workload=self.workload, 
                                                                           queryFeaturizer=self.queryFeaturizer, nodeFeaturizer=self.nodeFeaturizer)
            if isinstance(trees, list):
                trees, indexes, = _batch(trees, indexes)
            return trees, indexes, queryfeature
    
    def inference(self, seqs, attns, QueryFeature=None):
        cali_all = self.get_calibrations(seqs, attns, QueryFeature)
        # cali_all = 1000 * torch.rand(cali_all.shape[0])
        # print(cali_all)
        # cali_str = ['{:.2f}'.format(i) for i in cali_all.tolist()] # 最后一次 cali
        def format_scientific_notation(number):
            str_number = "{:e}".format(number)
            mantissa, exponent = str_number.split('e')
            mantissa = 9.994 if float(mantissa) >= 9.995 else float(mantissa)
            mantissa = format(mantissa, '.2f')
            exponent = int(exponent)
            exponent = max(-9, min(9, exponent))
            result = "{},{},{:d}".format(mantissa, '1' if exponent >= 0 else '0', abs(exponent))
            return result
        cali_str = [format_scientific_notation(i) for i in cali_all.tolist()] # 最后一次 cali
        
        # print("cali_str len", len(cali_str))
        cali_strs = ';'.join(cali_str)
        return cali_strs
    
    def load_model(self, path):
        if not os.path.exists(path):
            if self.model_type == "Transformer":
                print("load transformer model")
                model = SeqFormer(
                    input_dim=configs['node_length'],
                    hidden_dim=256,
                    output_dim=1,
                    mlp_activation="ReLU",
                    transformer_activation="gelu",
                    mlp_dropout=0.1,
                    transformer_dropout=0.1,
                    query_dim=configs['query_dim'],
                    padding_size=configs['pad_length']
                    ).to(DEVICE) # server.py 和 train.py 中的模型初始化也需要相同, 这里还没加上！！！
            elif self.model_type == "TreeConv":
                print("load treeconv model")
                model = treeconv.TreeConvolution(666, 50, 1).to(DEVICE)
            torch.save(model, path)
        else:
            model = torch.load(path, map_location=DEVICE)
            print(f"load checkpoint {path} Successfully!")
        return model
    
    def infer_equ(self, messages):
        temp, self.eqset, self.eq_summary = ray.get(self.task_counter.load_model.remote())
        if temp:
            print(self.eqset)
            self.__model = self.load_model("./log/model.pth")
        X = messages
        if not isinstance(X, list):
            X = [X]
        Relation_IDs = X[0]['Relation IDs']
        self.Current_Level = X[0]['Current Level']
        self.Levels_Needed = X[0]['Levels Needed']
        self.Query_Id = X[0]['QueryId']
        if self.Query_Id != self.Old_Query_Id:
            print("1")
            self.Old_Query_Id = self.Query_Id
            self.continuous_eqset = []
            self.explain_flag = True
        # if self.query_dict_flag:
        #     self.query_dict_flag = ray.get(self.query_dict.write_query_id.remote(X[0]['QueryId']))
        out = ','.join(sorted(Relation_IDs.split()))
        if (out in self.eqset) and (self.explain_flag or (out in self.continuous_eqset)): # and (self.Current_Level == self.Levels_Needed)
            if self.explain_flag:
                if self.continuous_eqset == []:
                    self.continuous_eqset.append(out)
                    self.Old_Current_Level = self.Current_Level
                else:
                    if self.Current_Level - self.Old_Current_Level <= 1:
                        self.continuous_eqset.append(out)
                        self.Old_Current_Level = self.Current_Level
                    else:
                        self.continuous_eqset = []
                        self.continuous_eqset.append(out)
                        self.Old_Current_Level = self.Current_Level
                if self.Current_Level == self.Levels_Needed:
                    self.explain_flag = False
                    temp_list = [0] * len(self.continuous_eqset)
                    temp_list[-1] = 1
                    for i, eq_1 in enumerate(self.continuous_eqset):
                        for eq_2 in self.continuous_eqset[i+1:]:
                            if is_subset(eq_1, eq_2):
                                temp_list[i] = 1
                    new_list = []
                    for i in range(0, len(self.continuous_eqset)):
                        if temp_list[i]:
                            new_list.append(self.continuous_eqset[i])
                    self.continuous_eqset = new_list
                    print(self.continuous_eqset)
            if not self.eq_summary:
                self.current_eq_summary = 1
            else:
                self.current_eq_summary = self.eq_summary.get(out)
            self.curr_eqset = out
            print(X[0]['QueryId'], out)
            print(self.Current_Level, self.Levels_Needed)
            return '1'
        else:
            return '0'


    def predict_plan(self, messages):
        
        # json解析
        print("Predicting plan for ", len(messages))
        X = messages
        if not isinstance(X, list):
            X = [X]
        # for x in X:
        #     if not x:
        #         return ','.join(['1.00' for _ in X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        # print(X[0])

        if self.Query_Id.startswith("picknode:"):
            temp_id = self.Query_Id[len("picknode:"):]
            parts = temp_id.split(";")
            curr_level = parts[0]
            pick_plan = float(parts[1])
            # print(curr_level, self.curr_eqset)
            if curr_level == self.curr_eqset:
                # for x in X:
                #     if(float(x['Plan']['Total Cost']) == 763602.06796):
                #         print(postgres.ParsePostgresPlanJson_1(x, self.workload.workload_info.alias_to_names))
                #         # print(x)
                #     print(x['Plan']['Total Cost'])
                # change_flag = False
                # print("Success begin hint leon", self.Query_Id)
                # for j, i in enumerate(X):
                #     if i['Plan']['Total Cost'] == pick_plan:
                #         print(i['Plan']['Total Cost'], pick_plan, j)
                #         change_flag = True
                # if change_flag == False:
                #     print("change_flag",self.Query_Id)
                #     with open("./error.pkl", 'wb') as f:
                #         pickle.dump(X, f) 
                # print("Success exec hint leon", self.Query_Id)
                return ';'.join(['1.00,1,0' if i['Plan']['Total Cost'] != pick_plan else '0.01,0,9' for i in X]) + ';'
            else:
                return ';'.join(['1.00,1,0' for _ in X]) + ';'

        try:
            # if ray.get(self.task_counter.GetOnline.remote()) and self.Current_Level == self.Levels_Needed:
            if ray.get(self.task_counter.GetOnline.remote()):
                self.task_counter.Add_task.remote()
                # self.writer_hander.recieved_task += 1 需要试一下能不能直接成员变量 +1?
                self.writer_hander.write_file.remote(X)
        except:
            print("The ray writer_hander cannot write file.")

        # Validation Accuracy
        # TODO: 可能不在Eq Summary里面？确实有可能，有些等价类没有被训练到，因为没有收集message
        # if self.current_eq_summary is None or \
        #     self.current_eq_summary[0] < 0.9:
        #     print("out")
        #     return ';'.join(['1.00,1,0' for _ in X])
        # 新添加的等价类，但没有训练    
        if self.current_eq_summary is None:
            print("out")
            return ';'.join(['1.00,1,0' for _ in X]) + ';'

        # 编码
        seqs, attns, QueryFeature = self.encoding(X)

        # 推理
        cali_strs = self.inference(seqs, attns, QueryFeature)
        # del seqs, attns, QueryFeature
        # gc.collect()
        # torch.cuda.empty_cache()
        print("out")
        return cali_strs + ';'
    

class SimpleLeonModel:

    def __init__(self, model_port):
        # 初始化
        self.__model = None
        with open ("./conf/namespace.txt", "r") as file:
            namespace = file.read().replace('\n', '')
        with open ("./conf/ray_address.txt", "r") as file:
            ray_address = file.read().replace('\n', '')
        context = ray.init(address=ray_address, namespace=namespace, _temp_dir=conf['leon']['ray_path'] + "/log/ray") # init only once
        print(context.address_info)
        
        self.eqset = None
        self.workload = envs.wordload_init(conf['leon']['workload_type'])
        self.queryFeaturizer = plans_lib.QueryFeaturizer(self.workload.workload_info)
        self.model_type = conf['leon']['model_type']
        if self.model_type == "TreeConv":
            self.nodeFeaturizer = plans_lib.TreeNodeFeaturizer_V2(self.workload.workload_info)
        elif self.model_type == "Transformer":
            statistics_file_path = "./statistics.json"
            self.feature_statistics = load_json(statistics_file_path)
            add_numerical_scalers(self.feature_statistics)
            self.op_name_to_one_hot = get_op_name_to_one_hot(self.feature_statistics)
        self.task_counter = ray.get_actor('counter')
        self.eq_summary = dict()
        self.current_eq_summary = None
        self.Current_Level = None
        self.Levels_Needed = None
        self.Query_Id = None
        self.Old_Query_Id = None
        self.explain_flag = False
        self.Old_Current_Level = None
        self.continuous_eqset = []
        self.model_port = model_port
        print("finish init")
            

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
    
    def get_calibrations(self, seqs, attns, 
                         QueryFeature):
        with torch.no_grad():
            # cost_iter
            self.__model.eval() # 关闭 drop out，否则模型波动大    
            QueryFeature = QueryFeature.to(DEVICE)
            seqs = seqs.to(DEVICE)
            attns = attns.to(DEVICE)
            if self.model_type == "TreeConv":
                cali_all = torch.tanh(self.__model(QueryFeature, seqs, attns)).add(1).squeeze(1)
            elif self.model_type == "Transformer":
                cali = self.__model(seqs, attns, QueryFeature) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
                cali_all = cali[:, 0] # [# of plan] -> [# of plan, 1] cali_all plan （cost_iter次）基数估计（归一化后）结果
        
        return cali_all
    
    def encoding(self, X): 
        if self.model_type == "Transformer":
            encoded_plans, attns, queryfeature, _ = envs.leon_encoding(self.model_type, X, 
                                                                           require_nodes=False, workload=self.workload, 
                                                                           configs=configs, op_name_to_one_hot=self.op_name_to_one_hot,
                                                                           plan_parameters=plan_parameters, feature_statistics=self.feature_statistics)
            return encoded_plans, attns, queryfeature

        elif self.model_type == "TreeConv":
            trees, indexes, queryfeature, _ = envs.leon_encoding(self.model_type, X, 
                                                                           require_nodes=False, workload=self.workload, 
                                                                           queryFeaturizer=self.queryFeaturizer, nodeFeaturizer=self.nodeFeaturizer)
            if isinstance(trees, list):
                trees, indexes, = _batch(trees, indexes)
            return trees, indexes, queryfeature
    
    def inference(self, seqs, attns, QueryFeature=None):
        cali_all = self.get_calibrations(seqs, attns, QueryFeature)
        def format_scientific_notation(number):
            str_number = "{:e}".format(number)
            mantissa, exponent = str_number.split('e')
            mantissa = 9.994 if float(mantissa) >= 9.995 else float(mantissa)
            mantissa = format(mantissa, '.2f')
            exponent = int(exponent)
            exponent = max(-9, min(9, exponent))
            result = "{},{},{:d}".format(mantissa, '1' if exponent >= 0 else '0', abs(exponent))
            return result
        cali_str = [format_scientific_notation(i) for i in cali_all.tolist()] # 最后一次 cali
        cali_strs = ';'.join(cali_str)
        return cali_strs
    
    def load_model(self, path):
        if not os.path.exists(path):
            if self.model_type == "Transformer":
                print("load transformer model")
                model = SeqFormer(
                    input_dim=configs['node_length'],
                    hidden_dim=256,
                    output_dim=1,
                    mlp_activation="ReLU",
                    transformer_activation="gelu",
                    mlp_dropout=0.1,
                    transformer_dropout=0.1,
                    query_dim=configs['query_dim'],
                    padding_size=configs['pad_length']
                    ).to(DEVICE) # server.py 和 train.py 中的模型初始化也需要相同, 这里还没加上！！！
            elif self.model_type == "TreeConv":
                print("load treeconv model")
                model = treeconv.TreeConvolution(666, 50, 1).to(DEVICE)
            torch.save(model, path)
        else:
            model = torch.load(path, map_location=DEVICE)
            print(f"load checkpoint {path} Successfully!")
        return model
    
    def infer_equ(self, messages):
        temp, self.eqset, self.eq_summary = ray.get(self.task_counter.load_port.remote(self.model_port))
        if temp:
            print(self.eqset)
            self.__model = self.load_model("./log/model.pth")
        X = messages
        if not isinstance(X, list):
            X = [X]
        Relation_IDs = X[0]['Relation IDs']
        self.Current_Level = X[0]['Current Level']
        self.Levels_Needed = X[0]['Levels Needed']
        self.Query_Id = X[0]['QueryId']
        if self.Query_Id != self.Old_Query_Id:
            print("1")
            self.Old_Query_Id = self.Query_Id
            self.continuous_eqset = []
            self.explain_flag = True
        # if self.query_dict_flag:
        #     self.query_dict_flag = ray.get(self.query_dict.write_query_id.remote(X[0]['QueryId']))
        out = ','.join(sorted(Relation_IDs.split()))
        if (out in self.eqset) and (self.explain_flag or (out in self.continuous_eqset)): # and (self.Current_Level == self.Levels_Needed)
            if self.explain_flag:
                if self.continuous_eqset == []:
                    self.continuous_eqset.append(out)
                    self.Old_Current_Level = self.Current_Level
                else:
                    if self.Current_Level - self.Old_Current_Level <= 1:
                        self.continuous_eqset.append(out)
                        self.Old_Current_Level = self.Current_Level
                    else:
                        self.continuous_eqset = []
                        self.continuous_eqset.append(out)
                        self.Old_Current_Level = self.Current_Level
                if self.Current_Level == self.Levels_Needed:
                    self.explain_flag = False
                    temp_list = [0] * len(self.continuous_eqset)
                    temp_list[-1] = 1
                    for i, eq_1 in enumerate(self.continuous_eqset):
                        for eq_2 in self.continuous_eqset[i+1:]:
                            if is_subset(eq_1, eq_2):
                                temp_list[i] = 1
                    new_list = []
                    for i in range(0, len(self.continuous_eqset)):
                        if temp_list[i]:
                            new_list.append(self.continuous_eqset[i])
                    self.continuous_eqset = new_list
                    print(self.continuous_eqset)
            if not self.eq_summary:
                self.current_eq_summary = 1
            else:
                self.current_eq_summary = self.eq_summary.get(out)
            self.curr_eqset = out
            print(X[0]['QueryId'], out)
            print(self.Current_Level, self.Levels_Needed)
            return '1'
        else:
            return '0'


    def predict_plan(self, messages):
        
        # json解析
        print("Predicting plan for ", len(messages), self.model_port)
        X = messages
        if not isinstance(X, list):
            X = [X]
        # for x in X:
        #     if not x:
        #         return ','.join(['1.00' for _ in X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        # print(X[0])
        if self.Query_Id.startswith("picknode:"):
            temp_id = self.Query_Id[len("picknode:"):]
            parts = temp_id.split(";")
            curr_level = parts[0]
            pick_plan = float(parts[1])
            if curr_level == self.curr_eqset:
                # change_flag = False
                # print("Success begin hint leon", self.Query_Id)
                # for j, i in enumerate(X):
                #     if i['Plan']['Total Cost'] == pick_plan:
                #         print(i['Plan']['Total Cost'], pick_plan, j)
                #         change_flag = True
                # if change_flag == False:
                #     print("change_flag",self.Query_Id)
                #     with open("./error.pkl", 'wb') as f:
                #         pickle.dump(X, f) 
                # print("Success exec hint leon", self.Query_Id)
                return ';'.join(['1.00,1,0' if i['Plan']['Total Cost'] != pick_plan else '0.01,0,9' for i in X]) + ';'
            else:
                return ';'.join(['1.00,1,0' for _ in X]) + ';'

        # Validation Accuracy
        # TODO: 可能不在Eq Summary里面？确实有可能，有些等价类没有被训练到，因为没有收集message
        # if self.current_eq_summary is None or \
        #     self.current_eq_summary[0] < 0.9:
        #     print("out")
        #     return ';'.join(['1.00,1,0' for _ in X])
        # 新添加的等价类，但没有训练    
        if self.current_eq_summary is None:
            print("out")
            return ';'.join(['1.00,1,0' for _ in X]) + ';'

        # 编码
        seqs, attns, QueryFeature = self.encoding(X)

        # 推理
        cali_strs = self.inference(seqs, attns, QueryFeature)
        # del seqs, attns, QueryFeature
        # gc.collect()
        # torch.cuda.empty_cache()
        print("out")
        return cali_strs + ';'

class JSONTCPHandler(socketserver.BaseRequestHandler):
    def handle(self):
        str_buf = ""
        while True:
            # 这里只有断连才会退出
            str_buf += self.request.recv(81960).decode("UTF-8")
            if not str_buf:
                # no more data, connection is finished.
                return
            
            if (null_loc := str_buf.find("\n")) != -1:
                json_msg = str_buf[:null_loc].strip()
                str_buf = str_buf[null_loc + 1:]
                if json_msg:
                    try:    
                        def fix_json_msg(json):
                            pattern = r'(ANY|ALL) \((.*?):text\[\]\)'
                            matches = re.findall(pattern, json)
                            for _, match in matches:
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
                print(len(result)/9)
                response = str(result).encode()
                # self.request.sendall(struct.pack("I", result))
                try:
                    self.request.sendall(response)
                except Exception as e:
                    print(f"发送响应时出错：{e}")
                finally:
                    self.request.close()

                # chunk_size = 1024  # 设置每个块的大小

                # try:
                #     while response:
                #         chunk = response[:chunk_size]
                #         self.request.sendall(chunk)
                #         response = response[chunk_size:]
                # except Exception as e:
                #     print(f"发送响应时出错：{e}")
                # finally:
                #     self.request.close()
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

def other_server(listen_on, port):
    model = SimpleLeonModel(port)
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((listen_on, port), LeonJSONHandler) as server:
        server.leon_model = model
        server.serve_forever()


if __name__ == "__main__":
    from multiprocessing import Process
    from config import read_config
    torch.multiprocessing.set_start_method('spawn')


    config = read_config()
    port = int(config['leon']["Port"])
    listen_on = config['leon']["ListenOn"]
    other_leon_port = eval(config['leon']["other_leon_port"])
    print(f"Listening on {listen_on} port {port}")
    
    server = Process(target=start_server, args=[listen_on, port])
    othersevers = []
    for i in other_leon_port:
        othersevers.append(Process(target=other_server, args=[listen_on, i]))

    print("Spawning server process...")
    server.start()
    time.sleep(10)
    for otherserver in othersevers:
        otherserver.start()