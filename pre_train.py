from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc
import torch.nn.functional as F
import torch.nn as nn
import os
from ray.util import ActorPool
from util import postgres
from util.pg_executor import ActorThatQueries, actor_call
from util import envs
from util.envs import load_sql, load_training_query
from util import plans_lib
import pytorch_lightning.loggers as pl_loggers
import pickle
import math
import json
from test_case import SeqFormer
from test_case import get_plan_encoding, configs, load_json, get_op_name_to_one_hot, plan_parameters, add_numerical_scalers
from leon_experience import Experience, TIME_OUT
from util.model import PL_Leon
import numpy as np
from ray.util.iter import from_items
import ray
import time
from argparse import ArgumentParser
import copy
import wandb
import time
import random
from tqdm import tqdm
from config import read_config

conf = read_config()
DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
Transformer_model = SeqFormer(
                        input_dim=configs['node_length'],
                        hidden_dim=256,
                        output_dim=1,
                        mlp_activation="ReLU",
                        transformer_activation="gelu",
                        mlp_dropout=0.1,
                        transformer_dropout=0.1,
                        query_dim=configs['query_dim'],
                        padding_size=configs['pad_length']
                    )


# UNFINISHED hint sql -> pgcosts
def plans_encoding(plans):
    '''
    input. a list of plans in type of json
    output. (seq_encoding, run_times, attention_mask, loss_mask)
        - seq_encoding torch.Size([1, 760])
    '''
    seqs = []
    attns = []
    for x in plans:        
        seq_encoding, run_times, attention_mask, loss_mask, database_id = get_plan_encoding(
            x, configs, op_name_to_one_hot, plan_parameters, feature_statistics
        ) # run_times 无法获取
        seqs.append(seq_encoding) 
        attns.append(attention_mask)

    seqs = torch.stack(seqs, dim=0)
    attns = torch.stack(attns, dim=0)

    return seqs, run_times, attns, loss_mask

def prepare_dataset(datas):
    labels = []
    encoded_plans = []
    attns = []
    for data in datas:
        label = data[0].cost
        labels.append(label)
        encoded_plans.append(data[1])
        attns.append(data[2])
    labels = torch.tensor(labels)
    encoded_plans = torch.cat(encoded_plans, dim=0)
    attns = torch.cat(attns, dim=0)
    dataset = PreTrainDataset(labels, encoded_plans, attns)
    return dataset

class PreTrainDataset(Dataset):
    def __init__(self, labels, encoded_plans, attns):
        self.labels = labels
        self.encoded_plans = encoded_plans
        self.attns = attns

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'labels': self.labels[idx],
            'encoded_plans': self.encoded_plans[idx],
            'attns': self.attns[idx],
            }

class PL_PreTrain(pl.LightningModule):
    def __init__(self, model, optimizer_state_dict=None, learning_rate=0.001):
        super(PL_PreTrain, self).__init__()
        self.model = model
        self.optimizer_state_dict = optimizer_state_dict
        self.learning_rate = learning_rate

    def forward(self, plans, attns):
        return self.model(plans, attns)[:, 0]

    def getBatchLoss(self, batch):
        labels = batch['labels']
        encoded_plans = batch['encoded_plans']
        attns = batch['attns']

        loss_fn = nn.MSELoss()
        output = self(encoded_plans, attns)
        loss = loss_fn(output, labels)
        
        return loss

    def training_step(self, batch):
        loss = self.getBatchLoss(batch)
        self.log_dict({'t_loss': loss}, on_epoch=True)
        return loss

    def validation_step(self, batch):
        loss = self.getBatchLoss(batch)
        self.log_dict({'v_loss': loss}, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        if self.optimizer_state_dict is not None:
            # Checks the params are the same.
            # 'params': [139581476513104, ...]
            curr = optimizer.state_dict()['param_groups'][0]['params']
            prev = self.optimizer_state_dict['param_groups'][0]['params']
            assert curr == prev, (curr, prev)
            # print('Loading last iter\'s optimizer state.')
            # Prev optimizer state's LR may be stale.
            optimizer.load_state_dict(self.optimizer_state_dict)
            for param_group in optimizer.param_groups:
                param_group['lr'] = self.learning_rate
            assert optimizer.state_dict(
            )['param_groups'][0]['lr'] == self.learning_rate
            # print('LR', self.learning_rate)
        return optimizer


if __name__ == '__main__':
    train_files = ['1a', '1b', '1c', '1d', '2a', '2b', '2c', '2d', '3a', '3b', '3c', '4a',
                    '4b', '4c', '5a', '5b', '5c', '6a', '6b', '6c', '6d', '6e', '6f', '7a', 
                    '7b', '7c', '8a', '8b', '8c', '8d', '9a', '9b', '9c', '9d', '10a', '10b', 
                    '10c', '11a', '11b', '11c', '11d', '12a', '12b', '12c', '13a', '13b', '13c', 
                    '13d', '14a', '14b', '14c', '15a', '15b', '15c', '15d', '16a', '16b', '16c',
                    '16d', '17a', '17b', '17c', '17d', '17e', '17f', '18a', '18b', '18c', '19a',
                    '19b', '19c', '19d', '20a', '20b', '20c', '21a', '21b', '21c', '22a', '22b',
                    '22c', '22d', '23a', '23b', '23c', '24a', '24b', '25a', '25b', '25c', '26a', 
                    '26b', '26c', '27a', '27b', '27c', '28a', '28b', '28c', '29a', '29b', '29c',
                    '30a', '30b', '30c', '31a', '31b', '31c', '32a', '32b', '33a', '33b', '33c']
    sqls_chunk = load_sql(train_files)
    training_data = []
    prev_optimizer_state_dict = None
    model = Transformer_model
    model = PL_PreTrain(model)
    statistics_file_path = "./statistics1.json"
    feature_statistics = load_json(statistics_file_path)
    add_numerical_scalers(feature_statistics)
    op_name_to_one_hot = get_op_name_to_one_hot(feature_statistics)
    max_cost = -float('inf')
    min_cost = float('inf')
    for i, q_send_cnt in enumerate(sqls_chunk):
        result = postgres.getPlans(q_send_cnt, None, check_hint_used=False, ENABLE_LEON=False, curr_file=train_files[i])
        json_dict = result[0][0][0]
        node = postgres.ParsePostgresPlanJson(json_dict)
        encoded_plans, _, attns, _ = plans_encoding([json_dict])
        training_data.append((node, encoded_plans, attns))
        max_cost = max(max_cost, node.cost)
        min_cost = min(min_cost, node.cost)

    for i, q_send_cnt in enumerate(sqls_chunk):
        training_data[i][0].cost = (training_data[i][0].cost - min_cost) / (max_cost - min_cost) * 2

    
    logger =  pl_loggers.WandbLogger(save_dir=os.getcwd() + '/logs', name="pretrain", project='leon4')
    my_step = 0

    dataset = prepare_dataset(training_data)
    dataloader_train = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    model.optimizer_state_dict = prev_optimizer_state_dict
    trainer = pl.Trainer(accelerator="gpu",
                        devices=[3],
                        enable_progress_bar=True,
                        max_epochs=30,
                        logger=logger,
                        fast_dev_run=False)
    trainer.fit(model, dataloader_train, dataloader_val)
    prev_optimizer_state_dict = trainer.optimizers[0].state_dict()
    model_path = "./log/model.pth" 
    torch.save(model.model, model_path)


    
