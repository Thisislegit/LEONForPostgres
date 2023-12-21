import sys
sys.path.append('./')
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import lightning.pytorch as pl
import lightning.pytorch.callbacks as plc
import torch.nn.functional as F
import torch.nn as nn
import os

import pytorch_lightning.loggers as pl_loggers
import pickle
import math
import json
from test_case import SeqFormer
from test_case import configs
import numpy as np
from argparse import ArgumentParser
import copy
import wandb
import random

DEVICE = 'cuda:2' if torch.cuda.is_available() else 'cpu'
Transformer_model = SeqFormer(
                        input_dim=configs['node_length'],
                        hidden_dim=128,
                        output_dim=1,
                        mlp_activation="ReLU",
                        transformer_activation="gelu",
                        mlp_dropout=0.3,
                        transformer_dropout=0.2,
                    )

def load_model(prev_optimizer_state_dict=None):
    model = Transformer_model.to(DEVICE)
    model = PL_Leon(model, prev_optimizer_state_dict)
    return model

class LeonDataset(Dataset):
    def __init__(self, labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2):
        self.labels = labels
        self.costs1 = costs1
        self.costs2 = costs2
        self.encoded_plans1 = encoded_plans1
        self.encoded_plans2 = encoded_plans2
        self.attns1 = attns1
        self.attns2 = attns2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.labels[idx],
                self.costs1[idx],
                self.costs2[idx],
                self.encoded_plans1[idx],
                self.encoded_plans2[idx],
                self.attns1[idx],
                self.attns2[idx])
    
def prepare_dataset(pairs):
    labels = []
    costs1 = []
    costs2 = []
    encoded_plans1 = []
    encoded_plans2 = []
    attns1 = []
    attns2 = []
    for pair in pairs:
        if pair[0][0].info['latency'] > pair[1][0].info['latency']:
            label = 0
        else:
            label = 1
        labels.append(label)
        costs1.append(pair[0][0].cost)
        costs2.append(pair[1][0].cost)
        encoded_plans1.append(pair[0][1])
        encoded_plans2.append(pair[1][1])
        attns1.append(pair[0][2])
        attns2.append(pair[1][2])
    labels = torch.tensor(labels)
    costs1 = torch.tensor(costs1)
    costs2 = torch.tensor(costs2)
    encoded_plans1 = torch.stack(encoded_plans1)
    encoded_plans2 = torch.stack(encoded_plans2)
    attns1 = torch.stack(attns1)
    attns2 = torch.stack(attns2)
    dataset = LeonDataset(labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2)
    return dataset

def load_callbacks(logger):
    callbacks = []
    callbacks.append(plc.EarlyStopping(
        monitor='v_acc',
        mode='max',
        patience=3,
        min_delta=0.001
    ))
    if logger:
        callbacks.append(plc.ModelCheckpoint(
            dirpath= logger.experiment.dir,
            monitor='val_scan',
            filename='best-{epoch:02d}-{val_scan:.3f}',
            save_top_k=1,
            mode='min',
            save_last=False
        ))
    return callbacks

class PL_Leon(pl.LightningModule):
    def __init__(self, model, optimizer_state_dict=None, learning_rate=0.001):
        super(PL_Leon, self).__init__()
        self.model = model
        self.optimizer_state_dict = optimizer_state_dict
        self.learning_rate = 0.001

    def forward(self, batch_pairs):
        pass

    def getBatchPairsLoss(self, labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2):
        """
        batch_pairs: a batch of train pairs
        return. a batch of loss
        """
        loss_fn = nn.BCELoss()
        # step 1. retrieve encoded_plans and attns from pairs


        # step 2. calculate batch_cali and calied_cost
        # 0是前比后大 1是后比前大
        batsize = costs1.shape[0]
        encoded_plans = torch.cat((encoded_plans1, encoded_plans2), dim=0)
        attns = torch.cat((attns1, attns2), dim=0)
        cali = self.model(encoded_plans, attns) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
        cali = cali[:, 0]
        costs = torch.cat((costs1, costs2), dim=0)
        # print(costs1)
        # print(costs2)
        # print(labels)
        # print(cali)
        calied_cost = torch.log(costs) * cali
        try:
            sigmoid = F.sigmoid(-(calied_cost[:batsize] - calied_cost[batsize:]))
            loss = loss_fn(sigmoid, labels.float())
        except:
            print(calied_cost, sigmoid)
        # print(loss)
        with torch.no_grad():
            prediction = torch.round(sigmoid)
            # print(prediction)
            accuracy = torch.sum(prediction == labels).item() / len(labels)
        # print(softm[:, 1].shape, labels.shape)
        return loss, accuracy

    def training_step(self, batch):
        labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2 = batch
        loss, acc  = self.getBatchPairsLoss(labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2)
        self.log_dict({'t_loss': loss, 't_acc': acc}, on_epoch=True)
        return loss

    def validation_step(self, batch):
        labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2 = batch
        loss, acc  = self.getBatchPairsLoss(labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2)
        self.log_dict({'v_loss': loss, 'v_acc': acc}, on_epoch=True)
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

def Getpair(exp):
    pairs = []
    for eq in exp.keys():
        for j in exp[eq]:
            for k in exp[eq]:
                if (j[0].info['sql_str'] == k[0].info['sql_str']) and (j[0].hint_str() == k[0].hint_str()): # sql 和 hint 都相同
                    continue
                if (j[0].info['latency'] == k[0].info['latency']): # latency 相同 1s之内不把他train_pair
                # if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.2:
                    continue
                tem = [j, k]
                pairs.append(tem)
    return pairs

if __name__ == '__main__':
    with open('./log/exp.pkl', 'rb') as f:
        exp = pickle.load(f)
    logger =  pl_loggers.WandbLogger(save_dir=os.getcwd() + '/logs', name="base", project='leon3')
    prev_optimizer_state_dict = None
    model = load_model().to(DEVICE)
    callbacks = load_callbacks(logger=None)
    train_pairs = Getpair(exp)
    print("len(train_pairs)" ,len(train_pairs))
    leon_dataset = prepare_dataset(train_pairs)
    dataloader_train = DataLoader(leon_dataset, batch_size=256, shuffle=True, num_workers=0)
    dataloader_val = DataLoader(leon_dataset, batch_size=256, shuffle=False, num_workers=0)
    trainer = pl.Trainer(accelerator="gpu",
                        devices=[2],
                        max_epochs=100,
                        callbacks=callbacks,
                        logger=logger)
    trainer.fit(model, dataloader_train, dataloader_val)
    prev_optimizer_state_dict = trainer.optimizers[0].state_dict()



    
