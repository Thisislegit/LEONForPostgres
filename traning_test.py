import pickle
import util.postgres as postgres
from util import envs
from util import plans_lib
import os
import torch
from test_case import *
from util.model import PL_Leon
from util.dataset import *
from leon_experience import TIME_OUT
import pytorch_lightning.loggers as pl_loggers
from config import read_config
import pytorch_lightning as pl

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
                        padding_size=configs['pad_length']
                    )

def Getpair(exp, key=None):
    pairs = []
    if key:
        for j in exp[key]:
            for k in exp[key]:
                ############################NEW
                if j[0].info['sql_str'] != k[0].info['sql_str'] and (j[0].info['latency'] == 90000 or k[0].info['latency'] == 90000):
                    continue    
                if j[0].cost == k[0].cost:
                        continue
                # if max(j[0].cost,k[0].cost) / min(j[0].cost,k[0].cost) < 1.2:
                #         continue
                # if j[0].cost == k[0].cost:
                #     continue

                if (j[0].info['sql_str'] == k[0].info['sql_str']) and (j[0].hint_str() == k[0].hint_str()): # sql 和 hint 都相同
                    continue
                # if (j[0].info['latency'] == k[0].info['latency']): # latency 相同 1s之内不把他train_pair
                if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.2:
                    continue
                # if j[0].info['latency'] == 90000 or k[0].info['latency'] == 90000:
                #     continue
                tem = [j, k]
                pairs.append(tem)
    else:          
        for eq in exp.keys():
            for j in exp[eq]:
                for k in exp[eq]:
                    if j[0].info['sql_str'] != k[0].info['sql_str'] and (j[0].info['latency'] == 90000 or k[0].info['latency'] == 90000):
                        continue
                    
                    if j[0].cost == k[0].cost:
                        continue
                    # if max(j[0].cost,k[0].cost) / min(j[0].cost,k[0].cost) < 1.2:
                    #     continue
                        
                    if (j[0].info['sql_str'] == k[0].info['sql_str']) and (j[0].hint_str() == k[0].hint_str()): # sql 和 hint 都相同
                        continue
                    # if (j[0].info['latency'] == k[0].info['latency']): # latency 相同 1s之内不把他train_pair
                    if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.2:
                        continue
                    # if j[0].info['latency'] == 90000 or k[0].info['latency'] == 90000:
                    #     continue
                    tem = [j, k]
                    pairs.append(tem)
    return pairs



with open('./log/exp_v4.pkl', 'rb') as f:
    exp1 = pickle.load(f)

with open('./log/exp_v1.pkl', 'rb') as f:
    exp2 = pickle.load(f)

key = None
prev_optimizer_state_dict = None
logger = pl_loggers.WandbLogger(save_dir=os.getcwd() + '/logs', name="test", project='leon3')
model = PL_Leon(Transformer_model)


train_pairs1 = Getpair(exp1, key=key)
leon_dataset1 = prepare_dataset(train_pairs1, True)
train_pairs2 = Getpair(exp2, key=key)
leon_dataset2 = prepare_dataset(train_pairs2, True)
dataloader_train = DataLoader(leon_dataset1, batch_size=256, shuffle=True, num_workers=0)
dataloader_val = DataLoader(leon_dataset2, batch_size=256, shuffle=False, num_workers=0)
# dataset_val = BucketDataset(exp1, keys=key)
# batch_sampler = BucketBatchSampler(dataset_val.buckets, batch_size=1)
# dataloader_val = DataLoader(dataset_val, batch_sampler=batch_sampler)

# model = load_model(model_path, prev_optimizer_state_dict).to(DEVICE)
model.optimizer_state_dict = prev_optimizer_state_dict
trainer = pl.Trainer(accelerator="gpu",
                    devices=[2],
                    max_epochs=100,
                    logger=logger)
trainer.fit(model, dataloader_train, dataloader_val)

prev_optimizer_state_dict = trainer.optimizers[0].state_dict()