# import re
# json_msg = text='{"Plans": [{"Node Type": "SeqScan","Node Type ID": "169","Relation IDs": "kt","Base Restrict Info": "kt.kind::text = ANY (\'{movie,"tv movie","video movie","video game"}\'::text[])"}, ,"wyz": "ci.note = ANY (\'{(writer),"(head writer)","(written by)",(story),"(story editor)"}\'::text[])"}'
# def fix_json_msg(json):
#     # pattern = r'ANY \((.*?)\)'
#     pattern = r'ANY \((.*?):text\[\]\)'
#     matches = re.findall(pattern, json)
#     for match in matches:
#         print(match)
#         extracted_string = match
#         cleaned_string = extracted_string.replace('"', '')
#         json = json.replace(extracted_string, cleaned_string)
#     return json
# json_msg = fix_json_msg(json_msg)
# print(json_msg)

# import torch
# import torch.nn as nn
# query_feats = torch.rand((30, 820))
# trees = torch.rand((30, 123, 8))
# indexes = torch.randint(0, 1 ,size=(30, 21, 1))

# query_mlp = nn.Sequential(
#             nn.Linear(820, 128),
#             nn.LayerNorm(128),
#             nn.LeakyReLU(),
#             nn.Linear(128, 64),
#             nn.LayerNorm(64),
#             nn.LeakyReLU(),
#             nn.Linear(64, 32),
#         )
# weights = nn.Conv1d(155, 512, kernel_size=3, stride=3)
# query_embs = query_mlp(query_feats.unsqueeze(1))
# query_embs = query_embs.transpose(1, 2)
# max_subtrees = trees.shape[-1]

# query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1], max_subtrees)
# print("query_embs", query_embs.shape)
# concat = torch.cat((query_embs, trees), axis=1)
# print("concat", concat.shape)
# gather = torch.gather(concat, 2, indexes.expand(-1, -1, 155).transpose(1, 2))
# print("gather", gather.shape)
# feats = weights(gather)
# print("feats", feats.shape)

import torch
import loralib as lora
import torch.nn as nn
import numpy as np
from torch.nn.functional import pad
import torch.nn.functional as F
import lightning.pytorch as pl

class SeqFormer(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        mlp_activation="ReLU",
        transformer_activation="gelu",
        mlp_dropout=0.3,
        transformer_dropout=0.2,
    ):
        super(SeqFormer, self).__init__()
        # input_dim: node bits
        self.tranformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=input_dim,
                dim_feedforward=hidden_dim,
                nhead=1,
                batch_first=True,
                activation=transformer_activation,
                dropout=transformer_dropout,
            ),
            num_layers=1,
        )
        self.node_length = input_dim
        if mlp_activation == "ReLU":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "GELU":
            self.mlp_activation = nn.GELU()
        elif mlp_activation == "LeakyReLU":
            self.mlp_activation = nn.LeakyReLU()
        # self.mlp_hidden_dims = [128, 64, 32]
        self.mlp_hidden_dims = [128, 64, 1]
        self.mlp = nn.Sequential(
            *[
                lora.Linear(self.node_length, self.mlp_hidden_dims[0], r=16),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                lora.Linear(self.mlp_hidden_dims[0], self.mlp_hidden_dims[1], r=8),
                nn.Dropout(mlp_dropout),
                self.mlp_activation,
                lora.Linear(self.mlp_hidden_dims[1], output_dim, r=4),
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attn_mask=None):
        # change x shape to (batch, seq_len, input_size) from (batch, len)
        # one node is 18 bits
        x = x.view(x.shape[0], -1, self.node_length)
        # attn_mask = attn_mask.repeat(4,1,1)
        out = self.tranformer_encoder(x, mask=attn_mask)
        # out = self.transformer_decoder(out, out, tgt_mask=attn_mask)
        out = self.mlp(out)
        out = self.sigmoid(out).squeeze(dim=2)
        return out * 2 # [0, 1] -> [1, 2] [??]
class PL_DACE(pl.LightningModule):
    def __init__(self, model):
        super(PL_DACE, self).__init__()
        self.model = model

    def forward(self, x, attn_mask=None):
        return self.model(x, attn_mask)

    def DACE_loss(self, est_run_times, run_times, loss_mask):
        # est_run_times: (batch, seq_len)
        # run_times: (batch, seq_len)
        # seqs_length: (batch,)
        # return: loss (batch,)
        # don't calculate the loss of padding nodes, set them to 0
        loss = torch.max(est_run_times / run_times, run_times / est_run_times)
        loss = loss * loss_mask
        loss = torch.log(torch.where(loss > 1, loss, 1))
        loss = torch.sum(loss, dim=1)
        return loss

    def training_step(self, batch, batch_idx):
        seqs_padded, attn_masks, loss_mask, run_times = batch
        est_run_times = self.model(seqs_padded, attn_masks)
        loss = self.DACE_loss(est_run_times, run_times, loss_mask)
        loss = torch.mean(loss)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        seqs_padded, attn_masks, loss_mask, run_times = batch
        est_run_times = self.model(seqs_padded, attn_masks)
        est_run_times = est_run_times[:, 0]
        run_times = run_times[:, 0]
        # calculate q-error
        loss = 0
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        seqs_padded, attn_masks, loss_mask, run_times = batch
        est_run_times = self.model(seqs_padded, attn_masks)
        loss = self.DACE_loss(est_run_times, run_times, loss_mask)
        loss = torch.mean(loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
model = SeqFormer(18, 128, 1)
model = PL_DACE(model)
model_dict = torch.load("/data1/wyz/online/LEONForPostgres/checkpoints/DACE.ckpt")
model.load_state_dict(model_dict["state_dict"])

# import json


# def read_to_json():
#     # 打开文件的模式: 常用的有’r’（读取模式，缺省值）、‘w’（写入模式）、‘a’（追加模式）等
#     with open('./data3.json', 'r') as f:
#         # 使用json.dump()函数将序列化后的JSON格式的数据写入到文件中
#         data = json.load(f)
#     return data


# json_data = read_to_json()

# # 初始化平均值变量
# average_analysis_time = 0
# average_encoding_time = 0
# average_inference_time = 0

# # 计算平均值
# for key, value in json_data.items():
#     for entry in value:
#         average_analysis_time += entry["analysis_time"]
#         average_encoding_time += entry["encoding_time"]
#         average_inference_time += entry["inference_time"]

# # 获取总条目数
# total_entries = sum(len(value) for value in json_data.values())

# # 计算平均值
# average_analysis_time /= total_entries
# average_encoding_time /= total_entries
# average_inference_time /= total_entries

# # 打印结果
# print("Average Analysis Time:", average_analysis_time)
# print("Average Encoding Time:", average_encoding_time)
# print("Average Inference Time:", average_inference_time)