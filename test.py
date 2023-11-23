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
import torch.nn.functional as F

# 假设有一个形状为 [32, 40, 155] 的张量
tensor = torch.rand((32, 40, 155))

# 计算填充的长度
padding_length = 128 - tensor.size(1)

# 使用 pad 函数进行填充
# 注意：这里假设 padding_length 是正数
tensor_padded = F.pad(tensor, (0, 0, 0, padding_length, 0, 0))

# 打印填充后的张量形状
print(tensor_padded.shape)