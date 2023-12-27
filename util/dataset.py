import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pickle
from leon_experience import Experience
import random
from util import plans_lib, postgres

# create dataset
class LeonDataset(Dataset):
    def __init__(self, labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2, queryfeature1=[], queryfeature2=[]):
        self.labels = labels
        self.costs1 = costs1
        self.costs2 = costs2
        self.encoded_plans1 = encoded_plans1
        self.encoded_plans2 = encoded_plans2
        self.attns1 = attns1
        self.attns2 = attns2

        self.queryfeature1 = queryfeature1
        self.queryfeature2 = queryfeature2

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        # if not self.queryfeature1:
        #     return {
        #         'labels': self.labels[idx],
        #         'costs1': self.costs1[idx],
        #         'costs2': self.costs2[idx],
        #         'encoded_plans1': self.encoded_plans1[idx],
        #         'encoded_plans2': self.encoded_plans2[idx],
        #         'attns1': self.attns1[idx],
        #         'attns2': self.attns2[idx]
        #     }
        # else:
        return {
            'labels': self.labels[idx],
            'costs1': self.costs1[idx],
            'costs2': self.costs2[idx],
            'encoded_plans1': self.encoded_plans1[idx],
            'encoded_plans2': self.encoded_plans2[idx],
            'attns1': self.attns1[idx],
            'attns2': self.attns2[idx],
            'queryfeature1': self.queryfeature1[idx],
            'queryfeature2': self.queryfeature2[idx]
            }
    
def prepare_dataset(pairs, queryFeaturizer=True):
    labels = []
    costs1 = []
    costs2 = []
    encoded_plans1 = []
    encoded_plans2 = []
    attns1 = []
    attns2 = []
    Nodes1 = []
    Nodes2 = []
    for pair in pairs:
        if pair[0][0].info['latency'] > pair[1][0].info['latency']:
            label = 0
        else:
            label = 1
        if queryFeaturizer:
            Nodes1.append(
                pair[0][0].info['query_feature']
            )
            Nodes2.append(
                pair[1][0].info['query_feature']
            )
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
    if Nodes1 or Nodes2:
        Nodes1 = torch.stack(Nodes1)
        Nodes2 = torch.stack(Nodes2)
    dataset = LeonDataset(labels, costs1, costs2, encoded_plans1, encoded_plans2, attns1, attns2, Nodes1, Nodes2)
    return dataset


class BucketDataset(Dataset):
    def __init__(self, buckets, keys=None):
        # filter buckets with in keys
        if keys:
            buckets = {key: value for key, value in buckets.items() if value and key in keys}
        else:
            buckets = {key: value for key, value in buckets.items() if value}
        self.buckets_dict = buckets
        
        self.buckets = list(buckets.values())
        # Flatten buckets
        self.buckets_item = [item for bucket in self.buckets_dict.values() for item in bucket]
        

    def __len__(self):
        return sum(len(bucket) for bucket in self.buckets)

    def __getitem__(self, idx):
        node, b, c = self.buckets_item[idx]
        item = {'join_tables': node.info['join_tables'], \
                'plan_encode': b, \
                'att_encode': c, \
                'latency': node.info['latency'], \
                'cost': node.cost,\
                'sql': node.info['sql_str'], \
                'queryfeature': node.info['query_feature']}
        return item

class BucketBatchSampler(Sampler):
    def __init__(self, buckets, batch_size):
        self.buckets = buckets
        self.bucket_indices = list(range(len(buckets)))
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.bucket_indices)
        for i in range(0, len(self.bucket_indices), self.batch_size):
            yield [item for bucket_idx in self.bucket_indices[i:i+self.batch_size] for item in range(sum(len(bucket) for bucket in self.buckets[:bucket_idx]), sum(len(bucket) for bucket in self.buckets[:bucket_idx+1]))]
            
    def __len__(self):
        return len(self.bucket_indices) // self.batch_size
    
    

