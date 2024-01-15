import pickle
import util.postgres as postgres
from util import envs
from util import plans_lib
import os
import torch
from test_case import *
from util.model import PL_Leon
from util.dataset import LeonDataset
from leon_experience import TIME_OUT
import pytorch_lightning.loggers as pl_loggers
from config import read_config
import pytorch_lightning as pl
from util import treeconv, encoding
from torch.utils.data import DataLoader
from leon_experience import TIME_OUT
from lightning.pytorch.strategies import DDPStrategy
import pytorch_lightning.callbacks as plc
conf = read_config()

pl.seed_everything(42)

def Getpair(exp, key=None):
    pairs = []
    if key:
        for j in exp[key]:
            for k in exp[key]:
                ############################NEW
                if j[0].info['sql_str'] != k[0].info['sql_str']: # (j[0].info['latency'] == TIME_OUT or k[0].info['latency'] == TIME_OUT)
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
                if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.05:
                    continue
                # if j[0].info['latency'] == 90000 or k[0].info['latency'] == 90000:
                #     continue
                tem = [j, k]
                pairs.append(tem)
    else:          
        for eq in exp.keys():
            for j in exp[eq]:
                for k in exp[eq]:
                    if j[0].info['sql_str'] != k[0].info['sql_str']:
                        continue
                    
                    if j[0].cost == k[0].cost:
                        continue
                    # if max(j[0].cost,k[0].cost) / min(j[0].cost,k[0].cost) < 1.2:
                    #     continue
                        
                    if (j[0].info['sql_str'] == k[0].info['sql_str']) and (j[0].hint_str() == k[0].hint_str()): # sql 和 hint 都相同
                        continue
                    # if (j[0].info['latency'] == k[0].info['latency']): # latency 相同 1s之内不把他train_pair
                    if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.05:
                        continue
                    # if j[0].info['latency'] == 90000 or k[0].info['latency'] == 90000:
                    #     continue
                    tem = [j, k]
                    pairs.append(tem)
    return pairs

def Getpair2(exp, key=None):
    pairs = []
    if key:
        for j in exp[key]:
                ############################NEW
                if j[0].info['sql_str'] != k[0].info['sql_str']:
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
                if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.05:
                    continue
                # if j[0].info['latency'] == 90000 or k[0].info['latency'] == 90000:
                #     continue
                tem = [j, k]
                pairs.append(tem)
    else:          
        for eq in exp.keys():
            min_dict = dict()
            for j in exp[eq]:
                if j[0].info['sql_str'] not in min_dict:
                    min_dict[j[0].info['sql_str']] = j
                else:
                    if j[0].cost < min_dict[j[0].info['sql_str']][0].cost:
                        min_dict[j[0].info['sql_str']] = j
            for j in exp[eq]:
                for k in min_dict.values():
                    if j[0].info['sql_str'] != k[0].info['sql_str']:
                        continue
                    if j[0].cost == k[0].cost:
                        continue
                    # if max(j[0].cost,k[0].cost) / min(j[0].cost,k[0].cost) < 1.2:
                    #     continue
                        
                    if (j[0].info['sql_str'] == k[0].info['sql_str']) and (j[0].hint_str() == k[0].hint_str()): # sql 和 hint 都相同
                        continue
                    # if (j[0].info['latency'] == k[0].info['latency']): # latency 相同 1s之内不把他train_pair
                    if max(j[0].info['latency'],k[0].info['latency']) / min(j[0].info['latency'],k[0].info['latency']) < 1.05:
                        continue
                    # if j[0].info['latency'] == 90000 or k[0].info['latency'] == 90000:
                    #     continue
                    tem = [j, k]
                    pairs.append(tem)
    return pairs

def prepare_dataset(pairs, Shouldquery, nodeFeaturizer, queryFeaturizer, dict=None):
    labels = []
    costs1 = []
    costs2 = []
    # encoded_plans1 = []
    # encoded_plans2 = []
    # attns1 = []
    # attns2 = []
    Nodes1 = []
    Nodes2 = []
    for pair in pairs:
        if pair[0][0].info['latency'] > pair[1][0].info['latency']:
            label = 0
        else:
            label = 1
        if Shouldquery:
            Nodes1.append(
                # pair[0][0].info['query_feature']
                pair[0][0]
            )
            Nodes2.append(
                # pair[1][0].info['query_feature']
                pair[1][0]
            )
        labels.append(label)
        costs1.append(pair[0][0].cost)
        costs2.append(pair[1][0].cost)
        # encoded_plans1.append(pair[0][1])
        # encoded_plans2.append(pair[1][1])
        # attns1.append(pair[0][2])
        # attns2.append(pair[1][2])
    labels = torch.tensor(labels)
    costs1 = torch.tensor(costs1)
    costs2 = torch.tensor(costs2)
    # encoded_plans1 = torch.stack(encoded_plans1)
    # encoded_plans2 = torch.stack(encoded_plans2)
    # attns1 = torch.stack(attns1)
    # attns2 = torch.stack(attns2)
    # if Nodes1 or Nodes2:
    #     Nodes1 = torch.stack(Nodes1)
    #     Nodes2 = torch.stack(Nodes2)
    dataset = TestDataset(labels, costs1, costs2, Nodes1, Nodes2, nodeFeaturizer, queryFeaturizer, dict)
    return dataset

class TestDataset(LeonDataset):
    def __init__(self, labels, costs1, costs2, nodes1, nodes2, nodeFeaturizer=None, queryFeaturizer=None, dict=None):
        super().__init__(labels, costs1, costs2, nodes1, nodes2, nodeFeaturizer, dict)
        self.queryFeaturizer = queryFeaturizer

    def __getitem__(self, idx):
        
        # null_nodes1 = plans_lib.Binarize(self.nodes1[idx])
        # trees1, indexes1 = encoding.TreeConvFeaturize(self.nodeFeaturizer, [null_nodes1])
        trees1 = self.dict[self.nodes1[idx].info['index']][0].squeeze(0)
        indexes1 = self.dict[self.nodes1[idx].info['index']][1].squeeze(0)
        trees2 = self.dict[self.nodes2[idx].info['index']][0].squeeze(0)
        indexes2 = self.dict[self.nodes2[idx].info['index']][1].squeeze(0)
        # null_nodes2 = plans_lib.Binarize(self.nodes2[idx])
        # trees2, indexes2 = encoding.TreeConvFeaturize(self.nodeFeaturizer, [null_nodes2])
        query_feats1 = self.nodes1[idx].info['query_feature']
        query_feats2 = self.nodes2[idx].info['query_feature']
        return {
            'labels': self.labels[idx],
            'costs1': self.costs1[idx],
            'costs2': self.costs2[idx],
            'encoded_plans1': trees1,
            'encoded_plans2': trees2,
            'attns1': indexes1,
            'attns2': indexes2,
            'queryfeature1': query_feats1,
            'queryfeature2': query_feats2
        }
    
    # def __getitem__(self, idx):
    #     null_nodes1 = plans_lib.Binarize(self.nodes1[idx])
    #     trees1, indexes1 = encoding.TreeConvFeaturize(self.nodeFeaturizer, [null_nodes1])
    #     null_nodes2 = plans_lib.Binarize(self.nodes2[idx])
    #     trees2, indexes2 = encoding.TreeConvFeaturize(self.nodeFeaturizer, [null_nodes2])
    #     query_feats1 = torch.from_numpy(queryFeaturizer(self.nodes1[idx]))
    #     query_feats2 = torch.from_numpy(queryFeaturizer(self.nodes2[idx]))
    #     # query_feats1 = self.nodes1[idx].info['query_feature']
    #     # query_feats2 = self.nodes2[idx].info['query_feature']
    #     return {
    #         'labels': self.labels[idx],
    #         'costs1': self.costs1[idx],
    #         'costs2': self.costs2[idx],
    #         'encoded_plans1': trees1.squeeze(0),
    #         'encoded_plans2': trees2.squeeze(0),
    #         'attns1': indexes1.squeeze(0),
    #         'attns2': indexes2.squeeze(0),
    #         'queryfeature1': query_feats1,
    #         'queryfeature2': query_feats2
    #     }



class TreeConvolution(nn.Module):
    """Balsa's tree convolution neural net: (query, plan) -> value.

    Value is either cost or latency.
    """

    def __init__(self, feature_size, plan_size, label_size, version=None):
        super(TreeConvolution, self).__init__()
        # None: default
        assert version is None, version
        self.query_p = 0.3
        # self.query_mlp = nn.Sequential(
        #     nn.Linear(feature_size, 128),
        #     nn.Dropout(p=self.query_p),
        #     nn.LayerNorm(128),
        #     nn.LeakyReLU(),
        #     nn.Linear(128, 64),
        #     nn.Dropout(p=self.query_p),
        #     nn.LayerNorm(64),
        #     nn.LeakyReLU(),
        #     nn.Linear(64, 32),
        # )
        self.conv = nn.Sequential(
            # TreeConv1d(32 + plan_size, 512),
            TreeConv1d(plan_size, 512),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(512, 512),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeConv1d(512, 256),
            TreeStandardize(),
            TreeAct(nn.LeakyReLU()),
            TreeMaxPool(),
        )
        self.plan_p = 0.2
        self.out_mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.Dropout(p=self.plan_p),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=self.plan_p),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, label_size),
        )
        # self.reset_weights()
        self.apply(self._init_weights)
        self.model_type = "TreeConv"

    def reset_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1:
                # Weights/embeddings.
                nn.init.normal_(p, std=0.02)
            elif 'bias' in name:
                # Layer norm bias; linear bias, etc.
                nn.init.zeros_(p)
            else:
                # Layer norm weight.
                # assert 'norm' in name and 'weight' in name, name
                nn.init.ones_(p)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, query_feats, trees, indexes):
        """Forward pass.

        Args:
          query_feats: Query encoding vectors.  Shaped as
            [batch size, query dims].
          trees: The input plan features.  Shaped as
            [batch size, plan dims, max tree nodes].
          indexes: For Tree convolution.

        Returns:
          Predicted costs: Tensor of float, sized [batch size, 1].
        """
        # Give larger dropout to query features.
        # query_embs = nn.functional.dropout(query_feats, p=0.5)
        # query_embs = self.query_mlp(query_feats.unsqueeze(1))
        # query_embs = query_embs.transpose(1, 2)
        # max_subtrees = trees.shape[-1]
        # #    print(query_embs.shape)
        # query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
        #                                max_subtrees)
        # concat = torch.cat((query_embs, trees), axis=1)

        out = self.conv((trees, indexes)) # batchsize * 54 * 200 batchsize * 1 * 200
        out = self.out_mlp(out)
        return out


class TreeConv1d(nn.Module):
    """Conv1d adapted to tree data."""

    def __init__(self, in_dims, out_dims):
        super().__init__()
        self._in_dims = in_dims
        self._out_dims = out_dims
        self.weights = nn.Conv1d(in_dims, out_dims, kernel_size=3, stride=3)

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        data, indexes = trees
        feats = self.weights(
            torch.gather(data, 2,
                         indexes.expand(-1, -1, self._in_dims).transpose(1, 2)))
        zeros = torch.zeros((data.shape[0], self._out_dims)).unsqueeze(2).to(feats.device)
        feats = torch.cat((zeros, feats), dim=2)
        return feats, indexes


class TreeMaxPool(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return trees[0].max(dim=2).values


class TreeAct(nn.Module):

    def __init__(self, activation):
        super().__init__()
        self.activation = activation

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return self.activation(trees[0]), trees[1]

class TreeMaxPool_With_Kernel_Stride(nn.Module):

    def __init__(self, kernel_size=3, stride=3, padding=0):
        super(TreeMaxPool_With_Kernel_Stride, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        data, indexes = trees
        feats = F.max_pool1d(torch.gather(data, 2,
                         indexes.expand(-1, -1, data.shape[1]).transpose(1, 2)), kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        zeros = torch.zeros((data.shape[0], data.shape[1])).unsqueeze(2).to(feats.device)
        feats = torch.cat((zeros, feats), dim=2)
        return (feats, indexes)



class TreeStandardize(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        mu = torch.mean(trees[0], dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        s = torch.std(trees[0], dim=(1, 2)).unsqueeze(1).unsqueeze(1)
        standardized = (trees[0] - mu) / (s + 1e-5)
        return standardized, trees[1]


def ReportModel(model, blacklist=None):
    ps = []
    for name, p in model.named_parameters():
        if blacklist is None or blacklist not in name:
            ps.append(np.prod(p.size()))
    num_params = sum(ps)
    mb = num_params * 4 / 1024 / 1024
    print('number of model parameters: {} (~= {:.1f}MB)'.format(num_params, mb))
    print(model)
    return mb


class Test_Leon(PL_Leon):
    def __init__(self, model, optimizer_state_dict=None, learning_rate=0.001):
        super().__init__(model, optimizer_state_dict, learning_rate)
    
    def forward(self, plans, attns, queryfeature=None):
        if self.model.model_type == 'Transformer':
            if queryfeature is None:
                return self.model(plans, attns)[:, 0]
            return self.model(plans, attns, queryfeature)[:, 0]
        elif self.model.model_type == 'TreeConv':
            return torch.tanh(self.model(queryfeature, plans, attns)).add(1).squeeze(1)

    def getBatchPairsLoss(self, batch):
        """
        batch_pairs: a batch of train pairs
        return. a batch of loss
        """
        labels = batch['labels']
        costs1 = batch['costs1']
        costs2 = batch['costs2']
        encoded_plans1 = batch['encoded_plans1']
        encoded_plans2 = batch['encoded_plans2']
        attns1 = batch['attns1']
        attns2 = batch['attns2']
        queryfeature1 = batch['queryfeature1']
        queryfeature2 = batch['queryfeature2']

        loss_fn = nn.BCELoss()
        # step 1. retrieve encoded_plans and attns from pairs


        # step 2. calculate batch_cali and calied_cost
        # 0是前比后大 1是后比前大
        batsize = costs1.shape[0]
        encoded_plans = torch.cat((encoded_plans1, encoded_plans2), dim=0)
        attns = torch.cat((attns1, attns2), dim=0)
        if queryfeature1 is not None and queryfeature2 is not None:
            queryfeature = torch.cat((queryfeature1, queryfeature2), dim=0)
        else:
            queryfeature = None
        if queryfeature is None:
            cali = self(encoded_plans, attns)
        else:
            cali = self(encoded_plans, attns, queryfeature) # cali.shape [# of plan, pad_length] cali 是归一化后的基数估计
        costs = torch.cat((costs1, costs2), dim=0)
        # print(costs1)
        # print(costs2)
        # print(labels)
        # print(cali)
        calied_cost = torch.log(costs) * cali 
        # calied_cost = cali
        try:
            sigmoid = F.sigmoid(-(calied_cost[:batsize] - calied_cost[batsize:]))
            loss = loss_fn(sigmoid, labels.to(sigmoid.dtype)) + 0.2 * torch.abs(calied_cost - torch.log(costs)).mean()
        except:
            print(calied_cost, sigmoid)
        # print(loss)
        with torch.no_grad():
            prediction = torch.round(sigmoid)
            # print(prediction)
            accuracy = torch.sum(prediction == labels).item() / len(labels)
        # print(softm[:, 1].shape, labels.shape)
        
        
        return loss, accuracy
    
class TreeAvgPool(nn.Module):

    def forward(self, trees):
        # trees: Tuple of (data, indexes)
        return trees[0].mean(dim=2)
    
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=3):
        super(ResidualBlock, self).__init__()
        self.conv1 = TreeConv1d(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = TreeConv1d(out_channels, out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # self.shortcut = nn.Sequential(
            #     nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=3),
            #     nn.BatchNorm1d(out_channels)
            # )
            self.shortcut = TreeConv1d(in_channels, out_channels)
            self.shortcut_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        trees, indexes = x
        residual = trees

        out, indexes = self.conv1((trees, indexes))
        out = self.bn1(out)
        out = self.relu(out)
        out, indexes = self.conv2((out, indexes))
        out = self.bn2(out)
        residual, index = self.shortcut((residual, indexes))
        residual = self.shortcut_bn(residual)

        out += residual
        out = F.relu(out)
        return out, indexes
    
class ResNet(nn.Module):
    def __init__(self, feature_size, plan_size, label_size, \
                 block, num_blocks, num_classes=1):
        super(ResNet, self).__init__()
        self.in_channels = plan_size + 32

        self.conv1 = TreeConv1d(self.in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)

        self.in_channels = 64
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1])
        self.layer3 = self._make_layer(block, 256, num_blocks[2])
        self.layer4 = self._make_layer(block, 512, num_blocks[3])
        # self.linear = nn.Linear(512, num_classes)
        # self.max_pool = TreeMaxPool_With_Kernel_Stride()

        self.query_p = 0.3
        self.query_mlp = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.Dropout(p=self.query_p),
            nn.LayerNorm(128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.Dropout(p=self.query_p),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
        )

        # self.plan_p = 0.2
        self.out_mlp = nn.Sequential(
            nn.Linear(512, 32),
            # nn.Dropout(p=self.plan_p),
            nn.LeakyReLU(),
            nn.Linear(32, label_size),
        )

        self.tree_pool = TreeMaxPool()

        self.apply(self._init_weights)
        self.model_type = "TreeConv"

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _make_layer(self, block, out_channels, num_blocks, stride=3):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, query_feats, trees, indexes):
        query_embs = nn.functional.dropout(query_feats, p=0.5)
        query_embs = self.query_mlp(query_feats.unsqueeze(1))
        query_embs = query_embs.transpose(1, 2)
        max_subtrees = trees.shape[-1]
        #    print(query_embs.shape)
        query_embs = query_embs.expand(query_embs.shape[0], query_embs.shape[1],
                                       max_subtrees)
        concat = torch.cat((query_embs, trees), axis=1)


        out, indexes = self.conv1((concat, indexes))
        out = self.bn1(out)
        out = F.relu(out)
        # out, indexes = self.max_pool((out, indexes))
        out, indexes = self.layer1((out, indexes))
        out, indexes = self.layer2((out, indexes))
        out, indexes = self.layer3((out, indexes))
        out, indexes = self.layer4((out, indexes))
        out = self.tree_pool((out, indexes))
        # out = out.view(out.size(0), -1)
        out = self.out_mlp(out)
        return out

if __name__ == '__main__':

    with open('./log/exp_cx202.pkl', 'rb') as f:
        exp1 = pickle.load(f)

    with open('./log/exp_wyz203.pkl', 'rb') as f:
        exp2 = pickle.load(f)

    dict_1 = dict()
    exp1_new = dict()
    exp2_new = dict()
    key = None
    prev_optimizer_state_dict = None
    model = ResNet(820, 54, 1, ResidualBlock, [1, 1, 1, 1])
    model = Test_Leon(model)
    
    workload = envs.wordload_init('job')
    nodeFeaturizer = plans_lib.TreeNodeFeaturizer_V2(workload.workload_info)
    queryFeaturizer = plans_lib.QueryFeaturizer(workload.workload_info)
    i = 0
    for key in exp1.keys():
        grouped_data = {}
        for item in exp1[key]:
            sql_str_value = item[0].info['sql_str']
            if sql_str_value not in grouped_data:
                grouped_data[sql_str_value] = []
            grouped_data[sql_str_value].append(item)
        for key, group in grouped_data.items():
            group.sort(key=lambda x: x[0].cost, reverse=False)
            to_remove = len(group) // 2
            group = group[:-to_remove]
            grouped_data[key] = group
        exp1_new[key] = []
        for values in grouped_data.values():
            exp1_new[key].extend(values)
        for j, plan in enumerate(exp1_new[key]):
            null_nodes = plans_lib.Binarize(plan[0])
            tree, index = encoding.PreTreeConvFeaturize(nodeFeaturizer, [null_nodes]) 
            exp1_new[key][j][0].info['index'] = i
            exp1_new[key][j][0].info['query_feature'] = torch.from_numpy(queryFeaturizer(plan[0]))
            dict_1[i] = (tree, index)
            i += 1
    for key in exp2.keys():
        grouped_data = {}
        for item in exp2[key]:
            sql_str_value = item[0].info['sql_str']
            if sql_str_value not in grouped_data:
                grouped_data[sql_str_value] = []
            grouped_data[sql_str_value].append(item)
        for key, group in grouped_data.items():
            group.sort(key=lambda x: x[0].cost, reverse=False)
            to_remove = len(group) // 2
            group = group[:-to_remove]
            grouped_data[key] = group
        exp2_new[key] = []
        for values in grouped_data.values():
            exp2_new[key].extend(values)
        for j, plan in enumerate(exp2_new[key]):
            null_nodes = plans_lib.Binarize(plan[0])
            tree, index = encoding.PreTreeConvFeaturize(nodeFeaturizer, [null_nodes]) 
            exp2_new[key][j][0].info['index'] = i
            exp2_new[key][j][0].info['query_feature'] = torch.from_numpy(queryFeaturizer(plan[0]))
            dict_1[i] = (tree, index)
            i += 1
    # for key in exp1.keys():
    #     for j, plan in enumerate(exp1[key]):
    #         null_nodes = plans_lib.Binarize(plan[0])
    #         tree, index = encoding.PreTreeConvFeaturize(nodeFeaturizer, [null_nodes]) 
    #         exp1[key][j][0].info['index'] = i
    #         exp1[key][j][0].info['query_feature'] = torch.from_numpy(queryFeaturizer(plan[0]))
    #         dict_1[i] = (tree, index)
    #         i += 1
    # for key in exp2.keys():
    #     for j, plan in enumerate(exp2[key]):
    #         null_nodes = plans_lib.Binarize(plan[0])
    #         tree, index = encoding.PreTreeConvFeaturize(nodeFeaturizer, [null_nodes]) 
    #         exp2[key][j][0].info['index'] = i
    #         exp2[key][j][0].info['query_feature'] = torch.from_numpy(queryFeaturizer(plan[0]))
    #         dict_1[i] = (tree, index)
    #         i += 1

    train_pairs1 = Getpair(exp1_new, key=None)
    leon_dataset1 = prepare_dataset(train_pairs1, True, nodeFeaturizer, queryFeaturizer, dict=dict_1)
    train_pairs2 = Getpair(exp2_new, key=None)
    leon_dataset2 = prepare_dataset(train_pairs2, True, nodeFeaturizer, queryFeaturizer, dict=dict_1)

    def collate_fn(batch):
        # 获取每个batch中的最大长度
        max_len1 = max(item['encoded_plans1'].shape[1] for item in batch)
        max_len2 = max(item['encoded_plans2'].shape[1] for item in batch)
        max_len3 = max(max_len1, max_len2)
        max_len4 = max(item['attns1'].shape[0] for item in batch)
        max_len5 = max(item['attns2'].shape[0] for item in batch)
        max_len6 = max(max_len4, max_len5)
        # 在collate_fn中对每个batch进行padding
        padded_batch = {}
        for key in batch[0].keys():
            if key.startswith('encoded_plans'):
                # 处理树结构等需要padding的数据
                padded_batch[key] = torch.stack([F.pad(item[key], pad=(0, max_len3 - item[key].shape[1])) for item in batch])
            elif key.startswith('attns'):
                # 处理attention等需要padding的数据
                padded_batch[key] = torch.stack([F.pad(item[key], pad=(0, 0, 0, max_len6 - item[key].shape[0])) for item in batch])
            else:
                padded_batch[key] = torch.stack([item[key] for item in batch])

        return padded_batch
    dataloader_train = DataLoader(leon_dataset1, batch_size=1024, shuffle=True, num_workers=7, collate_fn=collate_fn)
    dataloader_val = DataLoader(leon_dataset2, batch_size=1024, shuffle=False, num_workers=7, collate_fn=collate_fn)
    # dataset_val = BucketDataset(exp1, keys=key)
    # batch_sampler = BucketBatchSampler(dataset_val.buckets, batch_size=1)
    # dataloader_val = DataLoader(dataset_val, batch_sampler=batch_sampler)

    # model = load_model(model_path, prev_optimizer_state_dict).to(DEVICE)
    model.optimizer_state_dict = prev_optimizer_state_dict
    logger = pl_loggers.WandbLogger(save_dir=os.getcwd() + '/logs', name="同sql,resnet_torchabs", project='leon3')

    trainer = pl.Trainer(accelerator="gpu",
                         strategy="ddp",
                         sync_batchnorm=True,
                         devices=[0, 3],
                        # devices=[3],
                        max_epochs=100,
                        logger=logger,
                        callbacks=[plc.ModelCheckpoint(
                        dirpath= logger.experiment.dir,
                        monitor='val_acc',
                        filename='best-{epoch:02d}-{val_acc:.3f}',
                        save_top_k=1,
                        mode='max',
                        save_last=True
                        )])
    trainer.fit(model, dataloader_train, dataloader_val)

    prev_optimizer_state_dict = trainer.optimizers[0].state_dict()