import numpy as np
import torch

def _get_onehot_mask(vec):
    tmask = ~np.array(vec, dtype="bool")
    print(tmask)
    ptrue = 0.8
    pfalse = 1- ptrue

    # probabilities are switched
    bools = np.random.choice(a=[False, True], size=(len(tmask),),
            p=[ptrue,pfalse])
    tmask *= bools
    tmask = ~tmask
    tmask = torch.from_numpy(tmask).float()
    return tmask * torch.tensor(vec).float()

def probabilistic_mask(features, mask_prob):
    # 创建一个与输入特征形状相同的随机张量
    random_tensor = torch.rand(features.shape, dtype=features.dtype, device=features.device)
    
    # 创建一个掩码，其中输入特征不为0的地方，按照给定的概率mask_prob置为False
    mask = (random_tensor >= mask_prob) | (features == 0)
    
    # 应用掩码
    masked_features = features * mask.float()
    return masked_features

if __name__ == '__main__':
    print(probabilistic_mask(torch.tensor([2, 0, 0, 2]).float(), 0.2))