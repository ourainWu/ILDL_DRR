import scipy.io as sio
import torch
from torch.utils.data import Dataset
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pickle

eps = np.finfo(np.float64).eps

def load_dataset(name):
    data = sio.loadmat('datasets/' + name)
    return data['features'], data['labels']

def load_missing_dataset(name, obr_rate = 0.1):
    data = sio.loadmat('D:/LDL_LE代码/data/' + name + '_incomplete_' + str(obr_rate))
    return data['features'], data['labels'], data['obrT']


class GetDataset(Dataset):
    def __init__(self, X, Log_Label):
        self.data = X
        self.Log_Label = Log_Label

    def __getitem__(self, index):
        data = self.data[index]
        log_l = self.Log_Label[index]
        return data, log_l, index

    def __len__(self):
        return len(self.Log_Label.T[0])

class GetMissingDataset(Dataset):
    def __init__(self, X, Label, mask):
        self.data = X
        self.Label = Label
        self.mask = mask

    def __getitem__(self, index):
        data = self.data[index]
        y = self.Label[index]
        mask = self.mask[index]
        return data, y, mask, index

    def __len__(self):
        return len(self.Label.T[0])

def mask_kl(y, y_pred, mask):
    y = torch.clamp(y, min=eps, max=1.0)
    y_pred = torch.clamp(y_pred, min=eps, max=1.0)
    per_sample_per_label_kl = y * torch.log(y / y_pred)
    valid_kl = per_sample_per_label_kl[mask > 0]
    return torch.sum(valid_kl)

def mask_loss(y, pred, mask):
    y_masked = y * mask
    pred_masked = pred * mask
    fnorm = torch.norm(y_masked - pred_masked, p='fro', dim=1) ** 2
    # loss = mask.squeeze() * fnorm
    return 1/2 * torch.sum(fnorm)


def compute_P_and_W(y, mask):
    y = y
    P = torch.sigmoid(y.unsqueeze(-1) - y.unsqueeze(1))  # (n, d, d)
    P = torch.where(P > 0.5, torch.tensor(1.0, device=P.device), P)  # 大于 0.5 设置为 1.0
    P = torch.where(P < 0.5, torch.tensor(0.0, device=P.device), P)  # 小于 0.5 设置为 0.0

    # 计算权重矩阵 W
    W_full = torch.square(y.unsqueeze(-1) - y.unsqueeze(1))  # (n, d, d)
    mask_3d = mask.unsqueeze(-1) * mask.unsqueeze(-2)  # 广播到 (n, d, d)
    W = torch.where(mask_3d > 0, W_full, torch.tensor(-1.0, device=W_full.device))

    return P, W

def ranking_loss(y, y_pred, mask):
    P, W = compute_P_and_W(y, mask)
    Phat = torch.sigmoid((y_pred.unsqueeze(-1) - y_pred.unsqueeze(1)) * 100)  # (n, d, d)

    # 计算损失矩阵 l
    l = ((1 - P) * torch.log(torch.clamp(1 - Phat, min=1e-9, max=1.0)) +
         P * torch.log(torch.clamp(Phat, min=1e-9, max=1.0))) * W  # (n, d, d)

    # 获取 W != -1 的有效索引
    valid_indices = torch.nonzero(W != -1, as_tuple=True)  # 获取有效索引
    valid_l = l[valid_indices]  # 提取有效的损失值

    # 计算有效损失的总和
    return - 1/2 * torch.sum(valid_l)

def dpa_mask_origin(y, y_pred, mask):
    num_missing = (mask == 0).sum(dim=1).unsqueeze(dim=1)
    R = torch.argsort(torch.argsort(y, axis=1), axis=1).to(torch.float32) - num_missing
    # R = torch.argsort(torch.argsort(y, axis=1), axis=1).to(torch.float32)
    R = R * mask
    loss = -torch.sum(torch.mean(R * y_pred, dim=1))
    return loss

def dpa_mask_normalized(y, y_pred, mask):
    num_missing = (mask == 0).sum(dim=1).unsqueeze(dim=1)
    results_num = y.size(1) - num_missing
    V = ((results_num > 1).to(torch.float32)).T
    R = torch.argsort(torch.argsort(y, axis=1), axis=1).to(torch.float32) - num_missing
    R = R * mask
    # R = F.softmax(R, dim=1)  ## 归一化
    R_normalized = R / (torch.sum(R, dim=1, keepdim=True) + eps)
    R_normalized = torch.clamp(R_normalized, min=0, max=1.0)
    # a = torch.sum(R_normalized * y_pred, dim=1)
    # b = torch.sum(R_normalized * y_pred, dim=1) * V
    # c = -torch.sum(b)
    loss = -torch.sum(torch.sum(R_normalized * y_pred, dim=1) * V)
    return loss

def dpa_mask(y, y_pred, mask):
    num_missing = (mask == 0).sum(dim=1).unsqueeze(dim=1)
    V = ((num_missing > 1).to(torch.float32)).T
    R = torch.argsort(torch.argsort(y, axis=1), axis=1).to(torch.float32) - num_missing
    R = R * mask
    loss_before = torch.mean(R * y_pred, dim=1)
    loss_origin = -torch.sum(loss_before)
    loss_med = torch.mean(R * y_pred, dim=1) * V
    loss = -torch.sum(loss_med)
    return loss

def identify_implicit(y):
    y_remain_keep = 1 - torch.sum(y, dim = 1, keepdim=True)
    y_remain = 1 - torch.sum(y, dim=1)
    indicator = y > y_remain_keep
    identify_matrix = torch.any(y > y_remain_keep, dim=1).int()
    identify_matrix = identify_matrix.unsqueeze(1)

    # 将不存在True的行填充一个较大的索引值
    masked_indices = torch.where(indicator, torch.arange(y.size(1)).expand_as(y), float('inf'))
    min_indices = masked_indices.min(dim=1).values

    # 将 inf（无True的行）替换为0
    min_indices[min_indices == float('inf')] = 0
    min_indices = min_indices.long()

    rho = min_indices - y_remain
    # rho = torch.any(rho > 0, dim=1).int()
    # rho = rho.unsqueeze(1)
    return identify_matrix, min_indices, rho

def margin_loss(y, y_pred, mask):
    identify_matrix, min_indices, rho = identify_implicit(y)
    all_rows = torch.arange(y.shape[0], dtype=torch.long)
    rho = rho.long()
    low_bound = y_pred[all_rows, rho].reshape(-1, 1)
    Delta =  low_bound - y_pred
    # hinge = (Delta < rho.unsqueeze(1)).long()
    mask_hat = 1 - mask

    # loss1 = mask_hat * y_pred
    # loss2 = identify_matrix * y_pred
    # loss3 = mask_hat * y_pred * identify_matrix
    rho_margin_loss = torch.clamp(1 - (Delta) / (low_bound + eps), min = 0)
    loss = rho_margin_loss * mask_hat * identify_matrix
    a = torch.mean(loss, dim=1)
    return torch.sum(torch.mean(loss, dim=1))

def l2_reg(model):
    reg = 0.0
    for param in model.parameters():
        if param.requires_grad:  # 只对可训练参数进行正则化
            reg += torch.sum(param ** 2)
    return reg / 2.0