import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from monai.losses import DiceCELoss


def compute_distances(tensor):
    b, c, h, w = tensor.shape
    hw = h * w
    tensor = tensor.view(b, c, hw).permute(0, 2, 1)  # 形状 [B, H*W, C]
    distances = torch.norm(tensor[:, :, None, :] - tensor[:, None, :, :], dim=-1)
    return distances

def compute_distance_stats(distances):
    mean_distances = distances.mean(dim=[1, 2])
    std_distances = distances.std(dim=[1, 2])
    return mean_distances, std_distances



class InstanceAttentionModule(nn.Module):
    def __init__(self, input_channels, feature_channels):
        super(InstanceAttentionModule, self).__init__()
        
        self.input_channels = input_channels
        self.feature_channels = feature_channels

        # input_channels=768, feature_channels=192
        # self.embeddings_conv = nn.Sequential(
        #     nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=1, padding=1, groups=input_channels, bias=False),
        #     nn.Conv2d(input_channels, feature_channels, kernel_size=1, stride=1, padding=0, bias=False)
        # )
        self.embeddings_conv = nn.Conv2d(input_channels, feature_channels, kernel_size=1)
        self.attention_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)

        self.instance_threshold_conv = nn.Conv2d(input_channels, 1, kernel_size=1)
    
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_map, layer):
        feature_map = feature_map.permute(0, 3, 1, 2)

        b, c, h, w = feature_map.shape
        hw = h * w 
        instance_embeddings = self.embeddings_conv(feature_map)
        instance_embeddings = instance_embeddings.view(b, self.feature_channels, hw).permute(0, 2, 1)

        # # 使用局部性敏感哈希 (LSH) 近似计算相似度，替代欧几里得距离
        distances = torch.cdist(instance_embeddings, instance_embeddings, p=2)  # 计算pairwise距离
        sim_scores = torch.exp(-distances**2)
        # distances = torch.norm(instance_embeddings[:, :, None, :] - instance_embeddings[:, None, :, :], dim=-1)
        # sim_scores = torch.exp(-distances**2)

        shortcut = self.attention_conv(feature_map)

        Tinst = self.instance_threshold_conv(shortcut)
        Tinst = Tinst.view(b, -1)  # 维度 [B, H*W]

        A_inst = sim_scores - Tinst[:, None, :]
        A_inst = torch.max(A_inst, A_inst.new_zeros(A_inst.shape))
        A_inst = self.softmax(A_inst)

        shortcut = shortcut.view(b, self.input_channels, hw)
        instance_attention = torch.bmm(shortcut, A_inst.permute(0, 2, 1))
        instance_attention = instance_attention.view(b, c, h, w)

        
        res = feature_map + instance_attention

        res = res.permute(0, 2, 3, 1)
        return res
