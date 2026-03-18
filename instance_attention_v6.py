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
        self.embeddings_conv = nn.Conv2d(input_channels, feature_channels, kernel_size=1)
        self.attention_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1)

        self.instance_threshold_conv = nn.Conv2d(input_channels, 1, kernel_size=1)
    
        self.softmax = nn.Softmax(dim=1)

    def forward(self, feature_map, layer):
        feature_map = feature_map.permute(0, 3, 1, 2)

        # if layer == 8:

        #     original_distances = compute_distances(feature_map)
        #     original_mean, original_std = compute_distance_stats(original_distances)

        #     print("Original Feature Map - Mean Distance:", original_mean.mean().item())
        #     print("Original Feature Map - Distance Std:", original_std.mean().item())

        #     breakpoint()

        b, c, h, w = feature_map.shape
        hw = h * w

        # print('\nfeature_map shape: ', feature_map.shape)
        
        instance_embeddings = self.embeddings_conv(feature_map)
        # print('instance_embeddings shape: ', instance_embeddings.shape)
        instance_embeddings = instance_embeddings.view(b, self.feature_channels, hw).permute(0, 2, 1)
        # print('instance_embeddings shape: ', instance_embeddings.shape)

        
        distances = torch.norm(instance_embeddings[:, :, None, :] - instance_embeddings[:, None, :, :], dim=-1)
        # print('distances shape: ', distances.shape)
        sim_scores = torch.exp(-distances**2)
        # print('sim_scores shape: ', sim_scores.shape)
        # embeddings_norm = instance_embeddings.norm(dim=-1, keepdim=True)
        # distances = ((instance_embeddings[:, :, None, :] - instance_embeddings[:, None, :, :]) / embeddings_norm).pow_(2).neg_()
        # sim_scores = torch.exp(distances)

        shortcut = self.attention_conv(feature_map)

        # 计算Tinst
        # Tinst = torch.sigmoid(self.instance_threshold_conv(feature_map))
        Tinst = self.instance_threshold_conv(shortcut)
        Tinst = Tinst.view(b, -1)  # 维度 [B, H*W]

        # print('Tinst shape: ', Tinst.shape)
        
        # 应用Weight Clipping Strategy``
        A_inst = sim_scores - Tinst[:, None, :]
        A_inst = torch.max(A_inst, A_inst.new_zeros(A_inst.shape))  # 应用ReLU函数保证非负

        A_inst = self.softmax(A_inst)

        # print('A_inst shape: ', A_inst.shape)

        
        shortcut = shortcut.view(b, self.input_channels, hw)
        instance_attention = torch.bmm(shortcut, A_inst.permute(0, 2, 1))
        instance_attention = instance_attention.view(b, c, h, w)

        
        res = feature_map + instance_attention

        # if layer == 10:

        #     attended_distances = compute_distances(res)
        #     attended_mean, attended_std = compute_distance_stats(attended_distances)

            
        #     print("Attended Feature Map - Mean Distance:", attended_mean.mean().item())
        #     print("Attended Feature Map - Distance Std:", attended_std.mean().item())

        #     breakpoint()

        res = res.permute(0, 2, 3, 1)
        return res
