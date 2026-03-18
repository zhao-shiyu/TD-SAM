import torch
import torch.nn as nn

class InstanceAttentionModule(nn.Module):
    def __init__(self, input_channels, feature_channels, h = 32, w = 32):
        super(InstanceAttentionModule, self).__init__()
        
        self.input_channels = input_channels
        self.feature_channels = feature_channels
        self.h, self.w = h, w
        hw = h * w

        # 初始化卷积层
        self.embeddings_conv = nn.Conv2d(input_channels, feature_channels, kernel_size=1)
        self.attention_conv = nn.Conv2d(input_channels, input_channels, kernel_size=1, stride=1, padding=0)
        self.instance_threshold_conv = nn.Conv2d(input_channels, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

        # 生成距离矩阵并保存
        # y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        # coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2).float()
        # distance_matrix = torch.cdist(coords, coords, p=1) / max(h, w) 
        # self.register_buffer("distance_mask", -distance_matrix)


        # 生成阈值距离矩阵
        k = 32  # 16
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
        coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2).float()
        # 这里已经归一化了 是否需要归一化得考虑考虑
        distance_matrix = torch.cdist(coords, coords, p=1) / max(h, w)

        diff_matrix = torch.abs(coords[:, None, :] - coords[None, :, :]).sum(dim=-1)
        Mdistance = torch.zeros_like(distance_matrix)
        Mdistance[diff_matrix <= k] = distance_matrix[diff_matrix <= k]
        Mdistance[diff_matrix > k] = 0

        # 改为缓存直接存储
        self.register_buffer("distance_mask", Mdistance)


    def forward(self, feature_map, layer):
        feature_map = feature_map.permute(0, 3, 1, 2)
        b, c, h, w = feature_map.shape
        hw = h * w

        # 获得实例嵌入
        instance_embeddings = self.embeddings_conv(feature_map)
        instance_embeddings = instance_embeddings.view(b, self.feature_channels, hw).permute(0, 2, 1)

        # 计算 pairwise 距离并生成相似度分数
        distances = torch.cdist(instance_embeddings, instance_embeddings, p=2)
        sim_scores = torch.exp(-distances ** 2)

        # sim_scores = sim_scores + self.distance_mask
        # sim_scores = sim_scores * self.distance_mask
        sim_scores = sim_scores - (1 - self.distance_mask)
        sim_scores = torch.softmax(sim_scores, dim=-1)

        # 计算注意力机制
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
