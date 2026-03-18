from segment_anything import build_sam, SamPredictor
from segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from segment_anything.modeling import Sam

from typing import Optional, Tuple, Type

from instance_attention_v9 import InstanceAttentionModule

from torch.nn.modules.utils import _pair
from defcor import DefAgg, DefCorFixW

def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """
    Partition into non-overlapping windows with padding if needed.
    Args:
        x (tensor): input tokens with [B, H, W, C].
        window_size (int): window size.

    Returns:
        windows: windows after partition with [B * num_windows, window_size, window_size, C].
        (Hp, Wp): padded height and width before partition
    """
    B, H, W, C = x.shape

    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w

    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """
    Window unpartition into original sequences and removing padding.
    Args:
        x (tensor): input tokens with [B * num_windows, window_size, window_size, C].
        window_size (int): window size.
        pad_hw (Tuple): padded height and width (Hp, Wp).
        hw (Tuple): original height and width (H, W) before padding.

    Returns:
        x: unpartitioned sequences with [B, H, W, C].
    """
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, Hp, Wp, -1)

    if Hp > H or Wp > W:
        x = x[:, :H, :W, :].contiguous()
    return x

def get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """
    Get relative positional embeddings according to the relative positions of
        query and key sizes.
    Args:
        q_size (int): size of query q.
        k_size (int): size of key k.
        rel_pos (Tensor): relative position embeddings (L, C).

    Returns:
        Extracted positional embeddings according to relative positions.
    """
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    # Interpolate rel pos if needed.
    if rel_pos.shape[0] != max_rel_dist:
        # Interpolate rel pos.
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist,
            mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos

    # Scale the coords with short length if shapes for q and k are different.
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    return rel_pos_resized[relative_coords.long()]

def add_decomposed_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int],
    k_size: Tuple[int, int],
) -> torch.Tensor:
    """
    Calculate decomposed Relative Positional Embeddings from :paper:`mvitv2`.
    https://github.com/facebookresearch/mvit/blob/19786631e330df9f3622e5402b4a419a263a2c80/mvit/models/attention.py   # noqa B950
    Args:
        attn (Tensor): attention map.
        q (Tensor): query q in the attention layer with shape (B, q_h * q_w, C).
        rel_pos_h (Tensor): relative position embeddings (Lh, C) for height axis.
        rel_pos_w (Tensor): relative position embeddings (Lw, C) for width axis.
        q_size (Tuple): spatial sequence size of query q with (q_h, q_w).
        k_size (Tuple): spatial sequence size of key k with (k_h, k_w).

    Returns:
        attn (Tensor): attention map with added relative positional embeddings.
    """
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = get_rel_pos(q_w, k_w, rel_pos_w)

    B, _, dim = q.shape
    r_q = q.reshape(B, q_h, q_w, dim)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)

    attn = (
        attn.view(B, q_h, q_w, k_h, k_w) + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    ).view(B, q_h * q_w, k_h * k_w)

    return attn

class _Fact_tt_ImageEncoderViT(nn.Module):
    def __init__(
            self,
            ImageEncoderViT: nn.Module,
            FacTu: nn.Module,
            FacTv: nn.Module,
    ):
        super().__init__()
        self.ImageEncoderViT = ImageEncoderViT
        self.FacTu = FacTu
        self.FacTv = FacTv
        self.img_size = self.ImageEncoderViT.img_size

    
    def forward(self, x: torch.Tensor, d_size) -> torch.Tensor:
        x = self.ImageEncoderViT.patch_embed(x)  
        if self.ImageEncoderViT.pos_embed is not None:
            x = x + self.ImageEncoderViT.pos_embed

        for blk in self.ImageEncoderViT.blocks:
            x = blk(x, self.FacTu, self.FacTv, d_size)

        x = self.ImageEncoderViT.neck(x.permute(0, 3, 1, 2))  

        return x

class _Fact_tt_Block(nn.Module):
    def __init__(
            self,
            Block: nn.Module,
            layer: int
    ):
        super().__init__()
        self.Block = Block
        self.layer = layer

        self.instance_attention = InstanceAttentionModule(768, 192)

    
    def forward(self, x: torch.Tensor, FacTu, FacTv, d_size) -> torch.Tensor:

        shortcut = x
        x = self.Block.norm1(x)
        # Window partition
        if self.Block.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.Block.window_size)  # [B * num_windows, window_size, window_size, C]

        x = self.Block.attn(x, FacTu, FacTv)
        # breakpoint()
        # Reverse window partition
        if self.Block.window_size > 0:
            x = window_unpartition(x, self.Block.window_size, pad_hw, (H, W))

        x = shortcut + x

        x = self.instance_attention(x, self.layer)

        x = x + self.Block.mlp(self.Block.norm2(x))

        return x

class   _Fact_tt_Block_SIFA(nn.Module):
    def __init__(
        self,
        Block: nn.Module,
        layer: int,

        clip_len=5,
        K=3, 
        cor_dilation=1,
        cor_group=1,
        dim = 768,
        norm_layer: Type[nn.Module] = nn.LayerNorm
    ):
        super().__init__()
        self.Block = Block
        self.layer = layer


        self.sifa_pad_num = (cor_dilation * (K - 1) + 1) // 2
        self.sifa_off_channels_ = 2 * K * K
        self.sifa_kernel_size = _pair(K)
        self.sifa_width = dim
        self.sifa_clip_len = clip_len

        # dim的值需要根据参数文件的不同修改 具体参照defcor.py中26行注释
        self.sifa_def_cor = DefCorFixW(in_channels=self.sifa_width, times=self.sifa_clip_len, kernel_size=(K,K), stride=1, padding=self.sifa_pad_num, 
                          dilation=cor_dilation, defcor_groups=cor_group)
        self.sifa_def_agg = DefAgg(in_channels=self.sifa_width, times=self.sifa_clip_len, kernel_size=(K,K), stride=1, padding=self.sifa_pad_num, 
                          dilation=cor_dilation, defagg_groups=cor_group)
        self.sifa_conv = nn.Conv2d(dim, dim, kernel_size=[1, 3], padding=[0, 1], groups=1, bias=False)
        self.sifa_tda_norm = norm_layer(self.sifa_width)


        self.instance_attention = InstanceAttentionModule(768, 192)


    def forward_rtc(self, x):
        NT, W, H, C = x.size()
        L = W * H
        x = x.view(NT, L, C)

        n_batch = NT // self.sifa_clip_len
        shortcut = x
        x = x.transpose(1,2) # nt x c x l
        x = x.view(n_batch, self.sifa_clip_len, C, W, H).transpose(1,2) # NCTHW

        x_tmp = x.clone()
        x_tmp_clone = x_tmp.clone()
        x_tmp_clone[:,:,1:,:,:] = x_tmp[:,:,:-1,:,:]
        x_tmp = (torch.sigmoid(x - x_tmp_clone) * x) + x    # [sig(f_{t}-f_{t-1})*f_{t}]*f_{t}
        #offset = self.conv_offset(x_tmp)
        offset = nn.Parameter(torch.zeros(n_batch, self.sifa_off_channels_, self.sifa_clip_len, W, H)).cuda()

        corre_weight = self.sifa_def_cor(x, offset)
        x_agg = self.sifa_def_agg(x, offset, corre_weight)

        mask = torch.ones(x.size()).cuda()
        mask[:,:,-1,:,:] = 0
        mask.requires_grad = False
        x_shift = x_agg.clone()
        x_shift[:,:,:-1,:,:] = x_shift[:,:,1:,:,:]
        x = x_shift * mask
        
        x = x.transpose(1,2).reshape(n_batch*self.sifa_clip_len, C, -1).transpose(1,2)
        #x = self.fc_tda(x)
        x = self.sifa_tda_norm(x)
        x = shortcut + x

        x = x.view(NT, W, H, C)
        return x
    
    def forward_vtc(self, x):
        n_segment = self.sifa_clip_len
        NT, W, H, C = x.size()
        L = W * H
        x = x.view(NT, L, C)
        nt, l, c = x.size()
        n_batch = nt // n_segment
        x = x.view(n_batch, n_segment, l, c).transpose(1, 3)
        x = self.sifa_conv(x)
        x = x.transpose(1, 3).reshape(n_batch * n_segment, l, c)

        x = x.view(NT, W, H, C)
        return x
    

    def forward(self, x: torch.Tensor, FacTu, FacTv, d_size) -> torch.Tensor:

        b_size, hw_size = x.shape[0], x.shape[1]

        # # 3D adapter
        # shortcut = x
        # x = self.Block.adapter_norm(x)
        # x = self.Block.adapter_linear_down(x)
        # x = x.contiguous().view(int(b_size/d_size), d_size, hw_size, hw_size, self.Block.adapter_channels)
        # x = torch.permute(x, (0, -1, 1, 2, 3))
        # x = self.Block.adapter_conv(x)
        # x = torch.permute(x, (0, 2, 3, 4, 1))
        # x = x.contiguous().view(b_size, hw_size, hw_size, self.Block.adapter_channels)
        # x = self.Block.adapter_act(x)
        # x = self.Block.adapter_linear_up(x)
        # x = shortcut + x
        # # end 3D adapter

        shortcut = x
        x = self.Block.norm1(x)
        # Window partition
        if self.Block.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.Block.window_size)  # [B * num_windows, window_size, window_size, C]

        x = self.Block.attn(x, FacTu, FacTv)
        # Reverse window partition
        if self.Block.window_size > 0:
            x = window_unpartition(x, self.Block.window_size, pad_hw, (H, W))

        x = shortcut + x

        # # 3D adapter
        # shortcut = x
        # x = self.Block.adapter_norm_2(x)
        # x = self.Block.adapter_linear_down_2(x)
        # x = x.contiguous().view(int(b_size/d_size), d_size, hw_size, hw_size, self.Block.adapter_channels)
        # x = torch.permute(x, (0, -1, 1, 2, 3))
        # x = self.Block.adapter_conv_2(x)
        # x = torch.permute(x, (0, 2, 3, 4, 1))
        # x = x.contiguous().view(b_size, hw_size, hw_size, self.Block.adapter_channels)
        # x = self.Block.adapter_act_2(x)
        # x = self.Block.adapter_linear_up_2(x)
        # x = shortcut + x
        # # end 3D adapter

        # x = self.instance_attention(x, self.layer)
        # if self.layer in [6, 8, 10]:
        #     x = self.instance_attention(x, self.layer)
        # if self.layer in [6, 7, 8, 9, 10, 11]:
        #     x = self.instance_attention(x, self.layer)
        # if self.layer in [3, 4, 5, 6, 7, 8, 9, 10, 11]:
        #     x = self.instance_attention(x, self.layer)

        # if self.layer in [6, 8, 10]:
        #     x = self.forward_rtc(x)
        # elif self.layer in [7, 9, 11]:
        #     x += self.forward_vtc(x)

        x = x + self.Block.mlp(self.Block.norm2(x))

        return x

class _Fact_tt_Block_InstanceAttention(nn.Module):
    def __init__(
            self,
            Block: nn.Module,
            layer: int
    ):
        super().__init__()
        self.Block = Block
        self.layer = layer

        self.instance_attention = InstanceAttentionModule(768, 192)

    
    def forward(self, x: torch.Tensor, FacTu, FacTv, d_size) -> torch.Tensor:

        # x = self.instance_attention(x)

        shortcut = x
        x = self.Block.norm1(x)
        # Window partition
        if self.Block.window_size > 0:
            H, W = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.Block.window_size)  # [B * num_windows, window_size, window_size, C]

        x = self.Block.attn(x, FacTu, FacTv)
        # Reverse window partition
        if self.Block.window_size > 0:
            x = window_unpartition(x, self.Block.window_size, pad_hw, (H, W))

        x = shortcut + x

        if self.layer in [6, 8, 10]:
            x = self.instance_attention(x, self.layer)

        x = x + self.Block.mlp(self.Block.norm2(x))

        return x

class _Fact_tt_Attention(nn.Module):
    def __init__(
            self,
            Attention: nn.Module,
    ):
        super().__init__()
        self.Attention = Attention
    
    def forward(self, x: torch.Tensor, FacTu, FacTv) -> torch.Tensor:
        B, H, W, _ = x.shape
        # qkv with shape (3, B, nHead, H * W, C)
        qkv = self.Attention.qkv(x, FacTu, FacTv).reshape(B, H * W, 3, self.Attention.num_heads, -1).permute(2, 0, 3, 1, 4)
        # q, k, v with shape (B * nHead, H * W, C)
        q, k, v = qkv.reshape(3, B * self.Attention.num_heads, H * W, -1).unbind(0)

        attn = (q * self.Attention.scale) @ k.transpose(-2, -1)

        if self.Attention.use_rel_pos:
            attn = add_decomposed_rel_pos(attn, q, self.Attention.rel_pos_h, self.Attention.rel_pos_w, (H, W), (H, W))

        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.Attention.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        x = self.Attention.proj(x)

        return x

class _Fact_tt_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            q_FacTs: nn.Module,
            v_FacTs: nn.Module,
            s,
    ):
        super().__init__()
        self.qkv = qkv
        self.q_FacTs = q_FacTs
        self.v_FacTs = v_FacTs
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)
        self.dp_q = nn.Dropout(0.1)
        self.dp_v = nn.Dropout(0.1)
        self.s = s

    def forward(self, x, FacTu, FacTv):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = FacTv(self.dp_q(self.q_FacTs(FacTu(x))))  
        new_v = FacTv(self.dp_v(self.v_FacTs(FacTu(x))))
        qkv[:, :, :, : self.dim] += new_q*self.s
        qkv[:, :, :, -self.dim:] += new_v*self.s
        return qkv

class Fact_tt_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of FacT_tt
        num_classes: how many classes the model output, default to the vit model
        FacT_tt_layer: which layer we apply FacT_tt.

    """

    def __init__(self, sam_model: Sam, r: int, fact_layer=None, s=1):  # s是尺度系数
        super(Fact_tt_Sam, self).__init__()

        assert r > 0
        base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        
        # dim = base_vit_dim
        if fact_layer:
            self.fact_layer = fact_layer
        else:
            self.fact_layer = list(
                range(len(sam_model.image_encoder.blocks)))  
        # create for storage, then we can init them or load weights
        self.q_FacTs = []  # These are linear layers
        self.v_FacTs = []

        self.FacTu = nn.Linear(base_vit_dim, r, bias=False)
        self.FacTv = nn.Linear(r, base_vit_dim, bias=False)
        nn.init.zeros_(self.FacTv.weight)

        # lets freeze pre-trained weights
        for k, v in sam_model.image_encoder.named_parameters():
            if not '.adapter_' in k and not '.sifa_' in k and not '.instance_' in k:
                v.requires_grad = False

        # add factors
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            if t_layer_i not in self.fact_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            q_FacTs = nn.Linear(r, r, bias=False)
            v_FacTs = nn.Linear(r, r, bias=False)
            self.q_FacTs.append(q_FacTs)
            self.v_FacTs.append(v_FacTs)
            blk.attn.qkv = _Fact_tt_qkv(
                w_qkv_linear,
                q_FacTs,
                v_FacTs,
                s
            )

            blk.attn = _Fact_tt_Attention(blk.attn) 


            sam_model.image_encoder.blocks[t_layer_i] = _Fact_tt_Block_SIFA(blk, t_layer_i)

            # if t_layer_i in [0, 1, 2, 3, 4, 5]: 
            #     sam_model.image_encoder.blocks[t_layer_i] = _Fact_tt_Block(blk, t_layer_i)
            # elif t_layer_i in  [6, 7, 8, 9, 10, 11]: 
            #     sam_model.image_encoder.blocks[t_layer_i] = _Fact_tt_Block_SIFA(blk, t_layer_i)  
        
            # if t_layer_i in [0, 1, 2, 3, 4]: 
            #     sam_model.image_encoder.blocks[t_layer_i] = _Fact_tt_Block(blk, t_layer_i)
            # elif t_layer_i in  [5]: 
            #     sam_model.image_encoder.blocks[t_layer_i] = _Fact_tt_Block_InstanceAttention(blk, t_layer_i)  
            # elif t_layer_i in  [6, 7, 8, 9, 10, 11]: 
            #     sam_model.image_encoder.blocks[t_layer_i] = _Fact_tt_Block_SIFA(blk, t_layer_i) 

        sam_model.image_encoder = _Fact_tt_ImageEncoderViT(sam_model.image_encoder, self.FacTu, self.FacTv)
        self.sam = sam_model

    def save_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both FacT_tt and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.q_FacTs)  # actually, it is half
        a_tensors = {f"q_FacTs_{i:03d}": self.q_FacTs[i].weight for i in range(num_layer)}
        b_tensors = {f"v_FacTs_{i:03d}": self.v_FacTs[i].weight for i in range(num_layer)}

        prompt_encoder_tensors = {}
        mask_decoder_tensors = {}
        adapter_tensor = {}

        FacTu_tensors = {}
        FacTv_tensors = {}

        sifa_tensor= {}
        instance_tensor = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if '.adapter_' in key:
                adapter_tensor[key] = value
            if 'FacTu' in key:
                FacTu_tensors[key] = value
            if 'FacTv' in key:
                FacTv_tensors[key] = value

            if '.sifa_' in key:
                sifa_tensor[key] = value
            if '.instance_' in key:
                instance_tensor[key] = value

        merged_dict = {**a_tensors, **b_tensors, **FacTu_tensors, **FacTv_tensors, **prompt_encoder_tensors, **mask_decoder_tensors, 
                       **adapter_tensor, **sifa_tensor, **instance_tensor}
        torch.save(merged_dict, filename)

    def load_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both FacT_tt and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, q_FacTs in enumerate(self.q_FacTs):
            saved_key = f"q_FacTs_{i:03d}"
            saved_tensor = state_dict[saved_key]
            q_FacTs.weight = Parameter(saved_tensor)

        for i, v_FacTs in enumerate(self.v_FacTs):
            saved_key = f"v_FacTs_{i:03d}"
            saved_tensor = state_dict[saved_key]
            v_FacTs.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        FacTu_keys = [k for k in sam_keys if 'FacTu' in k]
        FacTu_values = [state_dict[k] for k in FacTu_keys]
        FacTu_new_state_dict = {k: v for k, v in zip(FacTu_keys, FacTu_values)}
        sam_dict.update(FacTu_new_state_dict)

        FacTv_keys = [k for k in sam_keys if 'FacTv' in k]
        FacTv_values = [state_dict[k] for k in FacTv_keys]
        FacTv_new_state_dict = {k: v for k, v in zip(FacTv_keys, FacTv_values)}
        sam_dict.update(FacTv_new_state_dict)

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)

        # load adapter
        adapter_keys = [k for k in sam_keys if '.adapter_' in k]
        adapter_values = [state_dict[k] for k in adapter_keys]
        adapter_new_state_dict = {k: v for k, v in zip(adapter_keys, adapter_values)}
        sam_dict.update(adapter_new_state_dict)


        # load sifa
        sifa_keys = [k for k in sam_keys if '.sifa_' in k]
        sifa_values = [state_dict[k] for k in sifa_keys]
        sifa_new_state_dict = {k: v for k, v in zip(sifa_keys, sifa_values)}
        sam_dict.update(sifa_new_state_dict)

        # load instance_attention
        instance_keys = [k for k in sam_keys if '.instance_' in k]
        instance_values = [state_dict[k] for k in instance_keys]
        instance_new_state_dict = {k: v for k, v in zip(instance_keys, instance_values)}
        sam_dict.update(instance_new_state_dict)


        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output, image_size):

        return self.sam(batched_input, multimask_output, image_size)

