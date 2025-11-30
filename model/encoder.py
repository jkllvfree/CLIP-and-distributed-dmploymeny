import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from utils.setup import configure_logger
from typing import Optional

import logging
client_logger = logging.getLogger('client')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#瓶颈残差块
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x
        #主路径
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        #旁路
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        # 修改：同时对齐 dtype 和 device，确保在 GPU 上不会再出现 device 不一致
        x = x + self.positional_embedding[:, None, :].to(dtype=x.dtype, device=x.device)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)

#transformer中的核心模块
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None,
                 offload_handler = None,
                 layer_id = 0,
                 encoder_type = 'visual'):
        super().__init__()
        # 多头自注意力计算，两个参数：模型维度，注意力头数量
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # 保存卸载配置
        self.offload_handler = offload_handler  # 这是一个函数，负责发网络请求
        self.layer_id = layer_id
        self.encoder_type = encoder_type

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None

        # 卸载逻辑
        if self.offload_handler and self.offload_handler.should_offload('attention', self.layer_id, self.encoder_type):
            return self.offload_handler.call_remote(
                endpoint='attention',
                data={'x': x,
                      'attn_mask': self.attn_mask,
                      'layer_id': self.layer_id,
                      'encoder_type': self.encoder_type
                },
                device = x.device
            )
        else:
            # 记录开始时间
            t_infer_start = time.perf_counter()
            # 本地计算，Q,K,V 都是 x，返回值是一个元组，第一个元素是注意力输出
            out = self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]
            # 记录推理结束时间
            t_infer_end = time.perf_counter()
            client_logger.info(
                "[attention] infer_ms=%.3f type=%s",
                (t_infer_end - t_infer_start) * 1000,
                "推理",
            )

            return out

    def mlp_forward(self,x:torch.Tensor):
        if self.offload_handler and self.offload_handler.should_offload('mlp', self.layer_id, self.encoder_type):
            return self.offload_handler.call_remote(
                endpoint='mlp',
                data_dict={
                    'x': x,
                    'layer_id': self.layer_id,
                    'encoder_type': self.encoder_type
                },
                device=x.device
            )
        else:
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            # 记录开始时间
            t_infer_start = time.perf_counter()
            out = self.mlp(x)
            # 记录推理结束时间
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t_infer_end = time.perf_counter()
            client_logger.info(
                "[mlp] infer_ms=%.3f type=%s",
                (t_infer_end - t_infer_start) * 1000,
                "推理",
            )
            return out

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp_forward(self.ln_2(x))
        return x


#用来处理浮点数16的，先转化为float32规范化，再转回来16
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)

#用sigmoid乘一个系数，来近似正态分布的累计分布函数
class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)




