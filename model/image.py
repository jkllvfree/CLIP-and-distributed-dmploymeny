import time
from collections import OrderedDict
from typing import Tuple, Union
import torch
import torch.nn as nn
from model.encoder import LayerNorm, AttentionPool2d, Bottleneck, ResidualAttentionBlock
# 假设 Bottleneck 在 encoder 里，这里省略 Bottleneck 具体代码以免太长，逻辑同 ResNet
from model.text import Transformer
from utils.setup import configure_logger
import logging

client_logger = logging.getLogger('client')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 offload_handler=None):
        super().__init__()
        self.input_resolution = input_resolution    #输入图像的边长
        self.output_dim = output_dim              #输出特征维度

        # Conv1 (Patch Embedding)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))    #CLS令牌，一个可学习参数
        #CLS 令牌 + 所有patch位置的位置信息
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        # Transformer Body
        self.transformer = Transformer(           #width：每个token的维度；layers：transformer层数；heads：多头注意力头数
            width, layers, heads,
            offload_handler=offload_handler,
            encoder_type='visual'
        )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))     #线性投影层，用于特征对齐

        self.offload_handler = offload_handler

    def forward(self, x: torch.Tensor):
        #图像块嵌入
        # [卸载逻辑] Vision Conv1
        if self.offload_handler and self.offload_handler.should_offload('vision_conv', encoder_type='visual'):
            x = self.offload_handler.call_remote(
                endpoint='vision_conv',
                data_dict={'x': x},
                device=x.device
            )
        else:
            # 记录开始时间
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t_infer_start = time.perf_counter()
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            # 记录推理结束时间
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t_infer_end = time.perf_counter()
            client_logger.info(
                "[vision_conv] infer_ms=%.3f type=%s",
                (t_infer_end - t_infer_start)*1000,
                "推理",
            )

        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # Class Embed + Pos Embed (通常在本地做，计算量极小)
        # 添加分类令牌
        cls = self.class_embedding.to(dtype=x.dtype, device=x.device)
        cls = cls + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
        x = torch.cat([cls,x], dim=1)

        #增加位置编码
        x = x + self.positional_embedding.to(dtype = x.dtype, device=x.device)
        x = self.ln_pre(x)

        #维度转换
        x = x.permute(1, 0, 2)  # NLD -> LND
        # Transformer Forward
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            # [卸载逻辑] Visual Projection
            if self.offload_handler and self.offload_handler.should_offload('visual_proj', encoder_type='visual'):
                x = self.offload_handler.call_remote(
                    endpoint='visual_projection',
                    data_dict={'x': x},
                    device=x.device
                )
            else:
                # 记录开始时间
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                t_infer_start = time.perf_counter()
                x = x @ self.proj
                # 记录推理结束时间
                if DEVICE.type == 'cuda':
                    torch.cuda.synchronize()
                t_infer_end = time.perf_counter()
                client_logger.info(
                    "[visual_projection] infer_ms=%.3f type=%s",
                    (t_infer_end - t_infer_start) * 1000,
                    "推理",
                )

        return x


#有3个主干卷积层，用平均池化代替最大池化，末尾池化用 QKV 注意力机制替代平均池化
class ModifiedResNet(nn.Module):
    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        #四个残差层连接，然后是注意力池化层，得到最终的向量表示
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x