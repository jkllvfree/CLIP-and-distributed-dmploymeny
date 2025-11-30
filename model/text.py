# --- START OF FILE inference/models/text.py ---
import time

import torch
import torch.nn as nn
from model.encoder import ResidualAttentionBlock, LayerNorm
from utils.setup import configure_logger
import logging

client_logger = logging.getLogger('client')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None,
                 offload_handler=None, encoder_type: str = 'text'):
        super().__init__()
        self.width = width
        self.layers = layers
        self.offload_handler = offload_handler
        self.encoder_type = encoder_type

        # 将 offload_handler 传递给每一个 ResBlock
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlock(
                width, heads, attn_mask,
                offload_handler=offload_handler,
                layer_id=i,
                encoder_type=encoder_type
            )
            for i in range(layers)
        ])

    def forward(self, x: torch.Tensor):
        # [卸载逻辑] 整个 ResBlocks 块级卸载 (中粒度)
        if self.offload_handler and self.offload_handler.should_offload('text_encoder'):
            return self.offload_handler.call_remote(
                endpoint='encoder_blocks',
                data_dict={
                    'x': x,
                    'encoder_type': self.encoder_type
                },
                device=x.device
            )

        # [本地计算]
        else:
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t_infer_start = time.perf_counter()
            out = self.resblocks(x)
            # 记录推理结束时间
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            t_infer_end = time.perf_counter()
            client_logger.info(
                "[encoder_blocks] infer_ms=%.3f type=%s",
                (t_infer_end - t_infer_start) * 1000,
                "推理",
            )
            return out
