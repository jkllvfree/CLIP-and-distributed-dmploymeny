import time

import numpy as np
from packaging import version
from typing import Union, List, Tuple

import torch
from PIL import Image
from torch import nn
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from model.encoder import LayerNorm
from model.image import ModifiedResNet, VisionTransformer
from model.text import Transformer
from model.simple_tokenizer import SimpleTokenizer as _Tokenizer

from utils.setup import configure_logger
import logging
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

client_logger = logging.getLogger('client')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_tokenizer = _Tokenizer()

class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 # offload
                 offload_handler=None
                 ):
        super().__init__()
        self.context_length = context_length
        self.offload_handler = offload_handler

        # === 构建 Vision Encoder ===
        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width,
                offload_handler=offload_handler
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                offload_handler=offload_handler  # 传入 handler
            )

        # === 构建 Text Encoder ===
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            offload_handler=offload_handler,  # 传入 handler
            encoder_type='text'
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)
        return mask

    @property
    #此处有修改
    def dtype(self):
        return self.visual.conv1.weight.dtype if isinstance(self.visual,
                                                            VisionTransformer) else self.visual.conv1.weight.dtype

    def encode_image(self, image):
        # 移除了 image.to(device) 的硬编码，由外部控制或 auto-cast
        return self.visual(image.type(self.dtype))
        # 原版：把输入 image 搬到模型所在的 device，并且对齐 dtype
        #device = self._device()
        #image = image.to(device=device, dtype=self.dtype)
        #return self.visual(image)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Transformer forward (内部处理卸载)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # 取 EOT token 特征
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        # [卸载逻辑] Text Projection
        if self.offload_handler and self.offload_handler.should_offload('text_proj', encoder_type='text'):
            x = self.offload_handler.call_remote(
                endpoint='text_projection',
                data_dict={'x': x},
                device=x.device
            )
        else:
            # 记录开始时间
            t_infer_start = time.perf_counter()
            x = x @ self.text_projection
            # 记录推理结束时间
            t_infer_end = time.perf_counter()
            client_logger.info(
                "[text_projection] infer_ms=%.3f type=%s",
                (t_infer_end - t_infer_start) * 1000,
                "推理",
            )

        return x

    def forward(self, image, text):
        # [卸载逻辑] 全量模型卸载
        if self.offload_handler and self.offload_handler.should_offload('complete_encoders', encoder_type='all'):
            result = self.offload_handler.call_remote(
                endpoint='complete_encoders',
                data_dict={'image': image, 'text': text},
                device=image.device
            )
            return result['logits_per_image'], result['logits_per_text']  # 假设返回字典

        # image_features = self.encode_image(image)
        # text_features = self.encode_text(text)
        else:
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            # 记录开始时间
            t_infer_start = time.perf_counter()
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            # 记录推理结束时间
            t_infer_end = time.perf_counter()
            client_logger.info(
                "[complete_encoders] infer_ms=%.3f type=%s",
                (t_infer_end - t_infer_start) * 1000,
                "推理",
            )

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()

        # [卸载逻辑] Cosine Similarity
        if self.offload_handler and self.offload_handler.should_offload('cos_sim', encoder_type='all'):
            logits_per_image = self.offload_handler.call_remote(
                endpoint='cos_sim',
                data_dict={
                    'image_features': image_features,
                    'text_features': text_features,
                    'logit_scale': logit_scale
                },
                device=image_features.device
            )
        else:
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            # 记录开始时间
            t_infer_start = time.perf_counter()
            logits_per_image = logit_scale * image_features @ text_features.t()
            if DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            # 记录推理结束时间
            t_infer_end = time.perf_counter()
            client_logger.info(
                "[cos_sim] infer_ms=%.3f type=%s",
                (t_infer_end - t_infer_start) * 1000,
                "推理",
            )
        logits_per_text = logits_per_image.t()

        return logits_per_image, logits_per_text

#采用混合精度计算，转换为float32
def convert_weights(model: nn.Module):
    def _convert_weights_to_fp16(l):
        #标准线性层和卷积层
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()        #权重
            if l.bias is not None:
                l.bias.data = l.bias.data.half()        #偏置

        #多头注意力层
        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        #其他的投影参数
        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, offload_handler=None):
    """
    模型构建入口。
    Client 端传入 offload_handler; Server 端传入 None。
    """
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in
                        [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        offload_handler=offload_handler  # 注入 handler
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict)
    return model.eval()


# def extract_model_components(model: CLIP) -> Dict:
#     """
#     【Server端专用】
#     将完整的 CLIP 模型对象拆解为零部件字典。
#     Server 启动时调用此函数初始化全局组件库。
#     """
#     components = {
#         'visual_attns': [],
#         'visual_mlps': [],
#         'text_attns': [],
#         'text_mlps': [],
#         'visual_conv': None,
#         'visual_proj': None,
#         'text_proj': model.text_projection,
#         'visual_blocks': None,  # 块级
#         'text_blocks': model.transformer.resblocks  # 块级
#     }
#
#     # 1. 提取 Vision 部分 (假设是 ViT)
#     if isinstance(model.visual, VisionTransformer):
#         components['visual_conv'] = model.visual.conv1
#         components['visual_proj'] = model.visual.proj
#         components['visual_blocks'] = model.visual.transformer.resblocks
#
#         for block in model.visual.transformer.resblocks:
#             components['visual_attns'].append(block.attn)
#             components['visual_mlps'].append(block.mlp)
#
#     # 2. 提取 Text 部分
#     for block in model.transformer.resblocks:
#         components['text_attns'].append(block.attn)
#         components['text_mlps'].append(block.mlp)
#
#     return components


def _convert_image_to_rgb(image):
    return image.convert("RGB")

def get_transform(n_px: int):
    """获取 CLIP 标准的图像预处理管线"""
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
        返回给定输入字符串的标记化表示形式
        参数
        ----------
        texts : Union[str, List[str]]
            要进行标记化的单个输入字符串或输入字符串列表

        context_length : int
            使用的上下文长度；所有CLIP模型均使用77作为上下文长度

        truncate: bool
            当文本的编码长度超过上下文长度时，是否截断文本

        返回
        -------
        包含生成标记的二维张量，形状 = [输入字符串数量, context_length]。
        当torch版本低于1.8.0时，我们返回LongTensor，因为旧版本的index_select要求索引为long类型。
        """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if version.parse(torch.__version__) < version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result