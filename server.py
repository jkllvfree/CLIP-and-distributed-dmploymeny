import torch.nn as nn
import time
import sys

import logging
from flask import Flask, request, jsonify
import torch
import base64
import io
from utils.offloader import OffloadHandler
from model.clip_loader import build_model
from utils import config
from utils.setup import get_device_and_ip,configure_logger
from utils.pred import predict
import manager.db_manager as db
from utils import config as cfg

from utils.speed_measurement import run_timed_inference

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

app = Flask(__name__)

DEVICE, local_ip, server_ip = get_device_and_ip()

logger = logging.getLogger('server')
logger = configure_logger(logger, __file__, propagate=False)

# 1. 加载完整模型 (不传 handler，即纯本地)
state_dict = torch.jit.load('ViT-L-14.pt', map_location='cpu').state_dict()
model = build_model(state_dict, offload_handler=None).to(DEVICE)  # 假设服务端用 CUDA

# 2. 拆解组件
#COMPONENTS = extract_model_components(full_model)

# model = build_model(state_dict).to(DEVICE)

# 2. 准备卸载处理器
offloader = OffloadHandler(
    server_ip=cfg.SERVER_IP,
    server_port=cfg.SERVER_PORT,
    config=cfg.OFFLOAD_CONFIG,
    logger=logger
)

model_explict = build_model(state_dict, offload_handler=offloader).to(DEVICE)


model = model.to(DEVICE)
model = model.eval()
print("模型加载完成")

# 残差块的注意力机制与MLP
attns = {'visual': [], 'text': []}
mlps = {'visual': [], 'text': []}
for i in range(24):
    attn = list(model.visual.transformer.resblocks.children())[i].attn
    attn = attn.to(DEVICE)
    attn.eval()
    attns['visual'].append(attn)

    mlp = list(model.visual.transformer.resblocks.children())[i].mlp
    mlp = mlp.to(DEVICE)
    mlp.eval()
    mlps['visual'].append(mlp)

for i in range(12):
    attn = list(model.transformer.resblocks.children())[i].attn
    attn = attn.to(DEVICE)
    attn.eval()
    attns['text'].append(attn)

    mlp = list(model.transformer.resblocks.children())[i].mlp
    mlp = mlp.to(DEVICE)
    mlp.eval()
    mlps['text'].append(mlp)

# 视觉编码器的卷积层
conv = model.visual.conv1.to(DEVICE)
conv.eval()

# 文本编码器和视觉编码器的末端投影层
text_proj = model.text_projection.to(DEVICE)
visual_proj = model.visual.proj

# 文本和编码器的encoder blocks
text_blocks = model.transformer.resblocks.to(DEVICE)
visual_blocks = model.visual.transformer.resblocks.to(DEVICE)


def decode_tensor(data_str, device='cuda'):
    buffer = io.BytesIO(base64.b64decode(data_str))
    data = torch.load(buffer)
    # 将字典里的 tensor 移到 GPU
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}


def encode_tensor(tensor_out):
    buffer = io.BytesIO()
    torch.save({'output': tensor_out.cpu()}, buffer)
    return base64.b64encode(buffer.getvalue()).decode()


@app.route('/attention', methods=['POST'])
def attention():

    server_recv_ts = time.time()
    # 接收base64编码的数据
    data_json = request.json

    # 解析 JSON，拿到 client_send_ts
    client_send_ts = data_json.get('client_send_ts', None)
    one_way_ms = server_recv_ts - client_send_ts

    data = base64.b64decode(data_json['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE) # 输入的张量
    encoder_type = tensor_dict['encoder_type'] # 文本编码器 or 视觉编码器
    block_num = tensor_dict['block_num'] # 第几个残差块
    attn = attns[encoder_type][block_num] # 提取对应的attention模块
    attn_mask = tensor_dict['attn_mask']
    if attn_mask is not None:
        attn_mask = attn_mask.to(DEVICE)

    output = run_timed_inference(
        tag="attention",
        local_ip=local_ip,
        server_ip=server_ip,
        one_way_ms=one_way_ms*1000,
        logger=logger,
        device=DEVICE,
        infer_func=lambda: attn(
            query=x,
            key=x,
            value=x,
            need_weights=False,
            attn_mask=attn_mask
        )[0],
        use_no_grad=True,
    )

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()


    return jsonify({'output': output_str})

@app.route('/cos_sim', methods=['POST'])
def cos_sim():
    server_recv_ts = time.time()
    # 接收base64编码的数据
    data_json = request.json

    # 解析 JSON，拿到 client_send_ts
    client_send_ts = data_json.get('client_send_ts', None)
    one_way_ms = server_recv_ts - client_send_ts
    data = base64.b64decode(data_json['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')
    image_features = tensor_dict['image_features'].to(DEVICE)
    text_features = tensor_dict['text_features'].to(DEVICE)
    logit_scale = tensor_dict['logit_scale'].to(DEVICE)

    output = run_timed_inference(
        tag="cos_sim",
        local_ip=local_ip,
        server_ip=server_ip,
        one_way_ms=one_way_ms*1000,
        logger=logger,
        device=DEVICE,
        infer_func=lambda: logit_scale * image_features @ text_features.t(),
        use_no_grad=False,
    )

    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/mlp', methods=['POST'])
def mlp():
    server_recv_ts = time.time()
    # 接收base64编码的数据
    data_json = request.json

    # 解析 JSON，拿到 client_send_ts
    client_send_ts = data_json.get('client_send_ts', None)
    one_way_ms = server_recv_ts - client_send_ts
    data = base64.b64decode(data_json['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量
    encoder_type = tensor_dict['encoder_type']  # 文本编码器 or 视觉编码器
    block_num = tensor_dict['block_num']  # 第几个残差块
    mlp = mlps[encoder_type][block_num]  # 提取对应的mlp模块

    output = run_timed_inference(
        tag="mlp",
        local_ip=local_ip,
        server_ip=server_ip,
        one_way_ms=one_way_ms*1000,
        logger=logger,
        device=DEVICE,
        infer_func=lambda: mlp(x),
        use_no_grad=True,
    )

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/vision_conv', methods=['POST'])
def vision_conv():
    server_recv_ts = time.time()
    # 接收base64编码的数据
    data_json = request.json

    # 解析 JSON，拿到 client_send_ts
    client_send_ts = data_json.get('client_send_ts', None)
    one_way_ms = server_recv_ts - client_send_ts
    data = base64.b64decode(data_json['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量

    output = run_timed_inference(
        tag="vision_conv",
        local_ip=local_ip,
        server_ip=server_ip,
        one_way_ms=one_way_ms*1000,
        logger=logger,
        device=DEVICE,
        infer_func=lambda: conv(x),
        use_no_grad=True,
    )

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/text_projection', methods=['POST'])
def text_projection():
    server_recv_ts = time.time()
    # 接收base64编码的数据
    data_json = request.json

    # 解析 JSON，拿到 client_send_ts
    client_send_ts = data_json.get('client_send_ts', None)
    one_way_ms = server_recv_ts - client_send_ts
    data = base64.b64decode(data_json['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量

    output = run_timed_inference(
        tag="text_projection",
        local_ip=local_ip,
        server_ip=server_ip,
        one_way_ms=one_way_ms*1000,
        logger=logger,
        device=DEVICE,
        infer_func=lambda: x @ text_proj,
        use_no_grad=True,
    )

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/visual_projection', methods=['POST'])
def visual_projection():
    server_recv_ts = time.time()
    # 接收base64编码的数据
    data_json = request.json

    # 解析 JSON，拿到 client_send_ts
    client_send_ts = data_json.get('client_send_ts', None)
    one_way_ms = server_recv_ts - client_send_ts
    data = base64.b64decode(data_json['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量

    output = run_timed_inference(
        tag="vis_projection",
        local_ip=local_ip,
        server_ip=server_ip,
        one_way_ms=one_way_ms*1000,
        logger=logger,
        device=DEVICE,
        infer_func=lambda: x @ visual_proj,
        use_no_grad=True,
    )

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/encoder_blocks', methods=['POST'])
def encoder_blocks():
    server_recv_ts = time.time()
    # 接收base64编码的数据
    data_json = request.json

    # 解析 JSON，拿到 client_send_ts
    client_send_ts = data_json.get('client_send_ts', None)
    one_way_ms = server_recv_ts - client_send_ts
    data = base64.b64decode(data_json['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    x = tensor_dict['x'].to(DEVICE)  # 输入的张量
    encoder_type = tensor_dict['encoder_type']  # 文本编码器 or 视觉编码器

    if encoder_type == 'text':
        blocks = text_blocks
    else:
        blocks = visual_blocks

    output = run_timed_inference(
        tag="encoder_blocks",
        local_ip=local_ip,
        server_ip=server_ip,
        one_way_ms=one_way_ms*1000,
        logger=logger,
        device=DEVICE,
        infer_func=lambda: blocks(x),
        use_no_grad=True,
    )

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({'output': output.cpu()}, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/complete_encoders', methods=['POST'])
def complete_encoders():
    server_recv_ts = time.time()
    # 接收base64编码的数据
    data_json = request.json

    # 解析 JSON，拿到 client_send_ts
    client_send_ts = data_json.get('client_send_ts', None)
    one_way_ms = server_recv_ts - client_send_ts
    data = base64.b64decode(data_json['data'])
    buffer = io.BytesIO(data)
    tensor_dict = torch.load(buffer, map_location='cpu')

    image = tensor_dict['image'].to(DEVICE)  # 输入的张量
    text = tensor_dict['text'].to(DEVICE)  # 文本编码器 or 视觉编码器

    def encode_both():
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:  # 并行执行
            inputs = [(image,), (text,)]
            models = [model.encode_image, model.encode_text]
            outputs = nn.parallel.parallel_apply(models, inputs)
            image_features_, text_features_ = outputs
        else:
            image_features_ = model.encode_image(image)
            text_features_ = model.encode_text(text)
        return image_features_, text_features_

    image_features, text_features = run_timed_inference(
        tag="complete_encoders",
        local_ip=local_ip,
        server_ip=server_ip,
        one_way_ms=one_way_ms*1000,
        logger=logger,
        device=DEVICE,
        infer_func=encode_both,
        use_no_grad=True,
    )

    # 将特征张量转换为base64返回
    buffer = io.BytesIO()
    torch.save({
        'image_features': image_features.cpu(),
        'text_features': text_features.cpu()
    }, buffer)
    output_str = base64.b64encode(buffer.getvalue()).decode()

    return jsonify({'output': output_str})

@app.route('/health', methods=['GET'])            #客户端测试服务器健康状态的接口
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'visual_blocks': len(attns['visual']),
        'text_blocks': len(attns['text'])
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.SERVER_PORT)