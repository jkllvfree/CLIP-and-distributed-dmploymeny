#服务器处理时候，接口写这里
# api/server.py
import logging
import threading
from flask import Flask, request, jsonify
import torch
import base64
import numpy as np
import io

from model.clip_loader import build_model
from utils import config
from utils.setup import get_device_and_ip,configure_logger
from utils.pred import predict
import manager.db_manager as db

app = Flask(__name__)

DEVICE, local_ip, server_ip = get_device_and_ip()

logger = logging.getLogger('client')
logger = configure_logger(logger, __file__, propagate=False)

# 1. 加载完整模型 (不传 handler，即纯本地)
state_dict = torch.jit.load('ViT-L-14.pt', map_location='cpu').state_dict()
full_model = build_model(state_dict, offload_handler=None).to('cuda')  # 假设服务端用 CUDA

# 2. 拆解组件
#COMPONENTS = extract_model_components(full_model)

state_dict = torch.jit.load('ViT-L-14.pt', map_location='cpu').state_dict()
model = build_model(state_dict).to(DEVICE)

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


# @app.route('/attention', methods=['POST'])
# def attention_endpoint():
#     req_data = request.json
#     input_data = decode_tensor(req_data['data'])
#
#     # 解析参数
#     x = input_data['x']
#     attn_mask = input_data.get('attn_mask')
#     layer_id = input_data['layer_id']
#     encoder_type = input_data['encoder_type']  # 'visual' or 'text'
#
#     # 从组件库中获取对应的 Attention 模块
#     if encoder_type == 'visual':
#         attn_module = COMPONENTS['visual_attns'][layer_id]
#     else:
#         attn_module = COMPONENTS['text_attns'][layer_id]
#
#     # 执行计算
#     with torch.no_grad():
#         output = attn_module(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
#
#     return jsonify({'output': encode_tensor(output)})

@app.route('/health', methods=['GET'])            #客户端测试服务器健康状态的接口
def health_check():
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'visual_blocks': len(attns['visual']),
        'text_blocks': len(attns['text'])
    })

@app.route("/predict", methods=["POST"])
def predict_():
    data = request.get_json(force=True)

    image_urls = data.get("image_urls", None)
    class_names = data.get("class_names", None)

    if not image_urls or not isinstance(image_urls, list):
        return jsonify({"error": "image_urls 必须是非空列表"}), 400

    if not class_names or not isinstance(class_names, list):
        return jsonify({"error": "class_names 必须是非空列表"}), 400

    try:
        logits_per_image, _ = predict(model, image_urls, class_names)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    # logits => 概率
    probs = torch.softmax(logits_per_image, dim=-1)  # [B, N]
    probs_np = probs.cpu().numpy()

    predictions = []
    for i, url in enumerate(image_urls):
        top1_idx = int(probs_np[i].argmax())
        predictions.append({
            "url": url,
            "class": class_names[top1_idx]
        })

    return jsonify({"results": predictions})

def func1_process(
    photo_list,
    photo_id_list,
    description_list,
    description_id_list,
    logger
 ):
    conn = None
    cursor = None


    logits_per_image, _ = predict(model, photo_list, description_list)
    probs = torch.softmax(logits_per_image, dim=-1)
    probs_np = probs.cpu().numpy()
    insert_rows = []
    num_photos = probs_np.shape[0]

    for i in range(num_photos):
        photo_id = photo_id_list[i]
        top1_desc_idx = probs_np[i].argmax()
        matched_description_id = description_id_list[top1_desc_idx]

        insert_rows.append((
            photo_id,
            matched_description_id,
            1
        ))

    sql_insert = """
            INSERT INTO photo_description (
                photo_id,
                description_id,
                function_type
            ) VALUES (%s, %s, %s)
            """
    db.execute_insert(sql_insert, insert_rows, app.logger)



@app.route("/upload", methods=["POST"])
def insert_photo_descriptions():
    conn = None
    cursor = None
    #读取json
    try:
        data = request.get_json(force=True)
        photo_list = data.get("photoList")
        photo_id_list = data.get("photoId")
        description_list = data.get("descriptionList")
        description_id_list = data.get("descriptionId")

        if not (photo_list and photo_id_list and description_list and description_id_list):
            return jsonify(
                {"code": 0, "msg": "参数不完整：photoList, photoId, descriptionList, descriptionId 均为必须"}), 400
        if len(description_list) != len(description_id_list):
            return jsonify(
                {"code": 0, "msg": "描述文本列表和描述ID列表长度不一致"}), 400
        if len(photo_list) != len(photo_id_list):
            return jsonify(
                {"code": 0, "msg": "图片地址列表和图片ID列表长度不一致"}), 400

    except Exception as e:
        app.logger.error(f"参数解析错误: {e}")
        return jsonify({"code": 0, "msg": "请求数据格式错误"}), 400

    try:
        thread = threading.Thread(
            target = func1_process,
            args=(photo_list, photo_id_list, description_list, description_id_list,app.logger)
        )
        thread.start()

        return jsonify({"code": 1, "msg": "图片上传及匹配任务已接收，正在后台处理"}), 200

    except Exception as e:
        app.logger.error(f"启动后台线程失败: {e}")
        return jsonify({"code": 0, "msg": f"启动后台任务失败：{str(e)}"}), 500


def func2_process(
        photo_records,
        description_list,
        description_id_list,
        user_id,
        logger
):
    conn = None
    cursor = None

    photo_id_list = [record['id'] for record in photo_records]
    photo_url_list = [record['image'] for record in photo_records]

    logits_per_image, logits_per_text = predict(model, photo_url_list, description_list)

    probs = torch.softmax(logits_per_text, dim=-1)
    probs_np = probs.cpu().numpy()

    insert_rows = []
    num_descriptions = probs_np.shape[0]

    for desc_idx in range(num_descriptions):
        description_id = description_id_list[desc_idx]
        desc_probs = probs_np[desc_idx]
        # 找到得分最高的 Top 5 图片的**索引**
        top_5_photo_indices = np.argsort(desc_probs)[::-1][:5]

        for photo_index in top_5_photo_indices:
            photo_id = photo_id_list[photo_index]
            insert_rows.append((photo_id, description_id, 2))

    sql_write = """
            INSERT INTO photo_description (
                photo_id, 
                description_id,
                function_type
            ) VALUES (%s, %s, %s)
            """
    db.execute_insert(sql_write, insert_rows, app.logger)

#以文搜图
@app.route("/getPhotos", methods=["POST"])
def search_photo_by_description():
    try:
        data = request.get_json(force=True)
        description_list = data.get("description", None)
        description_id_list = data.get("descriptionId", None)
        user_id = data.get("userId", None)

        if not (description_list and description_id_list and user_id):
            return jsonify(
                {"code": 0, "msg": "参数不完整：description, descriptionId, userId 均为必须"}), 400
        if len(description_list) != len(description_id_list):
            return jsonify(
                {"code": 0, "msg": "描述文本列表和描述ID列表长度不一致"}), 400

    except Exception as e:
        app.logger.error(f"参数解析错误: {e}")
        return jsonify({"code": 0, "msg": "请求数据格式错误"}), 400

    sql_read = """
                SELECT id, image 
                FROM photos
                WHERE user_id = %s
            """

    photo_records = db.execute_query(sql_read,(user_id,),app.logger)
    if not photo_records:
        return jsonify({"code": 0, "msg": "该用户没有图片记录"}), 200

        # 启动后台线程
        # 将获取到的 photo_records 和其他参数传递给后台处理函数
    thread = threading.Thread(
        target = func2_process,
        args = (photo_records, description_list, description_id_list, user_id,app.logger)
    )
    thread.start()

        # 立即返回响应给客户端
    return jsonify({"code": 1, "msg": "任务已接收，正在后台进行推理和匹配，请稍后查询结果"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.SERVER_PORT)