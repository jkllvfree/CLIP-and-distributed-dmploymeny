# client.py
import torch
import logging
from model.clip_loader import build_model
from utils.setup import get_device_and_ip,configure_logger
from utils.offloader import OffloadHandler
from utils import config as cfg

# 1. 初始化

device, local_ip, server_ip = get_device_and_ip()
client_logger = logging.getLogger('client')
configure_logger(client_logger,__file__)

# 2. 准备卸载处理器
offloader = OffloadHandler(
    server_ip=cfg.SERVER_IP,
    server_port=cfg.SERVER_PORT,
    config=cfg.OFFLOAD_CONFIG,
    logger=client_logger
)

# 3. 加载模型 (注入 handler)
state_dict = torch.jit.load('ViT-L-14.pt', map_location='cpu').state_dict()
model = build_model(state_dict, offload_handler=offloader).to(device)

# 4. 推理
print("Start Inference...")
dummy_image = torch.randn(1, 3, 224, 224).to(device)
dummy_text = torch.randint(0, 49408, (1, 77)).to(device)

with torch.no_grad():
    # 当运行到需要卸载的层时，会自动调用 offloader 发送请求
    img_feat, txt_feat = model(dummy_image, dummy_text)

print("Done.")