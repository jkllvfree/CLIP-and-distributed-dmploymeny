# utils/offloader.py
import torch
import io
import base64
import requests
import time
import logging


class OffloadHandler:
    def __init__(self, server_ip, server_port, config, logger=None):
        self.server_ip = server_ip
        self.server_port = server_port
        self.config = config
        self.logger = logger or logging.getLogger('client')

    def should_offload(self, module_type: str, layer_id: int = None, ) -> bool:
        """
        判断当前模块是否需要卸载到服务器。
        例如: module_type='attn', encoder_type='visual'
        """
        key = module_type  # e.g. "visual_attn"
        return self.config.get(key, False)

    def call_remote(self, endpoint: str, data_dict: dict, device: torch.device):
        """
        序列化数据 -> 发送请求 -> 反序列化结果
        """
        # 1. 序列化
        buffer = io.BytesIO()
        # 将 tensor 转为 cpu 以便序列化
        cpu_data = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in data_dict.items()}
        torch.save(cpu_data, buffer)
        data_str = base64.b64encode(buffer.getvalue()).decode()

        # 2. 发送请求
        url = f"http://{self.server_ip}:{self.server_port}/{endpoint}"
        payload = {
            "data": data_str,
            "client_send_ts": time.time()
        }

        t_start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            resp_json = resp.json()
            http_ms = (time.perf_counter() - t_start) * 1000

            if self.logger:
                self.logger.info(f"[{endpoint}] server={self.server_ip} rtt={http_ms:.2f}ms type=传输")

            # 3. 反序列化
            output_str = resp_json['output']
            output_buffer = io.BytesIO(base64.b64decode(output_str))
            output_dict = torch.load(output_buffer)

            # 移回原设备
            return output_dict['output'].to(device)

        except Exception as e:
            if self.logger:
                self.logger.error(f"[{endpoint}] Failed: {e}")
            raise e