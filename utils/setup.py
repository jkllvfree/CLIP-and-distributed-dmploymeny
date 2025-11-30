import os
import socket
import logging
from logging.handlers import RotatingFileHandler
import torch
from utils import config


#获取设备和 IP 信息
def get_device_and_ip():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    local_ip = socket.gethostbyname(socket.gethostname())
    server_ip = config.SERVER_IP
    return device, local_ip, server_ip

#这里传入了logger，引用对象，仍然返回logger是为了方便链式调用
def configure_logger(logger, caller_file_path, propagate=True):
    """
    通用日志配置函数：给传入的 logger 挂载文件处理器

    参数:
        logger: 要配置的 logger 对象 (可以是 app.logger 也可以是 client_logger)
        caller_file_path: 调用者的路径，用于确定日志文件位置
        propagate: 是否允许日志向上传播 (决定是否在控制台打印)默认为true，Flask调用时候手动传入False
    """
    # 1. 确定日志路径
    base_dir = os.path.dirname(os.path.abspath(caller_file_path))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'client.log')

    # 2. 创建 Handler (如果 logger 还没有 handler 的话)
    # 注意：为了防止多次调用导致添加重复 handler，这里做一个判断
    # 但由于 logging 的 handler 比较复杂，最简单的防重方法是看 handlers 列表是否为空
    # 或者如果不介意重建 handler 对象，也可以。

    has_file_handler = any(isinstance(h, RotatingFileHandler) for h in logger.handlers)

    if not has_file_handler:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
            encoding='utf-8'
        )
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s %(message)s')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)

        logger.addHandler(file_handler)

    # 3. 设置级别
    logger.setLevel(logging.INFO)

    # 4. 设置传播属性
    logger.propagate = propagate

    return logger
