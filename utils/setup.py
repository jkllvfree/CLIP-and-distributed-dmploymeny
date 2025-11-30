import os
import socket
import logging
import sys
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
def configure_logger(logger, caller_file_path, log_filename=None, propagate=True):
    """
    通用日志配置函数

    参数:
        logger: Logger对象
        caller_file_path: 调用者的路径 (__file__)
        log_filename: 指定日志文件名 (如 'server.log')，如果不传则自动根据 caller_file_path 生成
        propagate: 是否允许向上传播。
                   建议：如果想看控制台日志，要么设为 True，要么在该函数内手动添加 StreamHandler。
    """
    # 1. 智能确定日志文件名
    base_dir = os.path.dirname(os.path.abspath(caller_file_path))
    log_dir = os.path.join(base_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    if log_filename is None:
        # 自动根据调用文件的名字生成，例如 server.py -> server.log
        filename = os.path.splitext(os.path.basename(caller_file_path))[0]
        log_filename = f"{filename}.log"

    log_file_path = os.path.join(log_dir, log_filename)

    # 定义统一的格式
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')

    # 2. 添加 FileHandler (防止重复添加)
    # 检查是否已经有了写向 [同一路径] 的 Handler，比只检查类型更安全
    has_target_handler = False
    for h in logger.handlers:
        if isinstance(h, RotatingFileHandler) and h.baseFilename == log_file_path:
            has_target_handler = True
            break

    if not has_target_handler:
        file_handler = RotatingFileHandler(
            log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)

    # 3. 确保控制台有输出 (如果禁止了传播，或者 Logger 本身没有 StreamHandler)
    # 只有当 propagate 为 False 时，我们需要手动保底加一个 StreamHandler，否则控制台就瞎了
    if not propagate:
        has_console = any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        if not has_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            console_handler.setLevel(logging.INFO)
            logger.addHandler(console_handler)

    # 4. 设置 Logger 级别
    logger.setLevel(logging.INFO)

    # 5. 设置传播
    logger.propagate = propagate

    return logger
