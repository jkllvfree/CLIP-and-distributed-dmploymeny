import time
import torch
import requests
from utils import config


def run_timed_inference(
    tag: str,
    local_ip,
    server_ip,
    one_way_ms,
    logger,
    device: torch.device,
    infer_func,
    extra_info: str = "",
    sync_cuda: bool = True,
    use_no_grad: bool = False,
):
    """
    通用推理测速函数：

    1. 可选 CUDA 同步（避免异步导致计时不准）
    2. 前后计时（time.perf_counter）
    3. 可选 torch.no_grad()
    4. 统一在 logger 里打印 "[tag] infer_ms=xxx type=推理 [extra_info]"
    5. 返回 infer_func() 的输出

    参数：
        tag        : 日志标签，如 "attention" / "mlp" / "cos_sim"
        logger     : 日志对象，如 app.logger / client_logger
        device     : torch.device
        infer_func : 无参函数，在其中写具体前向推理逻辑
        extra_info : 附加日志信息字符串
        sync_cuda  : 是否在计时前后做 torch.cuda.synchronize()
        use_no_grad: 是否在 torch.no_grad() 下执行 infer_func
    """
    if sync_cuda and device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    if use_no_grad:
        with torch.no_grad():
            output = infer_func()
    else:
        output = infer_func()

    if sync_cuda and device.type == "cuda":
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    infer_ms = (t_end - t_start) * 1000.0

    if extra_info:
        logger.info(
            "[%s] local_ip=%s server_ip=%s infer_ms=%.3f one_way_ms=%.3f type=%s %s",
            tag,
            local_ip,
            server_ip,
            infer_ms,
            one_way_ms,
            device,
            extra_info,
        )
    else:
        logger.info(
            "[%s] local_ip=%s server_ip=%s infer_ms=%.3f one_way_ms=%.3f type=%s",
            tag,
            local_ip,
            server_ip,
            infer_ms,
            one_way_ms,
            device,
        )

    return output

#暂时未用到的函数
def call_api(endpoint: str, data_str):
    url = f"http://{config.SERVER_HOST}:{config.SERVER_PORT}/{endpoint}"

    # 2. 记录“我准备发送”的时间（客户端时间）
    client_send_ts = time.time()  # 秒为单位，float，小数部分是毫秒

    payload = {
        'data': data_str,
        'client_send_ts': client_send_ts,  # 带到服务器
    }

    # 3. 正常发请求（这个是整体 RTT，你也可以照样记）
    http_start = time.perf_counter()
    resp = requests.post(url, json=payload)
    http_end = time.perf_counter()
    http_elapsed_ms = (http_end - http_start) * 1000.0

    resp_json = resp.json()
    # 后面还可以从 resp_json 里取单向延迟，稍后说明

    return resp_json, http_elapsed_ms


def call_api_with_log(endpoint: str, data_str: str, server_ip: str, port: int,
                      local_ip: str, logger):
    """
    调用服务器 API 并写入日志。

    endpoint: "attention" / "mlp" / "vision_conv" 等（不带 /）
    data_str: 客户端已经构造好的 base64 字符串
    server_ip: 服务器 IP
    port: 服务器端口
    local_ip: 本机 IP（用于日志记录）
    logger: 你传入的 client_logger
    """

    url = f"http://{server_ip}:{port}/{endpoint}"

    payload = {
        "data": data_str,
        "client_send_ts": time.time(),  # 用于单向延迟计算
    }

    t_http_start = time.perf_counter()
    try:
        # ===== 计算 HTTP 往返时间 ====
        resp = requests.post(url, json=payload, timeout=30)
        t_http_end = time.perf_counter()
        http_ms = (t_http_end - t_http_start) * 1000.0

        resp.raise_for_status()
        resp_json = resp.json()

        # ===== 写日志 =====
        logger.info(
            "[%s] local_ip=%s server_ip=%s http_rtt_ms=%.3f type=%s",
            endpoint,
            local_ip,
            server_ip,
            http_ms,
            "传输",
        )

        return resp_json, http_ms
    except Exception as e:
        # 失败时也记一条日志，方便排查
        t_http_end = time.perf_counter()
        http_ms = (t_http_end - t_http_start) * 1000.0
        logger.error(
            "[%s] local_ip=%s server_ip=%s http_double_ms=%.3f type=%s error=%r",
            endpoint,
            local_ip,
            server_ip,
            http_ms,
            "传输失败",
            e,
        )
        raise
