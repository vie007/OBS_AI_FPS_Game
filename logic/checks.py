import torch
import os
import sys
import importlib.util

from logic.config_watcher import cfg
from logic.logger import logger
from logic.platform import IS_WINDOWS, IS_LINUX

def convert_onnx_to_fp16():
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(f"models/{cfg.AI_model_name}")
    model_fp16 = float16.convert_float_to_float16(model)
    new_model_name = cfg.AI_model_name.replace(".onnx", "_fp16.onnx")
    onnx.save(model_fp16, f"models/{new_model_name}")

    logger.info(f"""
    [AI] 转换后的模型已保存为 'models/{new_model_name}'。
    请将 config.ini 中的 ai_model_name 修改为转换后的模型名称 ({new_model_name})。
    """)

def check_model_fp16():
    try:
        import onnx
        from onnxconverter_common import float16
    except ModuleNotFoundError:
        logger.error("[AI] ONNX fp16 检查需要以下依赖包: onnx 和 onnxconverter-common。")
        raise SystemExit(1)

    model = onnx.load(f"models/{cfg.AI_model_name}")

    graph = model.graph

    for input_tensor in graph.input:
        tensor_type = input_tensor.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            return True

    for output_tensor in graph.output:
        tensor_type = output_tensor.type.tensor_type
        if tensor_type.elem_type == onnx.TensorProto.FLOAT16:
            return True

    return False

def Warnings():
    # 截取
    if cfg.capture_fps >= 120:
        logger.warning("[截取] 帧率过高可能影响自动瞄准的稳定性（画面抖动）。")
    if cfg.detection_window_width >= 600:
        logger.warning("[截取] 检测窗口宽度超过 600 像素，大窗口可能严重影响性能。")
    if cfg.detection_window_height >= 600:
        logger.warning("[截取] 检测窗口高度超过 600 像素，大窗口可能严重影响性能。")

    # AI
    is_onnx = cfg.AI_model_name.endswith(".onnx")
    if _is_cpu_device() and not is_onnx:
        logger.warning("[AI] ai_device 设置为 CPU，推理稳定但速度远慢于 CUDA。")
    if is_onnx:
        logger.info("[AI] ONNX 模式: 推理由 onnxruntime 后端处理（DirectML / CPU），ai_device 设置不影响实际推理设备")
    if cfg.AI_model_name.endswith(".pt"):
        logger.warning("[AI] 建议将模型导出为 .engine 以获得更好的性能！\n导出教程: 'https://github.com/SunOner/sunone_aimbot_docs/blob/main/ai_models/ai_models.md'")
    if cfg.AI_conf <= 0.10:
        logger.warning("[AI] ai_conf 值过低可能导致大量误检测。")

    # 鼠标
    if IS_WINDOWS:
        if not any([cfg.arduino_move, cfg.ch9329_move]):
            logger.warning("[鼠标] 使用 win32api 软件方式移动鼠标（未使用 Arduino/CH9329 硬件绕过）可能加速封号，请自行承担风险。")
        if not cfg.arduino_shoot and not cfg.ch9329_shoot and cfg.auto_shoot:
            logger.warning("[鼠标] 使用 win32api 软件方式自动射击（未使用 Arduino/CH9329 硬件绕过）可能加速封号，请自行承担风险。")
    elif IS_LINUX and importlib.util.find_spec("pynput") is None:
        logger.warning("[鼠标] pynput 未安装，Linux 下的热键监听和原生鼠标输入将不可用。")
    selected_methods = sum([cfg.arduino_move, cfg.ch9329_move])
    if selected_methods > 1:
        raise ValueError("[鼠标] 错误: 同时启用了多种鼠标输入方式，只能选择一种。")

    # 显示当前输入模式
    if cfg.ch9329_move or cfg.ch9329_shoot:
        logger.info(f"[输入模式] ✅ CH9329 硬件鼠标模拟（移动={cfg.ch9329_move}, 射击={cfg.ch9329_shoot}, 端口={cfg.ch9329_port}）")
    elif cfg.arduino_move or cfg.arduino_shoot:
        logger.info(f"[输入模式] ✅ Arduino 硬件鼠标模拟（移动={cfg.arduino_move}, 射击={cfg.arduino_shoot}）")
    else:
        logger.info("[输入模式] ⚠️ win32api 软件模拟（最易被检测，建议搭配硬件方案）")

    # CH9329 连接检查
    if cfg.ch9329_move or cfg.ch9329_shoot:
        try:
            from logic.ch9329 import ch9329
            if not ch9329.is_connected():
                logger.error("[CH9329] 设备未连接！请检查: 1) CH9329 模块已通过 USB 连接到电脑  2) 驱动已安装（CH340）  3) config.ini 中 ch9329_port 设置正确")
        except Exception as e:
            logger.error(f"[CH9329] 初始化失败: {e}")

    # 调试
    if cfg.show_window:
        logger.warning("[调试] 调试窗口已开启，会影响性能。正式使用时建议关闭。")

def _is_cpu_device():
    return str(cfg.AI_device).strip().lower().startswith("cpu")

def _cuda_device_index():
    raw = str(cfg.AI_device).strip().lower()
    if raw.startswith("cuda:"):
        raw = raw.split(":", 1)[1]
    return int(raw) if raw.isdigit() else None

def _validate_capture_config():
    selected = [
        name for name, enabled in (
            ("bettercam_capture", cfg.Bettercam_capture),
            ("obs_capture", cfg.Obs_capture),
            ("mss_capture", cfg.mss_capture),
        )
        if enabled
    ]

    if len(selected) < 1:
        logger.error("[截取] 请至少启用一种截取方式: bettercam_capture、obs_capture 或 mss_capture。")
        raise SystemExit(1)

    if len(selected) > 1:
        logger.error(f"[截取] 只能启用一种截取方式，当前启用了: {', '.join(selected)}。")
        raise SystemExit(1)

    if cfg.capture_fps <= 0:
        logger.error("[截取] capture_fps 必须大于 0。")
        raise SystemExit(1)

    if cfg.detection_window_width <= 0 or cfg.detection_window_height <= 0:
        logger.error("[截取] 检测窗口的宽度和高度必须大于 0。")
        raise SystemExit(1)

    if cfg.Bettercam_capture and not IS_WINDOWS:
        logger.error("[截取] BetterCam 仅支持 Windows。Linux 请使用 mss_capture 或 obs_capture。")
        raise SystemExit(1)

def _validate_torch_device():
    if _is_cpu_device() or cfg.AI_enable_AMD:
        return

    if not torch.cuda.is_available():
        torch_details = (
            f"Python 路径: {sys.executable}\n"
            f"Torch 版本: {getattr(torch, '__version__', '未安装')}\n"
            f"Torch CUDA 构建: {getattr(torch.version, 'cuda', None) or '无'}\n"
            f"Torch 文件: {getattr(torch, '__file__', '未知')}\n"
            f"CUDA 设备数量: {torch.cuda.device_count()}"
        )
        logger.error(
            f"[AI] ai_device 需要 CUDA，但当前 Python 环境中 CUDA 不可用。\n"
            f"{torch_details}\n"
            "如需 CPU 模式请设置 ai_device = cpu；如需 CUDA 请重装 CUDA 版 PyTorch。"
        )
        raise SystemExit(1)

    device_index = _cuda_device_index()
    if device_index is not None and device_index >= torch.cuda.device_count():
        logger.error(f"[AI] ai_device 指向 CUDA 设备 {device_index}，但仅有 {torch.cuda.device_count()} 个 CUDA 设备可用。")
        raise SystemExit(1)

def run_checks():
    os.makedirs("screenshots", exist_ok=True)

    _validate_capture_config()

    is_onnx = cfg.AI_model_name.endswith(".onnx")

    # ONNX + DirectML 不需要 CUDA，跳过 PyTorch CUDA 校验
    if not is_onnx:
        _validate_torch_device()

    if cfg.ai_model_use_enc:
        # 加密模式: 检查 .enc 文件是否存在
        enc_path = f"models/sunxds_0.8.0.enc"
        if not os.path.exists(enc_path):
            logger.error(f"[AI] 加密模型 {enc_path} 未找到！")
            raise SystemExit(1)
    elif not os.path.exists(f"models/{cfg.AI_model_name}"):
        logger.error(f"[AI] 模型文件 {cfg.AI_model_name} 未找到！请检查 config.ini 中 ai_model_name 的设置。")
        raise SystemExit(1)

    if not is_onnx and cfg.AI_model_name.endswith(".onnx"):
        fp16 = check_model_fp16()
        if fp16 == False:
            check_converted_model = cfg.AI_model_name.replace(".onnx", "_fp16.onnx")
            if not os.path.exists(f"models/{check_converted_model}"):
                logger.info(f"[AI] 当前模型 '{cfg.AI_model_name}' 为 fp32 格式，正在转换为 fp16...")
                convert_onnx_to_fp16()
                raise SystemExit(0)
            else:
                logger.info(f"[AI] 请使用转换后的模型 '{check_converted_model}'。\n修改 config.ini: ai_model_name = {check_converted_model}")
                raise SystemExit(0)
    Warnings()
