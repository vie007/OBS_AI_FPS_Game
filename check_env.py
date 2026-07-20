"""
环境检查脚本 — 验证所有依赖是否就绪
用法: python check_env.py
"""
import sys
import platform

print("=" * 60)
print("  Watermelon Aimbot 环境检查")
print("=" * 60)

errors = []
warnings = []

# 1. Python 版本
print(f"\n[1] Python 版本: {sys.version}")
if sys.version_info[:2] != (3, 12):
    warnings.append(f"推荐 Python 3.12，当前为 {sys.version_info.major}.{sys.version_info.minor}")

# 2. 操作系统
print(f"[2] 操作系统: {platform.system()} {platform.release()}")
if platform.system() != "Windows":
    warnings.append("此脚本针对 Windows 11 优化")

# 3. ONNX Runtime
print("\n[3] ONNX Runtime:")
try:
    import onnxruntime as ort
    print(f"    版本: {ort.__version__}")
    providers = ort.get_available_providers()
    print(f"    可用后端: {providers}")
    if "DmlExecutionProvider" in providers:
        print("    ✅ DirectML 可用 — GPU 加速就绪")
    else:
        warnings.append("DirectML 不可用，将使用 CPU 推理。请安装: pip install onnxruntime-directml")
except ImportError:
    errors.append("onnxruntime 未安装。请运行: pip install onnxruntime-directml")

# 4. PyTorch
print("\n[4] PyTorch:")
try:
    import torch
    print(f"    版本: {torch.__version__}")
    print(f"    CUDA 可用: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("    ℹ️  CUDA 不可用（ONNX 模式不需要 CUDA，这是正常的）")
except ImportError:
    warnings.append("torch 未安装（ONNX 推理不需要，但 Ultralytics 工具链需要）")

# 5. ONNX 模型文件
print("\n[5] ONNX 模型:")
import os
import configparser
_cfg = configparser.ConfigParser()
_cfg.read("config.ini", encoding="utf-8")
model_name = _cfg.get("AI", "ai_model_name", fallback="sunxds_0.8.0.onnx")
model_path = os.path.join("models", model_name)
if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / 1024 / 1024
    print(f"    ✅ {model_path} 存在 ({size_mb:.1f} MB)")

    # 检查模型输入输出
    try:
        sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        print(f"    输入名称: {inp.name}")
        print(f"    输入形状: {inp.shape}")
        print(f"    输入类型: {inp.type}")
        for out in sess.get_outputs():
            print(f"    输出名称: {out.name}, 形状: {out.shape}, 类型: {out.type}")
        del sess
    except Exception as e:
        errors.append(f"模型加载失败: {e}")
else:
    errors.append(f"模型文件未找到: {model_path}")

# 6. 核心依赖
print("\n[6] 核心依赖:")
core_deps = {
    "numpy": "numpy",
    "cv2": "opencv-python",
    "supervision": "supervision",
    "mss": "mss",
    "keyboard": "keyboard",
    "pynput": "pynput",
    "serial": "pyserial",
    "trackers": "trackers",
}
for module, pip_name in core_deps.items():
    try:
        __import__(module)
        print(f"    ✅ {module}")
    except ImportError:
        errors.append(f"{module} 未安装。请运行: pip install {pip_name}")

# 7. Windows 专属依赖
print("\n[7] Windows 专属依赖:")
if platform.system() == "Windows":
    win_deps = {
        "win32api": "pywin32",
        "win32gui": "pywin32",
        "bettercam": "bettercam",
    }
    for module, pip_name in win_deps.items():
        try:
            __import__(module)
            print(f"    ✅ {module}")
        except ImportError:
            warnings.append(f"{module} 未安装（可选）。安装: pip install {pip_name}")

# 8. config.ini
print("\n[8] 配置文件:")
if os.path.exists("config.ini"):
    import configparser
    config = configparser.ConfigParser()
    config.read("config.ini", encoding="utf-8")
    model = config.get("AI", "ai_model_name", fallback="未设置")
    device = config.get("AI", "ai_device", fallback="未设置")
    print(f"    模型名称: {model}")
    print(f"    AI 设备: {device}")
    if model.endswith(".onnx"):
        print("    ✅ 已配置为 ONNX 模式")
    else:
        warnings.append(f"ai_model_name = {model}，建议改为 .onnx 文件")
else:
    errors.append("config.ini 未找到")

# 总结
print("\n" + "=" * 60)
if errors:
    print("  ❌ 存在错误，请先修复：")
    for e in errors:
        print(f"    • {e}")
if warnings:
    print("  ⚠️  警告（不影响运行）：")
    for w in warnings:
        print(f"    • {w}")
if not errors and not warnings:
    print("  ✅ 所有检查通过，可以运行 python run.py")
elif not errors:
    print("  ✅ 无严重错误，可以尝试运行 python run.py")
print("=" * 60)
