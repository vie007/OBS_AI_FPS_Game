"""
Watermelon Aimbot 打包脚本
将项目打包为 Windows 可执行文件（无需安装 Python 环境）

前提条件:
  pip install pyinstaller

用法:
  python build_exe.py          # 打包
  python build_exe.py --clean  # 清理后重新打包
"""
import os
import sys
import shutil
import subprocess
import platform

# ──────────────────── 配置 ────────────────────

APP_NAME = "西瓜去水印文案配音助手"
ENTRY_POINT = "run.py"
ICON_FILE = os.path.join("icon", "icon.ico")  # 直接使用标准 .ico 文件

# 需要复制到 exe 旁边的外部资源（用户可编辑/大文件）
EXTERNAL_FILES = [
    "config.ini",
    "window_names.txt",
    "version",
]

EXTERNAL_DIRS = [
    "models",
]

# 需要创建的运行时目录
RUNTIME_DIRS = [
    "screenshots",
]

# PyInstaller 可能漏掉的隐式导入（仅添加确认需要的）
HIDDEN_IMPORTS = [
    "onnxruntime",
    "cv2",
    "numpy",
    "torch",
    "supervision",
    "scipy",
    "matplotlib",
    "mss",
    "serial",
    "pynput",
    "keyboard",
]

# 仅排除与项目完全无关的包
EXCLUDES = [
]

# DLL 文件（从源码目录复制到 _internal/logic/）
DLL_FILES = [
]


# ──────────────────── 工具函数 ────────────────────

def run_cmd(cmd, description):
    """执行命令并打印结果"""
    print(f"\n{'='*60}")
    print(f"  {description}")
    print(f"{'='*60}")
    print(f"  > {' '.join(cmd)}\n")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"\n[错误] {description} 失败 (返回码: {result.returncode})")
        sys.exit(1)
    print(f"\n[完成] {description}")


def copy_file(src, dst_dir):
    """复制文件，自动创建目标目录"""
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    shutil.copy2(src, dst)
    print(f"  复制: {src} → {dst}")


def copy_dir(src, dst):
    """复制整个目录"""
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    print(f"  复制目录: {src}/ → {dst}/")


# ──────────────────── 主流程 ────────────────────

def clean():
    """清理构建产物"""
    print("\n[清理] 删除构建目录...")
    for d in ["build", "dist", f"{APP_NAME}.spec"]:
        if os.path.isdir(d):
            shutil.rmtree(d)
            print(f"  已删除: {d}/")
        elif os.path.isfile(d):
            os.remove(d)
            print(f"  已删除: {d}")
    print("[清理] 完成")


def check_environment():
    """检查打包环境"""
    print("\n[检查] 打包环境...")

    if platform.system() != "Windows":
        print("[警告] 当前系统不是 Windows，打包的 exe 只能在 Windows 上运行。")
        print("       建议在 Windows 上执行打包操作。")

    try:
        import PyInstaller
        print(f"  PyInstaller: {PyInstaller.__version__} ✓")
    except ImportError:
        print("[错误] PyInstaller 未安装！请执行: pip install pyinstaller")
        sys.exit(1)

    if not os.path.isfile(ENTRY_POINT):
        print(f"[错误] 入口文件不存在: {ENTRY_POINT}")
        sys.exit(1)
    print(f"  入口文件: {ENTRY_POINT} ✓")

    for f in EXTERNAL_FILES:
        if not os.path.isfile(f):
            print(f"  [警告] 外部文件不存在: {f}")
        else:
            print(f"  外部文件: {f} ✓")

    for d in EXTERNAL_DIRS:
        if not os.path.isdir(d):
            print(f"  [警告] 外部目录不存在: {d}/")
        else:
            print(f"  外部目录: {d}/ ✓")


def build():
    """执行 PyInstaller 打包"""
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm",       # 覆盖已有的输出目录
        "--clean",           # 清理临时文件
        "--onedir",          # 目录模式（比单文件模式启动更快）
        "--console",         # 保留控制台窗口（需要看打印日志）
        f"--name={APP_NAME}",
    ]

    # 图标
    if ICON_FILE and os.path.isfile(ICON_FILE):
        cmd.append(f"--icon={ICON_FILE}")
        print(f"  PyInstaller 图标参数: {ICON_FILE}")
    else:
        print("  [警告] 图标文件不存在，exe 将使用默认图标")

    # 隐式导入
    for mod in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", mod])

    # 排除模块
    for mod in EXCLUDES:
        cmd.extend(["--exclude-module", mod])

    # DLL 文件打包到 _internal/logic/
    for src, dest in DLL_FILES:
        if os.path.isfile(src):
            cmd.extend([f"--add-binary={src}{os.pathsep}{dest}"])

    # 入口文件
    cmd.append(ENTRY_POINT)

    run_cmd(cmd, "PyInstaller 打包")


def post_process():
    """后处理：复制外部资源到输出目录"""
    dist_dir = os.path.join("dist", APP_NAME)

    if not os.path.isdir(dist_dir):
        print(f"[错误] 输出目录不存在: {dist_dir}/")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  后处理: 复制外部资源到 {dist_dir}/")
    print(f"{'='*60}")

    # 复制外部文件
    for f in EXTERNAL_FILES:
        if os.path.isfile(f):
            copy_file(f, dist_dir)

    # 复制外部目录
    for d in EXTERNAL_DIRS:
        if os.path.isdir(d):
            copy_dir(d, os.path.join(dist_dir, d))

    # 创建运行时目录
    for d in RUNTIME_DIRS:
        target = os.path.join(dist_dir, d)
        os.makedirs(target, exist_ok=True)
        print(f"  创建目录: {target}/")

    # 复制 DLL 到 _internal/logic/（PyInstaller 的 --add-binary 可能放在根目录）
    internal_logic = os.path.join(dist_dir, "_internal", "logic")
    os.makedirs(internal_logic, exist_ok=True)
    for src, _ in DLL_FILES:
        if os.path.isfile(src):
            # 复制到 _internal/logic/ 确保 __file__ 相对路径能找到
            copy_file(src, internal_logic)
            # 也复制到 exe 同级目录（备用）
            copy_file(src, dist_dir)


def print_summary():
    """打印构建摘要"""
    dist_dir = os.path.join("dist", APP_NAME)

    # 计算总大小
    total_size = 0
    file_count = 0
    for root, dirs, files in os.walk(dist_dir):
        for f in files:
            fp = os.path.join(root, f)
            total_size += os.path.getsize(fp)
            file_count += 1

    total_mb = total_size / (1024 * 1024)

    print(f"""
{'='*60}
  构建完成!
{'='*60}

  输出目录: {os.path.abspath(dist_dir)}
  文件数量: {file_count}
  总大小:   {total_mb:.1f} MB

  使用方法:
    1. 将 {dist_dir}/ 整个文件夹复制到目标电脑
    2. 确保 models/ 目录下有 AI 模型文件
    3. 根据需要修改 config.ini
    4. 双击 {APP_NAME}.exe 运行

  注意:
    - 目标电脑不需要安装 Python
    - 目标电脑需要安装 Visual C++ Redistributable (通常已有)
    - 如果使用 ONNX GPU 加速，目标电脑需要安装 DirectML 运行时
    - 首次运行可能需要 Windows Defender 放行
""")


def main():
    if "--clean" in sys.argv:
        clean()
        if len(sys.argv) == 2:  # 只有 --clean 参数
            return

    check_environment()
    build()
    post_process()
    print_summary()


if __name__ == "__main__":
    main()
