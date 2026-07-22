# Watermelon Aimbot

基于 ONNX Runtime 的 FPS 游戏人物识别，支持 DirectML GPU 加速，兼容 AMD / Intel / NVIDIA 任意 DX12 显卡。

---

## ✨ 功能特性

- **ONNX 推理引擎** — 使用 ONNX Runtime + DirectML，任意 DX12 GPU 均可加速
- **多截取后端** — BetterCam（DXGI，最低延迟）、MSS（跨平台截屏）、OBS 虚拟摄像头
- **ByteTrack 追踪** — 跨帧目标追踪，瞄准更稳定不跳目标
- **运动预测** — 根据目标速度/加速度预测未来位置，提前量补偿
- **目标锁定** — 基于 Tracker ID 锁定目标，开火时不跳到其他人物
- **硬件级输入** — 支持 CH9329 串口芯片 / Arduino
- **Raw Input 监听** — Windows Raw Input API 直接读取键盘鼠标
- **热重载配置** — 修改 `config.ini` 后按 F11 即时生效，无需重启

---

## 🎬 演示视频

下面是一个演示视频：

<video src="media/demo.mp4" controls width="100%"></video>
![演示视频](media/demo.gif)

如果视频无法播放，请[下载](media/demo.mp4)观看。

---

## 📋 系统要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Windows 10 / 11 64 位 |
| Python | **3.12.0**（[下载地址](https://www.python.org/downloads/release/python-3120/)） |
| GPU | 任意支持 DX12 的显卡（AMD / Intel / NVIDIA） |
| 运行时 | [Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe)（通常已预装） |

---

## 🚀 安装步骤

### 1. 克隆 / 下载项目

```powershell
cd C:\你的路径
git clone <仓库地址>
cd OBS_AI_FPS_Game
```

### 2. 创建 Python 3.12 虚拟环境

```powershell
python3.12 -m venv venv
venv\Scripts\activate
```

> ⚠️ 如果 `python3.12` 命令不可用，请确认 Python 3.12.0 已安装并加入 PATH。
> 安装时勾选 **"Add Python to PATH"**。

### 3. 安装依赖

```powershell
pip install -r requirements.txt
或者
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime-directml
pip install numpy pyserial opencv-python packaging ultralytics keyboard mss pynput supervision trackers hid pyusb psutil screeninfo pywin32 bettercam asyncio
```

### 4. 验证环境

```powershell
python check_env.py
```

### 5. 启动程序(注意以管理员权限运行cmd)

```powershell
python run.py
```

---

## 📦 pip 依赖清单

以下为 `requirements.txt` 中的全部依赖：

| 包名 | 用途 |
|------|------|
| `onnxruntime` | ONNX 推理引擎 |
| `onnxruntime-gpu` | ONNX GPU 加速（DirectML / CUDA） |
| `opencv-python` | 图像处理、NMS、绘制调试信息 |
| `numpy` | 数值计算 |
| `supervision` | 检测结果封装（Detections） |
| `trackers` | ByteTrack 目标追踪 |
| `torch` | PyTorch 张量运算（目标排序、距离计算） |
| `ultralytics` | YOLO 模型导出 |
| `mss` | 跨平台屏幕截取 |
| `bettercam` | DXGI 屏幕截取（Windows，最低延迟） |
| `screeninfo` | 获取显示器分辨率 |
| `pyserial` | 串口通信（CH9329 / Arduino） |
| `pywin32` | Windows API（Raw Input、鼠标注入） |
| `keyboard` | 键盘事件监听（备用） |
| `pynput` | 跨平台键鼠监听/注入（Linux 备用） |
| `hid` / `pyusb` | USB HID 设备通信 |
| `psutil` | 进程管理（Arduino 端口检测） |
| `packaging` | 版本号比较 |
| `cuda_python` | CUDA Python 绑定 |

---

## ⚙️ 配置说明

所有配置集中在 `config.ini`，修改后按 **F11** 热重载。关键配置项：

```ini
[Hotkeys]
hotkey_targeting = left,right,LeftControl  ; 瞄准激活键（支持鼠标按键）
hotkey_exit = F9              ; 退出程序
hotkey_pause = F10            ; 暂停瞄准
```

---

## 🗂️ 代码架构

```
cheats_FPS_Game/
├── run.py                  # 主入口 — ONNX 推理主循环
├── check_env.py            # 环境检查 — 验证依赖是否安装
├── config.ini              # 配置文件 — 所有参数集中管理
├── window_names.txt        # 窗口名称 — 用于窗口匹配
├── version                 # 版本文件
├── requirements.txt        # pip 依赖清单
│
├── logic/                  # 核心逻辑模块
│   ├── capture.py          # 屏幕截取 — BetterCam / MSS / OBS 三种后端
│   ├── frame_parser.py     # 目标解析 — 检测结果排序、选择最近目标、目标锁定
│   ├── mouse.py            # 瞄准计算 — 偏移量→角度→鼠标移动量、EMA平滑、速度控制
│   ├── shooting.py         # 射击控制 — 判断开火条件、触发 press/release
│   ├── ch9329.py           # CH9329 驱动 — 串口协议封装、鼠标相对/绝对移动
│   ├── arduino.py          # Arduino 驱动 — 串口通信、鼠标移动/点击
│   ├── platform.py         # 跨平台输入层 — 统一 Windows/Linux 键鼠监听和注入
│   ├── raw_input_listener.py  # Raw Input 监听器 — Windows 设备层键盘鼠标事件
│   ├── config_watcher.py   # 配置管理 — 读取 config.ini、热重载、类型转换
│   ├── hotkeys_watcher.py  # 热键监控 — 轮询热键状态、触发暂停/退出/重载
│   ├── visual.py           # 调试可视化 — OpenCV 窗口绘制、叠加层渲染
│   ├── overlay.py          # 透明叠加层 — Tkinter 全屏透明窗口（游戏画面上绘制）
│   ├── buttons.py          # 按键编码表 — 虚拟键码映射
│   ├── checks.py           # 启动检查 — 验证模型/环境/依赖
│   ├── logger.py           # 日志模块 — 统一日志格式
│   ├── model_classes.py    # 模型类别 — 定义 class ID 和名称
│   └── *.yaml              # 数据集和追踪器配置
```

### 主循环数据流

```
Capture 线程                主线程                        并行线程
─────────                  ────                        ────────
屏幕截取 ──→ frame_queue ──→ ONNX 推理 ──→ ByteTrack ──→ 目标选择
  (60FPS)                    (35ms)        (0.5ms)      (0.5ms)
                                                            │
                                    ┌───────────────────────┤
                                    ▼                       ▼
                              Shooting 线程           瞄准计算 + 鼠标移动
                              (press/release)        (EMA平滑 + CH9329串口)
                                    │                       │
                                    ▼                       ▼
                              Visuals 线程           CH9329 / Arduino
                              (调试窗口绘制)          (硬件级 HID 输出)
```

### 各模块职责详述

| 模块 | 职责 | 关键技术 |
|------|------|---------|
| `run.py` | 主入口，初始化 ONNX 检测器，驱动主循环 | ONNX Runtime, DirectML |
| `capture.py` | 独立线程按固定帧率截取屏幕区域 | DXGI Desktop Duplication, MSS, OpenCV |
| `frame_parser.py` | 解析检测结果，选择目标，实现 Tracker ID 锁定 | PyTorch 张量运算, supervision |
| `mouse.py` | 将像素偏移转换为鼠标移动量，动态平滑，速度控制 | FOV 换算, EMA 滤波, 运动预测 |
| `shooting.py` | 判断开火条件（bScope），异步触发 press/release | 队列通信, 多线程 |
| `ch9329.py` | CH9329 芯片串口协议驱动，支持相对/绝对移动和按键 | 串口通信, CH9329 协议 |
| `arduino.py` | Arduino 串口通信驱动 | 串口通信, 端口自动检测 |
| `platform.py` | 跨平台输入抽象层，统一键鼠监听和注入接口 | Raw Input (Win), pynput (Linux) |
| `raw_input_listener.py` | Windows Raw Input API 键盘鼠标监听 | Win32 API, WM_INPUT 消息 |
| `config_watcher.py` | 配置读取、类型转换、文件变更监控热重载 | configparser, mtime 轮询 |
| `hotkeys_watcher.py` | 热键状态轮询，触发暂停/退出/配置重载 | 线程轮询 |
| `visual.py` | OpenCV 调试窗口绘制（框、线、FPS、速度） | OpenCV imshow |
| `overlay.py` | Tkinter 全屏透明窗口，在游戏画面上叠加绘制 | Tkinter, win32gui 透明窗口 |

---

## 🙏 致谢

特别感谢微信小程序 **「西瓜去水印文案配音助手」** 开源的卡密校验系统。

<p align="center">
  <img src="media/小程序-下载码.jpg" alt="西瓜去水印文案配音助手小程序码" width="300">
</p>

为避免本项目被二次售卖或用于商业牟利，我们接入了该卡密校验登录机制。所有用户通过校验后即可**永久免费使用**全部功能，不收取任何费用。

| 项目 | 说明 |
|------|------|
| 来源 | 微信小程序「西瓜去水印文案配音助手」 |
| 用途 | API卡密生成校验登录，防止二次售卖 |
| 费用 | **永久免费**，不收取任何费用 |

> 💡 卡密校验仅用于验证合法用户身份，不收集任何个人隐私数据。

---

## ⚠️ 免责声明

### 🚨 重要警告

在使用本项目前，请仔细阅读并完全理解以下免责声明。继续使用即表示您接受所有条款和风险。

---

### 1. 项目性质与目的声明

本项目是一个**技术研究性质**的开源项目，主要用于以下目的：

- 编程技术学习和研究
- 计算机视觉与图像识别算法研究（YOLO / ONNX Runtime）
- 目标追踪技术探索（ByteTrack）
- 人机交互技术实验（Raw Input / HID 设备通信）
- 跨平台输入系统架构设计

**本项目并非为游戏作弊而设计，也不鼓励任何形式的游戏违规行为。**

---

### 2. 法律合规性警告

#### 2.1 遵守法律法规

用户在使用本项目时，必须严格遵守以下法律法规：

- 《中华人民共和国网络安全法》
- 《中华人民共和国著作权法》
- 《计算机软件保护条例》
- 相关国际法律法规
- 用户所在地适用的所有法律法规

#### 2.2 游戏服务条款遵守

> **重要提醒：** 使用本程序可能违反相关游戏的服务条款，包括但不限于：
>
> - 各 FPS 游戏（如 PUBG、CS2、Apex Legends、Valorant、Call of Duty 等）的用户协议
> - Steam / Epic Games / Riot Games 等平台用户协议
> - 其他相关游戏平台的使用条款
>
> 本项目适用于**大部分 FPS 类游戏**的技术研究场景，但不对任何特定游戏做合规性保证。

---

### 3. 使用风险提示

#### 3.1 账号风险评估

本项目采用硬件级输入方案（CH9329 串口芯片 / Arduino）结合 Raw Input API 监听，在技术层面绕过软件层检测，经过长期稳定性测试，目前未收到账号安全问题的反馈。

**风险提示：**

- 本项目无法保证与未来游戏版本的完全兼容性
- 游戏厂商可能随时更新检测机制，存在理论上的封号风险
- 如硬件级输入方案被游戏厂商认定为违规，则所有同类方案均可能面临同样风险
- 不同游戏的反作弊策略差异较大，本项目不对特定游戏的安全性做保证

#### 3.2 潜在法律风险

**当前状况：** 本项目基于公开技术栈（ONNX Runtime、DirectML、开源串口协议）开发，使用通用硬件接口。

**理论风险提示：**

- 如游戏厂商认定使用本程序违反服务条款，可能产生民事责任
- 极端情况下可能涉及知识产权争议
- 建议用户了解并遵守相关平台的使用政策

---

### 4. 用户责任声明

#### 4.1 用户义务

使用本项目的用户必须：

- 年满 18 周岁，具有完全民事行为能力
- 自行评估使用风险并**承担全部责任**
- 确保使用行为符合当地法律法规
- 不得用于任何商业用途
- 不得分发、销售或用于盈利目的

#### 4.2 禁止行为

严格禁止以下行为：

- 用于线上竞技游戏的排名赛、锦标赛等正式比赛
- 用于任何盈利性活动（代练、直播盈利等）
- 修改后用于恶意目的
- 用于攻击他人计算机系统

---

### 5. 开发者免责条款

#### 5.1 责任限制

开发者对以下情况**不承担任何责任**：

- 因使用本项目导致的账号封禁、游戏内处罚
- 程序使用造成的任何直接或间接损失
- 程序错误导致的数据丢失或设备损坏
- 用户违反法律法规所产生的后果
- 第三方使用本项目造成的任何损失

#### 5.2 无担保声明

本项目按 **"原样"** 提供，不提供任何明示或暗示的担保，包括但不限于：

- 适销性担保
- 特定用途适用性担保
- 不侵犯第三方权利担保
- 无病毒或错误担保

---

### 6. 知识产权声明

#### 6.1 使用许可

用户获得的是**有限使用许可**，仅限于：

- 个人学习和技术研究
- 非商业性的学术研究
- 符合开源协议的使用

---

### 7. 技术免责声明

#### 7.1 程序局限性

本项目存在以下技术局限性：

- 可能无法在所有系统环境下正常运行（依赖 DX12 GPU、特定串口驱动）
- 可能存在未发现的程序错误
- 不保证与所有游戏版本的兼容性
- ONNX 模型检测精度受训练数据集限制
- 功能可能随时变更或终止

#### 7.2 更新和维护

开发者没有义务：

- 提供程序更新和维护
- 提供技术支持和售后服务
- 保证程序的长期可用性

---

### 8. 隐私和数据安全

#### 8.1 数据收集

本项目**不收集、不存储、不传输**以下信息：

- 用户个人信息
- 游戏账号和密码
- 任何敏感数据

#### 8.2 安全警告

用户需要自行确保：

- 使用环境的安全性
- 防病毒软件的保护
- 系统漏洞的修补

---

### 9. 国际使用条款

#### 9.1 出口管制

用户需遵守所有适用的出口管制法律和法规，不得将本项目用于或被用于：

- 受制裁国家或地区
- 受限制的最终用途
- 受管制的最终用户

---

### 10. 免责声明的变更和更新

开发者保留随时修改本免责声明的权利，恕不另行通知。用户有责任定期查看并遵守最新版本的免责声明。

---

### 11. 管辖法律和争议解决

本免责声明受中华人民共和国法律管辖并据其解释。任何争议应首先通过友好协商解决，协商不成的，提交开发者所在地有管辖权的人民法院诉讼解决。

---

### 📝 最终确认条款

> **重要提示：** 继续使用本项目即表示用户已经：
>
> - ✅ 仔细阅读并完全理解本免责声明的所有内容
> - ✅ 接受所有条款和条件
> - ✅ 自愿承担使用本项目可能产生的一切风险和责任
> - ✅ 承诺遵守相关法律法规和游戏服务条款

> **技术说明：** 本项目采用硬件级输入方案（CH9329 / Arduino）结合 Raw Input API，如该方案被游戏厂商认定为违规，则所有同类硬件级输入方案均可能面临同样风险。
>
> **建议：** 如对任何条款有疑问，请咨询法律专业人士，并在完全理解风险前不要使用本项目。

---

> ⚠️ **风险提示：** 基于通用硬件接口开发，但无法保证与所有游戏未来版本的兼容性 ⚠️

