"""
CH9329 串口转 HID 键鼠芯片驱动
通过串口发送 CH9329 协议指令，实现硬件级鼠标/键盘模拟

协议格式:
  HEAD(0x57 0xAB) + ADDR(0x00) + CMD + LEN + DATA + SUM(校验和)

CMD:
  0x04 = 鼠标绝对坐标移动
  0x05 = 鼠标相对坐标移动

用法:
  from logic.ch9329 import ch9329
  ch9329.move(dx, dy)       # 相对移动
  ch9329.press()             # 按下左键
  ch9329.release()           # 释放左键
"""
import os
import time
import random
import threading
import serial
import serial.tools.list_ports
from itertools import zip_longest

from logic.config_watcher import cfg
from logic.logger import logger


# CH9329 协议常量
HEAD = b'\x57\xAB'       # 帧头
ADDR = b'\x00'           # 地址

# 鼠标命令
CMD_MOUSE_ABS = b'\x04'  # 绝对坐标移动
CMD_MOUSE_REL = b'\x05'  # 相对坐标移动

# 鼠标按键标志
BTN_NONE = 0x00
BTN_LEFT = 0x01
BTN_RIGHT = 0x02
BTN_MIDDLE = 0x04


def _checksum(*parts: bytes) -> int:
    """计算校验和: 所有字节之和 % 256"""
    total = 0
    for part in parts:
        for b in part:
            total += b
    return total % 256


def _build_packet(cmd: bytes, data: bytes) -> bytes:
    """构建 CH9329 数据包"""
    length = bytes([len(data)])
    cs = _checksum(HEAD, ADDR, cmd, length, data)
    return HEAD + ADDR + cmd + length + data + bytes([cs])


class CH9329Mouse:
    """CH9329 硬件鼠标模拟器"""

    def __init__(self):
        self.serial_port = serial.Serial()
        self.serial_port.baudrate = cfg.ch9329_baudrate
        self.serial_port.timeout = 0
        self.serial_port.write_timeout = 0
        self._write_lock = threading.Lock()

        # 屏幕分辨率（绝对坐标模式需要）
        self.screen_width, self.screen_height = self._get_screen_resolution()
        logger.info(f"[CH9329] 屏幕分辨率: {self.screen_width}x{self.screen_height}")

        # 自动检测或手动指定串口
        if cfg.ch9329_port == 'auto':
            port = self._detect_port()
            if port:
                self.serial_port.port = port
            else:
                logger.error("[CH9329] 未自动检测到 CH9329 设备，请在 config.ini 中手动设置 ch9329_port")
                return
        else:
            self.serial_port.port = cfg.ch9329_port

        try:
            self.serial_port.open()
            logger.info(f"[CH9329] 已连接! 端口: {self.serial_port.port}, 波特率: {self.serial_port.baudrate}")
        except Exception as e:
            logger.error(f"[CH9329] 连接失败: {e}")

    def is_connected(self) -> bool:
        """检查串口是否已连接"""
        return self.serial_port.is_open

    def move(self, x: int, y: int) -> None:
        """
        相对移动鼠标（增量模式）
        x: 水平偏移（正=右，负=左）
        y: 垂直偏移（正=下，负=上）

        CH9329 相对移动协议:
          DATA[0] = 0x01 (鼠标模式)
          DATA[1] = 按键标志
          DATA[2] = X 偏移 (有符号 int8)
          DATA[3] = Y 偏移 (有符号 int8, CH9329 正=下，与屏幕坐标一致)
          DATA[4] = 滚轮
        """
        if not self.is_connected():
            return

        # 拆分为多次发送（单次最大 ±127）
        x_parts = self._split_value(x)
        y_parts = self._split_value(y)

        for dx, dy in zip_longest(x_parts, y_parts, fillvalue=0):
            self._send_relative(dx, dy, BTN_NONE)

    def _send_relative(self, x: int, y: int, button: int = BTN_NONE) -> None:
        """发送相对移动指令"""
        # CH9329 的 Y 轴方向: 正=下, 负=上，与屏幕坐标方向一致，无需取反
        data = bytearray([
            0x01,                          # 鼠标模式
            button,                        # 按键标志
            x & 0xFF,                      # X 偏移 (有符号, 截断到 8 位)
            y & 0xFF,                      # Y 偏移 (有符号)
            0x00,                          # 滚轮
        ])

        packet = _build_packet(CMD_MOUSE_REL, bytes(data))
        try:
            with self._write_lock:
                self.serial_port.write(packet)
                self.serial_port.reset_input_buffer()  # 丢弃 CH9329 应答帧，防止缓冲区堆积
                time.sleep(random.uniform(0.001, 0.002))  # 1~2ms 间隔，防止 CH9329 缓冲区溢出
        except Exception as e:
            logger.warning(f"[CH9329] 串口写入失败: {e}")

    def send_absolute(self, x: int, y: int, button: int = BTN_NONE) -> None:
        """
        绝对坐标移动（屏幕上的绝对位置）
        x, y: 屏幕像素坐标
        """
        if not self.is_connected():
            return

        # 转换为 CH9329 的 0-4096 范围
        abs_x = (4096 * x) // max(self.screen_width, 1)
        abs_y = (4096 * y) // max(self.screen_height, 1)

        data = bytearray([
            0x02,                                      # 绝对坐标模式
            button,                                    # 按键标志
            abs_x & 0xFF, (abs_x >> 8) & 0xFF,        # X 坐标 (16 位小端)
            abs_y & 0xFF, (abs_y >> 8) & 0xFF,        # Y 坐标 (16 位小端)
            0x00,                                      # 滚轮
        ])

        packet = _build_packet(CMD_MOUSE_ABS, bytes(data))
        try:
            with self._write_lock:
                self.serial_port.write(packet)
                self.serial_port.reset_input_buffer()
                time.sleep(random.uniform(0.001, 0.002))
        except Exception as e:
            logger.warning(f"[CH9329] 串口写入失败: {e}")

    def press(self) -> None:
        """按下鼠标左键"""
        if not self.is_connected():
            return
        self._send_relative(0, 0, BTN_LEFT)

    def release(self) -> None:
        """释放鼠标左键"""
        if not self.is_connected():
            return
        self._send_relative(0, 0, BTN_NONE)

    def click(self) -> None:
        """点击鼠标左键（按下 + 释放）"""
        self.press()
        import time
        time.sleep(0.05)
        self.release()

    def close(self) -> None:
        """关闭串口"""
        if self.serial_port.is_open:
            self.serial_port.close()
            logger.info("[CH9329] 串口已关闭")

    def __del__(self):
        self.close()

    @staticmethod
    def _split_value(value: int) -> list[int]:
        """将大偏移量拆分为多个 ±127 以内的值"""
        if value == 0:
            return [0]

        parts = []
        sign = -1 if value < 0 else 1
        remaining = abs(value)

        while remaining > 127:
            parts.append(sign * 127)
            remaining -= 127

        if remaining > 0:
            parts.append(sign * remaining)

        return parts

    @staticmethod
    def _detect_port() -> str | None:
        """自动检测 CH9329 设备所在的串口"""
        ports = serial.tools.list_ports.comports()

        # CH9329 的 USB VID:PID 通常为 1A86:7523 (CH340) 或 0416:5710
        CH9329_KEYWORDS = ['ch340', 'ch9329', '1a86', 'wch', 'usb-serial', 'cp2102']

        for port in ports:
            desc = (port.description or '').lower()
            hwid = (port.hwid or '').lower()
            for keyword in CH9329_KEYWORDS:
                if keyword in desc or keyword in hwid:
                    logger.info(f"[CH9329] 自动检测到设备: {port.device} ({port.description})")
                    return port.device

        # 如果关键字匹配不到，返回第一个可用串口
        if ports:
            logger.info(f"[CH9329] 未匹配到已知设备，使用第一个串口: {ports[0].device}")
            return ports[0].device

        return None

    @staticmethod
    def _get_screen_resolution() -> tuple[int, int]:
        """获取主显示器分辨率，支持 auto 自动检测和手动指定"""
        # 解析配置值
        width_str = str(cfg.ch9329_screen_width).strip().lower()
        height_str = str(cfg.ch9329_screen_height).strip().lower()

        if width_str == 'auto' or height_str == 'auto':
            # 自动检测主显示器分辨率
            try:
                from screeninfo import get_monitors
                monitors = list(get_monitors())
                primary = next((m for m in monitors if m.is_primary), monitors[0])
                logger.info(f"[CH9329] 自动检测到主显示器: {primary.name} ({primary.width}x{primary.height})")
                return primary.width, primary.height
            except Exception as e:
                logger.warning(f"[CH9329] 自动检测分辨率失败: {e}，使用默认值 1920x1080")
                return 1920, 1080
        else:
            # 手动指定
            try:
                return int(width_str), int(height_str)
            except ValueError:
                logger.warning(f"[CH9329] 分辨率配置无效: {width_str}x{height_str}，使用默认值 1920x1080")
                return 1920, 1080


# ──────────────────── 全局实例（延迟初始化） ────────────────────

_ch9329_instance: CH9329Mouse | None = None


def _get_instance() -> CH9329Mouse:
    global _ch9329_instance
    if _ch9329_instance is None:
        _ch9329_instance = CH9329Mouse()
    return _ch9329_instance


class CH9329Proxy:
    """延迟初始化代理，避免模块导入时就打开串口"""

    @property
    def _inst(self) -> CH9329Mouse:
        return _get_instance()

    def move(self, x: int, y: int) -> None:
        self._inst.move(x, y)

    def send_absolute(self, x: int, y: int) -> None:
        self._inst.send_absolute(x, y)

    def press(self) -> None:
        self._inst.press()

    def release(self) -> None:
        self._inst.release()

    def click(self) -> None:
        self._inst.click()

    def close(self) -> None:
        self._inst.close()

    def is_connected(self) -> bool:
        return self._inst.is_connected()


ch9329 = CH9329Proxy()
