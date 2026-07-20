"""
Raw Input 键盘监听器
使用 Windows Raw Input API 直接从设备层获取键盘事件
绕过 Vanguard 对 GetAsyncKeyState / 低级钩子的拦截

调试开关: config.ini → [Debug window] → log_debug_raw_input = True
          运行时切换: from logic.raw_input_listener import toggle_debug; toggle_debug()
"""
import ctypes
import ctypes.wintypes as wt
import threading

from logic.logger import logger

# ──────────────────── Win32 常量 ────────────────────

WM_INPUT = 0x00FF
WM_DESTROY = 0x0002

RIM_TYPEMOUSE = 0
RIM_TYPEKEYBOARD = 1

RIDEV_INPUTSINK = 0x00000100

RI_KEY_BREAK = 1

# 鼠标按键标志
RI_MOUSE_LEFT_BUTTON_DOWN = 0x0001
RI_MOUSE_LEFT_BUTTON_UP = 0x0002
RI_MOUSE_RIGHT_BUTTON_DOWN = 0x0004
RI_MOUSE_RIGHT_BUTTON_UP = 0x0008
RI_MOUSE_MIDDLE_BUTTON_DOWN = 0x0010
RI_MOUSE_MIDDLE_BUTTON_UP = 0x0020

# ──────────────────── 调试开关 ────────────────────

_debug_enabled = True  # 默认开启

def toggle_debug(enabled: bool = None):
    """切换调试日志开关。不传参数则取反"""
    global _debug_enabled
    if enabled is None:
        _debug_enabled = not _debug_enabled
    else:
        _debug_enabled = bool(enabled)
    state = "开启" if _debug_enabled else "关闭"
    print(f"[RawInput] 调试日志已{state}")

def _dbg(msg: str):
    """仅在调试开启时打印"""
    if _debug_enabled:
        print(msg)

# ──────────────────── VK 映射 ────────────────────

VK_MAP = {
    "Backspace": 0x08, "Tab": 0x09, "Enter": 0x0D,
    "LeftShift": 0xA0, "RightShift": 0xA1,
    "LeftControl": 0xA2, "RightControl": 0xA3,
    "LeftAlt": 0xA4, "RightAlt": 0xA5,
    "Escape": 0x1B, "Space": 0x20,
    "PageUp": 0x21, "PageDown": 0x22, "End": 0x23, "Home": 0x24,
    "LeftArrow": 0x25, "UpArrow": 0x26, "RightArrow": 0x27, "DownArrow": 0x28,
    "Ins": 0x2D, "Delete": 0x2E,
}
for i in range(1, 25):
    VK_MAP[f"F{i}"] = 0x6F + i
for i in range(10):
    VK_MAP[f"Key{i}"] = 0x30 + i
for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    VK_MAP[c] = ord(c)

VK_TO_NAMES: dict[int, set[str]] = {}
for name, vk in VK_MAP.items():
    VK_TO_NAMES.setdefault(vk, set()).add(name)

# ──────────────────── Win32 结构体 ────────────────────

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", wt.USHORT),
        ("usUsage", wt.USHORT),
        ("dwFlags", wt.DWORD),
        ("hwndTarget", wt.HWND),
    ]

class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", wt.DWORD),
        ("dwSize", wt.DWORD),
        ("hDevice", wt.HANDLE),
        ("wParam", wt.WPARAM),
    ]

class RAWKEYBOARD(ctypes.Structure):
    _fields_ = [
        ("MakeCode", wt.USHORT),
        ("Flags", wt.USHORT),
        ("Reserved", wt.USHORT),
        ("VKey", wt.USHORT),
        ("Message", wt.UINT),
        ("ExtraInformation", wt.ULONG),
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", wt.USHORT),
        ("_padding", wt.USHORT),       # C 的 union{ULONG; struct{USHORT,USHORT}} 需要 4 字节对齐
        ("usButtonFlags", wt.USHORT),  # offset 4 — 与 Windows SDK 一致
        ("usButtonData", wt.USHORT),   # offset 6
        ("ulRawButtons", wt.ULONG),    # offset 8
        ("lLastX", wt.LONG),           # offset 12
        ("lLastY", wt.LONG),           # offset 16
        ("ulExtraInformation", wt.ULONG),  # offset 20
    ]

class RAWINPUTUNION(ctypes.Union):
    _fields_ = [
        ("keyboard", RAWKEYBOARD),
        ("mouse", RAWMOUSE),
    ]

class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("union", RAWINPUTUNION)]

WNDPROC = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, wt.UINT, ctypes.c_void_p, ctypes.c_void_p)

class WNDCLASS(ctypes.Structure):
    _fields_ = [
        ("style", wt.UINT),
        ("lpfnWndProc", WNDPROC),
        ("cbClsExtra", ctypes.c_int),
        ("cbWndExtra", ctypes.c_int),
        ("hInstance", wt.HINSTANCE),
        ("hIcon", wt.HICON),
        ("hCursor", wt.HANDLE),
        ("hbrBackground", wt.HBRUSH),
        ("lpszMenuName", wt.LPCWSTR),
        ("lpszClassName", wt.LPCWSTR),
    ]


# ──────────────────── 键盘监听器 ────────────────────

class RawInputListener:
    """
    基于 Windows Raw Input API 的键盘监听器。
    仅监听键盘事件，不监听鼠标。
    """

    def __init__(self):
        self._pressed_keys: set[str] = set()
        self._mouse_buttons: set[str] = set()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._hwnd: int | None = None
        self._running = False
        self._ready = threading.Event()
        self._header_size = ctypes.sizeof(RAWINPUTHEADER)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="RawInput")
        self._thread.start()
        self._ready.wait(timeout=5)
        if not self._ready.is_set():
            logger.warning("[RawInput] 键盘监听器启动超时")

    def stop(self):
        self._running = False
        if self._hwnd:
            ctypes.windll.user32.PostMessageW(self._hwnd, WM_DESTROY, 0, 0)
        if self._thread:
            self._thread.join(timeout=3)

    def is_key_pressed(self, key_name: str) -> bool:
        """检查某个键是否处于按下状态"""
        if not key_name or key_name == "None":
            return False
        with self._lock:
            return key_name in self._pressed_keys

    def get_pressed_keys(self) -> set[str]:
        """获取当前所有按下的键名（调试用）"""
        with self._lock:
            return set(self._pressed_keys)

    def is_mouse_button_pressed(self, button_name: str) -> bool:
        """检查某个鼠标按键是否处于按下状态 (left / right / middle)"""
        if not button_name:
            return False
        with self._lock:
            return button_name in self._mouse_buttons

    def get_pressed_mouse_buttons(self) -> set[str]:
        """获取当前所有按下的鼠标键名（调试用）"""
        with self._lock:
            return set(self._mouse_buttons)

    def _run_loop(self):
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        hinstance = kernel32.GetModuleHandleW(None)

        user32.DefWindowProcW.argtypes = [ctypes.c_void_p, wt.UINT, ctypes.c_void_p, ctypes.c_void_p]
        user32.DefWindowProcW.restype = ctypes.c_long

        self._buf = (ctypes.c_byte * 512)()
        self._size = wt.UINT()

        @WNDPROC
        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == WM_INPUT:
                self._handle_raw_input(lparam)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        class_name = "SunoneRawInputKB"
        wc = WNDCLASS()
        wc.lpfnWndProc = wnd_proc
        wc.hInstance = hinstance
        wc.lpszClassName = class_name
        user32.RegisterClassW(ctypes.byref(wc))

        HWND_MESSAGE = ctypes.c_void_p(-3)
        self._hwnd = user32.CreateWindowExW(
            0, class_name, "SunoneRawInputKB",
            0, 0, 0, 0, 0,
            HWND_MESSAGE,
            None, hinstance, None,
        )

        if not self._hwnd:
            logger.error("[RawInput] 创建消息窗口失败")
            return

        # 注册键盘 + 鼠标设备
        devices = (RAWINPUTDEVICE * 2)()

        # 键盘
        devices[0].usUsagePage = 0x01
        devices[0].usUsage = 0x06        # 键盘
        devices[0].dwFlags = RIDEV_INPUTSINK
        devices[0].hwndTarget = self._hwnd

        # 鼠标
        devices[1].usUsagePage = 0x01
        devices[1].usUsage = 0x02        # 鼠标
        devices[1].dwFlags = RIDEV_INPUTSINK
        devices[1].hwndTarget = self._hwnd

        if not user32.RegisterRawInputDevices(
            ctypes.byref(devices), 2, ctypes.sizeof(RAWINPUTDEVICE)
        ):
            error_code = kernel32.GetLastError()
            logger.error(f"[RawInput] 注册键盘+鼠标设备失败 (错误码: {error_code})")
            return

        logger.info("[RawInput] ✅ 键盘+鼠标监听器已启动（Raw Input 模式）")
        self._ready.set()

        msg = wt.MSG()
        while self._running:
            ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if ret == 0 or ret == -1:
                break
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

    def _handle_raw_input(self, lparam):
        user32 = ctypes.windll.user32

        self._size.value = 0
        user32.GetRawInputData(lparam, 0x10000003, None, ctypes.byref(self._size), self._header_size)

        if self._size.value > len(self._buf):
            return

        result = user32.GetRawInputData(
            lparam, 0x10000003, self._buf, ctypes.byref(self._size), self._header_size
        )
        if result == -1:
            return

        raw = ctypes.cast(self._buf, ctypes.POINTER(RAWINPUT)).contents

        if raw.header.dwType == RIM_TYPEKEYBOARD:
            self._handle_keyboard(raw.union.keyboard)
        elif raw.header.dwType == RIM_TYPEMOUSE:
            self._handle_mouse(raw.union.mouse)

    def _handle_keyboard(self, kb: RAWKEYBOARD):
        vk = kb.VKey
        is_break = (kb.Flags & RI_KEY_BREAK) != 0

        # 区分左右修饰键（通过扩展键标志 E0）
        e0 = (kb.Flags & 0x02) != 0
        if vk == 0x10:       # VK_SHIFT
            resolved_name = "RightShift" if e0 else "LeftShift"
        elif vk == 0x11:     # VK_CONTROL
            resolved_name = "RightControl" if e0 else "LeftControl"
        elif vk == 0x12:     # VK_MENU (Alt)
            resolved_name = "RightAlt" if e0 else "LeftAlt"
        else:
            names = VK_TO_NAMES.get(vk, set())
            if names:
                resolved_name = sorted(names)[0]
            else:
                resolved_name = f"VK_{hex(vk)}"

        action = "松开" if is_break else "按下"
        _dbg(f"[RawInput] 键盘 {action}: {resolved_name:<20s} vk={hex(vk)}")

        with self._lock:
            if is_break:
                self._pressed_keys.discard(resolved_name)
            else:
                self._pressed_keys.add(resolved_name)

    def _handle_mouse(self, mouse: RAWMOUSE):
        flags = mouse.usButtonFlags

        with self._lock:
            if flags & RI_MOUSE_LEFT_BUTTON_DOWN:
                self._mouse_buttons.add("left")
                _dbg("[RawInput] 鼠标 按下: left")
            if flags & RI_MOUSE_LEFT_BUTTON_UP:
                self._mouse_buttons.discard("left")
                _dbg("[RawInput] 鼠标 松开: left")
            if flags & RI_MOUSE_RIGHT_BUTTON_DOWN:
                self._mouse_buttons.add("right")
                _dbg("[RawInput] 鼠标 按下: right")
            if flags & RI_MOUSE_RIGHT_BUTTON_UP:
                self._mouse_buttons.discard("right")
                _dbg("[RawInput] 鼠标 松开: right")
            if flags & RI_MOUSE_MIDDLE_BUTTON_DOWN:
                self._mouse_buttons.add("middle")
                _dbg("[RawInput] 鼠标 按下: middle")
            if flags & RI_MOUSE_MIDDLE_BUTTON_UP:
                self._mouse_buttons.discard("middle")
                _dbg("[RawInput] 鼠标 松开: middle")


# ──────────────────── 全局实例 ────────────────────
raw_listener = RawInputListener()

# ──────────────────── 从 config.ini 初始化调试开关 ────────────────────
def _init_debug_from_config():
    try:
        from logic.config_watcher import cfg
        toggle_debug(getattr(cfg, 'log_debug_raw_input', True))
    except Exception:
        pass

_init_debug_from_config()
