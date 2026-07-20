"""
跨平台输入抽象层
  Windows: Raw Input 监听按键 + win32api 注入鼠标
  Linux:   pynput 监听 + 注入
"""
import platform
import threading

from logic.logger import logger

IS_WINDOWS = platform.system() == "Windows"
IS_LINUX = platform.system() == "Linux"

# ──────────────────── 键名标准化映射 ────────────────────
# 将用户配置中的各种写法统一为 Raw Input / pynput 使用的标准名称

_MOUSE_BTN_ALIASES = {
    "left": "LeftMouseButton",
    "right": "RightMouseButton",
    "middle": "MiddleMouseButton",
}

# 小写形式 → 标准键名（覆盖所有 buttons.py 中定义的键名）
_KEY_NAME_MAP = {
    # 修饰键
    "leftshift": "LeftShift", "rightshift": "RightShift",
    "leftcontrol": "LeftControl", "rightcontrol": "RightControl",
    "leftalt": "LeftAlt", "rightalt": "RightAlt",
    "leftwindowskey": "LeftWindowsKey", "rightwindowskey": "RightWindowsKey",
    # 特殊键
    "backspace": "Backspace", "tab": "Tab", "enter": "Enter",
    "escape": "Escape", "space": "Space", "clear": "Clear",
    "pause": "Pause", "capslock": "CapsLock",
    "pageup": "PageUp", "pagedown": "PageDown",
    "end": "End", "home": "Home", "ins": "Ins", "delete": "Delete",
    "help": "Help", "print": "Print", "execute": "Execute",
    "printscreen": "PrintScreen",
    # 方向键
    "leftarrow": "LeftArrow", "uparrow": "UpArrow",
    "rightarrow": "RightArrow", "downarrow": "DownArrow",
    # 小键盘
    "numlock": "NumLock", "scrolllock": "ScrollLock",
    "multiply": "Multiply", "add": "Add", "separator": "Separator",
    "subtract": "Subtract", "decimal": "Decimal", "divide": "Divide",
    # 浏览器 / 媒体键
    "browserback": "BrowserBack", "browserrefresh": "BrowserRefresh",
    "browserstop": "BrowserStop", "browsersearch": "BrowserSearch",
    "browserfavorites": "BrowserFavorites", "browserhome": "BrowserHome",
    "volumemute": "VolumeMute", "volumedown": "VolumeDown", "volumeup": "VolumeUp",
    "nexttrack": "NextTrack", "previoustrack": "PreviousTrack",
    "stopmedia": "StopMedia", "playmedia": "PlayMedia",
    "startmailkey": "StartMailKey", "selectmedia": "SelectMedia",
    "startapplication1": "StartApplication1", "startapplication2": "StartApplication2",
    "application": "Application", "sleep": "Sleep", "select": "Select",
    # 鼠标按键（额外的 X1/X2 按钮）
    "leftmousebutton": "LeftMouseButton",
    "rightmousebutton": "RightMouseButton",
    "middlemousebutton": "MiddleMouseButton",
    "x1mousebutton": "X1MouseButton",
    "x2mousebutton": "X2MouseButton",
    "controlbreak": "ControlBreak",
}
# 数字键 Key0-Key9
for _i in range(10):
    _KEY_NAME_MAP[f"key{_i}"] = f"Key{_i}"
    _KEY_NAME_MAP[f"numpadkey{_i}"] = f"NumpadKey{_i}"
# 功能键 F1-F24
for _i in range(1, 25):
    _KEY_NAME_MAP[f"f{_i}"] = f"F{_i}"
# 字母键 A-Z
for _c in "abcdefghijklmnopqrstuvwxyz":
    _KEY_NAME_MAP[_c] = _c.upper()


def _normalize_key_name(key_name: str) -> str:
    """将用户配置的键名标准化为 PascalCase 格式。

    支持的写法示例:
      'left' / 'LeftMouseButton'  → 'LeftMouseButton'
      'leftshift' / 'LeftShift'   → 'LeftShift'
      'f1' / 'F1'                 → 'F1'
      'a' / 'A'                   → 'A'
      'space' / 'Space'           → 'Space'
    """
    if not key_name:
        return key_name

    # 鼠标按钮快捷别名
    lower = key_name.strip().lower()
    if lower in _MOUSE_BTN_ALIASES:
        return _MOUSE_BTN_ALIASES[lower]

    # 查标准映射表
    canonical = _KEY_NAME_MAP.get(lower)
    if canonical:
        return canonical

    # 未匹配到 → 首字母大写尝试（兼容 'Space'、'Enter' 等直接传入）
    return key_name[0].upper() + key_name[1:]


class InputBackend:
    def __init__(self):
        self.mode = "none"
        self.available = False
        self.error = None
        self._pressed = set()
        self._lock = threading.Lock()
        self._mouse_controller = None
        self._mouse_button = None

        if IS_WINDOWS:
            self._init_win32()
        else:
            self._init_pynput()

    def _init_win32(self):
        try:
            import win32api
            import win32con

            self.win32api = win32api
            self.win32con = win32con
            self.mode = "win32"
            self.available = True

            # 启动 Raw Input 作为唯一按键监听源
            self._raw_listener = None
            self._init_raw_listener()

        except Exception as exc:
            self.error = str(exc)
            logger.warning(f"[输入] Win32 输入后端不可用: {exc}")

    def _init_raw_listener(self):
        """启动 Raw Input 监听器作为唯一按键输入源"""
        try:
            from logic.raw_input_listener import raw_listener
            raw_listener.start()
            self._raw_listener = raw_listener
        except Exception as exc:
            logger.warning(f"[输入] Raw Input 监听器不可用: {exc}")

    def _init_pynput(self):
        """Linux 平台使用 pynput 监听 + 注入"""
        try:
            from pynput import keyboard, mouse

            self.keyboard = keyboard
            self.mouse = mouse
            self._mouse_controller = mouse.Controller()
            self._mouse_button = mouse.Button
            self._keyboard_listener = keyboard.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
            self._mouse_listener = mouse.Listener(on_click=self._on_click)
            self._keyboard_listener.daemon = True
            self._mouse_listener.daemon = True
            self._keyboard_listener.start()
            self._mouse_listener.start()
            self.mode = "pynput"
            self.available = True
        except Exception as exc:
            self.error = str(exc)
            logger.warning(f"[输入] pynput 后端不可用: {exc}")

    def _on_key_press(self, key):
        name = self._normalize_key(key)
        if name:
            with self._lock:
                self._pressed.add(name)

    def _on_key_release(self, key):
        name = self._normalize_key(key)
        if name:
            with self._lock:
                self._pressed.discard(name)

    def _on_click(self, x, y, button, pressed):
        name = self._normalize_mouse_button(button)
        if not name:
            return
        with self._lock:
            if pressed:
                self._pressed.add(name)
            else:
                self._pressed.discard(name)

    # ──────────────────── 按键监听（统一入口） ────────────────────

    def is_pressed(self, key_name):
        """
        检查某个键是否处于按下状态。
        Windows: 通过 Raw Input 监听（绕过 Vanguard 对 GetAsyncKeyState 的拦截）
                 同时支持键盘按键和鼠标按键 (left/right/middle)
        Linux:   通过 pynput 监听
        """
        if not key_name or key_name == "None":
            return False

        key_name = _normalize_key_name(key_name)

        if self.mode == "win32":
            if self._raw_listener:
                # 鼠标按键名称直接查鼠标状态
                if key_name in ("LeftMouseButton", "RightMouseButton", "MiddleMouseButton"):
                    btn_name = {"LeftMouseButton": "left", "RightMouseButton": "right", "MiddleMouseButton": "middle"}[key_name]
                    return self._raw_listener.is_mouse_button_pressed(btn_name)
                return self._raw_listener.is_key_pressed(key_name)
            return False

        # Linux 使用 pynput
        with self._lock:
            return key_name in self._pressed

    # ──────────────────── 鼠标注入（移动/点击） ────────────────────

    def move_mouse(self, x, y):
        """注入鼠标移动事件（非监听，属于输出）"""
        if self.mode == "win32":
            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_MOVE, int(x), int(y), 0, 0)
            return True
        if self._mouse_controller is not None:
            self._mouse_controller.move(int(x), int(y))
            return True
        return False

    def left_down(self):
        """注入鼠标左键按下"""
        if self.mode == "win32":
            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            return True
        if self._mouse_controller is not None and self._mouse_button is not None:
            self._mouse_controller.press(self._mouse_button.left)
            return True
        return False

    def left_up(self):
        """注入鼠标左键释放"""
        if self.mode == "win32":
            self.win32api.mouse_event(self.win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            return True
        if self._mouse_controller is not None and self._mouse_button is not None:
            self._mouse_controller.release(self._mouse_button.left)
            return True
        return False

    # ──────────────────── 键名标准化 ────────────────────

    def _normalize_mouse_button(self, button):
        if not hasattr(button, "name"):
            return None
        return {
            "left": "LeftMouseButton",
            "right": "RightMouseButton",
            "middle": "MiddleMouseButton",
        }.get(button.name)

    def _normalize_key(self, key):
        char = getattr(key, "char", None)
        if char:
            if char.isalpha():
                return char.upper()
            if char.isdigit():
                return f"Key{char}"
            if char == " ":
                return "Space"

        name = getattr(key, "name", None)
        if not name:
            return None

        if name.startswith("f") and name[1:].isdigit():
            return name.upper()

        return {
            "esc": "Escape",
            "space": "Space",
            "enter": "Enter",
            "tab": "Tab",
            "backspace": "Backspace",
            "delete": "Delete",
            "insert": "Ins",
            "home": "Home",
            "end": "End",
            "page_up": "PageUp",
            "page_down": "PageDown",
            "up": "UpArrow",
            "down": "DownArrow",
            "left": "LeftArrow",
            "right": "RightArrow",
            "shift": "LeftShift",
            "shift_l": "LeftShift",
            "shift_r": "RightShift",
            "ctrl": "LeftControl",
            "ctrl_l": "LeftControl",
            "ctrl_r": "RightControl",
            "alt": "LeftAlt",
            "alt_l": "LeftAlt",
            "alt_r": "RightAlt",
            "caps_lock": "CapsLock",
            "num_lock": "NumLock",
            "scroll_lock": "ScrollLock",
        }.get(name)


input_backend = InputBackend()
