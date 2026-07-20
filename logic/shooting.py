import queue
import threading
import time

from logic.config_watcher import cfg
from logic.logger import logger
from logic.platform import input_backend

if cfg.arduino_move or cfg.arduino_shoot:
    from logic.arduino import arduino

class Shooting(threading.Thread):
    # 退避策略参数
    MAX_RETRIES = 5           # 连续失败最大次数，超过后停止重试
    BACKOFF_BASE = 0.1        # 基础退避时间（秒）
    BACKOFF_MULTIPLIER = 2.0  # 退避倍数（0.1 → 0.2 → 0.4 → 0.8 → 1.6）
    BACKOFF_MAX = 5.0         # 退避时间上限（秒）

    def __init__(self):
        super(Shooting, self).__init__()
        self.queue = queue.Queue(maxsize=1)
        self.daemon = True
        self.name = 'Shooting'
        self.button_pressed = False
        self.lock = threading.Lock()
        self._consecutive_errors = 0
        self._backoff_until = 0.0  # 退避截止时间戳

        self.start()

    def submit(self, bScope, shooting_state):
        try:
            self.queue.put_nowait((bScope, shooting_state))
        except queue.Full:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                pass
            self.queue.put_nowait((bScope, shooting_state))

    def run(self):
        while True:
            try:
                bScope, shooting_state = self.queue.get()

                # 退避中：丢弃指令，等待退避结束
                if self._consecutive_errors >= self.MAX_RETRIES:
                    if time.monotonic() < self._backoff_until:
                        continue
                    # 退避期满，重置计数器，恢复重试
                    logger.info("[射击] 退避期满，恢复射击线程")
                    self._consecutive_errors = 0

                self.shoot(bScope, shooting_state)
                # 成功执行，重置错误计数
                self._consecutive_errors = 0

            except Exception as e:
                self._consecutive_errors += 1
                if self._consecutive_errors < self.MAX_RETRIES:
                    backoff = min(
                        self.BACKOFF_BASE * (self.BACKOFF_MULTIPLIER ** (self._consecutive_errors - 1)),
                        self.BACKOFF_MAX,
                    )
                    self._backoff_until = time.monotonic() + backoff
                    logger.error(
                        "[射击] 射击线程异常 (%d/%d)，%.1f 秒后重试: %s",
                        self._consecutive_errors, self.MAX_RETRIES, backoff, e,
                    )
                else:
                    logger.error(
                        "[射击] 射击线程连续失败 %d 次，已暂停。"
                        "等待 %.1f 秒后自动恢复。错误: %s",
                        self._consecutive_errors, self.BACKOFF_MAX, e,
                    )
                    self._backoff_until = time.monotonic() + self.BACKOFF_MAX

    def shoot(self, bScope, shooting_state):
        with self.lock:
            should_press = False

            if cfg.mouse_auto_aim and bScope:
                should_press = True
            elif cfg.auto_shoot and cfg.triggerbot:
                should_press = bScope
            elif cfg.auto_shoot:
                should_press = shooting_state and bScope

            if should_press and not self.button_pressed:
                self._press()
                self.button_pressed = True
            elif not should_press and self.button_pressed:
                self._release()
                self.button_pressed = False

    def _press(self):
        if cfg.ch9329_shoot:
            from logic.ch9329 import ch9329
            ch9329.press()
        elif cfg.arduino_shoot:
            from logic.arduino import arduino
            arduino.press()
        else:
            if not input_backend.left_down():
                logger.warning("[射击] 原生鼠标点击后端在当前平台不可用。")

    def _release(self):
        if cfg.ch9329_shoot:
            from logic.ch9329 import ch9329
            ch9329.release()
        elif cfg.arduino_shoot:
            from logic.arduino import arduino
            arduino.release()
        else:
            if not input_backend.left_up():
                logger.warning("[射击] 原生鼠标点击后端在当前平台不可用。")

shooting = Shooting()
