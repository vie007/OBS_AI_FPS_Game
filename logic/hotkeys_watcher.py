import threading
import time
from typing import List
import cv2

from logic.config_watcher import cfg
from logic.capture import capture
from logic.mouse import mouse
from logic.shooting import shooting
from logic.visual import visuals
from logic.platform import input_backend
from logic.model_classes import (
    HEAD_CLASS_ID,
    HIDEOUT_TARGET_CLASS_IDS,
    PLAYER_CLASS_ID,
)
from logic.logger import logger

# 全局退出信号：主循环检测到此事件后自行退出，避免 os._exit 跳过清理
shutdown_event = threading.Event()


class HotkeysWatcher(threading.Thread):
    def __init__(self):
        super(HotkeysWatcher, self).__init__()
        self.daemon = True
        self.name = 'HotkeysWatcher'

        self.app_pause = 0
        self.clss = self.active_classes()
        self._next_config_poll_at = 0.0

        self.start()

    def run(self):
        cfg_reload_prev_state = False
        while not shutdown_event.is_set():
            cfg_reload_prev_state = self.process_hotkeys(cfg_reload_prev_state)
            self.reload_config_if_changed()

            # terminate
            if input_backend.is_pressed(cfg.hotkey_exit):
                self._graceful_shutdown()
            time.sleep(0.01)

    def _graceful_shutdown(self):
        """优雅退出：依次关闭各子系统，通知主循环退出。

        不再调用 os._exit(0)。改为:
          1. 设置 shutdown_event 通知主循环停止
          2. 执行清理（释放鼠标、停止截帧、关闭串口等）
          3. 主循环检测到 shutdown_event 后自行退出，进程自然结束
        """
        # 先通知主循环停止，防止主循环在清理期间继续调用已关闭的子系统
        shutdown_event.set()

        logger.info("[退出] 正在关闭程序...")

        # 0. 释放鼠标左键（防止自动射击状态下退出时按键卡住）
        try:
            shooting.shoot(False, False)
        except Exception:
            pass

        # 1. 停止截帧
        try:
            capture.Quit()
        except Exception as e:
            logger.warning(f"[退出] 关闭截帧失败: {e}")

        # 2. 关闭调试窗口
        if cfg.show_window:
            try:
                visuals.stop()
            except Exception as e:
                logger.warning(f"[退出] 关闭调试窗口失败: {e}")

        # 3. 关闭 Arduino 串口
        if cfg.arduino_move or cfg.arduino_shoot:
            try:
                from logic.arduino import arduino
                arduino.close()
            except Exception as e:
                logger.warning(f"[退出] 关闭 Arduino 串口失败: {e}")

        # 4. 关闭 CH9329 串口
        if cfg.ch9329_move or cfg.ch9329_shoot:
            try:
                from logic.ch9329 import ch9329
                ch9329.close()
            except Exception as e:
                logger.warning(f"[退出] 关闭 CH9329 串口失败: {e}")

        logger.info("[退出] 清理完成，程序即将退出")

    def process_hotkeys(self, cfg_reload_prev_state):
        self.app_pause = -1 if input_backend.is_pressed(cfg.hotkey_pause) else 0
        app_reload_cfg = input_backend.is_pressed(cfg.hotkey_reload_config)

        if app_reload_cfg and not cfg_reload_prev_state:
            cfg.Read(verbose=True)
            self.apply_config_changes()

        return app_reload_cfg

    def reload_config_if_changed(self):
        now = time.monotonic()
        if now < self._next_config_poll_at:
            return
        self._next_config_poll_at = now + 0.25
        if cfg.reload_if_changed(verbose=True):
            self.apply_config_changes()

    def apply_config_changes(self):
        capture.restart()
        mouse.update_settings()
        try:
            from logic.frame_parser import frameParser
            frameParser.update_settings()
        except ImportError:
            pass
        self.clss = self.active_classes()
        if cfg.show_window == False:
            cv2.destroyAllWindows()

    def active_classes(self) -> List[int]:
        clss = [PLAYER_CLASS_ID]

        if cfg.hideout_targets:
            clss.extend(HIDEOUT_TARGET_CLASS_IDS)

        if not cfg.disable_headshot:
            clss.append(HEAD_CLASS_ID)

        self.clss = sorted(set(clss))
        return self.clss

hotkeys_watcher = HotkeysWatcher()
