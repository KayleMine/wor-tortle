from time import time, sleep
from settings import Settings


class FpsManager:
    def __init__(self, task_name, target_fps=None):
        self.task_name = task_name
        self.last_execute = None
        if target_fps is None:
            self.target_delay = 1 / Settings.Render.target_fps
        else:
            self.target_delay = 1 / target_fps

    def delay(self):
        if self.last_execute is not None:
            time_diff = time() - self.last_execute
            if time_diff < self.target_delay:
                sleep(self.target_delay - time_diff)
        self.last_execute = time()
