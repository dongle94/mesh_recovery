import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from core.media_loader import MediaLoader
from core.mp import mp_queue, task


class VideoInputTask(task.Job):
    def __init__(self, source, realtime=False, bgr=True):
        super().__init__()

        self.media_loader = None
        self.f_count = 0

        self.source = source
        self.realtime = realtime
        self.bgr = bgr

    def init(self):
        self.media_loader = MediaLoader(self.source)
        self.f_count = 0

    def process(self, item=None):
        frame = self.media_loader.get_frame()

        stop = True if frame is None else False

        item = {
            'idx': self.f_count,
            'image': frame,
            'end': stop
        }
        self.f_count += 1
        return item
