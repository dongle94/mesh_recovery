import os
import sys

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from core.mp import task


class VideoInputTask(task.Job):
    def __init__(self, source, realtime=False, bgr=True):
        super().__init__()

        self.media_loader = None
        self.f_count = 0

        self.source = source
        self.realtime = realtime
        self.bgr = bgr

    def init(self):
        from core.media_loader import MediaLoader
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


class ObjectDetectionTask(task.Job):
    def __init__(self, opt, i=0):
        super().__init__(i=i)
        self.opt = opt

        self.detector = None
        self.f_count = 0

    def init(self):
        from core.obj_detector import ObjectDetector
        from utils.logger import init_logger

        init_logger(self.opt)

        self.detector = ObjectDetector(cfg=self.opt)
        self.f_count = 0

    def process(self, item=None):
        if item is None or item['end'] is True:
            return None

        frame = item['image']

        det = self.detector.run(frame)

        det_ret = []
        for d in det:
            x1, y1, x2, y2 = map(int, d[:4])
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            det_ret.append([x1, y1, x2, y2])

        item['det_ret'] = det_ret

        return item
