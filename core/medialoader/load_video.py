import os
import cv2
import math
import time
from threading import Thread

from core.medialoader import LoadSample

VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'


class LoadVideo(LoadSample):
    """
    Class to load and process video files frame by frame.

    Attributes:
        path (str): Absolute path to the video file.
        stride (int): Frame stride to skip frames for processing.
        realtime (bool): Flag to enable real-time frame streaming.
        bgr (bool): Flag to keep the frames in BGR format (default: True).
        logger (Logger, optional): Logger instance for logging information.
        mode (str): Mode for the loader, set to 'video'.
        w (int): Width of the video frame.
        h (int): Height of the video frame.
        fps (float): Frames per second of the video.
        frame (int): Current frame number being processed.
        frames (int): Total number of frames in the video file.
        wait_ms (float): Time interval to wait between frames (real-time mode).
        img (numpy.ndarray or None): Holds the current frame in real-time mode.
        cap (cv2.VideoCapture): OpenCV video capture object for reading the video.
    """

    def __init__(self, path, stride=1, realtime=False, bgr=True, logger=None):
        """
        Initializes the LoadVideo instance.

        Args:
            path (str): Path to the video file.
            stride (int, optional): Number of frames to skip. Defaults to 1.
            realtime (bool, optional): Enables real-time video streaming. Defaults to False.
            bgr (bool, optional): Keeps frames in BGR format if True; otherwise, RGB. Defaults to True.
            logger (Logger, optional): Logger instance for logging information.

        Raises:
            FileNotFoundError: If the video file format is invalid.
            RuntimeError: If the video file fails to open.
            ValueError: If no frames are found in the video file.
        """
        super().__init__()

        self.logger = logger
        self.stride = stride
        self.realtime = realtime
        self.bgr = bgr

        path = os.path.abspath(path)
        if path.split('.')[-1].lower() not in VID_FORMATS:
            raise FileNotFoundError(f"File ext is invalid: {path}")

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {path}")

        self.cap = cap
        self.mode = 'video'
        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = 0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.stride)
        if self.frames <= 0:
            raise ValueError(f"No frames found in the video: {path}")

        if self.logger:
            self.logger.info(f"-- Load Video: {self.w}*{self.h}, FPS: {self.fps} --")
        else:
            print(f"-- Load Video: {self.w}*{self.h}, FPS: {self.fps} --")

        self.wait_ms = 1 / self.fps
        self.img = None

        if self.realtime:
            _, self.img = cap.read()
            self.thread = Thread(target=self.update, args=(cap,), daemon=True)
            self.thread.start()

    def update(self, cap):
        """
        Real-time frame update for video playback.

        Args:
            cap (cv2.VideoCapture): OpenCV video capture object.

        Raises:
            StopIteration: When the video ends.
        """
        while cap.isOpened() and self.frame < self.frames:
            self.frame += 1

            st = time.time()
            for _ in range(self.stride):
                cap.grab()
            ret, im = cap.retrieve()
            while not ret:
                cap.release()
                break

            self.img = im
            wait_t = max(self.wait_ms - (time.time() - st), 0)
            time.sleep(wait_t)

    def __iter__(self):
        """
        Initialize the video iterator.

        Returns:
            LoadVideo: Instance of the current class.
        """
        self.frame = 0
        return self

    def __next__(self):
        """
        Retrieves the next video frame.

        Returns:
            numpy.ndarray: Video frame.

        Raises:
            StopIteration: If the video ends or 'q' is pressed.
        """
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration("User Stop Video")

        if self.realtime is True:
            if self.img is None:
                raise StopIteration("End of video file")
            im = self.img.copy()
        else:
            for _ in range(self.stride):
                self.cap.grab()
            ret, im = self.cap.retrieve()
            while not ret:
                self.cap.release()
                raise StopIteration("End of video file")

        if not self.bgr:
            im = im[..., ::-1]

        return im

    def __len__(self):
        """
        Returns the total number of frames in the video.

        Returns:
            int: Total frame count.
        """
        return self.frames


if __name__ == "__main__":
    p1 = './data/videos/sample.mp4'
    loader = LoadVideo(p1)
    for _im in loader:
        _im = _im[..., ::-1]
        cv2.imshow('.', _im)
