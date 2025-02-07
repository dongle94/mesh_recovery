import cv2
import math
import time
import numpy as np
import platform
from threading import Thread

from core.medialoader import LoadSample


class LoadStream(LoadSample):
    """
    LoadStream class manages video streams such as webcam or RTSP streams.
    It loads the video source, captures frames in a separate thread, and allows iteration over frames.

    Attributes:
        source (str): The video source (e.g., webcam index, RTSP URL, or file path).
        opt (object): Configuration options for custom video parameters.
        bgr (bool): Whether the frames should remain in BGR format.
        logger (object): Logger instance for debugging and information logging.
        cap (cv2.VideoCapture): OpenCV object for video capture.
        w (int): Width of the video stream.
        h (int): Height of the video stream.
        fps (int): Frames per second of the stream.
        img (ndarray): Most recently retrieved video frame.
        thread (Thread): Background thread for updating frames.
        frame (int): Counter for the total number of frames processed.
    """

    def __init__(self, source, opt=None, bgr=True, logger=None):
        """
        Initializes the LoadStream instance.

        Args:
            source (str): The video source (e.g., webcam index as string or file path).
            opt (object, optional): Configuration for custom video parameters. Defaults to None.
            bgr (bool, optional): Whether the stream should remain in BGR format. Defaults to True.
            logger (object, optional): Logger for debugging and info. Defaults to None.

        Raises:
            ValueError: If the video source cannot be opened.
        """
        super().__init__()

        self.logger = logger
        self.bgr = bgr
        source = int(source) if source.isnumeric() else source

        if platform.system() == 'Windows':
            cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Failed to open source: {source}")

        self.cap = cap
        self.mode = 'webcam'

        if opt is not None and opt.media_opt_auto is False:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*opt.media_fourcc))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, opt.media_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, opt.media_height)
            cap.set(cv2.CAP_PROP_FPS, opt.media_fps)

        self.w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
        self.frame = 0
        self.enable_param = opt.media_enable_param if opt is not None else False
        if self.enable_param:
            for param in opt.media_cv2_params:
                k, v = param.popitem()
                cap.set(eval(k), v)
        else:
            cap.set(cv2.CAP_PROP_SETTINGS, 0)

        if self.logger is not None:
            self.logger.info(f"-- Load Stream: {self.w}*{self.h}, FPS: {self.fps} --")
        else:
            print(f"-- Load Stream: {self.w}*{self.h}, FPS: {self.fps} --")

        _, self.img = cap.read()
        self.thread = Thread(target=self.update, args=(cap, ), daemon=True)
        self.thread.start()

    def update(self, cap):
        """
        Continuously grabs frames from the video source in a separate thread.

        Args:
            cap (cv2.VideoCapture): The OpenCV video capture object.
        """
        while cap.isOpened():
            self.frame += 1

            cap.grab()
            success, im = cap.retrieve()
            if success:
                self.img = im
            else:
                self.img = np.zeros_like(self.img)
            time.sleep(0.0)  # wait time

    def __iter__(self):
        """
        Resets the frame counter for iteration.

        Returns:
            LoadStream: The instance itself, allowing it to be used in loops.
        """
        self.frame = 0
        return self

    def __next__(self):
        """
        Returns the next frame in the stream.

        Returns:
            ndarray: A single video frame (BGR or RGB).

        Raises:
            StopIteration: If the stream has stopped or the 'q' key is pressed.
        """
        if not self.thread.is_alive() or cv2.waitKey(1) == ord('q'):
            self.close()
            cv2.destroyAllWindows()
            raise StopIteration

        im = self.img.copy()
        if self.bgr is False:       # rgb
            im = im[..., ::-1]

        return im

    def __len__(self):
        """
        Placeholder for length computation.
        """
        pass

    def close(self):
        """
        Closes the video stream, releases the video capture object, and safely terminates the thread.
        """
        self.cap.release()
        if self.thread.is_alive():
            self.thread.join(5)

if __name__ == "__main__":
    s = '0'
    loader = LoadStream(s)
    for _im in loader:
        _im = _im[..., ::-1]
        cv2.imshow('.', _im)
