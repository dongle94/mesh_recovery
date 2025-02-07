import os
import sys
import cv2
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
os.chdir(ROOT)

from core.medialoader import load_images, load_video, load_stream
from utils.logger import get_logger


def check_sources(source):
    img_formats = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'
    vid_formats = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv', 'webm'
    is_imgs, is_vid, is_stream = False, False, False
    if os.path.isdir(source) or '*' in source:
        is_imgs = True
    elif os.path.isfile(source) and Path(source).suffix[1:] in img_formats:
        is_imgs = True
    elif os.path.isfile(source) and Path(source).suffix[1:] in vid_formats:
        is_vid = True
    elif source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
        is_stream = True
    elif source.isnumeric() or source.endswith('.streams') or source.startswith('/dev'):
        is_stream = True

    return is_imgs, is_vid, is_stream


class MediaLoader(object):
    """
    A utility class to load and handle various media types including images, videos, and streams.

    Args:
        source (str): The media source path (file, directory, or stream URL).
        stride (int, optional): Frame sampling stride. Defaults to 1.
        logger (Logger, optional): Logger instance for logging. Defaults to None.
        realtime (bool, optional): Enable real-time streaming. Defaults to False.
        opt (Namespace, optional): Configuration options. Defaults to None.
        bgr (bool, optional): Load frames in BGR format if True. Defaults to True.

    Attributes:
        is_imgs (bool): Whether the source is a collection of images.
        is_vid (bool): Whether the source is a video file.
        is_stream (bool): Whether the source is a video stream.
        dataset (object): The dataset loader instance for the selected source.
        width (int): Width of the frames.
        height (int): Height of the frames.
    """
    def __init__(self, source, stride=1, logger=None, realtime=False, opt=None, bgr=True):

        self.stride = stride
        self.logger = logger if logger is not None else get_logger()
        self.realtime = realtime
        self.opt = opt
        self.bgr = bgr

        self.is_imgs, self.is_vid, self.is_stream = check_sources(source)

        if self.is_imgs:
            dataset = load_images.LoadImages(source, bgr=self.bgr)
        elif self.is_vid:
            dataset = load_video.LoadVideo(source, stride=self.stride, realtime=self.realtime, bgr=self.bgr,
                                           logger=logger)
        elif self.is_stream:
            dataset = load_stream.LoadStream(source, opt=self.opt, bgr=self.bgr, logger=logger)
        else:
            raise NotImplementedError(f'Invalid input: {source}')

        if self.is_vid or self.is_stream:
            self.width, self.height = dataset.w, dataset.h
        else:       # self.is_imgs:
            self.width, self.height = opt.media_width, opt.media_height

        self.dataset = dataset

        self.logger.info(f"-- Frame Metadata: {self.width}x{self.height}, FPS: {self.dataset.fps}")
        self.logger.info("-- MediaLoader is ready")

    def get_frame(self):
        """
        Fetch the next frame from the loaded media source.

        Returns:
            numpy.ndarray or None: The next frame as an image array if available, else None.

        Raises:
            StopIteration: If the dataset has no more frames to provide.
        """
        try:
            frame = next(self.dataset)  # Attempt to fetch the next frame
        except StopIteration:
            self.logger.info("Media source has been fully consumed.")
            raise StopIteration
        except Exception as e:
            # Handle other potential errors
            self.logger.error(f"Unexpected error while fetching frame: {e}")
            raise e
        return frame

    def show_frame(self, title: str = 'frame', wait_sec: int = 0):
        """
        Display the current frame in a window.

        Args:
            title (str, optional): Title of the display window. Defaults to 'frame'.
            wait_sec (int, optional): Delay in milliseconds for window display. Defaults to 0.
        """
        frame = self.get_frame()
        if self.bgr is False:
            frame = frame[..., ::-1]
        cv2.imshow(title, frame)
        if cv2.waitKey(wait_sec) == ord('q'):
            if self.logger is not None:
                self.logger.info("-- Quit Show frames")
            raise StopIteration

    def __del__(self):
        """
        Destructor to release resources.
        """
        if hasattr(self, 'dataset'):
            del self.dataset
        cv2.destroyAllWindows()


if __name__ == "__main__":
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _media_loader = MediaLoader(_cfg.media_source,
                                logger=_logger,
                                realtime=_cfg.media_realtime,
                                bgr=_cfg.media_bgr,
                                opt=_cfg)

    _title = 'frame'
    wt = int((0 if _media_loader.is_imgs else 1 / _media_loader.dataset.fps) * 1000)
    while True:
        try:
            _frame = _media_loader.show_frame(title=_title, wait_sec=wt)
        except StopIteration:
            print("All frames have been processed.")
            break
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            break
