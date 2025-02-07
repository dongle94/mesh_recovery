import os
import glob
import cv2

from core.medialoader import LoadSample

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp', 'pfm'


class LoadImages(LoadSample):
    """
    LoadImages: A class to load image files sequentially from a given path or list of paths.

    This class is an iterator that allows loading and processing of images from specified directories
    or files. It supports filtering image files based on the extension and converting color formats
    between BGR and RGB.

    Attributes:
        path (str or list): Path to the image folder or file(s).
        bgr (bool): Whether to load images in BGR format (default: True).
        files (list): List of image file paths.
        filename (str): The current image filename being loaded.
        num_files (int): The number of valid image files found.
        count (int): The current index of the image being processed.
        mode (str): Mode of operation, set to 'image'.
        w (int): Width of the current image.
        h (int): Height of the current image.
        fps (int): Frame per second (default: -1, not used in image loading).

    Methods:
        __init__(self, path, bgr=True): Initializes the class with a path to images and a flag to control color format.
        __iter__(self): Resets the iterator's count and prepares for iteration.
        __next__(self): Loads the next image from the list and returns it.
        __len__(self): Returns the total number of image files.
    """

    def __init__(self, path, bgr=True):
        """
        Initialize the LoadImages class.

        Args:
            path (str or list): Path(s) to the image file(s) or folder(s).
                It supports single file paths, directory paths, or wildcard paths.
            bgr (bool, optional): Whether to load images in BGR format.
                Defaults to True.

        Raises:
            FileNotFoundError: If the specified path does not exist.
        """
        super().__init__()

        self.bgr = bgr

        files = []
        path = sorted(path) if isinstance(path, (list, tuple)) else [path]
        for p in path:
            p = os.path.abspath(p)
            if '*' in p:
                files = [os.path.join(os.path.dirname(p), f) for f in os.listdir(os.path.dirname(p))]
            elif os.path.isdir(p):
                files = glob.glob(os.path.join(p, '*.*'))
            elif os.path.isfile(p):
                files.append(p)
            else:
                raise FileNotFoundError(f'{p} does not exist')

        images = [x for x in files if x.lower().endswith(IMG_FORMATS)]
        ni = len(images)

        self.mode = 'image'
        self.w, self.h = -1, -1
        self.fps = -1
        self.files = images
        self.filename = "-"
        self.num_files = ni
        self.count = 0

    def __iter__(self):
        """
        Resets the iteration counter to 0 when starting a new iteration.

        Returns:
            LoadImages: The instance of the class itself.
        """
        self.count = 0
        return self

    def __next__(self):
        """
        Load the next image in the sequence.

        Returns:
            numpy.ndarray: The loaded image in BGR or RGB format as a NumPy array.

        Raises:
            StopIteration: If all files have been processed.
            RuntimeError: If an image cannot be loaded.
        """
        if self.count == self.num_files:
            raise StopIteration

        path = self.files[self.count]

        self.count += 1
        self.filename = os.path.basename(path)
        im = cv2.imread(path)
        if im is None:
            raise RuntimeError(f'Failed to load image: {path}')
        self.h, self.w = im.shape[:2]
        if self.bgr is False:
            im = im[..., ::-1]

        return im

    def __len__(self):
        """
        Get the total number of valid image files.

        Returns:
            int: The total number of image files.
        """
        return self.num_files  # number of files


if __name__ == "__main__":
    loader = LoadImages('./data/images/')
    for img in loader:
        cv2.imshow('Loaded Image', img)
        if cv2.waitKey(0) == ord('q'):
            break
    cv2.destroyAllWindows()
