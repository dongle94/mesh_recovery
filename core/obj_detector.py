import os
import sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.logger import get_logger


class ObjectDetector(object):
    """
    A general object detection class that supports various YOLO frameworks such as YOLOv5, YOLOv8, YOLOv10, and YOLOv11.

    Attributes:
        logger (logging.Logger): Logger instance for logging.
        cfg (dict): Configuration dictionary loaded from a YAML file.
        detector_type (str): The type of detector being used (e.g., YOLOv5, YOLOv8).
        detector (object): Instance of the specific YOLO detector class.
        names (list): List of class names used by the detector.
        f_cnt (int): Frame counter for performance logging.
        ts (list): List of times for pre-processing, inference, and post-processing.

    Raises:
        FileNotFoundError: If the specified weight file is not found.
        NotImplementedError: If an unsupported detector type is provided.
    """
    def __init__(self, cfg=None):
        """
        Initialize the ObjectDetector instance.

        Args:
            cfg (dict): Configuration dictionary containing detector settings.

        Raises:
            FileNotFoundError: If the weight file does not exist.
            NotImplementedError: If the detector type is unsupported.
        """
        self.logger = get_logger()
        self.cfg = cfg

        weight = os.path.abspath(cfg.det_model_path)
        if not os.path.isfile(weight):
            raise FileNotFoundError(f"Weight file not found: {weight}")
        self.detector_type = cfg.det_model_type.lower()

        device = cfg.device
        gpu_num = cfg.gpu_num
        fp16 = cfg.det_half
        conf_thres = cfg.det_conf_thres
        classes = cfg.det_obj_classes

        self.framework = None

        if self.detector_type in ["yolov5", "yolov8", "yolov10", "yolov11"]:
            img_size = cfg.yolo_img_size
            iou_thres = cfg.yolo_nms_iou
            agnostic = cfg.yolo_agnostic_nms
            max_det = cfg.yolo_max_det
            self.im_shape = None
            self.im0_shape = None

            # model load with weight
            ext = os.path.splitext(weight)[1]
            if ext in ['.pt', '.pth']:
                from core.yolo.yolo_pt import YoloTorch
                model = YoloTorch
                self.framework = 'torch'
            elif ext == '.onnx':
                from core.yolo.yolo_ort import YoloORT
                model = YoloORT
                self.framework = 'onnx'
            elif ext in ['.engine', '.bin']:
                from core.yolo.yolo_trt import YoloTRT
                model = YoloTRT
                self.framework = 'trt'
            else:
                raise FileNotFoundError('No YOLO(v5,v8,v10,v11) weight File!')

            self.detector = model(
                weight=weight,
                device=device,
                img_size=img_size,
                fp16=fp16,
                gpu_num=gpu_num,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                agnostic=agnostic,
                max_det=max_det,
                classes=classes,
                model_type=self.detector_type
            )
            self.names = self.detector.names

            # warm up
            self.detector.warmup()
            self.logger.info(f"Successfully loaded weight from {weight}")
        else:
            raise NotImplementedError(f'Unknown detector type: {self.detector_type}')

        # logging
        self.f_cnt = 0
        self.ts = [0., 0., 0.]

    def run(self, img):
        """
        Perform object detection on a single image.

        Args:
            img (numpy.ndarray): The input image for detection.

        Returns:
            list: List of detection results, including bounding box coordinates, class IDs, and scores.

        Raises:
            Exception: If an error occurs during detection.
        """
        try:
            if self.detector_type in ["yolov5", "yolov8", "yolov10", "yolov11"]:
                t0 = self.detector.get_time()

                img, orig_img = self.detector.preprocess(img)
                im_shape = img.shape
                im0_shape = orig_img.shape
                t1 = self.detector.get_time()

                preds = self.detector.infer(img)
                t2 = self.detector.get_time()

                det = self.detector.postprocess(preds, im_shape, im0_shape)
                t3 = self.detector.get_time()

                # calculate time & logging
                self.f_cnt += 1
                self.ts[0] += t1 - t0
                self.ts[1] += t2 - t1
                self.ts[2] += t3 - t2
                if self.f_cnt % self.cfg.console_log_interval == 0:
                    self.logger.debug(
                        f"{self.detector_type} detector {self.f_cnt} Frames average time - "
                        f"preproc: {self.ts[0]/self.f_cnt:.6f} sec / "
                        f"infer: {self.ts[1] / self.f_cnt:.6f} sec / " 
                        f"postproc: {self.ts[2] / self.f_cnt:.6f} sec")

            else:
                pred, det = None, None

            return det
        except Exception as e:
            self.logger.error(f"Error during detection: {e}")
            raise

    def run_np(self, img):
        """
        Perform object detection on a single image and return results as a numpy array.

        Args:
            img (numpy.ndarray): The input image for detection.

        Returns:
            numpy.ndarray: Detection results in numpy array format.
        """
        return self.run(img).cpu().numpy() if self.framework == 'torch' else self.run(img)


if __name__ == "__main__":
    import cv2
    import time
    from core.media_loader import MediaLoader
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _detector = ObjectDetector(cfg=_cfg)

    _bgr = getattr(_cfg, 'media_bgr', True)
    _realtime = getattr(_cfg, 'media_realtime', False)
    media_loader = MediaLoader(_cfg.media_source,
                               logger=_logger,
                               realtime=_realtime,
                               bgr=_bgr,
                               opt=_cfg)
    wt = 0 if media_loader.is_imgs else 1 / media_loader.dataset.fps

    while True:
        st = time.time()
        frame = media_loader.get_frame()

        _det = _detector.run(frame)

        for d in _det:
            x1, y1, x2, y2 = map(int, d[:4])
            cls = int(d[5])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            cv2.putText(frame, str(_detector.names[cls]), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)

        et = time.time()
        if media_loader.is_imgs:
            t = 0
        else:
            if et - st < wt:
                t = int((wt - (et - st)) * 1000) + 1
            else:
                t = 1

        cv2.imshow('_', frame)
        if cv2.waitKey(t) == ord('q'):
            print("-- CV2 Stop --")
            break

    print("-- Stop program --")
