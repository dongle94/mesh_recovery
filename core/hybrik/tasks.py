import os
import sys
import cv2
import time
import math
import torch
import numpy as np
import datetime

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from core.mp import task
from utils.logger import get_logger, init_logger


class VideoInputTask(task.Job):
    def __init__(self, cfg):
        super().__init__(cfg=cfg, i=0)

        self.media_loader = None

        self.source = cfg.media_source
        self.realtime = cfg.media_realtime
        self.bgr = cfg.media_bgr

        self.logger = None
        self.total_time = 0.
        self.f_cnt = 0

    def init(self):
        from core.media_loader import MediaLoader
        self.media_loader = MediaLoader(self.source, logger=self.logger, realtime=self.realtime, bgr=self.bgr)

        init_logger(self.cfg)

        self.logger = get_logger()
        self.total_time = 0.
        self.f_cnt = 0

    def process(self, item=None):
        self.f_cnt += 1
        st = time.time()

        frame = self.media_loader.get_frame()

        stop = True if frame is None else False

        item = {
            'idx': self.f_cnt,
            'video_meta': [self.media_loader.dataset.fps, (self.media_loader.dataset.w, self.media_loader.dataset.h)],
            'image': frame,
            'end': stop
        }

        et = time.time()
        self.total_time += (et - st)
        if self.f_cnt % self.cfg.console_log_interval == 0:
            self.logger.info(
                f"VideoInputTask {self.f_cnt} frames average time: {self.total_time / self.f_cnt:.6f} sec."
            )

        return item


class ObjectDetectionTask(task.Job):
    def __init__(self, cfg, i=0):
        super().__init__(cfg=cfg, i=i)

        self.detector = None

        self.logger = None
        self.total_time = 0.
        self.f_cnt = 0

    def init(self):
        from core.obj_detector import ObjectDetector
        from utils.logger import init_logger

        init_logger(self.cfg)

        self.detector = ObjectDetector(cfg=self.cfg)

        self.logger = get_logger()
        self.total_time = 0.
        self.f_cnt = 0

    def process(self, item=None):
        if item is None or item['end'] is True:
            return item
        self.f_cnt += 1
        st = time.time()

        frame = item['image']

        det = self.detector.run(frame)

        det_ret = []
        for d in det:
            x1, y1, x2, y2 = map(int, d[:4])
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            det_ret.append([x1, y1, x2, y2])

        # det_ret = self.get_center_closed_box(det_ret, frame)
        item['det_ret'] = det_ret

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        et = time.time()
        self.total_time += (et - st)
        if self.f_cnt % self.cfg.console_log_interval == 0:
            self.logger.info(
                f"ObjectDetectionTask {self.f_cnt} frames average time: {self.total_time / self.f_cnt:.6f} sec."
            )

        return item

    @staticmethod
    def get_center_closed_box(boxes, img):
        h, w = img.shape[:2]
        ch, cw = h // 2, w // 2
        box = None
        dist_to_center = 0
        for b in boxes:
            x1, y1, x2, y2 = b
            cx, cy = int(x1 + x2) // 2, int(y1 + y2) // 2
            if box is None:
                box = b
                dist_to_center = math.dist((cy, cx), (ch, cw))
                continue

            _dist_to_center = math.dist((cy, cx), (ch, cw))
            if _dist_to_center < dist_to_center:
                dist_to_center = _dist_to_center
                box = b
        return [box]


class HybrIKXTask(task.Job):
    def __init__(self, cfg, i=0):
        super().__init__(cfg=cfg, i=i)

        self.device = cfg.device
        self.gpu_num = cfg.gpu_num

        self.hybrik_cfg = None
        self.transformation = None
        self.hybrik_model = None
        self._is_face_sent = False

        self.logger = None
        self.total_time = 0.
        self.f_cnt = 0

    def init(self):
        from core.hybrik.utils.presets.simple_transform_3d_smplx import SimpleTransform3DSMPLX
        from core.hybrik.utils.config import update_config
        from core.hybrik.models import builder
        from easydict import EasyDict

        init_logger(self.cfg)

        cfg_file = self.cfg.hybrik_cfg
        hybrik_ckpt = self.cfg.hybrik_ckpt

        self.hybrik_cfg = update_config(cfg_file)
        self.hybrik_cfg['MODEL']['EXTRA']['USE_KID'] = self.hybrik_cfg['DATASET'].get('USE_KID', False)

        bbox_3d_shape = getattr(self.hybrik_cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
        bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]
        dummy_set = EasyDict({
            'joint_pairs_17': None,
            'joint_pairs_24': None,
            'joint_pairs_29': None,
            'bbox_3d_shape': bbox_3d_shape
        })
        self.transformation = SimpleTransform3DSMPLX(
            dataset=dummy_set,
            scale_factor=self.hybrik_cfg.DATASET.SCALE_FACTOR,
            color_factor=self.hybrik_cfg.DATASET.COLOR_FACTOR,
            occlusion=self.hybrik_cfg.DATASET.OCCLUSION,
            add_dpg=False,
            input_size=self.hybrik_cfg.MODEL.IMAGE_SIZE,
            output_size=self.hybrik_cfg.MODEL.HEATMAP_SIZE,
            depth_dim=self.hybrik_cfg.MODEL.EXTRA.DEPTH_DIM,
            bbox_3d_shape=bbox_3d_shape,
            rot=self.hybrik_cfg.DATASET.ROT_FACTOR,
            sigma=self.hybrik_cfg.MODEL.EXTRA.SIGMA,
            train=False,
            loss_type=self.hybrik_cfg.LOSS['TYPE']
        )
        self.hybrik_model = builder.build_sppe(self.hybrik_cfg.MODEL)
        save_dict = torch.load(hybrik_ckpt, map_location='cpu')
        if type(save_dict) is dict:
            model_dict = save_dict['model']
            self.hybrik_model.load_state_dict(model_dict)
        else:
            self.hybrik_model.load_state_dict(save_dict)

        if self.device == 'cuda':
            self.hybrik_model.cuda(self.gpu_num)
        self.hybrik_model.eval()
        self._is_face_sent = False

        self.logger = get_logger()
        self.total_time = 0.
        self.f_cnt = 0

    def process(self, item=None):
        if item is None or item['end'] is True:
            return item
        self.f_cnt += 1
        st = time.time()

        frame = item['image'].copy()
        bboxes = item['det_ret']

        res = []
        for tight_box in bboxes:
            pose_input, bbox, img_center = self.transformation.test_transform(frame, tight_box)
            pose_input = pose_input[None, :, :, :]
            if self.device == 'cuda':
                pose_input = pose_input.to(self.gpu_num)

            pose_output = self.hybrik_model(
                x=pose_input,
                flip_test=True,
                bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
            )

            for k, v in pose_output.items():
                if type(v) is not int:
                    pose_output[k] = pose_output[k].detach().clone()
            bbox_xywh = self.xyxy2xywh(bbox)
            focal = 1000.0 / 256 * bbox_xywh[2]
            data = [pose_output, focal]
            res.append(data)
        item['hybrik_ret'] = res

        if self._is_face_sent is False:
            item['smplx_faces'] = torch.from_numpy(self.hybrik_model.smplx_layer.faces.astype(np.int32))
            self._is_face_sent = True

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        et = time.time()
        self.total_time += (et - st)
        if self.f_cnt % self.cfg.console_log_interval == 0:
            self.logger.info(
                f"HybrIKTask {self.f_cnt} frames average time: {self.total_time/self.f_cnt:.6f} sec."
            )

        return item

    @staticmethod
    def xyxy2xywh(bbox):
        x1, y1, x2, y2 = bbox

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return [cx, cy, w, h]


class PostHybrIKXTask(task.Job):
    def __init__(self, cfg, i=0):
        super().__init__(cfg=cfg, i=i)
        self.device = cfg.device
        self.torch_device = None
        self.gpu_num = cfg.gpu_num

        self.render_mesh = None
        self.smplx_faces = None
        self.write_stream = None
        self.write_mesh_stream = None
        self.dirname = ''
        self.savepath = ''
        self.savepath_mesh = ''

        self.logger = None
        self.total_time = 0.
        self.f_cnt = 0

    def init(self):
        from core.hybrik.utils.render_pytorch3d import render_mesh
        self.torch_device = torch.device(f'cuda:{self.gpu_num}') if self.cfg.device == 'cuda' else torch.device('cpu')
        self.render_mesh = render_mesh
        self.smplx_faces = None
        self.write_stream = None
        self.write_mesh_stream = None

        now = datetime.datetime.now()
        self.dirname = os.path.join('./out', now.strftime("%Y-%m-%d_%H-%M-%S"))
        if self.cfg.hybrik_save_img or self.cfg.hybrik_save_orig_img or self.cfg.hybrik_save_mesh_img or \
                self.cfg.hybrik_save_vid or self.cfg.hybrik_save_mesh_vid:
            os.makedirs(self.dirname)
        if self.cfg.hybrik_save_img is True:
            os.makedirs(os.path.join(self.dirname, 'res_images'))
        if self.cfg.hybrik_save_orig_img is True:
            os.makedirs(os.path.join(self.dirname, 'raw_images'))
        if self.cfg.hybrik_save_mesh_img is True:
            os.makedirs(os.path.join(self.dirname, 'res_mesh'))
        if self.cfg.hybrik_save_vid is True:
            self.savepath = f'{self.dirname}/res.mp4'
        if self.cfg.hybrik_save_mesh_vid is True:
            self.savepath_mesh = f'{self.dirname}/res_mesh.mp4'

        init_logger(self.cfg)
        self.logger = get_logger()
        self.total_time = 0.
        self.f_cnt = 0

    def process(self, item=None):
        if item is None or item['end'] is True:
            return item
        self.f_cnt += 1
        st = time.time()

        if self.smplx_faces is None:
            smplx_faces = item['smplx_faces']
            self.smplx_faces = smplx_faces.cuda() if self.cfg.device == 'cuda' else smplx_faces.cpu()
        if self.cfg.hybrik_save_vid is True and self.write_stream is None:
            fps, frame_size = item['video_meta'][0], item['video_meta'][1]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.write_stream = cv2.VideoWriter(
                self.savepath, fourcc, fps, frame_size)
        if self.cfg.hybrik_save_mesh_vid is True and self.write_mesh_stream is None:
            fps, frame_size = item['video_meta'][0], item['video_meta'][1]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.write_mesh_stream = cv2.VideoWriter(
                self.savepath_mesh, fourcc, fps, frame_size)

        frame = item['image'].copy()
        b_frame = None
        rets = item['hybrik_ret']
        for ret in rets:
            pose_output, focal = ret[0], ret[1]

            transl = pose_output.transl
            vertices = pose_output.pred_vertices
            verts_batch = vertices.cuda() if self.device == 'cuda' else vertices.cpu()
            transl_batch = transl.cuda() if self.device == 'cuda' else transl.cpu()

            color_batch = self.render_mesh(
                vertices=verts_batch,  # (1, 10475,3)
                faces=self.smplx_faces,  # (20908,3)
                translation=transl_batch,  # (1, 3)
                focal_length=focal,  # scalar
                height=frame.shape[0],
                width=frame.shape[1],
                device=self.torch_device
            )

            valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
            valid_mask = valid_mask_batch[0].cpu().numpy()

            image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
            image_vis_batch = (image_vis_batch * 255).cpu().numpy()
            color = image_vis_batch[0]

            alpha = 0.9
            frame = alpha * color[:, :, :3] * valid_mask + \
                (1 - alpha) * frame * valid_mask + \
                (1 - valid_mask) * frame
            if b_frame is None:
                b_frame = color
            else:
                b_frame = cv2.add(b_frame, color)

        image_vis = frame.astype(np.uint8)

        item['frame'] = image_vis
        item['b_frame'] = b_frame
        if b_frame is not None:
            b_frame = b_frame.astype(np.uint8)

        if self.cfg.hybrik_save_img is True:
            cv2.imwrite(os.path.join(self.dirname, 'res_images', f'image{self.f_cnt:06d}.jpg'), image_vis)
        if self.cfg.hybrik_save_orig_img is True:
            cv2.imwrite(os.path.join(self.dirname, 'raw_images', f'image{self.f_cnt:06d}.jpg'), item['image'])
        if self.cfg.hybrik_save_mesh_img is True:
            if b_frame is not None:
                cv2.imwrite(os.path.join(self.dirname, 'res_mesh', f'image{self.f_cnt:06d}.jpg'), b_frame)
        if self.cfg.hybrik_save_vid is True:
            self.write_stream.write(image_vis)
        if self.cfg.hybrik_save_mesh_vid is True:
            self.write_mesh_stream.write(b_frame)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        et = time.time()
        self.total_time += (et-st)
        if self.f_cnt % self.cfg.console_log_interval == 0:
            self.logger.info(
                f"PostHybrIKTask {self.f_cnt} frames average time: {self.total_time/self.f_cnt:.6f} sec."
            )

        del item['hybrik_ret']

        return item

    def close(self):
        if self.write_stream is not None:
            self.write_stream.release()
        if self.write_mesh_stream is not None:
            self.write_mesh_stream.release()
