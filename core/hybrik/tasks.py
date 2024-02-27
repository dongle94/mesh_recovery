import os
import sys
import cv2
import time
import torch
import numpy as np
import datetime

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_PATH)

from core.mp import task
from utils.logger import get_logger, init_logger


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
            'video_meta': [self.media_loader.dataset.fps, (self.media_loader.dataset.w, self.media_loader.dataset.h)],
            'image': frame,
            'end': stop
        }
        self.f_count += 1
        return item


class ObjectDetectionTask(task.Job):
    def __init__(self, cfg, i=0):
        super().__init__(i=i)
        self.cfg = cfg

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
            return None
        self.f_cnt += 1
        st = time.time()

        frame = item['image']

        det = self.detector.run(frame)

        det_ret = []
        for d in det:
            x1, y1, x2, y2 = map(int, d[:4])
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
            det_ret.append([x1, y1, x2, y2])

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


class HybrIKTask(task.Job):
    def __init__(self, cfg, i=0):
        super().__init__(i=i)

        self.cfg = cfg

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
        cfg_file = './configs/256x192_hrnet_rle_smplx_kid.yaml'
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
        self.hybrik_model.cuda(self.gpu_num)
        self.hybrik_model.eval()
        self._is_face_sent = False

        init_logger(self.cfg)
        self.logger = get_logger()
        self.total_time = 0.
        self.f_cnt = 0

    def process(self, item=None):
        if item is None or item['end'] is True:
            return None
        self.f_cnt += 1
        st = time.time()

        frame = item['image'].copy()
        bboxes = item['det_ret']

        res = []
        for tight_box in bboxes:
            pose_input, bbox, img_center = self.transformation.test_transform(frame.copy(), tight_box)
            pose_input = pose_input.to(self.gpu_num)[None, :, :, :]
            bbox_xywh = self.xyxy2xywh(bbox)

            pose_output = self.hybrik_model(
                x=pose_input,
                flip_test=True,
                bboxes=torch.from_numpy(np.array(bbox)).to(pose_input.device).unsqueeze(0).float(),
                img_center=torch.from_numpy(img_center).to(pose_input.device).unsqueeze(0).float(),
            )

            for k, v in pose_output.items():
                if type(v) is not int:
                    pose_output[k] = pose_output[k].clone().detach().cpu()
            focal = 1000.0 / 256 * bbox_xywh[2]
            data = {
                'pose_output': pose_output,
                'focal': focal
            }
            res.append(data)
        item['hybrik_ret'] = res

        if self._is_face_sent is False:
            item['smplx_faces'] = torch.from_numpy(self.hybrik_model.smplx_layer.faces.astype(np.int32))
            self._is_face_sent = True

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        et = time.time()
        self.total_time += (et-st)
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


class PostHybrIKTask(task.Job):
    def __init__(self, cfg, i=0):
        super().__init__(i=i)
        self.cfg = cfg
        self.gpu_num = cfg.gpu_num

        self.render_mesh = None
        self.smplx_faces = None
        self.write_stream = None
        self.write_mesh_stream = None

        self.logger = None
        self.total_time = 0.
        self.f_cnt = 0

    def init(self):
        from core.hybrik.utils.render_pytorch3d import render_mesh
        self.render_mesh = render_mesh
        self.smplx_faces = None
        self.write_stream = None
        self.write_mesh_stream = None

        now = datetime.datetime.now()
        dirname = os.path.join('./out', now.strftime("%Y-%m-%d_%H-%M-%S"))
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        if not os.path.exists(os.path.join(dirname, 'raw_images')):
            os.makedirs(os.path.join(dirname, 'raw_images'))
        if not os.path.exists(os.path.join(dirname, 'res_images')):
            os.makedirs(os.path.join(dirname, 'res_images'))
        if not os.path.exists(os.path.join(dirname, 'res_mesh')):
            os.makedirs(os.path.join(dirname, 'res_mesh'))
        self.savepath = f'./out/{dirname}/res.mp4'
        self.savepath_mesh = f'./out/{dirname}/res_mesh.mp4'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        init_logger(self.cfg)
        self.logger = get_logger()
        self.total_time = 0.
        self.f_cnt = 0

    def process(self, item=None):
        if item is None or item['end'] is True:
            return None
        self.f_cnt += 1
        st = time.time()

        # if self.smplx_faces is None:
        #     self.smplx_faces = item['smplx_faces']
        # if self.write_stream is None:
        #     fps, frame_size = item['video_meta'][0], item['video_meta'][1]
        #     self.write_stream = cv2.VideoWriter(
        #         self.savepath, self.fourcc, fps, frame_size)
        #     self.write_mesh_stream = cv2.VideoWriter(
        #         self.savepath_mesh, self.fourcc, fps, frame_size)

        frame = item['image']
        b_frame = None
        rets = item['hybrik_ret']
        # for ret in rets:
        #     pose_output, focal = ret['pose_output'], ret['focal']
        #     uv_jts = pose_output.pred_uvd_jts.reshape(-1, 3)[:, :2]
        #     transl = pose_output.transl.detach()
        #     vertices = pose_output.pred_vertices.detach()
        #     verts_batch = vertices
        #     transl_batch = transl
        #
        #     color_batch = self.render_mesh(
        #         vertices=verts_batch,  # (10475,3)
        #         faces=self.smplx_faces,  # (20908,3)
        #         translation=transl_batch,  # (1, 3)
        #         focal_length=focal,  # scalar: 3709
        #         height=frame.shape[0],  # 1024
        #         width=frame.shape[1],  # 576
        #         device=torch.device(f'cuda:{self.gpu_num}')
        #     )
        #
        #     valid_mask_batch = (color_batch[:, :, :, [-1]] > 0)
        #     valid_mask = valid_mask_batch[0].cpu().numpy()
        #
        #     image_vis_batch = color_batch[:, :, :, :3] * valid_mask_batch
        #     image_vis_batch = (image_vis_batch * 255).cpu().numpy()
        #     color = image_vis_batch[0]
        #
        #     input_img = frame
        #     alpha = 0.9
        #     image_vis = alpha * color[:, :, :3] * valid_mask + \
        #         (1 - alpha) * input_img * valid_mask + \
        #         (1 - valid_mask) * input_img
        #     if b_frame is None:
        #         b_frame = color[:, :, :3] * valid_mask
        #     else:
        #         b_frame = cv2.add(b_frame, color[:, :, :3] * valid_mask)
        #
        # b_frame = b_frame.astype(np.uint8)
        # b_frame = cv2.cvtColor(b_frame, cv2.COLOR_RGB2BGR)
        #
        # item['b_frame'] = b_frame
        #
        # if torch.cuda.is_available():
        #     torch.cuda.synchronize()
        # et = time.time()
        # self.total_time += (et-st)
        # if self.f_cnt % self.cfg.console_log_interval == 0:
        #     self.logger.info(
        #         f"PostHybrIKTask {self.f_cnt} frames average time: {self.total_time/self.f_cnt:.6f} sec."
        #     )

        return item
