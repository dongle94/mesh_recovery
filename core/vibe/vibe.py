# -*- coding: utf-8 -*-

"""
Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
holder of all proprietary rights on this computer program.
You can only use this computer program if you have closed
a license agreement with MPG or you get the right to use the computer
program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and
liable to prosecution.

Copyright©2019 Max-Planck-Gesellschaft zur Förderung
der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
for Intelligent Systems. All rights reserved.

Contact: ps-license@tuebingen.mpg.de
"""
import os
import sys
import cv2
import shutil
import colorsys
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from utils.logger import get_logger
from core.vibe.models.vibe import VIBE_Demo
from core.vibe.utils.renderer import Renderer
from core.vibe.dataset.inference import Inference
from core.vibe.models.mpt import run_tracker, prepare_output_tracks
from core.vibe.utils.demo_utils import (
	video_to_images,
    images_to_video,
    convert_crop_cam_to_orig_img,
    convert_crop_coords_to_orig_img,
    prepare_rendering_results
)


class VIBE(nn.Module):
    """
    Minimal VIBE wrapper skeleton for single-video 3D mesh extraction.

    This file provides a lightweight class and method stubs to be filled
    during the integration process. Methods intentionally raise
    NotImplementedError to mark integration points.
    """

    def __init__(self, config: Dict, device: str = 'cuda', logger=None):
        super(VIBE, self).__init__()

        self.config = config
        self.device = torch.device('cuda') if device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
        self.logger = get_logger() if logger is None else logger
        self.model = None
        self.is_initialized = False

        # Model components
        self.model = None

        # build model components (to be implemented)
        self._build_model()

    def _build_model(self) -> None:
        """Initialize network, load checkpoints and any required assets."""
        # placeholder: implement loading of VIBE backbone, temporal model, SMPL, etc.
        self.model = VIBE_Demo(
            seqlen=16,
            batch_size=self.config.vibe_batch_size,
            n_layers=2,
            hidden_size=1024,
            add_linear=True,
            use_residual=True,
            pretrained=os.path.abspath(os.path.join(self.config.vibe_data_dir, 'spin_model_checkpoint.pth.tar')),
            smpl_mean_params=os.path.abspath(os.path.join(self.config.vibe_data_dir, 'smpl_mean_params.npz')),
        ).to(self.device)

        if self.config.vibe_use_3dpw:
            pretrained_file = os.path.join(self.config.vibe_data_dir, 'vibe_model_w_3dpw.pth.tar')
        else:
            pretrained_file = os.path.join(self.config.vibe_data_dir, 'vibe_model_wo_3dpw.pth.tar')

        ckpt = torch.load(pretrained_file, weights_only=False)
        self.logger.info(f'Performance of pretrained model on 3DPW: {ckpt["performance"]}')
        ckpt = ckpt['gen_state_dict']
        self.model.load_state_dict(ckpt, strict=False)
        self.logger.info(f'Loaded pretrained weights from \"{pretrained_file}\"')

        self.is_initialized = True

    def preprocess_video(self, video_path: str) -> Tuple[np.ndarray, Dict]:
        """Load video, extract frames and apply required preprocessing.

        Returns:
            frames (np.ndarray): array of frames (T, H, W, C)
            meta (dict): metadata needed for postprocessing (e.g., original sizes)
        """
        raise NotImplementedError("VIBE.preprocess_video must be implemented during integration")

    def forward(self, frames: torch.Tensor) -> Dict:
        """Run model forward on a batch of frames.

        Returns a dict containing model raw outputs (poses, betas, cameras, etc.).
        """
        raise NotImplementedError("VIBE.forward must be implemented during integration")

    def postprocess_output(self, output: Dict, meta: Dict) -> Dict:
        """Convert raw model outputs to meshes, projected joints and friendly formats."""
        raise NotImplementedError("VIBE.postprocess_output must be implemented during integration")

    def visualize_result(self, frames: np.ndarray, results: Dict) -> np.ndarray:
        """Render or draw results on input frames and return a visualization buffer.

        This should return an image or video-like numpy array for debugging.
        """
        raise NotImplementedError("VIBE.visualize_result must be implemented during integration")

    def predict(self, video_path: str) -> Dict:
        """High-level helper: preprocess -> forward -> postprocess -> visualize.

        Returns a results dict with meshes, joints and optionally visualization.
        """
        # simple orchestration stub
        frames, meta = self.preprocess_video(video_path)
        output = self.forward(torch.from_numpy(frames).to(self.device))
        results = self.postprocess_output(output, meta)
        return results

    def __repr__(self) -> str:
        return f"VIBE(device={self.device}, initialized={self.is_initialized})"

    def process_detection(self, image_folder: str, detector):
        # run tracker
        trackers = run_tracker(image_folder, detector)
        
        tracking_results = prepare_output_tracks(trackers)

        return tracking_results

if __name__ == "__main__":
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from core.obj_detector import ObjectDetector

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    OUTPUT_FOLDER = Path("out") / os.path.basename(os.path.splitext(_cfg.media_source)[0])
    video_file = _cfg.media_source

    _detector = ObjectDetector(cfg=_cfg)
    _VIBE = VIBE(_cfg, device=_cfg.device, logger=_logger)

    # ========= Run tracking ========= #
    bbox_scale = 1.1
    image_folder, num_frames, img_shape = video_to_images(_cfg.media_source, return_info=True)
    print(f'Input video number of frames {num_frames}')
    orig_height, orig_width = img_shape[:2]
    tracking_results = _VIBE.process_detection(image_folder, _detector)

    vibe_results = {}
    # ========= Run VIBE on each person ========= #
    for person_id in tqdm(list(tracking_results.keys())):
        bboxes = joints2d = None

        bboxes = tracking_results[person_id]['bbox']
        frames = tracking_results[person_id]['frames']

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=bbox_scale,
        )

        bboxes = dataset.bboxes
        frames = dataset.frames

        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(dataset, batch_size=_cfg.vibe_batch_size, num_workers=16)

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d, norm_joints2d = \
                [], [], [], [], [], [], []
            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))

                batch = batch.unsqueeze(0)
                batch = batch.to(_VIBE.device)

                batch_size, seqlen = batch.shape[:2]
                output = _VIBE.model(batch)[-1]

                pred_cam.append(output['theta'][:, :, :3].reshape(batch_size * seqlen, -1))
                pred_verts.append(output['verts'].reshape(batch_size * seqlen, -1, 3))
                pred_pose.append(output['theta'][:,:,3:75].reshape(batch_size * seqlen, -1))
                pred_betas.append(output['theta'][:, :,75:].reshape(batch_size * seqlen, -1))
                pred_joints3d.append(output['kp_3d'].reshape(batch_size * seqlen, -1, 3))
                smpl_joints2d.append(output['kp_2d'].reshape(batch_size * seqlen, -1, 2))
            
            pred_cam = torch.cat(pred_cam, dim=0)
            pred_verts = torch.cat(pred_verts, dim=0)
            pred_pose = torch.cat(pred_pose, dim=0)
            pred_betas = torch.cat(pred_betas, dim=0)
            pred_joints3d = torch.cat(pred_joints3d, dim=0)
            smpl_joints2d = torch.cat(smpl_joints2d, dim=0)
            del batch

        # ========= Save results to a pickle file ========= #
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        print(pred_cam.shape, bboxes.shape)
        orig_cam = convert_crop_cam_to_orig_img(
            cam=pred_cam,
            bbox=bboxes,
            img_width=orig_width,
            img_height=orig_height
        )

        joints2d_img_coord = convert_crop_coords_to_orig_img(
            bbox=bboxes,
            keypoints=smpl_joints2d,
            crop_size=224,
        )

        output_dict = {
            'pred_cam': pred_cam,
            'orig_cam': orig_cam,
            'verts': pred_verts,
            'pose': pred_pose,
            'betas': pred_betas,
            'joints3d': pred_joints3d,
            'joints2d': joints2d,
            'joints2d_img_coord': joints2d_img_coord,
            'bboxes': bboxes,
            'frame_ids': frames,
        }

        vibe_results[person_id] = output_dict

    # ========= Render results as a single video ========= #
    renderer = Renderer(resolution=(orig_width, orig_height), orig_img=True, wireframe=False, faces=_VIBE.model.smpl.faces)

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f'Rendering output video, writing frames to {OUTPUT_FOLDER}')

    # prepare results for rendering
    frame_results = prepare_rendering_results(vibe_results, num_frames)
    mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in vibe_results.keys()}

    image_file_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.endswith('.png') or x.endswith('.jpg') or x.endswith('.jpeg')
    ])

    for frame_idx in tqdm(range(len(image_file_names))):
        img_fname = image_file_names[frame_idx]
        img = cv2.imread(img_fname)

        for person_id, person_data in frame_results[frame_idx].items():
            frame_verts = person_data['verts']
            frame_cam = person_data['cam']

            mc = mesh_color[person_id]

            mesh_filename = None

            img = renderer.render(
                img,
                frame_verts,
                cam=frame_cam,
                color=mc,
                mesh_filename=mesh_filename,
            )

        cv2.imwrite(os.path.join(OUTPUT_FOLDER, f'{frame_idx:06d}.png'), img)

        cv2.imshow('Video', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()

    vid_name = os.path.basename(video_file)
    save_name = f'{vid_name.replace(".mp4", "")}_vibe_result.mp4'
    save_name = os.path.join(OUTPUT_FOLDER, save_name)
    print(f'Saving result video to {save_name}')
    images_to_video(img_folder=OUTPUT_FOLDER, output_vid_file=save_name)
    # shutil.rmtree(OUTPUT_FOLDER)

    print("완료")