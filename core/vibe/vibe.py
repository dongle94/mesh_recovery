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
        
        self.bbox_scale = 1.1

    def _build_model(self) -> None:
        """Initialize network components and load pretrained weights.

        This will construct the VIBE model, load SMPL mean parameters and
        pretrained weights according to the configuration available at
        ``self.config`` and move the model to ``self.device``.

        Raises:
            RuntimeError: If checkpoint files are missing or loading fails.
        """
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

    def preprocess_video(self, video_path: str) -> Tuple[str, int, Tuple[int, int, int]]:
        """Extract frames from a video and return image folder metadata.

        Args:
            video_path: Path to the input video file.

        Returns:
            A tuple ``(image_folder, num_frames, img_shape)`` where:
            - image_folder: path containing extracted image frames
            - num_frames: number of extracted frames
            - img_shape: image shape as (height, width, channels)

        Raises:
            RuntimeError: If frame extraction fails or video file not found.
        """
        image_folder, num_frames, img_shape = video_to_images(video_path, return_info=True)
        self.logger.info(f'Input video number of frames {num_frames}')
                
        return image_folder, num_frames, img_shape
    
    def process_detection(self, image_folder: str, detector) -> Dict:
        """Run detection and tracking over extracted frames.

        Args:
            image_folder: Directory with extracted image frames.
            detector: Object detector instance with a ``run(img)`` method.

        Returns:
            A tracking_results dictionary mapping person_id to track info,
            typically containing keys like ``'bbox'`` and ``'frames'``.
        """
        # run tracker
        trackers = run_tracker(image_folder, detector)

        tracking_results = prepare_output_tracks(trackers)

        return tracking_results

    def run_person_inference(
        self,
        image_folder: str,
        frames: np.ndarray,
        bboxes: np.ndarray,
        joints2d: Optional[np.ndarray],
        orig_width: int,
        orig_height: int,
    ) -> Dict:
        """Run VIBE model on a single person's track and return results.

        Args:
            image_folder: Directory with extracted image frames.
            frames: Array of frame indices (0-based) belonging to this track.
            bboxes: Array of bounding boxes corresponding to ``frames`` with
                shape (N, 4) in [x, y, w, h] format (crop coordinates).
            joints2d: Optional 2D keypoints aligned with ``frames`` if available.
            orig_width: Original image width in pixels.
            orig_height: Original image height in pixels.

        Returns:
            A dictionary containing numpy arrays for predictions with keys like
            ``'pred_cam'``, ``'orig_cam'``, ``'verts'``, ``'pose'``, ``'betas'``,
            ``'joints3d'``, ``'joints2d'``, ``'joints2d_img_coord'``, ``'bboxes'``,
            and ``'frame_ids'``. All arrays are trimmed to the same leading
            dimension N (number of outputs).

        Raises:
            RuntimeError: If the model produces no outputs for the provided data.
        """
        # create dataset (keeps same filtering logic as Inference)

        dataset = Inference(
            image_folder=image_folder,
            frames=frames,
            bboxes=bboxes,
            joints2d=joints2d,
            scale=self.bbox_scale,
        )

        # use dataset's potentially reindexed values
        bboxes = dataset.bboxes
        frames = dataset.frames

        has_keypoints = True if joints2d is not None else False

        dataloader = DataLoader(
            dataset, 
            batch_size=self.config.vibe_batch_size, 
            num_workers=min(8, os.cpu_count() or 1),
        )

        with torch.no_grad():
            pred_cam, pred_verts, pred_pose, pred_betas, pred_joints3d, smpl_joints2d = \
                [], [], [], [], [], []
            norm_joints2d = [] if has_keypoints else None

            for batch in dataloader:
                if has_keypoints:
                    batch, nj2d = batch
                    norm_joints2d.append(nj2d.numpy().reshape(-1, 21, 3))
                
                # prepare batch for model (VIBE expects (B, T, C, H, W) depending on implementation)
                # here original code did unsqueeze and used self.model(batch)
                batch = batch.unsqueeze(0) if batch.dim() == 4 else batch
                batch = batch.to(self.device)

                batch_size, seqlen = batch.shape[:2]
                output = self.model(batch)[-1]

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

        # to numpy
        pred_cam = pred_cam.cpu().numpy()
        pred_verts = pred_verts.cpu().numpy()
        pred_pose = pred_pose.cpu().numpy()
        pred_betas = pred_betas.cpu().numpy()
        pred_joints3d = pred_joints3d.cpu().numpy()
        smpl_joints2d = smpl_joints2d.cpu().numpy()

        # convert to original image coordinates
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

        return output_dict

    def postprocess_output(self, vibe_results: Dict, meta: Dict) -> List[Dict]:
        """Aggregate per-person outputs into per-frame rendering structures.

        Args:
            vibe_results: Mapping person_id -> per-person output dict returned
                from :meth:`run_person_inference`.
            meta: Dictionary with metadata, must include ``'num_frames'``.

        Returns:
            A list ``frame_results`` of length ``num_frames`` where each entry
            is a dict mapping ``person_id`` to a small data dict with keys
            ``'verts'`` and ``'cam'`` (and optionally ``'color'``, etc.),
            suitable for the renderer.
        """
        num_frames = int(meta.get('num_frames'))
        # prepare_rendering_results does the heavy lifting (existing util)
        frame_results = prepare_rendering_results(vibe_results, num_frames)
    
        return frame_results

    def visualize_result(
        self,
        image_folder: str,
        frame_results: List[Dict],
        output_folder: str,
        show: bool = True,
    ) -> str:
        """Render meshes on frames and assemble a result video.

        Args:
            image_folder: Directory containing extracted frames (ordered).
            frame_results: List of per-frame dicts produced by
                :meth:`postprocess_output`.
            output_folder: Directory where rendered frames and video will be
                written. Will be created if it does not exist.
            show: If True, display frames in a window during rendering.

        Returns:
            Path to the saved video file (string).
        """
        os.makedirs(output_folder, exist_ok=True)

        # renderer uses SMPL faces from the model
        # faces attr assumed available at self.model.smpl.faces
        h, w = frame_results[0].get(list(frame_results[0].keys())[0], {}).get('img_size', (None, None))[:2] if len(frame_results) > 0 else (None, None)
        # fallback: try to infer from first image
        files = sorted([
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        def _num_key(p):
            name = Path(p).stem
            try:
                return int(name)
            except Exception:
                return name
        files = sorted(files, key=_num_key)
        if len(files) == 0:
            raise RuntimeError(f"No image files found in {image_folder}")

        # create renderer
        # use image size from first image if not provided
        sample_img = cv2.imread(files[0])
        rh, rw = sample_img.shape[:2]
        renderer = Renderer(
            resolution=(rw, rh), 
            orig_img=True, 
            wireframe=False, 
            faces=self.model.smpl.faces
        )

        mesh_color = {k: colorsys.hsv_to_rgb(np.random.rand(), 0.5, 1.0) for k in frame_results[0].keys()}

        for frame_idx, img_path in enumerate(tqdm(files, desc='rendering')):
            img = cv2.imread(img_path)
            persons = frame_results[frame_idx]
            for person_id, person_data in persons.items():
                frame_verts = person_data['verts']
                frame_cam = person_data['cam']
                color = mesh_color.get(person_id, (1.0, 1.0, 1.0))
                img = renderer.render(
                    img,
                    frame_verts,
                    cam=frame_cam,
                    color=color,
                    mesh_filename=None,
                )
            out_path = os.path.join(output_folder, f'{frame_idx:06d}.png')
            cv2.imwrite(out_path, img)
            if show:
                cv2.imshow('Video', img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break    
        cv2.destroyAllWindows()

        # assemble video
        vid_name = os.path.basename(image_folder.rstrip('/'))
        save_name = os.path.join(output_folder, f"{vid_name}_vibe_result.mp4")
        images_to_video(img_folder=output_folder, output_vid_file=save_name)
        return save_name


    def predict(self, video_path: str, detector=None, output_folder: Optional[str] = None) -> Dict:
        """Run the full VIBE pipeline on an input video.

        This is a convenience method that performs frame extraction,
        detection/tracking, per-person inference, postprocessing and
        visualization in sequence.

        Args:
            video_path: Path to the input video file.
            detector: Detector instance used by :meth:`process_detection`.
            output_folder: Optional path where visualization outputs will be
                written. If None, a default under ``out/`` is used.

        Returns:
            A dictionary with keys ``'vibe_results'``, ``'frame_results'`` and
            ``'video_path'`` providing the raw per-person outputs, the
            per-frame rendering structure and the path to the rendered video
            respectively.
        """
        image_folder, num_frames, img_shape = self.preprocess_video(self.config.media_source)
        tracking_results = self.process_detection(image_folder, detector)

        orig_height, orig_width = img_shape[:2]
        vibe_results = {}
        # ========= Run VIBE on each person ========= #
        for person_id in tqdm(list(tracking_results.keys())):
            bboxes = joints2d = None
            bboxes = tracking_results[person_id]['bbox']
            frames = tracking_results[person_id]['frames']

            output_dict = self.run_person_inference(
                image_folder=image_folder,
                frames=frames,
                bboxes=bboxes,
                joints2d=joints2d,
                orig_width=orig_width,
                orig_height=orig_height,
            )

            vibe_results[person_id] = output_dict

        # prepare results for rendering
        frame_results = self.postprocess_output(
            vibe_results,
            {'num_frames': num_frames, 'img_shape': img_shape}
        )

        # visualize
        if output_folder is None:
            output_folder = os.path.join('out', os.path.basename(os.path.splitext(video_path)[0]))
        video_path = self.visualize_result(
            image_folder, 
            frame_results, 
            output_folder, 
            show=True)
        self.logger.info(f'Saving result video to {video_path}')
        # shutil.rmtree(output_folder)

    def __repr__(self) -> str:
        return f"VIBE(device={self.device}, initialized={self.is_initialized})"


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

    _VIBE.predict(video_file, detector=_detector, output_folder=str(OUTPUT_FOLDER))

