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
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
	sys.path.append(str(ROOT))

from utils.logger import get_logger
from core.vibe.models.vibe import VIBE_Demo


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
            batch_size=64,
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


if __name__ == "__main__":
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    OUTPUT_FOLDER = Path("out") / os.path.basename(_cfg.media_source)

    _VIBE = VIBE(_cfg, device=_cfg.device, logger=_logger)