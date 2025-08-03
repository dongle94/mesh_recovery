# -*- coding: utf-8 -*-
"""
SPIN: SMPL IteratiVe Network for 3D Human Pose and Shape Recovery
"""

import sys
import cv2
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from torchvision.transforms import Normalize

FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from utils.logger import get_logger
from core.spin.models import hmr, SMPL
from core.spin.utils.imutils import crop
from core.spin.utils.renderer import Renderer
import core.spin.constants as constants


class SPIN(nn.Module):
    """
    SPIN (SMPL IteratiVe Network) class for 3D human pose and shape estimation.
    
    This class implements the SPIN model that iteratively refits SMPL parameters
    to improve 3D human pose and shape estimation from RGB images.
    
    Args:
        config (dict): Configuration dictionary containing model parameters
        device (str, optional): Device to run the model on. Defaults to 'cuda'.
        logger (Logger, optional): Logger instance for logging. Defaults to None.
    
    Attributes:
        config (dict): Model configuration
        device (str): Computing device
        logger (Logger): Logger instance
        model (nn.Module): The underlying neural network model
        smpl (SMPL): SMPL model instance
        is_initialized (bool): Whether the model is properly initialized
    """
    
    def __init__(self, config: Dict, device: str = 'cuda', logger=None):
        super(SPIN, self).__init__()
        
        self.config = config
        self.device = torch.device('cuda') if device == 'cuda' else torch.device('cpu')
        self.logger = get_logger() if logger is None else logger
        self.is_initialized = False
        
        # Model components
        self.model = None
        self.smpl = None
        
        # Initialize the model
        self._build_model()

        # Setup renderer for visualization
        self.renderer = Renderer(focal_length=constants.FOCAL_LENGTH, img_res=constants.IMG_RES, faces=self.smpl.faces)
        
    def _build_model(self):
        """
        Build the SPIN model architecture.
        This method should initialize the backbone network and regression heads.
        """
        try:
            # Initialize HMR model
            self.model = hmr(self.config.spin_smpl_mean_params).to(self.device)
            
            # Load pretrained weights if checkpoint path is provided
            if hasattr(self.config, 'spin_checkpoint') and self.config.spin_checkpoint:
                checkpoint = torch.load(self.config.spin_checkpoint, map_location=self.device, weights_only=False)
                self.model.load_state_dict(checkpoint['model'], strict=False)
                if self.logger:
                    self.logger.info(f"Loaded SPIN checkpoint from {self.config.spin_checkpoint}")
            
            # Load SMPL model
            self.smpl = SMPL(
                joint_regressor_extra_path=self.config.spin_joint_regressor_extra,
                model_path=self.config.spin_smpl_model_dir,
                batch_size=1,
                create_transl=False).to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            self.is_initialized = True
            
            if self.logger:
                self.logger.info("SPIN model initialized successfully")
                
        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize SPIN model: {e}")
            self.is_initialized = False
            raise
    
    def preprocess_image(self, image: np.ndarray, bbox: Optional[List] = None, input_res: int = 224) -> torch.Tensor:
        """
        Preprocess input image for model inference.
        
        Args:
            image (np.ndarray): Input image in BGR/RGB format
            bbox (List, optional): Bounding box coordinates [x, y, w, h]
            input_res (int): Input resolution for the model. Defaults to 224.
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        
        # Convert BGR to RGB if needed (assuming input is BGR from OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            img = image[:,:,::-1].copy()  # BGR to RGB
        else:
            img = image.copy()
        
        if bbox is None:
            # Assume that the person is centered in the image
            height = img.shape[0]
            width = img.shape[1]
            center = np.array([width // 2, height // 2])
            scale = max(height, width) / 200.0
        else:
            # Convert bbox from [x, y, w, h] to center and scale
            x, y, w, h = bbox
            center = np.array([x + w * 0.5, y + h * 0.5])
            scale = max(w, h) / 200.0
        
        # Crop and resize image
        img = crop(img, center, scale, (input_res, input_res))
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Convert to tensor and change dimension order (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        # Normalize with ImageNet stats
        norm_img = normalize_img(img.clone())[None]  # Add batch dimension
        
        return img, norm_img

    def forward(self, img, images: torch.Tensor, bbox: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the SPIN model.
        
        Args:
            images (torch.Tensor): Batch of input images
            bbox (torch.Tensor, optional): Bounding box coordinates
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing model outputs:
                - 'theta': SMPL parameters (pose + shape + camera)
                - 'vertices': 3D mesh vertices
                - 'joints': 3D joint coordinates
                - 'joints_2d': 2D joint projections
        """
        with torch.no_grad():
            pred_rotmat, pred_betas, pred_camera = self.model(images.to(self.device))
            pred_output = self.smpl(betas=pred_betas, body_pose=pred_rotmat[:,1:], global_orient=pred_rotmat[:,0].unsqueeze(1), pose2rot=False)
            pred_vertices = pred_output.vertices

        # Calculate camera parameters for rendering
        camera_translation = torch.stack([pred_camera[:,1], pred_camera[:,2], 2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)],dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()
        pred_vertices = pred_vertices[0].cpu().numpy()
        img = img.permute(1,2,0).cpu().numpy()

        
        # Render parametric shape
        img_shape = self.renderer(pred_vertices, camera_translation, img)
        cv2.imshow('Rendered Image', img_shape)
        cv2.waitKey(0)
    
    def predict(self, image: np.ndarray, bbox: Optional[List] = None) -> Dict[str, np.ndarray]:
        """
        Predict 3D pose and shape from a single image.
        
        Args:
            image (np.ndarray): Input image
            bbox (List, optional): Bounding box [x, y, w, h]
            
        Returns:
            Dict[str, np.ndarray]: Prediction results containing:
                - 'vertices': 3D mesh vertices
                - 'joints': 3D joint coordinates
                - 'pose': SMPL pose parameters
                - 'shape': SMPL shape parameters
                - 'camera': Camera parameters
        """
        # TODO: Implement single image prediction
        pass
    
    def iterative_fitting(self, image: torch.Tensor, initial_params: Dict[str, torch.Tensor], 
                         num_iterations: int = 3) -> Dict[str, torch.Tensor]:
        """
        Perform iterative fitting to refine SMPL parameters.
        
        Args:
            image (torch.Tensor): Input image tensor
            initial_params (Dict[str, torch.Tensor]): Initial SMPL parameters
            num_iterations (int): Number of iterations for refinement
            
        Returns:
            Dict[str, torch.Tensor]: Refined SMPL parameters
        """
        # TODO: Implement iterative fitting logic
        pass
    
    def postprocess_output(self, output: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
        """
        Postprocess model output for visualization or further processing.
        
        Args:
            output (Dict[str, torch.Tensor]): Raw model output
            
        Returns:
            Dict[str, np.ndarray]: Postprocessed results
        """
        # TODO: Implement output postprocessing
        pass
    
    def visualize_result(self, image: np.ndarray, vertices: np.ndarray, 
                        joints: np.ndarray, camera: np.ndarray) -> np.ndarray:
        """
        Visualize the prediction result on the input image.
        
        Args:
            image (np.ndarray): Original input image
            vertices (np.ndarray): 3D mesh vertices
            joints (np.ndarray): 3D joint coordinates
            camera (np.ndarray): Camera parameters
            
        Returns:
            np.ndarray: Image with overlaid 3D mesh and joints
        """
        # TODO: Implement visualization

    
    def __repr__(self):
        return f"SPIN(device={self.device}, initialized={self.is_initialized})"


if __name__ == "__main__":
    from utils.logger import init_logger
    from utils.config import set_config, get_config

    set_config('./configs/config.yaml')
    cfg = get_config()

    init_logger(cfg)

    _spin = SPIN(config=cfg, device=cfg.spin_device, logger=get_logger())

    _im = cv2.imread('./data/images/army.png')

    _im0, _norm_im = _spin.preprocess_image(_im, bbox=None)
    _output = _spin.forward(_im0, _norm_im)