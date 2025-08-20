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
    
    def preprocess_image(self, image: np.ndarray, bbox: Optional[List] = None, input_res: int = 224) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess input image for model inference.
        
        Args:
            image (np.ndarray): Input image in BGR format
            bbox (List, optional): Bounding box coordinates [x, y, w, h]
            input_res (int): Input resolution for the model. Defaults to 224.
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Preprocessed image tensors containing:
                - img: Unnormalized image tensor (3, 224, 224) 
                - norm_img: Normalized image tensor for model input (1, 3, 224, 224)
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

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the SPIN model.
        
        Args:
            images (torch.Tensor): Batch of preprocessed input images with shape (B, C, H, W)

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Model outputs containing:
                - pred_rotmat: SMPL pose parameters as rotation matrices
                - pred_betas: SMPL shape parameters 
                - pred_camera: Camera parameters
        """
        with torch.no_grad():
            output = self.model(images.to(self.device))

        return output
    
    def postprocess_output(self, output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[object, np.ndarray]:
        """
        Postprocess model output for visualization or further processing.
        
        Args:
            output (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Raw model output tuple containing:
                - pred_rotmat: SMPL pose parameters as rotation matrices
                - pred_betas: SMPL shape parameters 
                - pred_camera: Camera parameters
                
        Returns:
            Tuple[object, np.ndarray]: Postprocessed results containing:
                - pred_output: SMPL model output with vertices and mesh data
                - camera_translation: Camera translation parameters for rendering
        """
        # Unpack the output
        pred_rotmat, pred_betas, pred_camera = output
        pred_output = self.smpl(
            betas=pred_betas, 
            body_pose=pred_rotmat[:,1:], 
            global_orient=pred_rotmat[:,0].unsqueeze(1), 
            pose2rot=False
        )

        # Calculate camera parameters for renderer (tx, ty, depth)
        camera_translation = torch.stack([
            pred_camera[:,1], 
            pred_camera[:,2], 
            2*constants.FOCAL_LENGTH/(constants.IMG_RES * pred_camera[:,0] +1e-9)
        ], dim=-1)
        camera_translation = camera_translation[0].cpu().numpy()

        # Keep raw pred_camera (s, tx, ty) for 2D projection
        pred_camera_raw = pred_camera[0].cpu().numpy()

        return pred_output, camera_translation, pred_camera_raw
    
    @staticmethod
    def project_joints_weak(joints_3d: np.ndarray, pred_camera_raw: np.ndarray, img_res: int = constants.IMG_RES) -> np.ndarray:
        """
        Project 3D joints to 2D using weak-perspective camera predicted by network.
        pred_camera_raw: [s, tx, ty]
        - If tx,ty are in [-1,1] (normalized), convert to pixel coords.
        - Otherwise assume tx,ty are already in pixel units.
        """
        s, tx, ty = float(pred_camera_raw[0]), float(pred_camera_raw[1]), float(pred_camera_raw[2])
        proj = s * joints_3d[:, :2] + np.array([tx, ty])

        # heuristic: if translation small (~[-1,1]) treat as normalized coord -> convert to pixels
        if abs(tx) <= 1.5 and abs(ty) <= 1.5:
            proj = (proj + 1.0) * (img_res / 2.0)

        return proj

    def visualize_result(self, image: torch.Tensor, pred_output, camera_translation: np.ndarray, pred_camera_raw: np.ndarray = None) -> np.ndarray:
        """
        Visualize the prediction result on the input image.
        
        Args:
            image (torch.Tensor): Preprocessed image tensor with shape (3, 224, 224) in RGB format
            pred_output: SMPL model output containing vertices and other mesh data
            camera_translation (np.ndarray): Camera translation parameters for rendering
            
        Returns:
            np.ndarray: Rendered image with overlaid 3D mesh in BGR format
        """
        # Convert SMPL output vertices to numpy array
        pred_vertices = pred_output.vertices[0].cpu().numpy()

        # Convert tensor (RGB) to numpy (BGR) for rendering
        img = image.permute(1,2,0).cpu().numpy()[:,:,::-1]

        joints_3d = pred_output.joints[0].cpu().numpy()  # (N_joints, 3)

        j2d_px = self.project_joints_weak(joints_3d, pred_camera_raw, img_res=constants.IMG_RES)

        # optional: clip within image
        j2d_px[:, 0] = np.clip(j2d_px[:, 0], 0, img.shape[1]-1)
        j2d_px[:, 1] = np.clip(j2d_px[:, 1], 0, img.shape[0]-1)
        
        # now j2d_px are pixel coordinates in the cropped/resized image (IMG_RES x IMG_RES)
        img_pose = img.copy()
        for i in range(j2d_px.shape[0]):
            cv2.circle(img_pose, (int(j2d_px[i, 0]), int(j2d_px[i, 1])), 2, (0, 255, 0), -1)

        img_shape = self.renderer(pred_vertices, camera_translation, img)

        return img_shape, img_pose


    def predict(self, image: np.ndarray, bbox: Optional[List] = None) -> Tuple[torch.Tensor, object, np.ndarray]:
        """
        Predict 3D pose and shape from a single image.
        
        Args:
            image (np.ndarray): Input image in BGR format
            bbox (List, optional): Bounding box coordinates [x, y, w, h]
            
        Returns:
            Tuple[torch.Tensor, object, np.ndarray]: Prediction results containing:
                - im: Preprocessed image tensor (3, 224, 224)
                - pred_output: SMPL model output with vertices and mesh data
                - cam_transl: Camera translation parameters for rendering
        """
        # Preprocess the input image
        im, norm_im = self.preprocess_image(image, bbox=bbox)

        # Infer the model
        output = self.forward(norm_im)

        # Postprocess the output
        pred_output, cam_transl, pred_camera_raw = self.postprocess_output(output)

        return im, pred_output, cam_transl, pred_camera_raw


    def __repr__(self):
        return f"SPIN(device={self.device}, initialized={self.is_initialized})"


if __name__ == "__main__":
    from utils.logger import init_logger
    from utils.config import set_config, get_config
    from core.media_loader import MediaLoader
    from core.obj_detector import ObjectDetector

    set_config('./configs/config.yaml')
    _cfg = get_config()

    init_logger(_cfg)
    _logger = get_logger()

    _media_loader = MediaLoader(_cfg.media_source,
                               logger=_logger,
                               realtime=_cfg.media_realtime,
                               bgr=_cfg.media_bgr,
                               opt=_cfg)
    _detector = ObjectDetector(cfg=_cfg)
    _wt = 0 if _media_loader.is_imgs is True else 1
    _spin = SPIN(config=_cfg, device=_cfg.spin_device, logger=_logger)

    try:
        while True:
            _frame = _media_loader.get_frame()
            _det = _detector.run(_frame)

            d = _det[0] if len(_det) > 0 else None
            if d is not None:
                x1, y1, x2, y2 = map(int, d[:4])


                _im, _pred_output, _cam_transl, _pred_camera_raw = _spin.predict(_frame, bbox=[x1, y1, x2 - x1, y2 - y1])

                # Visualize the result
                _img_res, _img_pose = _spin.visualize_result(_im, _pred_output, _cam_transl, _pred_camera_raw)
                cv2.imshow('Rendered Image', _img_res)
                cv2.imshow('Pose Image', _img_pose)

                # Draw bounding box and label on the original frame
                cls = int(d[5])
                cv2.rectangle(_frame, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
                cv2.putText(_frame, str(_detector.names[cls]), (x1, y1+20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (96, 96, 96), thickness=1, lineType=cv2.LINE_AA)
            else:
                cv2.destroyWindow('Rendered Image')
            cv2.imshow('Original Image', _frame)
            
            # Display the original frame with bounding box
            if cv2.waitKey(_wt) == ord('q'):
                break

    except KeyboardInterrupt:
        _logger.info("Process interrupted by user")
    except Exception as e:
        _logger.error(f"Error during processing: {e}")
    finally:
        cv2.destroyAllWindows()
        _logger.info("Application closed")
