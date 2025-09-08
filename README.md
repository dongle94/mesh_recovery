# Mesh Recovery

## Structure
### Models
- **Object Detector**: YOLO for human detection
- **Mesh Recovery**: 
  - SPIN for 3D human pose and shape estimation
  - HybrIK-X for 3D human mesh reconstruction

## Installation
```shell
conda create -n mr python==3.10 -y
conda activate mr
./install.sh
```

## Model Weights
Download model weights and organize them as shown below for proper functionality:

```
weights/
├── yolov11/
│   └── yolo11m.pt
├── smpl/
│   ├── ...
│   └── SMPL_NEUTRAL.pkl
├── smplx/
│   ├── SMPLX_FEMALE.pkl
│   ├── SMPLX_MALE.pkl
│   └── SMPLX_NEUTRAL.pkl
├── spin/
│   ├── data/
│   │   ├── ...
│   │   └── smpl_mean_params.npz
│   └── hmr.pt
├── vibe/
│   ├── prepare_data.sh
│   ├── ...
│   ├── vibe_model_w_3dpw.pth.tar
│   └── vibe_model_wo_3dpw.pth.tar
├── hybrik/
    └──hybrikx_rle_hrnet.pth
```

Configure paths in `./configs/config.yaml` to match your setup.


## Third-party Licenses

This project incorporates code from SPIN:
- **SPIN (SMPL oPtimization IN the loop)** 
- Copyright (c) 2019, University of Pennsylvania, Max Planck Institute for Intelligent Systems
- Licensed under BSD 3-Clause License
- See [LICENSE_SPIN](core/spin/LICENSE) for full license text

This project also incorporates code from VIBE:
- **VIBE (Video Inference for Human Body Pose and Shape Estimation)**
- Copyright (c) 2019, Max Planck Institute for Intelligent Systems
- Licensed for non-commercial scientific research and education use
- See [LICENSE_VIBE](core/vibe/LICENSE) for full license text