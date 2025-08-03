# Mesh Recovery

## Structure
### Models
- **Object Detector**: YOLO for human detection
- **Mesh Recovery**: HybrIK-X for 3D human mesh reconstruction

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
├── hybrikx_rle_hrnet.pth
├── smplx/
│   ├── SMPLX_FEMALE.pkl
│   ├── SMPLX_MALE.pkl
│   └── SMPLX_NEUTRAL.pkl
├── yolov5/
│   └── yolov5m.pt
└── yolov11/
    └── yolo11m.pt
```

Configure paths in `./configs/config.yaml` to match your setup.


## Third-party Licenses

This project incorporates code from SPIN:
- **SPIN (SMPL oPtimization IN the loop)** 
- Copyright (c) 2019, University of Pennsylvania, Max Planck Institute for Intelligent Systems
- Licensed under BSD 3-Clause License
- See [LICENSE_SPIN](core/spin/LICENSE) for full license text