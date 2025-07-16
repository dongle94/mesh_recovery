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
