# mesh_recovery


## Structure
### Model
- Object Detector
  - YoloV5
- Mesh Recorvery
  - HybrIK-X

### Support Task
- VideoInputTask
- ObjeectDetectionTask
- HybrIKTask(HybrIK-X)
- PostHybrIKTask(visualization)

## Install
```shell
conda create -n mr python==3.8.x -y
conda activate mr
$ ./install.sh
```

download weights from release tab like below tree. And you should edit config(`./configs/config.yaml`)

```
.
└── weights
    ├── hybrikx_rle_hrnet.pth
    ├── smplx
    │   ├── SMPLX_FEMALE.npz
    │   ├── SMPLX_FEMALE.pkl
    │   ├── SMPLX_MALE.npz
    │   ├── SMPLX_MALE.pkl
    │   ├── SMPLX_NEUTRAL.npz
    │   ├── SMPLX_NEUTRAL.pkl
    │   ├── smplx_npz.zip
    │   └── version.txt
    ├── smplx_kid_template.npy
    ├── yolov5n.pt
    └── yolov5x6.pt
```