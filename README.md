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