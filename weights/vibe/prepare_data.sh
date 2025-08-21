#!/usr/bin/env bash

gdown "https://drive.google.com/uc?id=1untXhYOLQtpNEy4GTY_0fL_H-k6cTf_r"
unzip vibe_data.zip
rm vibe_data.zip
cd ..
mv vibe_data/sample_video.mp4 .
mv vibe_data/yolov3.weights .
