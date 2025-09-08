# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

import os
import cv2
import numpy as np
import math

import torch
from torch.utils.data import Dataset

from core.vibe.utils.smooth_bbox import get_all_bbox_params
from core.vibe.data_utils.img_utils import get_single_image_crop_demo


class Inference(Dataset):
    def __init__(self, image_folder, frames, bboxes=None, joints2d=None, scale=1.0, crop_size=224, seqlen: int = 16):
        self.image_file_names = [
            os.path.join(image_folder, x)
            for x in os.listdir(image_folder)
            if x.endswith('.png') or x.endswith('.jpg')
        ]
        self.image_file_names = sorted(self.image_file_names)
        n_imgs = len(self.image_file_names)
        valid_mask = (frames >= 0) & (frames < n_imgs)
        frames = frames[valid_mask]
        
        self.image_file_names = np.array(self.image_file_names)[frames]
        self.bboxes = bboxes
        self.joints2d = joints2d
        self.scale = scale
        self.crop_size = crop_size

        self.frames = frames
        self.has_keypoints = True if joints2d is not None else False
        self.seqlen = int(seqlen)

        self.norm_joints2d = np.zeros_like(self.joints2d)

        if self.has_keypoints:
            bboxes, time_pt1, time_pt2 = get_all_bbox_params(joints2d, vis_thresh=0.3)
            bboxes[:, 2:] = 150. / bboxes[:, 2:]
            self.bboxes = np.stack([bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 2]]).T

            # 재인덱싱된 배열에 time_pt1:time_pt2 적용
            self.image_file_names = self.image_file_names[time_pt1:time_pt2]
            self.joints2d = joints2d[time_pt1:time_pt2]
            self.frames = frames[time_pt1:time_pt2]
        else:
            self.norm_joints2d = None
        
        # 마지막 시퀀스 패딩을 위해 ceil 사용
        self.N = len(self.image_file_names)
        self.length = math.ceil(self.N / self.seqlen)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        start_idx = idx * self.seqlen
        end_idx = min(start_idx + self.seqlen, self.N)
        idxs = list(range(start_idx, end_idx))
        if len(idxs) < self.seqlen:
            idxs += [self.N - 1] * (self.seqlen - len(idxs))
 
        seq_imgs = []

        for i in idxs:
            img = cv2.cvtColor(cv2.imread(self.image_file_names[i]), cv2.COLOR_BGR2RGB)
            bbox = self.bboxes[i]
            j2d = self.joints2d[i] if self.has_keypoints else None

            norm_img, raw_img, kp_2d = get_single_image_crop_demo(
                img,
                bbox,
                kp_2d=j2d,
                scale=self.scale,
                crop_size=self.crop_size)

            seq_imgs.append(norm_img)
    
        seq_imgs = torch.stack(seq_imgs, dim=0)

        return seq_imgs
