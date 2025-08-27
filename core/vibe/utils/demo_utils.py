import os
import cv2
import subprocess
import numpy as np


def video_to_images(vid_file, img_folder=None, return_info=False):
    if img_folder is None:
        img_folder = os.path.join('/tmp', os.path.basename(vid_file).replace('.', '_'))

    os.makedirs(img_folder, exist_ok=True)

    command = ['ffmpeg',
               '-i', vid_file,
               '-f', 'image2',
               '-v', 'error',
               f'{img_folder}/%06d.png']
    print(f'Running \"{" ".join(command)}\"')
    subprocess.call(command)

    print(f'Images saved to \"{img_folder}\"')

    img_shape = cv2.imread(os.path.join(img_folder, '000001.png')).shape

    if return_info:
        return img_folder, len(os.listdir(img_folder)), img_shape
    else:
        return img_folder


def convert_crop_cam_to_orig_img(cam, bbox, img_width, img_height):
    '''
    Convert predicted camera from cropped image coordinates
    to original image coordinates
    :param cam (ndarray, shape=(3,)): weak perspective camera in cropped img coordinates
    :param bbox (ndarray, shape=(4,)): bbox coordinates (c_x, c_y, h)
    :param img_width (int): original image width
    :param img_height (int): original image height
    :return:
    '''
    cx, cy, h = bbox[:,0], bbox[:,1], bbox[:,2]
    hw, hh = img_width / 2., img_height / 2.
    sx = cam[:,0] * (1. / (img_width / h))
    sy = cam[:,0] * (1. / (img_height / h))
    tx = ((cx - hw) / hw / sx) + cam[:,1]
    ty = ((cy - hh) / hh / sy) + cam[:,2]
    orig_cam = np.stack([sx, sy, tx, ty]).T
    return orig_cam


def convert_crop_coords_to_orig_img(bbox, keypoints, crop_size):
    # import IPython; IPython.embed(); exit()
    cx, cy, h = bbox[:, 0], bbox[:, 1], bbox[:, 2]

    # unnormalize to crop coords
    keypoints = 0.5 * crop_size * (keypoints + 1.0)

    # rescale to orig img crop
    keypoints *= h[..., None, None] / crop_size

    # transform into original image coords
    keypoints[:,:,0] = (cx - h/2)[..., None] + keypoints[:,:,0]
    keypoints[:,:,1] = (cy - h/2)[..., None] + keypoints[:,:,1]
    return keypoints