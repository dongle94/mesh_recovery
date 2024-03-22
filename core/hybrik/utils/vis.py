import cv2
import numpy as np


def draw_heatmap(image, joints, indices, colors, mask=None, height=None):
    if height is None:
        height = image.shape[0]

    base = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    colors = get_heatmap_colors(colors)
    for idx, pt in enumerate(joints):
        if idx in indices:
            x, y = pt
            base = cv2.circle(base, (int(x), int(y)), int(height / 30), colors[idx][0], -1)
            base = cv2.circle(base, (int(x), int(y)), int(height / 45), colors[idx][1], -1)
            base = cv2.circle(base, (int(x), int(y)), int(height / 65), colors[idx][2], -1)
            base = cv2.circle(base, (int(x), int(y)), int(height / 100), colors[idx][3], -1)
    if mask is not None:
        frame = cv2.add((0.3 * base * mask).astype(np.uint8), image.astype(np.uint8))
    else:
        frame = cv2.add((0.3 * base).astype(np.uint8), image.astype(np.uint8))
    return frame


def get_heatmap_colors(string_color: dict):
    colors = {}
    for k, v in string_color.items():
        if v == 'red':
            colors[k] = [(0, 0, 64), (0, 0, 128), (0, 0, 192), (0, 0, 255)]
        elif v == 'green':
            colors[k] = [(0, 64, 0), (0, 128, 0), (0, 192, 0), (0, 255, 0)]
        elif v == 'blue':
            colors[k] = [(64, 0, 0), (128, 0, 0), (192, 0, 0), (255, 0, 0)]
        else:
            colors[k] = [(0, 0, 64), (0, 0, 128), (0, 0, 192), (0, 0, 255)]

    return colors
