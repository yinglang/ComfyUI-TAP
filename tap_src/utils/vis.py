import os
import imageio
import torch
import numpy as np
import subprocess
import colorsys
from matplotlib import cm
import cv2
from PIL import Image, ImageDraw


color_map = cm.get_cmap("jet")

# https://github.com/qianqianwang68/omnimotion/blob/main/viz.py
def vis_omnimotion_style(images, kpts_foreground, kpts_background, save_path):
    """
    images：（T, H, W, 3)
    kpts_foreground: shape=(T, N, 2)
    kpts_background: shape=(T, M, 2), 用于计算偏移的
    
    This function calculates the median motion of the background, which is subsequently
    subtracted from the foreground motion. This subtraction process "stabilizes" the camera and
    improves the interpretability of the foreground motion trails.
    """
    kpts_foreground = kpts_foreground[:, ::1]  # can adjust kpts sampling rate here
    num_imgs, num_pts = kpts_foreground.shape[:2]

    frames = []
    for i in range(num_imgs):
        # 按照背景参考点偏移的中值作为偏移
        kpts = kpts_foreground - np.median(kpts_background - kpts_background[i], axis=1, keepdims=True)
        img_curr = images[i]
        for t in range(i):
            img1 = img_curr.copy()
            
            # changing opacity，不同时间的点的连线使用不同的权重，最终加到图片上就是渐变色
            alpha = max(1 - 0.9 * ((i - t) / ((i + 1) * .99)), 0.1)
            for j in range(num_pts):
                color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255
                color_alpha = 1
                hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
                color = colorsys.hsv_to_rgb(hsv[0], hsv[1]*color_alpha, hsv[2])

                pt1 = kpts[t, j]
                pt2 = kpts[t+1, j]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                cv2.line(img1, p1, p2, color, thickness=1, lineType=16)
            img_curr = cv2.addWeighted(img1, alpha, img_curr, 1 - alpha, 0)

        for j in range(num_pts):
            color = np.array(color_map(j/max(1, float(num_pts - 1)))[:3]) * 255
            pt1 = kpts[i, j]
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            cv2.circle(img_curr, p1, 2, color, -1, lineType=16)

        frames.append(img_curr)

    imageio.mimwrite(save_path, frames, quality=90, fps=16)


def add_points_to_frames(images, query_points, radius=3):
    for image, points in zip(images, query_points):
        draw = ImageDraw.Draw(image)
        for x, y in points:
            draw.ellipse([(x-radius, y-radius), (x+radius, y+radius)], fill="red")
