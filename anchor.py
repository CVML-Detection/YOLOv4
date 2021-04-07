import numpy as np
import torch
from math import sqrt
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from config import device

import config as cfg
from utils import cxcy_to_xy


class Anchor(metaclass=ABCMeta):
    def __init__(self, model_name='yolo'):
        self.model_name = model_name.lower()
        assert model_name in ['yolo', 'ssd', 'retina', 'yolov3']

    @abstractmethod
    def create_anchors(self):
        pass


class YOLOv4_Anchor(Anchor):
    def __init__(self):
        super().__init__()

    def anchor_for_scale(self, anchor, stride, img_size):
        # Anchor (X, Y)
        stride = int(stride.item())
        grid_size = int(img_size/stride)
        # print('> creating Anchor...')
        # print('stride:{}'.format(stride))
        # print('grid_size:{}'.format(grid_size))
        
        grid_arange = np.arange(grid_size)
        xx, yy = np.meshgrid(grid_arange, grid_arange)
        d1 = np.expand_dims(xx, axis=-1)
        d2 = np.expand_dims(yy, axis=-1)
        m_grid = np.concatenate([d1,d2], axis=-1)
        m_grid = m_grid + 0.5                               # (g,g,2)
        xy = torch.from_numpy(m_grid)

        # Get xy & wh
        xy = xy.view(grid_size, grid_size, 1, 2).expand(grid_size, grid_size, 3, 2).type(torch.float32)  # centor ([g,g,2]=>[g,g,3,2])
        wh = anchor.view(1, 1, 3, 2).expand(grid_size, grid_size, 3, 2).type(torch.float32)  # w, h ([3,2]=>[g,g,3,2])
        wh = wh * stride
        center_anchors = torch.cat([xy, wh], dim=3).to(device)  # [g,g,3,4]
        return center_anchors


    def create_anchors(self, anchors, strides, img_size):

        center_anchors_0 = self.anchor_for_scale(anchors[0], strides[0], img_size)    #stage0
        center_anchors_1 = self.anchor_for_scale(anchors[1], strides[1], img_size)    #stage1
        center_anchors_2 = self.anchor_for_scale(anchors[2], strides[2], img_size)    #stage2

        return [center_anchors_0, center_anchors_1, center_anchors_2]


if __name__ == "__main__":
    anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
    strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
    img_size = cfg.MODEL["INPUT_IMG_SIZE"]

    ANCHOR_ = YOLOv4_Anchor()
    anchor_c = ANCHOR_.create_anchors(anchors, strides, img_size)

    print(anchor_c[0].shape)
    print(anchor_c[0][32][32])