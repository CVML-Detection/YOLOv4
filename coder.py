import numpy as np
import torch
import torch.nn.functional as F
from math import sqrt
from abc import ABCMeta, abstractmethod

from utils import cxcy_to_xy
from anchor import YOLOv4_Anchor
import config as cfg
from config import device


class Coder(metaclass=ABCMeta):
    @abstractmethod
    def encode(self):
        pass
    def decode(self):
        pass


class YOLOv4_Coder(Coder):
    def __init__(self, data_type):
        super().__init__()
        self.data_type = data_type
        self.anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.img_size = cfg.MODEL["INPUT_IMG_SIZE"]
        self.num_anchors = len(self.anchors[0])     # 3

        self.ANCHOR_ = YOLOv4_Anchor()
        self.c_anchor = self.ANCHOR_.create_anchors(self.anchors, self.strides, self.img_size)

        assert self.data_type in ['voc', 'coco']
        if self.data_type == 'voc':
            self.num_classes = 20
        elif self.data_type == 'coco':
            self.num_classes = 80


    def encode(self, gt_boxes, gt_labels, stage):
        """
        :param gt_boxes (list)  :   (N,4)
        :param gt_labels (list) :   (N)
        :len(gt_boxes) : 4
        """
        # print('len(boxes) : {}'.format(len(boxes)))
        # print('boxes: {}'.format(boxes[0].shape))
        # print('labels: {}'.format(labels[0].shape))
        # print('ANCHORS:{}'.format(self.anchors[0]))
        self.assign_anchors_to_device()

        stride = int(self.strides[stage].item())
        grid_size = int(self.img_size/stride)

        print('> In STAGE : {}'.format(stage))
        print('\t>> stride : {} & grid_size : {}'.format(stride, grid_size))
        print('\t>> c_anchor[{}]:{}'.format(stage, self.c_anchor[stage].shape))

        center_anchor = self.c_anchor[stage]
        corner_anchor = cxcy_to_xy(center_anchor).view(grid_size*grid_size*self.num_anchors, 4)     # (64, 64, 3, 4)
        batch_size = len(gt_boxes)

        # (순서대로 x, y, w, h, conf(1), mix(1), cls(80))
        gt_prop_txty = torch.zeros([batch_size, grid_size, grid_size, 3, 2])    # a proportion between (0 ~ 1) in a cell
        gt_twth = torch.zeros([batch_size, grid_size, grid_size, 3, 2])     # ratio of gt box and anchor box
        gt_objectness = torch.zeros([batch_size, grid_size, grid_size, 3, 1])   # maximum iou anchor (a obj assign a anc)
        ignore_mask = torch.zeros([batch_size, grid_size, grid_size, 3])
        gt_classes = torch.zeros([batch_size, grid_size, out_grid_sizesize, 3, self.num_classes])   # one-hot encoded class label


        for b in range(batch_size):
            break



        return 1, 2

    def decode(self, p, stage):
        p = p.view(
            p.shape[0],
            self.num_anchors,
            5+self.num_classes,
            p.shape[-1],
            p.shape[-1]).permute(0, 3, 4, 1, 2)     # [b, 255, 64, 64] => [b, 64, 64, 3, 85]

        pred = p.clone()
        
        batch_size, output_size = pred.shape[:2]        # output_size = 64, 32, 16
        device = pred.device
        anchors = (1.0 * self.anchors[stage]).to(device)
        stride = self.strides[stage]

        # 85 -> 2 + 2 + 1 + 80
        conv_raw_dxdy = pred[:, :, :, :, 0:2]       # [b, 64, 64, 3, 2]
        conv_raw_dwdh = pred[:, :, :, :, 2:4]       # [b, 64, 64, 3, 2]
        conv_raw_conf = pred[:, :, :, :, 4:5]       # [b, 64, 64, 3, 1]
        conv_raw_prob = pred[:, :, :, :, 5:]        # [b, 64, 64, 3, c]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)    # torch.Size([64, 64])  row0 : [0, 1 ... 64]
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)    # torch.Size([64, 64])  row0 : [0, 0 ... 0]
        grid_xy = torch.stack([x,y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)   # [b, 64, 64, 3, 2] 로 복사

        # print('conv_raw_dxdy.shape : {}'.format(conv_raw_dxdy.shape))
        # print('grid_xy.shape : {}'.format(grid_xy.shape))


        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride

        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob], dim=-1)

        # pred_bbox = pred_bbox.view(-1, 85)

        # return prediction값 , decode된 값
        return (p, pred_bbox)       # 둘 다 [b, o, o, 3, 85]    (o = 64, 32, 16)

            
    def assign_anchors_to_device(self):
        self.c_anchor[0] = self.c_anchor[0].to(device)
        self.c_anchor[1] = self.c_anchor[1].to(device)
        self.c_anchor[2] = self.c_anchor[2].to(device)


    def assign_anchors_to_device(self):
        self.c_anchor[0] = self.c_anchor[0].to('cpu')
        self.c_anchor[1] = self.c_anchor[1].to('cpu')
        self.c_anchor[2] = self.c_anchor[2].to('cpu')
        

