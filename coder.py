import numpy as np
import torch
import torch.nn.functional as F
from math import sqrt
from abc import ABCMeta, abstractmethod
import config as cfg


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
        self.num_anchors = len(self.anchors[0])

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

        batch_size = len(gt_boxes)
        for b in range(batch_size):



        return 1, 2, 3, 4, 5, 6

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
