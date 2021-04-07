import numpy as np
import torch
import torch.nn.functional as F
from math import sqrt
from abc import ABCMeta, abstractmethod

from anchor import YOLOv4_Anchor
import config as cfg
from config import device
from utils import cxcy_to_xy, xy_to_cxcy, find_jaccard_overlap


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

            
    def assign_anchors_to_device(self):
        for i in range(3):
            self.c_anchor[i] = self.c_anchor[i].to(device)
            self.xy_anchor[i] = self.xy_anchor[i].to(device)


    def assign_anchors_to_cpu(self):
        for i in range(3):
            self.c_anchor[i] = self.c_anchor[i].to('cpu')
            self.xy_anchor[i] = self.xy_anchor[i].to('cpu')


    def encode(self, gt_boxes, gt_labels):
        """
        :param gt_boxes (list)  :   (N,4)
        :param gt_labels (list) :   (N)
        :len(gt_boxes) : 4
        """
        batch_size = len(gt_boxes)
        stride=[]               # [8, 16, 32]   for each stage 0, 1, 2
        grid_size=[]            # [64, 32, 16]

        gt_ignore_mask = []
        gt_prop_txty = []
        gt_twth = []
        gt_objectness = []
        gt_classes = []

        self.xy_anchor=[]

        # Stage 0, 1, 2에 대해서
        for stg in range(3):
            stride.append(int(self.strides[stg].item()))
            grid_size.append(int(self.img_size/stride[stg]))
            
            # ---- 1. Container 만들기 ----
            gt_ignore_mask.append(torch.zeros([batch_size, grid_size[stg], grid_size[stg], self.num_anchors]))      # [b, 64, 64, 3]
            gt_prop_txty.append(torch.zeros([batch_size, grid_size[stg], grid_size[stg], self.num_anchors, 2]))
            gt_twth.append(torch.zeros([batch_size, grid_size[stg], grid_size[stg], self.num_anchors, 2]))
            gt_objectness.append(torch.zeros([batch_size, grid_size[stg], grid_size[stg], self.num_anchors, 1]))
            gt_classes.append(torch.zeros([batch_size, grid_size[stg], grid_size[stg], self.num_anchors, self.num_classes]))

            # ---- 2. Anchor 만들기 ----
            # self.c_anchor && self.xy_anchor      ex) Stage 0 Anchor : xy_anchor[0]  //  Stage 1 Anchor : xy_anchor[1]
            self.xy_anchor.append(cxcy_to_xy(self.c_anchor[stg]).view(grid_size[stg]*grid_size[stg]*self.num_anchors, 4))

        self.assign_anchors_to_device()

        # For Each IMAGE in Batch
        for b in range(batch_size):
            label = gt_labels[b]                        # [N]
            corner_gt_box = gt_boxes[b]                 # [N, 4] 
            center_gt_box = xy_to_cxcy(corner_gt_box)   # [N, 4] -> 비율로 되있음 (0 ~ 1)
            num_obj = corner_gt_box.size(0)
            scaled_corner_gt_box = []
            scaled_center_gt_box = []
            iou_anchors_gt = []
            bxby = []
            proportion_of_xy = []
            bwbh = []

            for stg in range(3):
                scaled_corner_gt_box.append(corner_gt_box * float(grid_size[stg]))  # grid size로 맞춰줘 (0 ~ 64)
                scaled_center_gt_box.append(center_gt_box * float(grid_size[stg]))

                bxby.append(scaled_center_gt_box[stg][..., :2])          # [N, 2] cx cy
                proportion_of_xy.append(bxby[stg] - bxby[stg].floor())   # [N, 2] (0~1)
                bwbh.append(scaled_center_gt_box[stg][..., 2:])          # [N, 2] w h
                
                iou_anchors_gt.append(find_jaccard_overlap(self.xy_anchor[stg], scaled_corner_gt_box[stg]))      # 각 앵커들에 대한 IOU 계산 [gsxgsx3 , num_obj]
                iou_anchors_gt[stg] = iou_anchors_gt[stg].view(grid_size[stg], grid_size[stg], self.num_anchors, -1)       # [gs, gs, 3, num_obj]
            
            # For Each Object 각 gt bbox에 대해
            for n_obj in range(num_obj):
                # best_stg : 0 -> Stage 0 has best IoU
                # best_stg : 1 -> Stage 1 has best IoU
                # best_stg : 2 -> Stage 2 has best IoU
                best_stg = torch.FloatTensor(
                    [iou_anchors_gt[0][..., n_obj].max(),
                    iou_anchors_gt[1][..., n_obj].max(),
                    iou_anchors_gt[2][..., n_obj].max()]).argmax()
                
                # 해당 gt box의 x, y
                #   FIXME gt box 0~1에서 0~64가 되었고, 완벽한 크기 (512)가 아닌 소수점이 있는 상황에서 int로 변형, 
                #   FIXME 그 후 grid 내에서의 위치 (proportion) 값을 Container에 넣는다.
                cx, cy = bxby[best_stg][n_obj]
                cx = int(cx)
                cy = int(cy)
                max_iou, max_idx = iou_anchors_gt[best_stg][cy, cx, :, n_obj].max(0)    # anchor 3 개중에 IoU 가 큰 것 쓰기
                gt_prop_txty[best_stg][b, cy, cx, max_idx, :] = proportion_of_xy[best_stg][n_obj]

                # FIXME
                # gt_twth[best_stg][b, cy, cx, max_idx, :] = torch.log(bwbh[best_stg][n_obj] / torch.from_numpy(np.array(self.anchors[best_stg][max_idx]) / stride[best_stg]).to(device))
                gt_twth[best_stg][b, cy, cx, max_idx, :] = torch.log(bwbh[best_stg][n_obj] / torch.from_numpy(np.array(self.anchors[best_stg][max_idx])).to(device))

                gt_objectness[best_stg][b, cy, cx, max_idx, 0] = 1
                gt_classes[best_stg][b, cy, cx, max_idx, int(label[n_obj].item())] = 1

            for i in range(3):
                gt_ignore_mask[i][b] = (iou_anchors_gt[i].max(-1)[0] < 0.5)

        result = []
        result_en = []  

        # 512 만들어 주기
        for stg in range(3):
            result.append(torch.cat([gt_prop_txty[stg], gt_twth[stg], gt_objectness[stg], gt_ignore_mask[stg].unsqueeze(-1), gt_classes[stg]], dim=-1).to(device))
            # FIXME 이렇게 해도 되나?
            result_en.append(result[stg].clone())

            xy_raw = result_en[stg][:,:,:,:,0:2] # [b, gs, gs, 3, 2]
            wh_raw = result_en[stg][:,:,:,:,2:4] # [b, gs, gs, 3, 2]
            rest_raw = result_en[stg][:,:,:,:,4:]  # rest [b, gs, gs, 3, 82]

            
            y = torch.arange(0, grid_size[stg]).unsqueeze(1).repeat(1, grid_size[stg])    # torch.Size([64, 64])  row0 : [0, 1 ... 64]
            x = torch.arange(0, grid_size[stg]).unsqueeze(0).repeat(grid_size[stg], 1)    # torch.Size([64, 64])  row0 : [0, 0 ... 0]
            grid_xy = torch.stack([x,y], dim=-1)
            grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, 3, 1).float().to(device)   # [b, 64, 64, 3, 2] 로 복사

            scaled_gt_xy = (xy_raw + grid_xy) * stride[stg]
            scaled_gt_wh = (torch.exp(wh_raw)*(self.anchors[stg].to(device))) * stride[stg]        #FIXME exp 때리는게 맞나?
            result_en[stg] = torch.cat([scaled_gt_xy, scaled_gt_wh, rest_raw], dim=-1)

        return result, result_en


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
