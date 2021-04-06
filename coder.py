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


    def encode_new(self, gt_boxes, gt_labels):
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

                gt_twth[best_stg][b, cy, cx, max_idx, :] = torch.log(bwbh[best_stg][n_obj] / torch.from_numpy(np.array(self.anchors[best_stg][max_idx]) / stride[best_stg]).to(device))

                gt_objectness[best_stg][b, cy, cx, max_idx, 0] = 1
                gt_classes[best_stg][b, cy, cx, max_idx, int(label[n_obj].item())] = 1

            for i in range(3):
                gt_ignore_mask[i][b] = (iou_anchors_gt[i].max(-1)[0] < 0.5)

        print('debugging')
        result = []
        result_en = []  # 여기에 512 넣기

        for stg in range(3):
            result.append(torch.cat([gt_prop_txty[stg], gt_twth[stg], gt_objectness[stg], gt_ignore_mask[stg].unsqueeze(-1), gt_classes[stg]], dim=-1).to(device))
        
        return result, result_en
        
    def encode(self, gt_boxes, gt_labels, stage):
        """
        :param gt_boxes (list)  :   (N,4)
        :param gt_labels (list) :   (N)
        :len(gt_boxes) : 4
        """
        self.assign_anchors_to_device()

        stride = int(self.strides[stage].item())
        grid_size = int(self.img_size/stride)

        center_anchor = self.c_anchor[stage]                                                        # (gs, gs, 3, 4)
        corner_anchor = cxcy_to_xy(center_anchor).view(grid_size*grid_size*self.num_anchors, 4)     # (gs, gs, 3, 4)
        
        print('\n==================================')
        print('> In STAGE : {}'.format(stage))
        print('\t>> stride : {} && grid_size : {}'.format(stride, grid_size))
        print('\t>> c_anchor[{}] : {} && corner anchor count{}'.format(stage, self.c_anchor[stage].shape, corner_anchor.shape))

        batch_size = len(gt_boxes)

        # (순서대로 x, y, w, h, conf(1), mix(1), cls(80))   (2+2+1+1+80 = 86)
        gt_prop_txty = torch.zeros([batch_size, grid_size, grid_size, 3, 2])    # a proportion between (0 ~ 1) in a cell
        gt_twth = torch.zeros([batch_size, grid_size, grid_size, 3, 2])     # ratio of gt box and anchor box
        gt_objectness = torch.zeros([batch_size, grid_size, grid_size, 3, 1])   # maximum iou anchor (a obj assign a anc)
        ignore_mask = torch.zeros([batch_size, grid_size, grid_size, 3])
        gt_classes = torch.zeros([batch_size, grid_size, grid_size, 3, self.num_classes])   # one-hot encoded class label

        # 한 이미지에 대해서
        for b in range(batch_size):
            label = gt_labels[b]
            corner_gt_box = gt_boxes[b]             # [N, 4] && corner bbox : (x1, y1, x2, y2) -> 비율로 되있음 (0 ~ 1)
            scaled_corner_gt_box = corner_gt_box * float(grid_size)     # grid size로 맞춰줘 (0 ~ 64)
            num_obj = corner_gt_box.size(0)

            center_gt_box = xy_to_cxcy(corner_gt_box)       # center bbox : (cx, cy, w, h) -> (0 ~ 1)
            scaled_center_gt_box = center_gt_box * float(grid_size)     # grid size로 맞춰줘 (0 ~ 64)
            
            bxby = scaled_center_gt_box[..., :2]    # [N, 2] - cxcy
            proportion_of_xy = bxby - bxby.floor()  # [N, 2] - 0 ~ 1
            bwbh = scaled_center_gt_box[..., 2:]    # [N, 2] - wh

            # (64*64*3 , 4) , (3, 4)
            iou_anchors_gt = find_jaccard_overlap(corner_anchor, scaled_corner_gt_box)
            iou_anchors_gt = iou_anchors_gt.view(grid_size, grid_size, 3, -1)   # [gs, gs, 3, 5]


            # print('\t\t ==== FOR One Image ====')
            # print('\t\t 1th box : {}'.format(scaled_corner_gt_box[0]))
            # print('\t\t box shape : {}'.format(scaled_corner_gt_box.shape))
            # print('\t\t labels : {}'.format(label))
            # print('\t\t iou_anchors_gt shape : {}'.format(iou_anchors_gt.shape))
            # print('\t\t num_obj : {}'.format(num_obj))


            for n_obj in range(num_obj):
                cx, cy = bxby[n_obj]
                cx = int(cx)
                cy = int(cy)

                max_iou, max_idx = iou_anchors_gt[cy, cx, :, n_obj].max(0)  # which anchor has maximum iou?
                j = max_idx  # j is idx.
                gt_objectness[b, cy, cx, j, 0] = 1
                gt_prop_txty[b, cy, cx, j, :] = proportion_of_xy[n_obj]
                ratio_of_wh = bwbh[n_obj] / torch.from_numpy(np.array(self.anchors[stage][j])).to(device)

                gt_twth[b, cy, cx, j, :] = torch.log(ratio_of_wh)
                gt_classes[b, cy, cx, j, int(label[n_obj].item())] = 1

            ignore_mask[b] = (iou_anchors_gt.max(-1)[0] < 0.5)

        # print('gt_prop_txty : {}'.format(gt_prop_txty.shape))
        # print('gt_twth : {}'.format(gt_twth.shape))
        # print('gt_objectness : {}'.format(gt_objectness.shape))
        # print('ignore_mask : {}'.format(ignore_mask.unsqueeze(-1).shape))
        # print('gt_classes : {}'.format(gt_classes.shape))
        gt_label = torch.cat([gt_prop_txty, gt_twth, gt_objectness, ignore_mask.unsqueeze(-1), gt_classes], dim=-1).to(device)
        # print('gt_label:{}'.format(gt_label.shape))

        # FIXME 임시 코드
        gt_box = torch.randn([batch_size,150,4]).to(device)

        return gt_label, gt_box

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
