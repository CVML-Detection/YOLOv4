import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch.nn as nn
import torch
import math

import config as cfg

class YOLOv4_Loss(nn.Module):
    def __init__(self, coder):
        super().__init__()
        self.coder = coder
        self.num_classes = self.coder.num_classes       # 80
        


    def forward(self, pred, gt_boxes, gt_labels):

        # Decode Pred
        # -----------------------------
        output = []
        [[f1, f2, f3], atten] = pred
        output.append(self.coder.decode(p=f1, stage=0))  # [4, 64, 64, 3, 85]   p[0], p_d[0]
        output.append(self.coder.decode(p=f2, stage=1))  # [4, 32, 32, 3, 85]   p[1], p_d[1]
        output.append(self.coder.decode(p=f3, stage=2))  # [4, 16, 16, 3, 85]   p[2], p_d[2]
        p, p_d = list(zip(*output))
        # -----------------------------


        # Encode GT (임시)
        # -----------------------------
        batch_size = p[0].shape[0]

        gt_labels_en_s = torch.randn([batch_size,64,64,3,86]).to(cfg.device)
        gt_labels_en_m = torch.randn([batch_size,32,32,3,86]).to(cfg.device)
        gt_labels_en_l = torch.randn([batch_size,16,16,3,86]).to(cfg.device)

        gt_boxes_en_s = torch.randn([batch_size,150,4]).to(cfg.device)
        gt_boxes_en_m = torch.randn([batch_size,150,4]).to(cfg.device)
        gt_boxes_en_l = torch.randn([batch_size,150,4]).to(cfg.device)
        # -----------------------------
        
        strides = self.coder.strides
        loss_s = self.loss_per_layer(p[0], p_d[0], gt_labels_en_s, gt_boxes_en_s, strides[0])
        loss_m = self.loss_per_layer(p[1], p_d[1], gt_labels_en_m, gt_boxes_en_m, strides[1])
        loss_l = self.loss_per_layer(p[2], p_d[2], gt_labels_en_l, gt_boxes_en_l, strides[2])

        print('cious1 : {}'.format(loss_s.shape))
        print('cious2 : {}'.format(loss_m.shape))
        print('cious3 : {}'.format(loss_l.shape))

        return 0

    def loss_per_layer(self, p, p_d, label, bboxes, stride):
        BCE = nn.BCEWithLogitsLoss(reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        # pred (decoded)
        p_d_xywh = p_d[..., :4]
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        # gt (encoded)
        label_xywh = label[..., :4]
        label_obj_mask = label[..., 4:5]
        label_mix = label[..., 5:6]
        label_cls = label[..., 6:]
        ciou = self.CIOU_xywh_torch(p_d_xywh, label_xywh)
        return ciou


    def CIOU_xywh_torch(self, boxes1, boxes2):
        # xywh -> xyxy
        # print('boxes1 : {}'.format(boxes1.shape))
        # print('boxes2 : {}'.format(boxes2.shape))

        boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                            boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)   # xy-(wh/2) -> x1,y1  &&  xy+(wh/2) -> x2, y2
        boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                            boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

        boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)   # x1 y1 to min  &&  x2 y2 to max
        boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)
        
        # ====== Calculate IOU ======
        boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])     # (x2-x1)*(y2-y1)
        boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
        inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        
        inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area
        ious = 1.0 * inter_area / union_area

        # ====== Calculate IOU ======
        # cal outer boxes
        outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
        outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
        outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
        outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2)

        # cal center distance
        boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
        boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
        center_dis = torch.pow(boxes1_center[...,0]-boxes2_center[...,0], 2) +\
                    torch.pow(boxes1_center[...,1]-boxes2_center[...,1], 2)

        # cal penalty term
        # cal width,height
        boxes1_size = torch.max(boxes1[..., 2:] - boxes1[..., :2], torch.zeros_like(inter_right_down))  # w, h
        boxes2_size = torch.max(boxes2[..., 2:] - boxes2[..., :2], torch.zeros_like(inter_right_down))  # w, h

        v = (4 / (math.pi ** 2)) * torch.pow(
                torch.atan((boxes1_size[...,0]/torch.clamp(boxes1_size[...,1],min = 1e-6))) -
                torch.atan((boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1],min = 1e-6))), 2)
        alpha = v / (1-ious+v)

        #cal ciou
        cious = ious - (center_dis / outer_diagonal_line + alpha*v)

        return cious


if __name__ == '__main__':
    from model.model import YOLOv4, CSPDarknet53
    from coder import YOLOv4_Coder
    import config as cfg
    
    bs = 2
    model = YOLOv4(CSPDarknet53(pretrained=True)).to(cfg.device)
    coder = YOLOv4_Coder(data_type='coco')
    criterion = YOLOv4_Loss(coder=coder)

    ## ---------------------
    img_size = 512
    num_obj = 5
    img = torch.randn([4, 3, img_size, img_size]).to(cfg.device)
    gt_boxes = []
    gt_labels = []
    for b in range(bs):
        gt_boxes.append(torch.randn([num_obj, 4]).to(cfg.device))
        gt_labels.append(torch.randn([num_obj]).to(cfg.device))
    
    pred = model(img)
    criterion(pred, gt_boxes, gt_labels)


    

