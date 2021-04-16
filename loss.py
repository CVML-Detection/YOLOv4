import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch.nn as nn
import torch
import math

import config as cfg


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(
            torch.abs(target - torch.sigmoid(input)), self.__gamma
        )
        return loss


class YOLOv4_Loss(nn.Module):
    def __init__(self, coder):
        super().__init__()
        self.coder = coder
        self.num_classes = self.coder.num_classes       # 80
        

    def forward(self, pred, gt_boxes, gt_labels):
        # -----------------------------
        # Decode Pred
        # -----------------------------
        output = []
        [[f1, f2, f3], atten] = pred        # [b,255,64,64], [b,255,32,32], [b,255,16,16]
        output.append(self.coder.decode(p=f1, stage=0))  # [4, 64, 64, 3, 85]   p[0], p_d[0]
        output.append(self.coder.decode(p=f2, stage=1))  # [4, 32, 32, 3, 85]   p[1], p_d[1]
        output.append(self.coder.decode(p=f3, stage=2))  # [4, 16, 16, 3, 85]   p[2], p_d[2]
        p, p_d, p_d_512 = list(zip(*output))
        
        # -----------------------------
        # Encode GT
        # -----------------------------
        batch_size = p[0].shape[0]
        gt, gt_en = self.coder.encode(gt_boxes, gt_labels)

        
        strides = self.coder.strides
        (loss_s, loss_s_ciou, loss_s_conf, loss_s_cls) = self.loss_per_layer(p[0], p_d[0], p_d_512[0], gt[0], gt_en[0], strides[0])
        (loss_m, loss_m_ciou, loss_m_conf, loss_m_cls) = self.loss_per_layer(p[1], p_d[1], p_d_512[1], gt[1], gt_en[1], strides[1])
        (loss_l, loss_l_ciou, loss_l_conf, loss_l_cls) = self.loss_per_layer(p[2], p_d[2], p_d_512[2], gt[2], gt_en[2], strides[2])
        loss = loss_s + loss_m + loss_l
        loss_ciou = loss_s_ciou + loss_m_ciou + loss_l_ciou
        loss_conf = loss_s_conf + loss_m_conf + loss_l_conf
        loss_cls = loss_s_cls + loss_m_cls + loss_l_cls
        return loss, loss_ciou, loss_conf, loss_cls


    def loss_per_layer(self, p, p_d, p_d_512, gt, gt_en, stride):
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        FOCAL = FocalLoss(gamma=2, alpha=1.0, reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        # pred (decoded)        [b, gs, gs, 3, 85]
        p_d_xywh = p_d_512[..., :4]     #[b, 64, 64, 3, 4]
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        # gt (encoded)          [b, gs, gs, 3, 86]  ( + ignore mask )
        label_xywh = gt_en[..., :4]
        label_obj_mask = gt[..., 4:5]
        label_ignore_mask = gt[..., 5:6]        # use in Data Aug 
        label_cls = gt[..., 6:]

        # Calculating Loss
        # 1) CIoU
        ciou = self.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)

        # The scaled weight of bbox is used to balance the impact of small objects and large objects on loss.
        bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size ** 2)     # [b,grid,grid,3,1]
        loss_ciou = label_obj_mask * bbox_loss_scale * (1.0 - ciou)

        # 2) Conf Loss      # Focal Loss
        label_noobj_mask = (1-label_obj_mask) * BCE(label_obj_mask, p_conf) * label_ignore_mask
        loss_conf = label_obj_mask * FOCAL(input=p_conf, target=label_obj_mask) + label_noobj_mask * FOCAL(input=p_conf, target=label_obj_mask)

        # 3) Cls Loss       # BCE
        loss_cls = (label_obj_mask * BCE(input=p_cls, target=label_cls))

        loss_ciou = (torch.sum(loss_ciou)) / batch_size
        loss_conf = (torch.sum(loss_conf)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size

        loss = loss_ciou + loss_conf + loss_cls
        
        return loss, loss_ciou, loss_conf, loss_cls


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
    
    bs = 4
    model = YOLOv4(CSPDarknet53(pretrained=True)).to(cfg.device)
    coder = YOLOv4_Coder(data_type='coco')
    criterion = YOLOv4_Loss(coder=coder)

    ## ---------------------
    img_size = 512
    num_obj = 6
    img = torch.randn([bs, 3, img_size, img_size]).to(cfg.device)
    gt_boxes = []
    gt_labels = []
    for b in range(bs):
        gt_boxes.append(torch.randn([num_obj, 4]).to(cfg.device))
        gt_labels.append(torch.randn([num_obj]).to(cfg.device))
    
    pred = model(img)
    criterion(pred, gt_boxes, gt_labels)


    


