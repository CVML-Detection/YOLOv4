import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch.nn as nn
import torch
import config as cfg

class YOLOv4_Loss(nn.Module):
    def __init__(self, coder):
        super().__init__()
        self.coder = coder
        self.num_classes = self.coder.num_classes       # 80
        


    def forward(self, pred, gt_boxes, gt_labels):
        output = []
        [[f1, f2, f3], atten] = pred
        output.append(self.coder.decode(p=f1, stage=0))  # [4, 64, 64, 3, 85]   p[0], p_d[0]
        output.append(self.coder.decode(p=f2, stage=1))  # [4, 32, 32, 3, 85]   p[1], p_d[1]
        output.append(self.coder.decode(p=f3, stage=2))  # [4, 16, 16, 3, 85]   p[2], p_d[2]
        p, p_d = list(zip(*output))
        strides = self.coder.strides

        self.CIoULoss(p[0], p_d[0], gt_labels, gt_boxes, strides[0])


        return 0

    def CIoULoss(self, p, p_d, label, bboxes, stride):
        BCE = nn.BCEWithLogitsLoss(reduction="none")

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        p_d_xywh = p_d[..., 4]


if __name__ == '__main__':
    from model.model import YOLOv4, CSPDarknet53
    from coder import YOLOv4_Coder
    import config as cfg

    model = YOLOv4(CSPDarknet53(pretrained=True)).to(cfg.device)
    coder = YOLOv4_Coder(data_type='coco')
    criterion = YOLOv4_Loss(coder=coder)

    ## ---------------------
    img_size = 512
    img = torch.randn([4, 3, img_size, img_size]).to(cfg.device)
    
    pred = model(img)
    criterion(pred, None, None)


    


