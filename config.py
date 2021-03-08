import argparse
import torch

# 2. device
device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')


def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150)  # 173
    parser.add_argument('--port', type=str, default='2005')  # 173
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--image_size', type=int, help='320, 416, 608', default=416)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolov3_darknet53_coco')  # FIXME
    parser.add_argument('--conf_thres', type=float, default=0.05)
    parser.add_argument('--start_epoch', type=int, default=0)

    # FIXME choose your dataset root
    # parser.add_argument('--data_root', type=str, default='D:\data\\voc')
    # parser.add_argument('--data_root', type=str, default='D:\data\coco')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/voc')
    parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/coco')
    # parser.add_argument('--data_root', type=str, default='/data/voc')

    parser.add_argument('--data_type', type=str, default='coco', help='choose voc or coco')  # FIXME
    parser.add_argument('--num_classes', type=int, default=80)

    opts = parser.parse_args(args)
    print(opts)
    return opts