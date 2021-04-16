import argparse
import torch

# 2. device
device_ids = [0]
device = torch.device('cuda:{}'.format(min(device_ids)) if torch.cuda.is_available() else 'cpu')


# model
MODEL = {
    "ANCHORS": [
        [
            (1.25, 1.625),
            (2.0, 3.75),
            (4.125, 2.875),
        ],  # Stage 0 )
            # Anchors for small obj(12,16),(19,36),(40,28)
            # (10, 13), (16, 30), (33, 23) (* STRIDE = 8)
        [
            (1.875, 3.8125),
            (3.875, 2.8125),
            (3.6875, 7.4375),
        ],  # Stage 1 ) 
            # Anchors for medium obj(36,75),(76,55),(72,146)
            # (30, 61), (62, 45), (59, 119) (* STRIDE = 16)
        [
            (3.625, 2.8125),
            (4.875, 6.1875),
            (11.65625, 10.1875)
        ],  # Stage 2 ) 
            # Anchors for big obj(142,110),(192,243),(459,401)
            # (116,90), (156,198), (373, 326) (* STRIDE = 32)
    ],
    "STRIDES": [8, 16, 32],
    "ANCHORS_PER_SCLAE": 3,
    "INPUT_IMG_SIZE":512,
}



def parse(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=150)  # 173
    parser.add_argument('--port', type=str, default='8097')  # 173
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--save_path', type=str, default='./saves')
    parser.add_argument('--save_file_name', type=str, default='yolov4_cspdkn53_coco')  # FIXME
    parser.add_argument('--conf_thres', type=float, default=0.05)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--test', dest='test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--test_epoch', type=int, default=0)

    # FIXME choose your dataset root
    # parser.add_argument('--data_root', type=str, default='D:\data\\voc')
    # parser.add_argument('--data_root', type=str, default='D:\data\coco')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/voc')
    # parser.add_argument('--data_root', type=str, default='/home/cvmlserver5/Sungmin/data/coco')
    # parser.add_argument('--data_root', type=str, default='/data/voc')
    parser.add_argument('--data_root', type=str, default='/data/voc')


    parser.add_argument('--data_type', type=str, default='voc', help='choose voc or coco')  # FIXME
    parser.add_argument('--num_classes', type=int, default=20)

    opts = parser.parse_args(args)
    print(opts)
    return opts