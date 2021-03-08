import torch

from config import device


class CSPDarknet53(nn.Module):

    def __init__(self, num_classes=1000, pretrained=True):
        super().__init__()
        
        self.num_classes = num_classes

        if pretrained:
            self.pretrained_path = "/data/weights/yolov4.weights"
            self.load_CSPDarknet_weights(pretrained_path)

    def load_CSPDarknet_weights(self, weights):
        print("load darknet weights : ", weight_file)

class YOLOv4(nn.Module):
    def __init__(self):

    def forward(self, x):


if __name__ == '__main__':
    img = torch.randn([2, 3, 512, 512]).to(device)
    model = YoloV4(CSPDarknet53(pretrained=True)).to(device)
