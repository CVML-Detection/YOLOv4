import torch
import torch.nn as nn
import os, sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.CSPDarknet53 import CSPDarknet53
from model.global_context_block import ContextBlock2d
from config import device


class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.conv(x)


class SpatialPyramidPooling(nn.Module):
    def __init__(self, feature_channels, pool_sizes=[5, 9, 13]):        # feature channels = [256, 512, 1024]
        super(SpatialPyramidPooling, self).__init__()

        # head conv
        self.head_conv = nn.Sequential(
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
            Conv(feature_channels[-1] // 2, feature_channels[-1], 3),
            Conv(feature_channels[-1], feature_channels[-1] // 2, 1),
        )

        self.maxpools = nn.ModuleList(
            [
                nn.MaxPool2d(pool_size, 1, pool_size // 2)
                for pool_size in pool_sizes
            ]
        )
        self.__initialize_weights()

    def forward(self, x):
        x = self.head_conv(x)   # torch.Size([1, 512, 16, 16])
        features = [maxpool(x) for maxpool in self.maxpools]    # torch.Size([1, 512, 16, 16])
        features = torch.cat([x] + features, dim=1)
        return features

    def __initialize_weights(self):
        # print("**" * 10, "Initing head_conv weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                # print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                # print("initing {}".format(m))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            Conv(in_channels, out_channels, 1), nn.Upsample(scale_factor=scale)
        )

    def forward(self, x):
        return self.upsample(x)


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super(Downsample, self).__init__()

        self.downsample = Conv(in_channels, out_channels, 3, 2)

    def forward(self, x):
        return self.downsample(x)


class PANet(nn.Module):
    def __init__(self, feature_channels):               # feature channels = [256, 512, 1024]
        super(PANet, self).__init__()

        self.feature_transform3 = Conv(
            feature_channels[0], feature_channels[0] // 2, 1
        )
        self.feature_transform4 = Conv(
            feature_channels[1], feature_channels[1] // 2, 1
        )
        self.resample5_4 = Upsample(
            feature_channels[2] // 2, feature_channels[1] // 2
        )
        self.resample4_3 = Upsample(
            feature_channels[1] // 2, feature_channels[0] // 2
        )
        self.resample3_4 = Downsample(
            feature_channels[0] // 2, feature_channels[1] // 2
        )
        self.resample4_5 = Downsample(
            feature_channels[1] // 2, feature_channels[2] // 2
        )
        self.downstream_conv5 = nn.Sequential(
            Conv(feature_channels[2] * 2, feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
        )
        self.downstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.downstream_conv3 = nn.Sequential(
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
            Conv(feature_channels[0] // 2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
            Conv(feature_channels[0] // 2, feature_channels[0], 3),
            Conv(feature_channels[0], feature_channels[0] // 2, 1),
        )
        self.upstream_conv4 = nn.Sequential(
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
            Conv(feature_channels[1] // 2, feature_channels[1], 3),
            Conv(feature_channels[1], feature_channels[1] // 2, 1),
        )
        self.upstream_conv5 = nn.Sequential(
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
            Conv(feature_channels[2] // 2, feature_channels[2], 3),
            Conv(feature_channels[2], feature_channels[2] // 2, 1),
        )
        self.__initialize_weights()

    def forward(self, features):
        features = [
            self.feature_transform3(features[0]),
            self.feature_transform4(features[1]),
            features[2],
        ]

        downstream_feature5 = self.downstream_conv5(features[2])

        # print('downstream_feature5: {}'.format(downstream_feature5.shape))
        downstream_feature4 = self.downstream_conv4(
            torch.cat(
                [features[1], self.resample5_4(downstream_feature5)], dim=1
            )
        )
        # print('downstream_feature4: {}'.format(downstream_feature4.shape))

        downstream_feature3 = self.downstream_conv3(
            torch.cat(
                [features[0], self.resample4_3(downstream_feature4)], dim=1
            )
        )
        # print('downstream_feature3: {}'.format(downstream_feature3.shape))

        upstream_feature4 = self.upstream_conv4(
            torch.cat(
                [self.resample3_4(downstream_feature3), downstream_feature4],
                dim=1,
            )
        )
        upstream_feature5 = self.upstream_conv5(
            torch.cat(
                [self.resample4_5(upstream_feature4), downstream_feature5],
                dim=1,
            )
        )
        # print('upstream_feature4: {}'.format(upstream_feature4.shape))
        # print('upstream_feature5: {}'.format(upstream_feature5.shape))


        return [downstream_feature3, upstream_feature4, upstream_feature5]

    def __initialize_weights(self):
        # print("**" * 10, "Initing PANet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                # print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                # print("initing {}".format(m))


class PredictNet(nn.Module):
    def __init__(self, feature_channels, target_channels):              # feature channels = [256, 512, 1024]
        super(PredictNet, self).__init__()

        self.predict_conv = nn.ModuleList(
            [
                nn.Sequential(
                    Conv(feature_channels[i] // 2, feature_channels[i], 3),
                    nn.Conv2d(feature_channels[i], target_channels, 1),
                )
                for i in range(len(feature_channels))
            ]
        )
        self.__initialize_weights()

    def forward(self, features):
        predicts = [
            predict_conv(feature)
            for predict_conv, feature in zip(self.predict_conv, features)
        ]

        return predicts

    def __initialize_weights(self):
        # print("**" * 10, "Initing PredictNet weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()

                # print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

                # print("initing {}".format(m))


class YOLOv4(nn.Module):
    def __init__(self, backbone, num_classes=80, showatt=False):
        super(YOLOv4, self).__init__()
        self.backbone = backbone
        feature_channels = backbone.feature_channels[-3:]       # [256, 512, 1024]

        self.showatt=showatt

        if self.showatt:
            self.attention = ContextBlock2d(feature_channels[-1], feature_channels[-1])

        self.SPP = SpatialPyramidPooling(feature_channels)

        self.PANet = PANet(feature_channels)

        self.predict_net = PredictNet(feature_channels, target_channels=3*(num_classes+5))

    def forward(self, x):
        atten = None
        features = self.backbone(x)     # CSPDarknet53      features[0, 1, 2]
        # print('features1 : {}'.format(features[0].shape))
        # print('features2 : {}'.format(features[1].shape))
        # print('features3 : {}'.format(features[2].shape))
        if self.showatt:
            features[-1], atten = self.attention(features[-1])
        
        att = False
        if att:
            print('Attention : ContextBlock2d here')
       
        features[-1] = self.SPP(features[-1])

        features = self.PANet(features)

        predicts = self.predict_net(features)

        # print('predicts1: {}'.format(predicts[0].shape))
        # print('predicts2: {}'.format(predicts[1].shape))
        # print('predicts3: {}'.format(predicts[2].shape))

        return [predicts, atten]




if __name__ == "__main__":
    img_size = 512
    img = torch.randn([2, 3, img_size, img_size]).to(device)
    model = YOLOv4(CSPDarknet53(pretrained=True)).to(device)

    #################################################################

    # [[f1, f2, f3], atten] = model(img)
    pred = model(img)
    [[f1, f2, f3], atten] = pred
    
    print('final1: {}'.format(f1.shape))
    print('final2: {}'.format(f2.shape))
    print('final3: {}'.format(f3.shape))