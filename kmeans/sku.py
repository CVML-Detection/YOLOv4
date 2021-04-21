import os
import glob
import torch
import csv
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as FT
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import matplotlib.pyplot as plt

from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans


def resize(image, boxes, dims=(256, 256), return_percent_coords=False):

    # torch to PIL
    image = FT.to_pil_image(image)

    # Resize image
    new_image = FT.resize(image, dims)

    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.width, image.height, image.width, image.height]).unsqueeze(0)
    new_boxes = boxes / old_dims  # percent coordinates

    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image, new_boxes


class SKU110K_Dataset(Dataset):
    def __init__(self, root='D:\data\SKU110K_fixed', split='train', resize=800):
        super().__init__()
        assert split in ['train', 'val', 'test']
        # train dataset : 8279

        self.root = root
        self.split = split
        self.resize = resize
        self.root = os.path.join(self.root, 'SKU110K_fixed')
        self.images_path = os.path.join(self.root, 'images')
        self.images_path = glob.glob(os.path.join(self.images_path, '{}_*.jpg').format(self.split))
        self.annotations_path = os.path.join(self.root, 'annotations')
        self.annotations_path = os.path.join(self.annotations_path, 'annotations_{}.csv'.format(self.split))

        self.bbox_dict = {}
        self.size_dict = {}
        self.bbox_w = []
        self.bbox_h = []
        print("Loading Dataset ...")
        with open(self.annotations_path, mode='r') as f:      # Load Annotation
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                # print(line)
                img_name = line[0]
                bbox = list(map(int, line[1:5]))              # string to int in list
                wh = list(map(int, line[-2:]))              # string to int in list
                
                self.bbox_w.append((bbox[2]-bbox[0])/wh[0])
                self.bbox_h.append((bbox[3]-bbox[1])/wh[1])

                # if img_name not in self.bbox_dict:
                #     self.bbox_dict[img_name] = []
                #     self.bbox_dict[img_name].append(bbox)
                #     # dict['wh'] = wh
                # else:
                #     self.bbox_dict[img_name].append(bbox)

                # if img_name not in self.size_dict:
                #     self.size_dict[img_name] = wh
                
        print("num of image :", len(self.bbox_dict))


    def kmeans(self, num_clusters=3, axis=0.6):
        print('\n=============\nKMEANs Clustering...')
        bbox_w_tensor = torch.from_numpy(np.array(self.bbox_w)).unsqueeze(-1)
        bbox_h_tensor = torch.from_numpy(np.array(self.bbox_h)).unsqueeze(-1)
        bbox_wh = torch.cat([bbox_w_tensor, bbox_h_tensor], dim=-1)
        print('bbox_wh:{}'.format(bbox_wh.size()))

        cluster_ids, cluster_centers = kmeans(
            X=bbox_wh, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
        )
        print('\nNum Cluster: {}'.format(num_clusters))
        for i, center in enumerate(cluster_centers):
            print('Center Point: {}'.format(center))

        plt.figure(figsize=(4,3), dpi=160)
        plt.scatter(bbox_wh[:, 0], bbox_wh[:, 1], c=cluster_ids, cmap='cool')
        plt.scatter(
            cluster_centers[:, 0], cluster_centers[:, 1],
            c='white',
            alpha=0.6,
            edgecolors='black',
            linewidths=2
        )
        plt.axis([0, axis, 0, axis])
        plt.tight_layout()
        result_name = 'result/cluster{}_{}.png'.format(num_clusters, axis)

        plt.savefig(result_name, dpi=300, bbox_inches='tight')


    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):

        image = Image.open(self.images_path[idx]).convert('RGB')
        img_name = os.path.basename(self.images_path[idx])
        boxes = self.bbox_dict[img_name]    # x1 y1 x2 y2

        # make add info and img name
        img_name = img_name.split('.')[0]  # .jpg, .png 등 제거
        img_name_to_ascii = [ord(c) for c in img_name]
        img_width, img_height = float(image.size[0]), float(image.size[1])
        img_name = torch.FloatTensor([img_name_to_ascii])
        additional_info = torch.FloatTensor([img_width, img_height])

        # convert to tensor
        image = FT.to_tensor(image)
        boxes = torch.FloatTensor(boxes)                               # x1 y1 x2 y2

        # --------------------- short transform ---------------------
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # x1 y1 x2 y2
        image, boxes = resize(image, boxes, dims=(self.resize, self.resize))

        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2    # x1 y1 x2 y2 -> c_x, c_y
        cycx = torch.cat([cxcy[:, 1:2], cxcy[:, 0:1]], dim=1)                      # y, x 변환
        locations = cycx
        counts = torch.tensor(locations.size(0), dtype=torch.get_default_dtype())
        print(locations)
        if counts == 0:
            locations = torch.tensor([-1, -1], dtype=torch.get_default_dtype())

        labels = torch.zeros(boxes.size(0), dtype=torch.int64)
        boxes = boxes / self.resize

        # Convert PIL image to Torch tensor
        image = FT.to_tensor(image)
        image = FT.normalize(image, mean=mean, std=std)

        visualize = False
        if visualize:

            # tensor to img
            img_vis = np.array(image.permute(1, 2, 0), np.float32)  # C, W, H
            img_vis *= std
            img_vis += mean
            img_vis = np.clip(img_vis, 0, 1)

            plt.figure('input')
            plt.imshow(img_vis)
            print('num objects : {}'.format(len(boxes)))
            for i in range(len(boxes)):

                x1 = boxes[i][0] * self.resize
                y1 = boxes[i][1] * self.resize
                x2 = boxes[i][2] * self.resize
                y2 = boxes[i][3] * self.resize

                # bounding box
                plt.gca().add_patch(Rectangle(xy=(x1, y1),
                                              width=x2 - x1,
                                              height=y2 - y1,
                                              linewidth=1,
                                              edgecolor=(1, 0, 1),
                                              facecolor='none'))

                cx = cxcy[i][0]  # y
                cy = cxcy[i][1]  # x

                # center of box
                plt.scatter(x=cx, y=cy, c='r')

            plt.show()
        if self.split == "test" or self.split == "val":
            return image, boxes, labels, locations, counts, img_name, additional_info

        return image, boxes, labels, locations, counts  # , bbox

    def collate_fn(self, batch):
        """
        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes
        """
        images = list()
        boxes = list()
        labels = list()
        locations = list()
        counts = list()
        img_name = list()
        additional_info = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])
            locations.append(b[3])
            counts.append(b[4])
            if self.split == "test" or self.split == "val":
                img_name.append(b[5])
                additional_info.append(b[6])

        images = torch.stack(images, dim=0)
        counts = torch.stack(counts, dim=0)

        if self.split == "test" or self.split == "val":
            return images, boxes, labels, locations, counts, img_name, additional_info
        return images, boxes, labels, locations, counts


if __name__ == '__main__':
    #data_root = 'D:\data\SKU110K_fixed'
    data_root = '/data/sku110k'
    dataset = SKU110K_Dataset(root=data_root, split='val', resize=800)
    dataset.kmeans(num_clusters=3, axis=0.3)


    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=dataset.collate_fn)
    # for i, (img, box, lab, loc, cnt) in enumerate(dataloader):
    #     print(i)
    #     break
