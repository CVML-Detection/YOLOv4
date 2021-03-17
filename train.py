import torch
import config as cfg
import time


def train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts):
    tic = time.time()
    model.train()

    for idx, datas in enumerate(train_loader):
        print('========= DEBUGGING ========')
        images = datas[0]
        boxes = datas[1]
        labels = datas[2]

        images = images.to(cfg.device)
        boxes = [b.to(cfg.device) for b in boxes]
        labels = [l.to(cfg.device) for l in labels]


        print('images : {}'.format(images.shape))
        print('boxes : {}'.format(boxes[0].shape))
        print('labels : {}'.format(labels[0].shape))
        
        # pred = model(images)
        # loss = criterion(pred, boxes, labels)

        break