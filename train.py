import torch
import config as cfg
import time


def train(epoch, vis, train_loader, model, criterion, optimizer, scheduler, opts):
    tic = time.time()
    model.train()

    for idx, datas in enumerate(train_loader):
        images = datas[0]
        boxes = datas[1]
        labels = datas[2]

        images = images.to(cfg.device)
        boxes = [b.to(cfg.device) for b in boxes]
        labels = [l.to(cfg.device) for l in labels]

        # print('batch_size : {}'.format(len(boxes)))
        # print('images : {}'.format(images.shape))
        # print('object count : {}'.format(boxes[0].shape[0]))
        # print('boxes : {}'.format(boxes[0].shape))
        # print('labels : {}'.format(labels[0].shape))

        pred = model(images)
        loss, loss_ciou, loss_cls = criterion(pred, boxes, labels)

        # sgd
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        toc = time.time()


        # for each steps
        if idx % 10 == 0 or idx == len(train_loader) - 1:
            print('Epoch: [{0}]\t'
                  'Step: [{1}/{2}]\t'
                  'Loss: {loss:.4f}\t'
                  'Time : {time:.4f}\t'
                  .format(epoch, idx, len(train_loader), loss=loss, time=toc - tic))

            if vis is not None:
                # loss plot
                vis.line(X=torch.ones((1, 3)).cpu() * idx + epoch * train_loader.__len__(),  # step
                         Y=torch.Tensor([loss, loss_ciou, loss_cls]).unsqueeze(0).cpu(),
                         win='train_loss',
                         update='append',
                         opts=dict(xlabel='step',
                                   ylabel='Loss',
                                   title='training loss',
                                   legend=['Total Loss', 'CIoU Loss', 'CLS Loss']))
    
    if not os.path.exist(opts.save_path):
        os.mkdir(opts.save_path)
    
    checkpoint = {'epoch': epoch,
                  'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict()}

    torch.save(checkpoint, os.path.join(opts.save_path, opts.save_file_name + '.{}.pth.tar'.format(epoch)))