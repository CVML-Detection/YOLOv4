import os
import time
import torch
from evaluator import Evaluator
from config import device
from utils import detect


def test(epoch, vis, test_loader, model, criterion, coder, opts):
    model.eval()
    check_point = torch.load(os.path.join(opts.save_path, opts.save_file_name) + '.{}.pth.tar'.format(epoch),map_location=device)
    state_dict = check_point['model_state_dict']
    model.load_state_dict(state_dict)

    tic = time.time()
    sum_loss = 0

    if hasattr(test_loader.dataset, 'coco'):
        print('COCO dataset evaluation...')
    else:
        print('VOC dataset evaluation...')
    
    evaluator = Evaluator(data_type = opts.data_type)

    with torch.no_grad():
        for idx, datas in enumerate(test_loader):

            images = datas[0]
            boxes = datas[1]
            labels = datas[2]

            # ---------- cuda ----------
            images = images.to(device)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # ---------- loss ----------
            pred = model(images)
            loss, loss_ciou, loss_conf, loss_cls = criterion(pred, boxes, labels)
            sum_loss += loss.item()

            # ---------- eval ----------

            pred_boxes, pred_labels, pred_scores = detect(pred=pred, coder=coder, opts=opts)

            if opts.data_type == 'voc':
                img_name = datas[3][0]
                img_info = datas[4][0]
                info = (pred_boxes, pred_labels, pred_scores, img_name, img_info)

            elif opts.data_type == 'coco':
                img_id = test_loader.dataset.img_id[idx]
                img_info = test_loader.dataset.coco.loadImgs(ids=img_id)[0]
                coco_ids = test_loader.dataset.coco_ids
                info = (pred_boxes, pred_labels, pred_scores, img_id, img_info, coco_ids)

            evaluator.get_info(info)

            toc = time.time()
            
            # ---------- print ----------
            if idx % 100 == 0 or idx == len(test_loader) - 1:
                print('Epoch: [{0}]\t'
                      'Step: [{1}/{2}]\t'
                      'Loss: {loss:.4f}\t'
                      'Time : {time:.4f}\t'
                      .format(epoch,
                              idx, len(test_loader),
                              loss=loss,
                              time=toc - tic))

        mAP = evaluator.evaluate(test_loader.dataset)
        mean_loss = sum_loss / len(test_loader)

        if vis is not None:
            # loss plot
            vis.line(X=torch.ones((1, 2)).cpu() * epoch,  # step
                     Y=torch.Tensor([mean_loss, mAP]).unsqueeze(0).cpu(),
                     win='test_loss',
                     update='append',
                     opts=dict(xlabel='step',
                               ylabel='test',
                               title='test loss',
                               legend=['test Loss', 'mAP']))

