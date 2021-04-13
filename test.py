import os
import time
import torch
from evaluator import Evaluator
from config import device


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
            loss, _ = criterion(pred, boxes, labels)
            sum_loss += loss.item()

            # ---------- eval ----------

            