import torch


if __name__ == '__main__':

    box1 = torch.tensor([[2., 2.], [5., 7.]])
    box2 = torch.tensor([[4., 1.], [6., 3.]])

    box_left_up = torch.max(box1[0], box2[0])
    box_right_down = torch.min(box1[1], box2[1])

    print('box1 : {}'.format(box1.shape))
    print('box2 : {}'.format(box2.shape))

    print(box_left_up)
    print(box_right_down)