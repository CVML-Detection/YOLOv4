

class Coder(metaclass=ABCMeta):
    @abstractmethod
    def encode(self):
        pass
    def decode(self):
        pass


class YOLOv4_Coder(Coder):
    def __init__(self, data_type):
        super().__init__()
        self.data_type = data_type

        assert self.data_type in ['voc', 'coco']
        if self.data_type == 'voc':
            self.num_classes = 20
        elif self.data_type == 'coco':
            self.num_classes = 80


if __name__ == '__main__':
    coder = YOLOv4_Coder('coco')