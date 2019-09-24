import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform(opt):
    """
    Image transformation for A pass training and testing.
    :param opt: Options
    :return: Transform lists
    """
    transform_list = []

    if opt.isTrain:
        # Resize, crop or flip for A pass while training
        if 'resize' in opt.resize_or_crop:
            transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))
        if 'crop' in opt.resize_or_crop:
            transform_list.append(transforms.RandomCrop(opt.fineSize))
        if not opt.no_flip:
            if opt.no_random_flip:
                transform_list.append(transforms.RandomHorizontalFlip(p=0))
            else:
                transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    else:
        # For testing
        if 'resize' in opt.resize_or_crop:
            if opt.resize_to_crop_size:
                transform_list.append(transforms.Resize([opt.fineSize, opt.fineSize], Image.BICUBIC))
            else:
                transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)
def get_transform_flip(opt):
    """
    Image transformation for B pass training.
    :param opt: Options
    :return: Transform lists
    """
    transform_list = []
    # Resize, crop or flip for A pass while training
    if 'resize' in opt.resize_or_crop:
        transform_list.append(transforms.Resize(opt.loadSize, Image.BICUBIC))
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    if not opt.no_flip:
        if opt.no_random_flip:
            transform_list.append(transforms.RandomHorizontalFlip(p=1.0))
        else:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)