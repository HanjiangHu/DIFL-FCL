import os.path, glob
from data.base_dataset import BaseDataset, get_transform,get_transform_flip
from data.image_folder import make_dataset
from PIL import Image
import random

class UnalignedDataset(BaseDataset):
    def __init__(self, opt):
        super(UnalignedDataset, self).__init__()
        self.opt = opt
        self.transform = get_transform(opt)
        # flip images of the other training branch, i.e. B pass
        self.transform_flip = get_transform_flip(opt)
        if opt.phase == 'train':
            datapath = os.path.join(opt.dataroot, opt.phase + '*')
        else:
            datapath = os.path.join(opt.dataroot,'s'+str(opt.which_slice), opt.phase + '*')
        self.dirs = sorted(glob.glob(datapath))

        self.paths = [sorted(make_dataset(d)) for d in self.dirs]
        self.sizes = [len(p) for p in self.paths]

    def load_image(self, dom, idx):
        path = self.paths[dom][idx]
        old_img = Image.open(path).convert('RGB')
        img = self.transform(old_img)
        return img, path

    def load_image_flip(self, dom, idx):
        path = self.paths[dom][idx]
        old_img = Image.open(path).convert('RGB')
        img = self.transform_flip(old_img)
        return img, path
    def __getitem__(self, index):
        if not self.opt.isTrain:
            # for testing
            if self.opt.serial_test:
                for d,s in enumerate(self.sizes):
                    if index < s:
                        DA = d; break
                    index -= s
                index_A = index
            else:
                DA = index % len(self.dirs)
                index_A = random.randint(0, self.sizes[DA] - 1)
        else:
            # Choose two of our domains to perform a pass on
            if self.opt.domains_overlap:
                # the two domains may be identical
                DA = random.randint(0, len(self.dirs) - 1)
                DB = random.randint(0, len(self.dirs) - 1)
            else:
                # two different domains
                DA, DB = random.sample(range(len(self.dirs)), 2)
            index_A = random.randint(0, self.sizes[DA] - 1)

        A_img, A_path = self.load_image(DA, index_A)
        bundle = {'A': A_img, 'DA': DA, 'path': A_path}

        if self.opt.isTrain:
            index_B = random.randint(0, self.sizes[DB] - 1)
            B_img, _ = self.load_image_flip(DB, index_B)
            bundle.update( {'B': B_img, 'DB': DB} )

        return bundle

    def __len__(self):
        if self.opt.isTrain:
            return max(self.sizes)
        return sum(self.sizes)

    def name(self):
        return 'UnalignedDataset'
