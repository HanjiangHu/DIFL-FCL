import os
from options.test_options import TestOptions
from data.data_loader import DataLoader
from models.DIFL_model import DIFLModel

import numpy as np


class Tester():
    def __init__(self):
        # Parse test options. Note test code only supports nThreads=1 and batchSize=1
        self.opt = TestOptions().parse()
        self.opt.nThreads = 1
        self.opt.batchSize = 1

        self.dataset = DataLoader(self.opt)
        self.model = DIFLModel(self.opt)
        # Read groundtruth poses of database from txt files for each slice, note that the poses has already been
        # transformed to R,t, not original R,c for CMU-Seasons dataset
        self.split_file = os.path.join(self.opt.dataroot, 's' + str(self.opt.which_slice),
                                       'pose_new_s' + str(self.opt.which_slice) + '.txt')
        self.names = np.loadtxt(self.split_file, dtype=str, delimiter=' ', skiprows=0, usecols=(0))
        with open(self.split_file, 'r') as pose_file:
            self.poses = pose_file.read().splitlines()

        if self.opt.test_using_cos:
            metric_mode = "cos"
        else:
            metric_mode = "l2"
        # Open the result txt file
        self.result_file = open("result_" + self.opt.name + "_" + str(self.opt.which_epoch) + '_s' + str(
            self.opt.which_slice) + "_" + metric_mode + ".txt", 'w')

    def test(self):
        for i, data in enumerate(self.dataset):
            if not self.opt.serial_test and i >= self.opt.how_many:
                break
            self.model.set_input(data)
            if self.opt.test_after_pca:
                retrieved_path = self.model.test_pca()
            else:
                retrieved_path = self.model.test()
            img_path = self.model.get_image_paths()

            if retrieved_path != "database":
                # find and write the corresponding pose for every retrieved path
                for k in range(len(self.names)):
                    if self.names[k].split('/')[-1] == retrieved_path.split('/')[-1]:
                        self.result_file.write(
                            img_path[0].split('/')[-1] + self.poses[k][len(self.poses[k].split(' ')[0]):] + '\n')
                print('Now  %s' % img_path[0].split('/')[-1])
            else:
                print('Building up database...  %s' % img_path[0].split('/')[-1])

        self.result_file.close()
        print("Done slice {}".format(self.opt.which_slice))


if __name__ == "__main__":
    tester = Tester()
    tester.test()
