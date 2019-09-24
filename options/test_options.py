from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.isTrain = False

        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

        self.parser.add_argument('--which_epoch', required=True, type=int, help='which epoch to load for inference?')
        self.parser.add_argument('--phase', type=str, default='test',
                                 help='train, test (determines name of folder to load from)')

        self.parser.add_argument('--how_many', type=int, default=50,
                                 help='how many test images to run (if serial_test not enabled)')
        self.parser.add_argument('--serial_test', action='store_true',
                                 help='read each image once from folders in sequential order')

        self.parser.add_argument('--which_slice', type=int, default=2,
                                 help='which slice of images to be test for CMU-Seasons dataset')
        self.parser.add_argument('--test_using_cos', action='store_true',
                                 help='use cosine distance as retrieval metric while testing')
        self.parser.add_argument('--resize_to_crop_size', action='store_true',
                                 help='resize the image to the same size as after square-cropping while testing')
        self.parser.add_argument('--test_after_pca', action='store_true',
                                 help='PCA technique is applied to latent features just before retrieval')
        self.parser.add_argument('--PCA_dimension', type=int, default=100,
                                 help='the dimension of feature after PCA is applied')
