import time
from options.train_options import TrainOptions
from data.data_loader import DataLoader
from models.DIFL_model import DIFLModel
from util.visualizer import Visualizer

class Trainer():
    def __init__(self):
        self.opt = TrainOptions().parse()
        self.dataset = DataLoader(self.opt)
        print('# training images = %d' % len(self.dataset))
        self.model = DIFLModel(self.opt)
        self.visualizer = Visualizer(self.opt)
        self.total_steps = 0

    def train(self):
        """
        Main loop for DIFL training.
        Training options are set through train_options and base_options.
        :return: None
        """

        # Update hyperparameters if continuing training
        if self.opt.which_epoch > 0:
            self.model.update_hyperparams(self.opt.which_epoch)

        for epoch in range(self.opt.which_epoch + 1, self.opt.niter + self.opt.niter_decay + 1):
            epoch_start_time = time.time()
            epoch_iter = 0
            for i, data in enumerate(self.dataset):
                iter_start_time = time.time()
                self.total_steps += self.opt.batchSize
                epoch_iter += self.opt.batchSize
                self.model.set_input(data)
                self.model.optimize_parameters()

                if self.total_steps % self.opt.display_freq == 0:
                    self.visualizer.display_current_results(self.model.get_current_visuals(), epoch)

                if self.total_steps % self.opt.print_freq == 0:
                    errors = self.model.get_current_errors()
                    t = (time.time() - iter_start_time) / self.opt.batchSize
                    self.visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                    # display on the visdom
                    if self.opt.display_id > 0:
                        self.visualizer.plot_current_errors(epoch, float(epoch_iter)/len(self.dataset), self.opt, errors)

            if epoch % self.opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' % (epoch, self.total_steps))
                self.model.save(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, self.opt.niter + self.opt.niter_decay, time.time() - epoch_start_time))

            # update hyperparameters every epoch
            self.model.update_hyperparams(epoch)

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()


