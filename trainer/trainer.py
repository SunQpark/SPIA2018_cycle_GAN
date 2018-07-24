import numpy as np
import torch
from torchvision.utils import make_grid
from torch.autograd import Variable
from base import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class

    Note:
        Inherited from BaseTrainer.
        Modify __init__() if you have additional arguments to pass.
    """
    def __init__(self, model, loss, recon_loss, metrics, data_loader, batch_size, optimizer, epochs,
                 save_dir, save_freq, resume, device, verbosity, training_name='',
                 valid_data_loader=None, train_logger=None, writer=None, lr_scheduler=None, monitor='loss', monitor_mode='min'):
        super(Trainer, self).__init__(model, loss, metrics, data_loader, valid_data_loader, optimizer, epochs,
                                      batch_size, save_dir, save_freq, resume, verbosity, training_name,
                                      device, train_logger, writer, monitor, monitor_mode)
        self.scheduler = lr_scheduler
        self.recon_loss = recon_loss

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current training epoch.
        :return: A log that contains all information you want to save.

        Note:
            If you have additional information to record, for example:
                > additional_log = {"x": x, "y": y}
            merge it with log before return. i.e.
                > log = {**log, **additional_log}
                > return log
        """
        self.model.train()
        self.model.to(self.device)

        total_loss = 0
        data_loader = self.data_loader(self.batch_size)
        # length = len(self.data_loader) #TODO: fix this
        length = 400

        total_metrics = np.zeros(len(self.metrics))
        for batch_idx, (data, label) in enumerate(data_loader):
            data = data.to(self.device)
            data_in_A = (label == 0)

            real_label = 1
            fake_label = 0
            for _, opt in self.optimizer.items():
                opt.zero_grad()
            #
            # update discriminator with real data
            #
            if data_in_A:
                direction = 'AtoB'
                other = 'BtoA'
                model = self.model.AtoB
                rev_model = self.model.BtoA
            else:
                direction = 'BtoA'
                other = 'AtoB'
                model = self.model.BtoA
                rev_model = self.model.AtoB

            real_dis_score = rev_model.dis(data)
            real_dis_loss = self.loss(real_dis_score, real_label)
            real_dis_loss.backward()
            self.optimizer[f'{other}_dis'].step()
            #
            # update gen, dis with fake data
            #
            fake_y = model.gen(data)
            fake_dis_score = model.dis(fake_y.detach())
            fake_dis_loss = self.loss(fake_dis_score, fake_label)
            fake_dis_loss.backward()
            self.optimizer[f'{direction}_dis'].step()

            self.optimizer[f'{direction}_gen'].zero_grad()
            fake_dis_score = model.dis(fake_y)
            # update generator with gan loss, reconstruction loss
            gen_loss = self.loss(fake_dis_score, real_label)

            recon_x = rev_model.gen(fake_y)
            recon_loss = self.recon_loss(recon_x, data)
            gen_loss += recon_loss
            gen_loss.backward()

            self.optimizer[f'{direction}_gen'].step()
            self.optimizer[f'{other}_gen'].step()

            self.train_iter += 1
            self.writer.add_scalar(f'{self.training_name}/gen_loss', gen_loss.item(), self.train_iter)
            self.writer.add_scalar(f'{self.training_name}/recon_loss', recon_loss.item(), self.train_iter)
            self.writer.add_scalar(f'{self.training_name}/dis_real_loss', real_dis_loss.item(), self.train_iter)
            self.writer.add_scalar(f'{self.training_name}/dis_fake_loss', fake_dis_loss.item(), self.train_iter)

            loss = real_dis_loss.item() + fake_dis_loss.item() + gen_loss.item()
            total_loss += loss
            log_step = int(np.sqrt(self.batch_size))
            if self.verbosity >= 2 and batch_idx % log_step == 0:
                img_orig = make_grid(data[0:16], nrow=4)
                img_fake = make_grid(fake_y[0:16], nrow=4)
                img_recon = make_grid(recon_x[0:16], nrow=4)
                self.writer.add_image(f'{self.training_name}/original', img_orig, self.train_iter)
                self.writer.add_image(f'{self.training_name}/transfer', img_fake, self.train_iter)
                self.writer.add_image(f'{self.training_name}/reconstruction', img_recon, self.train_iter)
                

                self.logger.info('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                    epoch, batch_idx * len(data), length * len(data),
                    100.0 * batch_idx / length, loss))

        avg_loss = total_loss / length
        avg_metrics = (total_metrics / length).tolist()
        log = {'loss': avg_loss, 'metrics': avg_metrics}

        if self.valid:
            val_log = self._valid_epoch()
            log = {**log, **val_log}

        return log

    def _valid_epoch(self):
        """
        Validate after training an epoch

        :return: A log that contains information about validation
        """
        self.model.eval()
        with torch.no_grad():
            total_val_loss = 0
            total_val_metrics = np.zeros(len(self.metrics))
            for batch_idx, (data, target) in enumerate(self.valid_data_loader):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.loss(output, target)
                total_val_loss += loss.item()

                self.valid_iter += 1
                self.writer.add_scalar(f'{self.training_name}/Valid/loss', loss.item(), self.valid_iter)
                for i, metric in enumerate(self.metrics):
                    score = metric(output, target)
                    total_val_metrics[i] += score
                    self.writer.add_scalar(f'{self.training_name}/Valid/{metric.__name__}', score, self.valid_iter)

            avg_val_loss = total_val_loss / len(self.valid_data_loader)
            if self.scheduler is not None:
                self.scheduler.step(avg_val_loss)
            avg_val_metrics = (total_val_metrics / len(self.valid_data_loader)).tolist()
        return {'val_loss': avg_val_loss, 'val_metrics': avg_val_metrics}
