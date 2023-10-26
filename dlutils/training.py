import torch
from torch import nn
import copy
import numpy as np
import matplotlib.pyplot as plt


# Note that criterion in this code includes the model as an argument, for more flexibility than standard output/target
def general_criterion(criterion):
    func = lambda model, data, target: criterion(model(data), target)
    return func


def mse_criterion():
    return general_criterion(nn.MSELoss())


def batch_apply(model, dataloader, device, return_targets=True):

    model.eval()
    with torch.no_grad():

        outputs = []
        targets = []
        for data, target in dataloader:
            data = data.to(device)
            output = model(data)
            outputs.append(output)
            targets.append(target)

        if return_targets:
            return torch.cat(outputs, dim=0), torch.cat(targets, dim=0)
        return torch.cat(outputs, dim=0)


class RegressionTrainer:

    def __init__(self,
                 model,
                 train_dataloader,
                 val_dataloader,
                 criterion=None,
                 optimizer=None,
                 scheduler=None,
                 no_targets=False,
                 device=torch.device("cpu")):
        """

        :param model: the model to train, call/forward method should take in data and return output
        :param train_dataloader: dataloader for training data. Should be in form (data, target)
        :param val_dataloader: dataloader for validation data. Should be in form (data, target)
        :param criterion: loss function to use. Should take in model, data, and target and return loss, or model, data, and return loss if no_targets is True
        :param optimizer: optimizer to use. Defaults to Adam
        :param scheduler: scheduler to use. Defaults to no scheduler
        :param device: device to use. Defaults to cpu
        """

        if criterion is None:
            print("No criterion provided, using MSE")
            criterion = mse_criterion

        if optimizer is None:
            print("No optimizer provided, using Adam")
            optimizer = torch.optim.Adam(model.parameters())

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_model_params = None
        self.best_model_loss = None
        self.train_losses = []
        self.val_losses = []
        self.no_targets = no_targets

    def train_step(self, model, dataloader, criterion, optimizer, device):
        model.train()
        epoch_loss = 0
        for item in dataloader:
            optimizer.zero_grad()
            if self.no_targets:
                (data,) = item
                data = data.to(device)
                loss = criterion(model, data)
            else:
                data, target = item
                data, target = data.to(device), target.to(device)
                loss = criterion(model, data, target)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(dataloader)

    def val_step(self, model, dataloader, criterion, device):
        model.eval()
        with torch.no_grad():
            epoch_loss = 0
            for item in dataloader:

                if self.no_targets:
                    (data,) = item
                    data = data.to(device)
                    loss = criterion(model, data)
                else:
                    data, target = item
                    data, target = data.to(device), target.to(device)
                    loss = criterion(model, data, target)

                epoch_loss += loss.item()
            return epoch_loss / len(dataloader)

    def train_loop(self, model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs, device,
                   print_progress=False):
        train_losses = []
        val_losses = []

        best_loss = np.inf
        print_every = epochs // 10

        if print_every == 0:
            print_every = 1

        if print_progress:
            print("Beginning training for {} epochs\n".format(epochs))

        for epoch in range(epochs):
            train_loss = self.train_step(model, train_dataloader, criterion, optimizer, device)
            if self.scheduler is not None:
                scheduler.step()

            val_loss = self.val_step(model, val_dataloader, criterion, device)

            if val_loss < best_loss:
                self.best_model_params = copy.deepcopy(model.state_dict())
                self.best_model_loss = val_loss

            if print_progress and epoch % print_every == 0:
                print(f"Epoch {epoch + 1} \t| Train Loss: {train_loss:.3e} \t| Val Loss: {val_loss:.3e}")

            train_losses.append(train_loss)
            val_losses.append(val_loss)

        if print_progress:
            print("\nTraining complete. Final train loss: {:.3e} | Final val loss: {:.3e}".format(train_losses[-1],
                                                                                                  val_losses[-1]))

        return train_losses, val_losses

    def run_training(self, epochs, lr, print_progress=False, show_loss_plot=False):

        # update lr for optimizer
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        train_losses, val_losses = self.train_loop(self.model,
                                                   self.train_dataloader,
                                                   self.val_dataloader,
                                                   self.criterion,
                                                   self.optimizer,
                                                   self.scheduler,
                                                   epochs,
                                                   self.device,
                                                   print_progress=print_progress)

        self.train_losses.extend(train_losses)
        self.val_losses.extend(val_losses)

        if show_loss_plot:
            self.loss_plot()

        return train_losses, val_losses

    def loss_plot(self, logx=False, logy=False, save_path=None):
        plt.figure(figsize=(12, 8))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')

        if logx and logy:
            plt.loglog()
        elif logx:
            plt.semilogx()
        elif logy:
            plt.semilogy()

        plt.legend()

        if save_path is None:
            plt.show()
        else:
            plt.savefig(save_path)

    def get_best_model(self):
        self.model.load_state_dict(self.best_model_params)
        return self.model, self.best_model_loss

    def show_model_performance(self, load_best=False):

        if load_best:
            self.model.load_state_dict(self.best_model_params)

        train_results, train_targets = batch_apply(self.model, self.train_dataloader, self.device)
        val_results, val_targets = batch_apply(self.model, self.val_dataloader, self.device)

        train_results, train_targets = train_results.cpu().numpy(), train_targets.cpu().numpy()
        val_results, val_targets = val_results.cpu().numpy(), val_targets.cpu().numpy()

        num_targets = train_targets.shape[1]

        fig, axes = plt.subplots(ncols=num_targets, figsize=(num_targets*4, 4))

        if num_targets == 1:
            axes = [axes]

        for i in range(num_targets):

            axes[i].scatter(train_targets[:, i], train_results[:, i], label="Train")
            axes[i].scatter(val_targets[:, i], val_results[:, i], label="Validation")
            axes[i].set_xlabel("Target")
            axes[i].set_ylabel("Prediction")
            axes[i].legend()

            # make a 1-1 line
            min_val = min(train_targets[:, i].min(), val_targets[:, i].min())
            max_val = max(train_targets[:, i].max(), val_targets[:, i].max())
            axes[i].plot([min_val, max_val], [min_val, max_val], color="black")

        plt.show()