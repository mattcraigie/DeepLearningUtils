import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt

# Datasets


class DataHandler:
    def __init__(self, data, targets=None):
        """
        This class takes the data and manipulates it into the form needed for training
        :param data:
        :param targets:
        """
        self.data = data
        self.targets = targets

        self.use_targets = True if targets is not None else False

    def make_dataloaders(self, batch_size, val_fraction=0.2, shuffle=True, dataset_class=None):
        """
        Makes dataloaders for the simulated data
        :param batch_size: the batch size for the dataloaders
        :param val_fraction: the fraction of the data to use for validation
        :param shuffle: whether or not to shuffle the data
        :param dataset_class: the class to use for the dataset. Defaults to TensorDataset
        :return:
        """

        if dataset_class is None:
            dataset_class = TensorDataset

        # shuffle data
        num_data = self.data.shape[0]
        train_fraction = 1 - val_fraction
        num_train = int(num_data * train_fraction)
        shuffle_idx = torch.randperm(num_data) if shuffle else torch.arange(num_data)
        train_idx = shuffle_idx[:num_train]
        val_idx = shuffle_idx[num_train:]

        train_data = self.data[train_idx]
        val_data = self.data[val_idx]

        if self.use_targets:
            val_targets = self.targets[val_idx]
            train_targets = self.targets[train_idx]
            train_dataset = dataset_class(train_data, train_targets)
            val_dataset = dataset_class(val_data, val_targets)
        else:
            train_dataset = dataset_class(train_data)
            val_dataset = dataset_class(val_data)

        # create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

        return train_loader, val_loader

    def show_data(self, num_to_show=1, style=None, extra_index=0, index_data=0, index_target=0):

        assert style in [None, 'image', 'scatter'], "Invalid style must be None, 'image', or 'scatter'"

        if style == None:
            print(self.data[:num_to_show])

        elif style == 'image':
            for i in range(num_to_show):
                data = self.data[i]

                for _ in range(extra_index):
                    data = data[0]

                plt.imshow(data)
                plt.show()

        elif style == 'scatter':
            plt.scatter(self.data[:, index_data], self.targets[:, index_target])
            plt.show()

    def make_single_dataloader(self, batch_size, shuffle=True, dataset_class=None):

        if dataset_class is None:
            dataset_class = TensorDataset

        if self.use_targets:
            dataset = dataset_class(self.data, self.targets)
        else:
            dataset = dataset_class(self.data)

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        return dataloader