import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

import build_dataset


class SatelliteData(Dataset):
    """
    Define the Satellite Dataset.
    """

    def __init__(self, data_dir, output_variable, split):
        """
        Store the data filtered by the selected output variable.
        :param data_dir: (str) path to split dataset locations
        :param output_variable: (str) output variable
        :param split: (str) one of ['train', 'val', 'test']
        """
        # Load file
        data_path = os.path.join(data_dir, '{}_{}_split.npz'.format(
            output_variable, split))
        try:
            data = np.load(data_path)
        except FileNotFoundError:
            print('[ERROR] Dataset not found.')

        # Get features and labels
        self.X = data['X']
        self.m = self.X.shape[0]
        self.Y = data['y'].reshape(self.m, 1)

        # Get image information
        self.res = self.X.shape[1]
        self.num_channels = self.X.shape[3]

    def __len__(self):
        """
        Returns the size of the dataset
        :return: (int)
        """
        return self.m

    def __getitem__(self, item):
        """
        Returns a single datapoint given an index
        :param item: (int)
        :return: a tuple containing the image (res, res, num_bands) and label
        """
        return self.X[item, :], self.Y[item, :]


def fetch_dataloader(dataset_types, data_dir, output_variable, params,
                     base_data_file, data_split):
    """
    Fetches the DataLoader object for each type of data.
    :param dataset_types: (list) list including ['train', 'val', 'test']
    :param data_dir: (str) path to the split dataset directory
    :param output_variable: (str) selected output variable
    :param params: (dict) a dictionary containing the model specifications
    :param base_data_file: (str) Path to the file generated by the GGEarth
        script
    :param data_split: (list) containing the % of each split in the order
        [size_train, size_val, size_test]
    :return: dataloaders (dict) a dictionary containing the DataLoader object
        for each type of data
    """

    # Build datasets for selected output variable if they do not exist
    file_path = os.path.join(data_dir,
                             '{}_{}_split.npz'.format(output_variable, 'train'))
    if not os.path.exists(file_path):
        print('[INFO] Building dataset...')
        build_dataset.process_sat_data(
            base_data_file, data_dir, output_variable, data_split)

    # Use GPU if available
    use_cuda = torch.cuda.is_available()

    # Get data loaders
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in dataset_types:
            data = SatelliteData(data_dir, output_variable, split)
            dl = DataLoader(
                dataset=data, batch_size=params['batch_size'], shuffle=True,
                num_workers=params['num_workers'], pin_memory=use_cuda)
            dataloaders[split] = dl

    return dataloaders
