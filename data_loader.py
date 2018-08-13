
from torch.utils.data import Dataset


class PascalVOC(Dataset):

    def __int__(self):
        """
        Initialize dataset
        """

        # TODO: get data
        # TODO: transform data
        # TODO: set self.N

    def __len__(self):
        """
        Number of data in the dataset
        """
        return self.N

    def __getitem__(self, index):
        """
        Return item from dataset
        """
        pass


def get_loader(data_path, mode='train'):
    """
    Get dataset loader
    """
    pass
