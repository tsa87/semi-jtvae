"""MNIST DataModule"""
import argparse

from torch.utils.data import random_split
from torch_geometric.datasets import ZINC

from molecule_optimizer.data.base_data_module import (
    BaseDataModule,
    load_and_print_info,
)

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


class MNIST(BaseDataModule):
    """
    MNIST DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.transform = 
        self.dims = (
            1,
            28,
            28,
        )  # dims are returned when calling `.size()` on this object.
        self.output_dims = (1,)
        self.mapping = list(range(10))

    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""
        

    def setup(self, stage=None) -> None:
        """Split into train, val, test, and set dims."""
        mnist_full = TorchMNIST(
            self.data_dir, train=True, transform=self.transform
        )
        self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])  # type: ignore
        self.data_test = TorchMNIST(
            self.data_dir, train=False, transform=self.transform
        )


if __name__ == "__main__":
    load_and_print_info(MNIST)
