from enum import Enum
from torchvision.datasets import MNIST, SVHN, USPS
import ada.datasets.preprocessing as proc
from ada.datasets.dataset_access import DatasetAccess


class DigitDataset(Enum):
    MNIST = "MNIST"
    USPS = "USPS"
    SVHN = "SVHN"

    @staticmethod
    def get_accesses(source: "DigitDataset", target: "DigitDataset", data_path):
        channel_numbers = {
            DigitDataset.MNIST: 3,
            DigitDataset.USPS: 3,
            DigitDataset.SVHN: 3,
        }

        transform_names = {
            (DigitDataset.MNIST, 3): "mnist32rgb",
            (DigitDataset.USPS, 3): "usps32rgb",
            (DigitDataset.SVHN, 3): "svhn",
        }

        factories = {
            DigitDataset.MNIST: MNISTDatasetAccess,
            DigitDataset.USPS: USPSDatasetAccess,
            DigitDataset.SVHN: SVHNDatasetAccess,
        }

        # handle color/nb channels
        num_channels = max(channel_numbers[source], channel_numbers[target])
        source_tf = transform_names[(source, num_channels)]
        target_tf = transform_names[(target, num_channels)]

        return (
            factories[source](data_path, source_tf),
            factories[target](data_path, target_tf),
            num_channels,
        )


class DigitDatasetAccess(DatasetAccess):
    def __init__(self, data_path, transform_kind):
        super().__init__(n_classes=10)
        self._data_path = data_path
        self._transform = proc.get_transform(transform_kind)


class MNISTDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        print("DATAPATH: ", self._data_path)
        #MNIST.resources = [
        #    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        #    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
        #    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
        #    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        #]
        return MNIST(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        print("DATAPATH: ", self._data_path)
        #MNIST.resources = [
        #    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz', 'f68b3c2dcbeaaa9fbdd348bbdeb94873'),
        #    ('https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz', 'd53e105ee54ea40749a09fcbcd1e9432'),
        #    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz', '9fb629c4189551a2d022fa330f9573f3'),
        #    ('https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz', 'ec29112dd5afa0611ce80d1b7f02629c')
        #]

        return MNIST(
            self._data_path, train=False, transform=self._transform, download=True
        )

class USPSDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return USPS(
            self._data_path, train=True, transform=self._transform, download=True
        )

    def get_test(self):
        
        return USPS(
            self._data_path, train=False, transform=self._transform, download=True
        )


class SVHNDatasetAccess(DigitDatasetAccess):
    def get_train(self):
        return SVHN(
            self._data_path, split="train", transform=self._transform, download=True
        )

    def get_test(self):
        return SVHN(
            self._data_path, split="test", transform=self._transform, download=True
        )
