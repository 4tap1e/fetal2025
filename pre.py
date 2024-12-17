import os
import itertools
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy.ndimage import zoom
from scipy import ndimage
from torch.utils.data.sampler import Sampler


class FetalDataset(Dataset):
    def __init__(self, labeled_data_dir, unlabeled_data_dir, transform=None):
        """
        初始化数据集。
        
        :param labeled_data_dir: 包含有标签数据的目录。
        :param unlabeled_data_dir: 包含无标签数据的目录。
        :param transform: 用于图像的预处理操作。
        """
        self.labeled_data_dir = labeled_data_dir
        self.unlabeled_data_dir = unlabeled_data_dir
        self.transform = transform
        
        # 加载有标签数据
        self.labeled_images = sorted(os.listdir(os.path.join(labeled_data_dir, 'images')))
        self.labeled_labels = sorted(os.listdir(os.path.join(labeled_data_dir, 'labels')))
        
        # 加载无标签数据
        self.unlabeled_images = sorted(os.listdir(unlabeled_data_dir))

        # 计算总的数据条数
        self.total_labeled = len(self.labeled_images)
        self.total_unlabeled = len(self.unlabeled_images)

    def __len__(self):
        """
        返回数据集的总大小，包括有标签和无标签的数据。
        """
        return self.total_labeled + self.total_unlabeled

    def __getitem__(self, idx):
        """
        根据索引返回一个样本。
        对于有标签的数据，返回图像和标签。
        对于无标签的数据，返回图像和空的标签。
        """
        if idx < self.total_labeled:
            # 处理有标签数据
            img_name = os.path.join(self.labeled_data_dir, 'images', self.labeled_images[idx])
            label_name = os.path.join(self.labeled_data_dir, 'labels', self.labeled_labels[idx])
            
            image = Image.open(img_name).convert('RGB')
            label = Image.open(label_name)

            # 对图像和标签进行预处理
            if self.transform:
                image = self.transform(image)
                label = self.transform(label)

            return image, label
        else:
            # 处理无标签数据
            img_name = os.path.join(self.unlabeled_data_dir, self.unlabeled_images[idx - self.total_labeled])
            
            image = Image.open(img_name).convert('RGB')

            # 对图像进行预处理
            if self.transform:
                image = self.transform(image)

            # 返回无标签数据时，标签为 None
            return image, None
        
class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image

def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label

class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        return sample
    
def iterate_once(iterable):
    return np.random.permutation(iterable)

def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)