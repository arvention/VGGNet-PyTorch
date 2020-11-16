import torch
from data.ip102 import IP102
from torch.utils.data import DataLoader
from data.augmentations import Augmentations, BaseTransform


def collate(batch):
    images = []
    targets = []
    for sample in batch:
        images.append(sample[0])
        targets.append(int(sample[1]))
    return torch.stack(images, 0), targets


def get_loader(config):

    dataset = None
    loader = None

    if config.dataset == 'ip102':
        if config.mode == 'train':
            image_transform = Augmentations(config.new_size, config.means)
            dataset = IP102(data_path=config.ip102_data_path,
                            mode='train',
                            new_size=config.new_size,
                            image_transform=image_transform)

        elif config.mode == 'val':
            image_transform = BaseTransform(config.new_size, config.means)
            dataset = IP102(data_path=config.ip102_data_path,
                            mode='val',
                            new_size=config.new_size,
                            image_transform=image_transform)

        elif config.mode == 'test':
            image_transform = BaseTransform(config.new_size, config.means)
            dataset = IP102(data_path=config.ip102_data_path,
                            mode='test',
                            new_size=config.new_size,
                            image_transform=image_transform)

    if dataset is not None:
        if config.mode == 'train':
            loader = DataLoader(dataset=dataset,
                                batch_size=config.batch_size,
                                shuffle=True,
                                collate_fn=collate,
                                num_workers=8,
                                pin_memory=True)

        elif config.mode == 'val' or config.mode == 'test':
            loader = DataLoader(dataset=dataset,
                                batch_size=config.batch_size,
                                shuffle=False,
                                collate_fn=collate,
                                num_workers=8,
                                pin_memory=True)

    return loader
