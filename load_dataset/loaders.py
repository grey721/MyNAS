from __future__ import annotations

from enum import Enum
from typing import Literal

import numpy as np
import torch
import torch.utils.data as tdata
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from load_dataset.autoaugment import CIFAR10Policy, ImageNetPolicy
from load_dataset.random_erasing import RandomErasing


_DEFAULT_DATA_ROOTS = {
    'cifar10':  '../datasets/CIFAR10_data',
    'cifar100': '../datasets/CIFAR100_data',
    'imagenet': '../datasets/imagenet/ILSVRC2012',
}


_NORM: dict[str, transforms.Normalize] = {
    'cifar10': transforms.Normalize(
        mean=[0.49139968, 0.48215827, 0.44653124],
        std=[0.24703233, 0.24348505, 0.26158768],
    ),
    'cifar100': transforms.Normalize(
        mean=[0.5071, 0.4865, 0.4409],
        std=[0.2673, 0.2564, 0.2762],
    ),
    'imagenet': transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
}


class AugLevel(Enum):
    """
    训练增强的强度级别，传给 get_train_loader / get_nas_loader 等函数。

    NONE
        仅 ToTensor + Normalize。
        适用场景：NAS 零成本代理评估（增强会干扰 kernel 矩阵评分稳定性）。

    BASIC
        CIFAR    : RandomCrop(32, padding=4) + RandomHorizontalFlip + Cutout
        ImageNet : RandomResizedCrop(224) + RandomHorizontalFlip
                   + ColorJitter + RandomErasing

    STRONG
        BASIC 的全部增强 + AutoAugment policy（CIFAR10Policy / ImageNetPolicy）。
        训练时间较长，但在 CIFAR-100 / ImageNet 上通常带来 0.5~1% 的精度提升。
    """
    NONE = 0
    BASIC = 1
    STRONG = 2


def _validate_dataset(dataset: str) -> None:
    if dataset not in _NORM:
        raise ValueError(f'不支持的数据集: {dataset!r}，可选: {list(_NORM)}')


class Cutout:
    def __init__(self, patch_size: int):
        self.patch_size = patch_size

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.size(1), img.size(2)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1 = max(cy - self.patch_size // 2, 0)
        y2 = min(cy + self.patch_size // 2, h)
        x1 = max(cx - self.patch_size // 2, 0)
        x2 = min(cx + self.patch_size // 2, w)
        mask = np.ones((h, w), np.float32)
        mask[y1:y2, x1:x2] = 0.
        return img * torch.from_numpy(mask).expand_as(img)


class BatchMixTransform:
    """
    Batch 级数据增强，在训练循环中手动调用（不挂在 DataLoader collate 上）。

    支持三种模式
    ------------
    - 纯 Mixup   (mixup_alpha > 0, cutmix_alpha = 0)
    - 纯 CutMix  (cutmix_alpha > 0, mixup_alpha = 0)
    - 随机混用   (两者均 > 0，每 batch 各以 50% 概率选一种)

    与 Cutout 的关系
    ----------------
    推荐配置：
      · 启用 CutMix  → 关闭 Cutout（传 use_cutout=False 给 get_train_loader）
      · 仅启用 Mixup → Cutout 可保留

    返回值与 CrossEntropyLoss 的兼容性
    -----------------------------------
    本变换将整型 label [B] 转为 soft one-hot float [B, C]。
    PyTorch ≥ 1.10 的 nn.CrossEntropyLoss 可直接接受，无需修改 criterion。

    用法示例
    --------
        mixer = BatchMixTransform(num_classes=100, cutmix_alpha=1.0)
        for inputs, labels in train_loader:
            inputs, soft_labels = mixer(inputs.cuda(), labels.cuda())
            loss = criterion(model(inputs), soft_labels)
    """

    def __init__(
            self,
            num_classes: int,
            mixup_alpha: float = 0.0,
            cutmix_alpha: float = 0.0,
    ):
        if mixup_alpha == 0.0 and cutmix_alpha == 0.0:
            raise ValueError('mixup_alpha 和 cutmix_alpha 至少有一个 > 0')
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(
            self,
            images: torch.Tensor,  # [B, C, H, W]
            labels: torch.Tensor,  # [B]，整型
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 (mixed_images [B,C,H,W], soft_labels [B, num_classes])。"""
        soft_labels = torch.zeros(
            labels.size(0), self.num_classes,
            dtype=torch.float32, device=labels.device,
        ).scatter_(1, labels.view(-1, 1), 1.0)

        use_both = self.mixup_alpha > 0 and self.cutmix_alpha > 0
        apply_cutmix = (use_both and torch.rand(1).item() < 0.5) or \
                       (not use_both and self.cutmix_alpha > 0)

        return self._cutmix(images, soft_labels) if apply_cutmix \
            else self._mixup(images, soft_labels)

    def _mixup(
            self, images: torch.Tensor, soft_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        perm = torch.randperm(images.size(0), device=images.device)
        return (
            lam * images + (1.0 - lam) * images[perm],
            lam * soft_labels + (1.0 - lam) * soft_labels[perm],
        )

    def _cutmix(
            self, images: torch.Tensor, soft_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        lam_nominal = float(np.random.beta(self.cutmix_alpha, self.cutmix_alpha))
        perm = torch.randperm(images.size(0), device=images.device)

        _, _, img_h, img_w = images.shape
        cut_h = int(img_h * np.sqrt(1.0 - lam_nominal))
        cut_w = int(img_w * np.sqrt(1.0 - lam_nominal))
        cy, cx = np.random.randint(img_h), np.random.randint(img_w)
        y1, y2 = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, img_h)
        x1, x2 = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, img_w)

        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[perm, :, y1:y2, x1:x2]

        # 用实际裁剪面积重算 lambda（边界 clip 后真实面积略小于名义值）
        lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (img_h * img_w)
        mixed_labels = lam_actual * soft_labels + (1.0 - lam_actual) * soft_labels[perm]
        return mixed_images, mixed_labels


def _build_train_transform(
        dataset: str,
        aug_level: AugLevel,
        use_cutout: bool = False,
) -> transforms.Compose:
    """
    aug_level = NONE
        [ToTensor, Normalize]

    aug_level = BASIC  (cifar)
        [RandomCrop, HFlip, ToTensor, Normalize, Cutout?]
    aug_level = BASIC  (imagenet)
        [RandomResizedCrop, HFlip, ColorJitter, ToTensor, Normalize, RandomErasing]

    aug_level = STRONG (cifar)
        [RandomCrop, HFlip, CIFAR10Policy, ToTensor, Normalize, Cutout?]
    aug_level = STRONG (imagenet)
        [RandomResizedCrop, HFlip, ColorJitter, ImageNetPolicy, ToTensor, Normalize, RandomErasing]
    """
    normalize = _NORM[dataset]
    if aug_level == AugLevel.NONE:
        if dataset == 'imagenet':
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                normalize,
            ])
        return transforms.Compose([transforms.ToTensor(), normalize])

    # autoaugment
    if dataset == 'imagenet':
        ops = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        ]
        if aug_level == AugLevel.STRONG:
            ops.append(ImageNetPolicy())
        ops += [
            transforms.ToTensor(),
            normalize,
            RandomErasing(probability=0.2, mode='pixel', max_count=1,
                          num_splits=False, device='cpu'),
        ]

    else:  # cifar
        ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
        if aug_level == AugLevel.STRONG:
            ops.append(CIFAR10Policy())
        ops += [transforms.ToTensor(), normalize]
        if use_cutout:
            ops.append(Cutout(patch_size=16))

    return transforms.Compose(ops)


def _build_eval_transform(
        dataset: str,
) -> transforms.Compose:
    normalize = _NORM[dataset]
    if dataset == 'imagenet':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    return transforms.Compose([transforms.ToTensor(), normalize])


def _make_torchvision_dataset(
        dataset: str,
        split: Literal['train', 'val'],
        transform,
) -> tdata.Dataset:
    is_train = (split == 'train')
    if dataset == 'cifar10':
        return datasets.CIFAR10(root=_DEFAULT_DATA_ROOTS[dataset], train=is_train, download=True, transform=transform)
    if dataset == 'cifar100':
        return datasets.CIFAR100(root=_DEFAULT_DATA_ROOTS[dataset], train=is_train, download=True, transform=transform)
    return datasets.ImageNet(root=_DEFAULT_DATA_ROOTS[dataset], split=split, transform=transform)


class _InfiniteRandomBatchSampler:
    def __init__(self, num_samples: int, batch_size: int, generator=None):
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.generator = generator

    def __iter__(self):
        while True:
            yield torch.randperm(
                self.num_samples, generator=self.generator
            )[:self.batch_size].tolist()


def get_train_loader(
        dataset: str,
        batch_size: int,
        aug_level: AugLevel = AugLevel.BASIC,
        use_cutout: bool = True,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
) -> tdata.DataLoader:
    _validate_dataset(dataset)
    tf = _build_train_transform(dataset, aug_level=aug_level, use_cutout=use_cutout)
    data = _make_torchvision_dataset(dataset, split='train', transform=tf)

    return tdata.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def get_test_loader(
        dataset: str,
        batch_size: int,
        shuffle: bool = False,
        num_workers: int = 4,
        pin_memory: bool = True,
) -> tdata.DataLoader:
    _validate_dataset(dataset)
    tf = _build_eval_transform(dataset)
    data = _make_torchvision_dataset(dataset, split='val', transform=tf)

    return tdata.DataLoader(
        data, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def get_nas_loader(
        dataset: str,
        batch_size: int,
        aug_level: AugLevel = AugLevel.NONE,
        num_workers: int = 4,
        pin_memory: bool = True,
        generator=None,
) -> tdata.DataLoader:
    """
    NAS 零成本代理专用 loader：无限随机单 batch，有放回采样。
    用法：
        loader = get_nas_loader('cifar100', batch_size=128)
        it = iter(loader)
        for individual in population:
            inputs, labels = next(it)
            score = proxy_score(individual, inputs, labels)
    """
    _validate_dataset(dataset)
    tf = _build_train_transform(
        dataset,
        aug_level=aug_level,
        use_cutout=False,
    )
    data = _make_torchvision_dataset(dataset, split='train', transform=tf)
    sampler = _InfiniteRandomBatchSampler(len(data), batch_size, generator=generator)

    return tdata.DataLoader(
        data, batch_sampler=sampler,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
    )


def get_debug_loaders(
        dataset: str,
        batch_size: int,
        sample_fraction: float = 0.01,
        aug_level: AugLevel = AugLevel.BASIC,
        num_workers: int = 4,
        pin_memory: bool = True,
        use_cutout: bool = False,
) -> tuple[tdata.DataLoader, tdata.DataLoader]:
    """
    快速 Debug 专用：返回 (train_loader, test_loader)，
    各自只包含约 sample_fraction 比例的随机子集。

    Parameters
    ----------
    sample_fraction : 采样比例，默认 0.01（即 1% 数据，约 500 张 CIFAR-100）
    aug_level       : 训练集的增强强度，测试集始终无增强
    """
    assert 0.0 < sample_fraction <= 1.0, 'sample_fraction 需在 (0, 1]'
    _validate_dataset(dataset)

    train_data = _make_torchvision_dataset(
        dataset, split='train',
        transform=_build_train_transform(dataset, aug_level=aug_level, use_cutout=use_cutout),
    )
    test_data = _make_torchvision_dataset(
        dataset, split='val',
        transform=_build_eval_transform(dataset),
    )

    def _random_subset_sampler(ds: tdata.Dataset) -> SubsetRandomSampler:
        n = max(1, int(len(ds) * sample_fraction))
        return SubsetRandomSampler(torch.randperm(len(ds))[:n].tolist())

    kw = dict(batch_size=batch_size, num_workers=num_workers,
              pin_memory=pin_memory, persistent_workers=(num_workers > 0))
    return (
        tdata.DataLoader(train_data, sampler=_random_subset_sampler(train_data), **kw),
        tdata.DataLoader(test_data, sampler=_random_subset_sampler(test_data), **kw),
    )
