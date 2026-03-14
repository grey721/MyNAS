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

DEFAULT_DATA_ROOTS = {
    'cifar10': '../datasets/CIFAR10_data',
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
    Training augmentation intensity level.

    Passed to ``get_train_loader`` / ``get_nas_loader``.

    NONE
        ToTensor + Normalize only.
        Recommended for NAS zero-cost proxy evaluation, where augmentation
        can destabilise kernel-matrix scoring.

    BASIC
        CIFAR    : RandomCrop(32, padding=4) + RandomHorizontalFlip + Cutout.
        ImageNet : RandomResizedCrop(224) + RandomHorizontalFlip
                   + ColorJitter + RandomErasing.

    STRONG
        All BASIC augmentations plus an AutoAugment policy
        (CIFAR10Policy / ImageNetPolicy).  Typically yields 0.5–1 % accuracy
        gains on CIFAR-100 / ImageNet at the cost of longer training.
    """
    NONE = 0
    BASIC = 1
    STRONG = 2


def _validate_dataset(dataset: str) -> None:
    if dataset not in _NORM:
        raise ValueError(f"Unsupported dataset: {dataset!r}. Available: {list(_NORM)}")


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
    Batch-level data augmentation, applied manually inside the training loop
    (not hooked into the DataLoader collate function).

    Supported modes
    ---------------
    - Pure Mixup   (mixup_alpha > 0, cutmix_alpha = 0)
    - Pure CutMix  (cutmix_alpha > 0, mixup_alpha = 0)
    - Random mix   (both > 0; each batch randomly selects one at 50/50)

    Interaction with Cutout
    -----------------------
    Recommended setup:
      - With CutMix  -> disable Cutout (pass ``use_cutout=False`` to ``get_train_loader``).
      - Mixup only   -> Cutout can remain enabled.

    Compatibility with CrossEntropyLoss
    ------------------------------------
    This transform converts integer labels ``[B]`` into soft one-hot floats
    ``[B, C]``.  PyTorch >= 1.10 ``nn.CrossEntropyLoss`` accepts this directly.

    Example
    -------
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
            raise ValueError("At least one of mixup_alpha or cutmix_alpha must be > 0.")
        self.num_classes = num_classes
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha

    def __call__(
            self,
            images: torch.Tensor,  # [B, C, H, W]
            labels: torch.Tensor,  # [B], integer
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(mixed_images [B,C,H,W], soft_labels [B, num_classes])``."""
        soft_labels = torch.zeros(
            labels.size(0), self.num_classes,
            dtype=torch.float32, device=labels.device,
        ).scatter_(1, labels.view(-1, 1), 1.0)

        use_both = self.mixup_alpha > 0 and self.cutmix_alpha > 0
        apply_cutmix = (use_both and torch.rand(1).item() < 0.5) or (not use_both and self.cutmix_alpha > 0)

        return self._cutmix(images, soft_labels) if apply_cutmix else self._mixup(images, soft_labels)

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

        # Recompute lambda from actual crop area (boundary clipping reduces it slightly).
        lam_actual = 1.0 - (y2 - y1) * (x2 - x1) / (img_h * img_w)
        mixed_labels = lam_actual * soft_labels + (1.0 - lam_actual) * soft_labels[perm]
        return mixed_images, mixed_labels


def _build_train_transform(
        dataset: str,
        aug_level: AugLevel,
        use_cutout: bool = False,
) -> transforms.Compose:
    """
    Build the training transform pipeline for a given augmentation level.

    NONE  : [ToTensor, Normalize]
    BASIC  (cifar): [RandomCrop, HFlip, ToTensor, Normalize, Cutout?]
    BASIC  (imagenet): [RandomResizedCrop, HFlip, ColorJitter, ToTensor, Normalize, RandomErasing]
    STRONG (cifar): [RandomCrop, HFlip, CIFAR10Policy, ToTensor, Normalize, Cutout?]
    STRONG (imagenet): [RandomResizedCrop, HFlip, ColorJitter, ImageNetPolicy,
                        ToTensor, Normalize, RandomErasing]
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
        return datasets.CIFAR10(root=DEFAULT_DATA_ROOTS[dataset], train=is_train, download=True, transform=transform)
    if dataset == 'cifar100':
        return datasets.CIFAR100(root=DEFAULT_DATA_ROOTS[dataset], train=is_train, download=True, transform=transform)
    return datasets.ImageNet(root=DEFAULT_DATA_ROOTS[dataset], split=split, transform=transform)


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
    Infinite random single-batch loader for NAS zero-cost proxy evaluation.

    Samples are drawn with replacement so the iterator never exhausts,
    regardless of how many generations the search runs.

    Example
    -------
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
    Fast-debug loaders: return ``(train_loader, test_loader)`` each containing
    a random subset of approximately ``sample_fraction`` of the full dataset.

    Parameters
    ----------
    sample_fraction:
        Fraction of data to include (default 0.01 ≈ 500 samples for CIFAR-100).
    aug_level:
        Augmentation level for the training split; the test split is always
        evaluated without augmentation.
    """
    assert 0.0 < sample_fraction <= 1.0, 'sample_fraction must be in (0, 1]'
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
