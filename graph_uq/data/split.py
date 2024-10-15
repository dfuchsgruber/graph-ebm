from typing import Tuple

import torch
from jaxtyping import Bool, jaxtyped
from torch import Tensor
from typeguard import typechecked

from graph_uq.config.data import DatasetSplit, DistributionType
from graph_uq.data import Data, Dataset
from graph_uq.experiment import experiment
from graph_uq.util.sample import sample_from_mask


@jaxtyped(typechecker=typechecked)
def sample_stratified_mask(
    dataset: Data,
    num: int | float,
    mask: Bool[Tensor, "num_nodes"] | None = None,
    generator: torch.Generator | None = None,
    allow_empty_mask: bool = False,
) -> Bool[Tensor, "num_nodes"]:
    """Sample a stratified mask from a dataset.

    Args:
        dataset (_type_): The dataset from which to sample
        num (int | float): How many (absolute or as a fraction) to sample
        mask (Bool[Tensor, &#39;num_nodes&#39;] | None, optional):  A mask of possible indices. Defaults to None.
        generator (torch.Generator | None, optional): A random number generator for sampling. Defaults to None.
        allow_empty_mask (bool, optional): Whether to allow an empty mask. Defaults to False.

    Returns:
        Bool[Tensor, &#39;num_nodes&#39;] | None: A mask of sampled indices
    """
    if mask is None:
        mask = torch.ones(dataset.num_nodes, dtype=torch.bool)
    mask_sampled = torch.zeros(dataset.num_nodes, dtype=torch.bool)
    for y in range(dataset.num_classes):
        mask_sampled |= sample_from_mask(
            (dataset.y == y) & mask,
            num,
            generator=generator,
            allow_empty_mask=allow_empty_mask,
        )
    return mask_sampled


@jaxtyped(typechecker=typechecked)
def apply_dataset_split(
    dataset: Dataset,
    train_size: float | int | None,
    val_size: float | int | None,
    test_size: float | int | None,
):
    """Sample a stratified train/val/test split from a dataset and set's the masks in-place.

    Args:
        dataset (Data): The dataset from which to sample
        train_size (float | int | None): How many (absolute or as a fraction) to sample for the train set.
        val_size (float | int | None): How many (absolute or as a fraction) to sample for the validation set.
        test_size (float | int | None): How many (absolute or as a fraction) to sample for the test set.
    """

    # Generate a fixed test mask on the base data and transfer it to all data derived from it
    rng_test = torch.Generator()
    rng_test.manual_seed(0x123456789)  # some fixed seed for test split
    match test_size:
        case None:
            assert dataset.data_base.test_mask is not None, "Test set is not available."
            test_mask = dataset.data_base.test_mask
        case num if isinstance(num, (int, float)):
            test_mask = sample_stratified_mask(
                dataset.data_base, num, generator=rng_test
            )
        case _:
            raise ValueError(f"Invalid test size {test_size}")

    # Transfer the test mask to all datasets
    dataset.data_base.test_mask = test_mask
    dataset.data_train.test_mask = dataset.data_base.transfer_mask_to(
        dataset.data_base.test_mask, dataset.data_train
    )
    for name, data_ood in dataset.data_shifted.items():
        data_ood.test_mask = dataset.data_base.transfer_mask_to(
            dataset.data_base.test_mask, data_ood
        )

    # Generate a training mask on the train data that can not contain nodes that are marked to be OOD in some shift and transfer to al other datasets (including the base data)
    match train_size:
        case None:
            assert (
                dataset.data_base.train_mask is not None
            ), "Train set is not available."
            train_mask = dataset.data_base.transfer_mask_to(
                dataset.data_base.train_mask, dataset.data_train
            ) & dataset.data_train.get_distribution_mask(DistributionType.ID)
        case num if isinstance(num, (int, float)):
            train_mask = sample_stratified_mask(
                dataset.data_train,
                num,
                mask=~dataset.data_train.test_mask
                & dataset.data_train.get_distribution_mask(DistributionType.ID),
                allow_empty_mask=True,
            )
        case _:
            raise ValueError(f"Invalid train size {train_size}")

    # Transfer the train mask to all datasets
    dataset.data_train.train_mask = train_mask
    dataset.data_base.train_mask = dataset.data_train.transfer_mask_to(
        train_mask, dataset.data_base
    )
    for name, data_ood in dataset.data_shifted.items():
        data_ood.train_mask = dataset.data_train.transfer_mask_to(train_mask, data_ood)

    match val_size:
        case -1:
            # The validation mask of each dataset is the remainder of test and train
            dataset.data_base.val_mask = ~(
                dataset.data_base.train_mask | dataset.data_base.test_mask
            )
            dataset.data_train.val_mask = dataset.data_base.transfer_mask_to(
                dataset.data_base.val_mask, dataset.data_train
            )
            for name, data_ood in dataset.data_shifted.items():
                data_ood.val_mask = dataset.data_base.transfer_mask_to(
                    dataset.data_base.val_mask, data_ood
                )
        case None:
            assert (
                dataset.data_base.val_mask is not None
            ), "Validation set is not available."
            val_mask = dataset.data_base.val_mask
            dataset.data_train.val_mask = dataset.data_base.transfer_mask_to(
                val_mask, dataset.data_train
            )
            for name, data_ood in dataset.data_shifted.items():
                data_ood.val_mask = dataset.data_base.transfer_mask_to(
                    val_mask, data_ood
                )
        case num if isinstance(num, (int, float)):
            dataset.data_base.val_mask = sample_stratified_mask(
                dataset.data_base,
                num,
                mask=~(dataset.data_base.train_mask | dataset.data_base.test_mask),
            )
            dataset.data_train.val_mask = dataset.data_base.transfer_mask_to(
                dataset.data_base.val_mask, dataset.data_train
            )
            for name, data_ood in dataset.data_shifted.items():
                data_ood.val_mask = dataset.data_base.transfer_mask_to(
                    dataset.data_base.val_mask, data_ood
                )
        case _:
            raise ValueError(f"Invalid validation size {val_size}")

    # For the training set, mask split masks to only concern in-distribution data
    # assert that the training mask contains only in-distribution data
    assert (
        dataset.data_train.get_distribution_mask(DistributionType.ID)[
            dataset.data_train.get_mask(DatasetSplit.TRAIN)
        ]
    ).all(), "Training mask contains out-of-distribution data"
    dataset.data_train.val_mask = (
        dataset.data_train.val_mask
        & dataset.data_train.get_distribution_mask(DistributionType.ID)
    )
    dataset.data_train.test_mask = (
        dataset.data_train.test_mask
        & dataset.data_train.get_distribution_mask(DistributionType.ID)
    )

    # Assert validity of all masks
    dataset.data_base.assert_masks_valid()
    dataset.data_train.assert_masks_valid()
    for name, data_ood in dataset.data_shifted.items():
        data_ood.assert_masks_valid()
