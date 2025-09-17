"""Data-loading helpers built on top of TensorFlow Datasets and `tf.data`."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import tensorflow as tf
import tensorflow_datasets as tfds


def load_tfds_dataset(
    name: str,
    *,
    split: str = "train",
    data_dir: str | None = None,
    as_supervised: bool = True,
    shuffle_files: bool = True,
    with_info: bool = False,
    try_gcs: bool = True,
) -> Any:
    """Download and load a TensorFlow Datasets dataset.

    Parameters
    ----------
    name:
        Dataset identifier, e.g. ``"mnist"`` or ``"imdb_reviews"``.
    split:
        Split spec passed directly to ``tfds.load``.
    data_dir:
        Optional directory for cached downloads.
    as_supervised:
        Return ``(features, label)`` pairs if supported.
    shuffle_files:
        Shuffle input files before reading.
    with_info:
        When ``True`` the :class:`tfds.core.DatasetInfo` is returned alongside the dataset.
    try_gcs:
        Allow TFDS to use public GCS mirrors when available.
    """

    ds, info = tfds.load(
        name,
        split=split,
        data_dir=data_dir,
        as_supervised=as_supervised,
        shuffle_files=shuffle_files,
        with_info=True,
        try_gcs=try_gcs,
    )
    return (ds, info) if with_info else ds


def prepare_for_training(
    dataset: tf.data.Dataset,
    *,
    batch_size: int = 32,
    shuffle_buffer: int | None = 1000,
    cache: bool = True,
    augment_fn: Callable[[Any], Any] | None = None,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """Apply standard ``tf.data`` transformations for training workflows."""

    ds = dataset
    if cache:
        ds = ds.cache()
    if shuffle_buffer:
        ds = ds.shuffle(int(shuffle_buffer))
    if augment_fn is not None:
        ds = ds.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
