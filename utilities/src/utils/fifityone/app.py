import os
import os.path
from pathlib import Path
from typing import Optional

import click

# isort: off
import numpy as np

import cv2

# isort: on
import fiftyone as fo
import fiftyone.utils.data as foud

from aasdatahub import ImageHub
from aasdatahub.structures import ImageTask


class AASDataHubImporter(foud.LabeledImageDatasetImporter):
    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        dataset_name: Optional[str] = None,
        destination_dir: Optional[str] = None,
        max_samples: Optional[int] = None,
    ):
        """Importer for AASDataHub datasets.

        Parameters
        ----------
        dataset_dir: str
            The dataset directory
        dataset_name: str
            The dataset name
        destination_dir: str
            The destination directory to save the imported dataset
        max_samples: int
            The maximum number of samples to import. By default, all samples are imported
        """
        assert dataset_dir is not None or dataset_name is not None, "dataset_dir and dataset_name must be specified."
        super().__init__(dataset_dir=dataset_dir, shuffle=False, seed=None, max_samples=max_samples)

        if not os.path.exists(destination_dir):
            os.makedirs(destination_dir)

        if not os.path.isdir(destination_dir):
            raise NotADirectoryError(f"{destination_dir} is not a directory")

        self._labels_file = None
        self._labels = None
        self._iter_labels = None
        self._destination_dir = destination_dir
        self._dataset_name = dataset_name
        self.hub = ImageHub()

    def __iter__(self):
        self._iter_labels = iter(self._labels)
        return self

    def __next__(self):
        """Returns information about the next sample in the dataset.

        Returns:
            an  ``(image_path, image_metadata, label)`` tuple, where

            -   ``image_path``: the path to the image on disk
            -   ``image_metadata``: an
                :class:`fiftyone.core.metadata.ImageMetadata` instances for the
                image, or ``None`` if :meth:`has_image_metadata` is ``False``
            -   ``label``: an instance of :meth:`label_cls`, or a dictionary
                mapping field names to :class:`fiftyone.core.labels.Label`
                instances, or ``None`` if the sample is unlabeled

        Raises:
            StopIteration: if there are no more samples to import
        """
        (
            filepath,
            size_bytes,
            mime_type,
            width,
            height,
            num_channels,
            index,
        ) = next(self._iter_labels)

        image_sample = self.hub.read(index)

        image_metadata = fo.ImageMetadata(
            size_bytes=size_bytes,
            mime_type=mime_type,
            width=width,
            height=height,
            num_channels=num_channels,
        )

        label = {"class_id": None, "bboxes": None, "mask": None}
        if image_sample.class_id.is_labeled:
            class_names = self.hub.get_classname_table(ImageTask.CLASSIFICATION)
            label["class_id"] = fo.Classification(label=class_names[int(image_sample.class_id.labels[0])])
        if image_sample.bboxes.is_labeled and len(image_sample.bboxes.bboxes) > 0:
            class_names = self.hub.get_classname_table(ImageTask.OBJECT_DETECTION)
            detections = []
            for b in image_sample.bboxes.bboxes:
                x, y, w, h, class_id, _ = b.data
                detections.append(
                    fo.Detection(label=class_names[int(class_id)], bounding_box=[x - w / 2, y - h / 2, w, h])
                )
            label["bboxes"] = fo.Detections(detections=detections)
        if image_sample.mask.is_labeled:
            label["mask"] = fo.Segmentation(mask=image_sample.mask.data[0, :, :, 0].astype(np.uint8))

        return filepath, image_metadata, label

    def __len__(self):
        """The total number of samples that will be imported.

        Raises:
            TypeError: if the total number is not known
        """
        return len(self._labels)

    @property
    def has_dataset_info(self):
        """Whether this importer produces a dataset info dictionary."""
        return False

    @property
    def has_image_metadata(self):
        """Whether this importer produces
        :class:`fiftyone.core.metadata.ImageMetadata` instances for each image.
        """
        return True

    @property
    def label_cls(self):
        """The :class:`fiftyone.core.labels.Label` class(es) returned by this
        importer.

        This can be any of the following:

        -   a :class:`fiftyone.core.labels.Label` class. In this case, the
            importer is guaranteed to return labels of this type
        -   a list or tuple of :class:`fiftyone.core.labels.Label` classes. In
            this case, the importer can produce a single label field of any of
            these types
        -   a dict mapping keys to :class:`fiftyone.core.labels.Label` classes.
            In this case, the importer will return label dictionaries with keys
            and value-types specified by this dictionary. Not all keys need be
            present in the imported labels
        -   ``None``. In this case, the importer makes no guarantees about the
            labels that it may return
        """
        return {
            "class_id": fo.Classification,
            "bboxes": fo.Detections,
            "mask": fo.Segmentation,
        }

    def setup(self):
        """Performs any necessary setup before importing the first sample in
        the dataset.

        This method is called when the importer's context manager interface is
        entered, :func:`DatasetImporter.__enter__`.
        """

        if self.dataset_dir:
            self.hub.load(self.dataset_dir)
        else:
            self.hub.load_remote(self._dataset_name)

        labels = []
        max_samples = self.max_samples if self.max_samples is not None else len(self.hub)
        for i in range(max_samples):
            sample = self.hub.read(i)
            filepath = os.path.join(self._destination_dir, f"{i:06d}.png")
            h, w, c = sample.image.shape

            cv2.imwrite(filepath, cv2.cvtColor(sample.image, cv2.COLOR_RGB2BGR))
            labels.append(
                (
                    filepath,
                    sample.image.size * 4,  # float = 4 bytes
                    "image/png",
                    w,
                    h,
                    c,
                    i,
                )
            )

        # The `_preprocess_list()` function is provided by the base class
        # and handles shuffling/max sample limits
        self._labels = self._preprocess_list(labels)


@click.command(help=__doc__)
@click.option(
    "-p",
    "--dataset-path",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="The dataset directory",
)
@click.option(
    "-n",
    "--dataset-name",
    default=None,
    help="The name of dataset in Datacenter",
)
@click.option(
    "--destination-dir",
    default=Path(""),
    type=click.Path(),
    help="The destination directory to store the artifacts",
)
@click.option(
    "--max-samples",
    default=None,
    type=int,
    help="A maximum number of samples to import. By default, all samples are imported",
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Whether to overwrite an existing dataset of the same name",
)
@click.option(
    "--persistent",
    is_flag=True,
    help="Whether the dataset should persist in the database after the session terminates",
)
def main(
    dataset_path: Optional[str],
    dataset_name: Optional[str],
    destination_dir: Optional[str],
    max_samples: Optional[int],
    overwrite: bool,
    persistent: bool,
):
    if dataset_path:
        importer = AASDataHubImporter(
            dataset_name=os.path.basename(dataset_path),
            dataset_dir=dataset_path,
            destination_dir=os.path.join(destination_dir, os.path.basename(dataset_path)),
            max_samples=max_samples,
        )
        dataset_name = os.path.basename(dataset_path) if dataset_name is None else dataset_name
    elif dataset_name:
        importer = AASDataHubImporter(
            dataset_name=dataset_name,
            destination_dir=os.path.join(destination_dir, dataset_name),
            max_samples=max_samples,
        )
    else:
        raise RuntimeError("dataset_path and dataset_name must be specified")

    dataset = fo.Dataset.from_importer(
        dataset_importer=importer,
        name=dataset_name,
        overwrite=overwrite,
        persistent=persistent,
    )

    # Print summary information about the dataset
    print(dataset)


if __name__ == "__main__":
    main()
