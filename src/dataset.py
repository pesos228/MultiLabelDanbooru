"""
Dataset class for Danbooru multi-label classification with GPU optimization
"""
import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Callable
from collections import Counter

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class DanbooruDataset(Dataset):
    """
    Dataset for multi-label classification of Danbooru image tags
    """

    def __init__(
        self,
        csv_path: Path,
        root_img_dir: Path,
        transform: Optional[Callable] = None,
        tag_vocab: Optional[List[str]] = None,
        mlb: Optional[MultiLabelBinarizer] = None,
        top_k_tags: Optional[int] = None,
        min_tag_frequency: int = 10,
        min_tags_per_image: int = 1,
        indices: Optional[List[int]] = None
    ):
        """
        Args:
            csv_path: path to CSV file with metadata
            root_img_dir: root directory with images
            transform: image transformations
            tag_vocab: ready tag vocabulary (if None - created automatically)
            mlb: ready MultiLabelBinarizer (if None - created automatically)
            top_k_tags: number of most frequent tags for vocabulary
            min_tag_frequency: minimum tag frequency for vocabulary inclusion
            min_tags_per_image: minimum number of relevant tags per image
            indices: row indices to use (for train/val split)
        """
        self.csv_path = Path(csv_path)
        self.root_img_dir = Path(root_img_dir)
        self.transform = transform
        self.min_tags_per_image = min_tags_per_image

        # Load CSV
        logger.info(f"Loading metadata from {csv_path}")
        self.df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(self.df)} records")

        # Apply index filtering if specified
        if indices is not None:
            self.df = self.df.iloc[indices].reset_index(drop=True)
            logger.info(f"Applied index filtering: {len(self.df)} records")

        # Process tags
        self.df['tags_list'] = self.df['tags'].apply(self._parse_tags)

        # Create or use ready tag vocabulary
        if tag_vocab is None or mlb is None:
            self.tag_vocab, self.mlb = self._create_tag_vocab(
                top_k_tags=top_k_tags,
                min_tag_frequency=min_tag_frequency
            )
        else:
            self.tag_vocab = tag_vocab
            self.mlb = mlb

        self.num_classes = len(self.tag_vocab)
        logger.info(f"Tag vocabulary size: {self.num_classes}")

        # Filter images by minimum number of relevant tags
        self._filter_by_relevant_tags()

        logger.info(f"Final dataset size: {len(self.filtered_df)} images")

    def _parse_tags(self, tags_str: str) -> List[str]:
        """Parse tag string into list"""
        if pd.isna(tags_str):
            return []

        tags = [tag.strip().lower() for tag in tags_str.split(', ')]
        return [tag for tag in tags if tag]  # remove empty tags

    def _create_tag_vocab(
        self,
        top_k_tags: Optional[int] = None,
        min_tag_frequency: int = 10
    ) -> Tuple[List[str], MultiLabelBinarizer]:
        """Create tag vocabulary based on frequency"""
        logger.info("Creating tag vocabulary...")

        # Collect all tags
        all_tags = []
        for tags_list in self.df['tags_list']:
            all_tags.extend(tags_list)

        # Count frequencies
        tag_counts = Counter(all_tags)
        logger.info(f"Found {len(tag_counts)} unique tags")

        # Filter by frequency
        frequent_tags = [
            tag for tag, count in tag_counts.items()
            if count >= min_tag_frequency
        ]
        logger.info(f"Tags with frequency >= {min_tag_frequency}: {len(frequent_tags)}")

        # Take top-K most frequent
        if top_k_tags is not None:
            most_common = tag_counts.most_common(top_k_tags)
            tag_vocab = [tag for tag, _ in most_common if tag in frequent_tags]
            tag_vocab = tag_vocab[:top_k_tags]
        else:
            # Sort by frequency (descending)
            tag_vocab = sorted(frequent_tags, key=lambda x: tag_counts[x], reverse=True)

        logger.info(f"Final vocabulary contains {len(tag_vocab)} tags")

        # Create and train MultiLabelBinarizer
        mlb = MultiLabelBinarizer(classes=tag_vocab)

        # Prepare data for MLB training (need to call fit at least once)
        sample_tags = []
        for tags_list in self.df['tags_list'][:1000]:  # use first 1000 for initialization
            relevant_tags = [tag for tag in tags_list if tag in tag_vocab]
            sample_tags.append(relevant_tags)

        mlb.fit(sample_tags)

        return tag_vocab, mlb

    def _filter_by_relevant_tags(self):
        """Filter images by minimum number of relevant tags"""
        def count_relevant_tags(tags_list):
            return len([tag for tag in tags_list if tag in self.tag_vocab])

        self.df['relevant_tag_count'] = self.df['tags_list'].apply(count_relevant_tags)

        before_count = len(self.df)
        self.filtered_df = self.df[
            self.df['relevant_tag_count'] >= self.min_tags_per_image
        ].reset_index(drop=True)
        after_count = len(self.filtered_df)

        logger.info(
            f"Filtering by minimum tag count "
            f"({self.min_tags_per_image}): {before_count} -> {after_count} images"
        )

    def __len__(self) -> int:
        return len(self.filtered_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.filtered_df.iloc[idx]

        # Path to image
        img_path = self.root_img_dir / row['filename']

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except (IOError, OSError) as e:
            logger.warning(f"Error loading image {img_path}: {e}")
            # Return black image in case of error
            image = Image.new('RGB', (256, 256), color='black')

        # Apply transformations
        if self.transform is not None:
            image = self.transform(image)

        # Process tags
        tags_list = row['tags_list']
        relevant_tags = [tag for tag in tags_list if tag in self.tag_vocab]

        # Convert to multi-hot vector
        labels = self.mlb.transform([relevant_tags])[0].astype(np.float32)
        labels = torch.from_numpy(labels)

        return image, labels


def get_train_transforms(image_size: int = 256) -> transforms.Compose:
    """Transformations for training set"""
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transforms(image_size: int = 256) -> transforms.Compose:
    """Transformations for validation set"""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def create_train_val_datasets(
    csv_path: Path,
    root_img_dir: Path,
    image_size: int = 256,
    top_k_tags: int = 1000,
    min_tag_frequency: int = 10,
    min_tags_per_image: int = 1,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[DanbooruDataset, DanbooruDataset]:
    """
    Create train and validation datasets with shared tag vocabulary
    """
    logger.info("Creating train/val datasets...")

    # First create full dataset for vocabulary building
    full_dataset = DanbooruDataset(
        csv_path=csv_path,
        root_img_dir=root_img_dir,
        transform=None,  # no transforms yet
        top_k_tags=top_k_tags,
        min_tag_frequency=min_tag_frequency,
        min_tags_per_image=min_tags_per_image
    )

    # Split filtered indices
    train_indices, val_indices = train_test_split(
        range(len(full_dataset.filtered_df)),
        test_size=val_size,
        random_state=random_state,
        stratify=None  # stratification is complex for multi-label
    )

    logger.info(f"Split: {len(train_indices)} train, {len(val_indices)} val")

    # Use filtered data and indices to create subsets
    train_df_indices = full_dataset.filtered_df.index[train_indices].tolist()
    val_df_indices = full_dataset.filtered_df.index[val_indices].tolist()

    # Create train dataset
    train_dataset = DanbooruDataset(
        csv_path=csv_path,
        root_img_dir=root_img_dir,
        transform=get_train_transforms(image_size),
        tag_vocab=full_dataset.tag_vocab,
        mlb=full_dataset.mlb,
        min_tags_per_image=min_tags_per_image,
        indices=train_df_indices
    )

    # Create val dataset
    val_dataset = DanbooruDataset(
        csv_path=csv_path,
        root_img_dir=root_img_dir,
        transform=get_val_transforms(image_size),
        tag_vocab=full_dataset.tag_vocab,
        mlb=full_dataset.mlb,
        min_tags_per_image=min_tags_per_image,
        indices=val_df_indices
    )
    
    return train_dataset, val_dataset