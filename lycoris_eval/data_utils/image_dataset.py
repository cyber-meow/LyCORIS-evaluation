import os
import warnings
from PIL import Image

from torch.utils.data import Dataset


class ImageDataset(Dataset):

    def __init__(self, image_paths, transform, is_generated):
        self.image_paths = image_paths
        self.transform = transform
        self.is_generated = is_generated

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        # Custom warning handler
        def custom_showwarning(message,
                               category,
                               filename,
                               lineno,
                               file=None,
                               line=None):
            print(f"{image_path}: {message}")

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            warnings.showwarning = custom_showwarning

            if self.is_generated:
                key = int(os.path.basename(image_path).split('-')[0])
            else:
                key = image_path
            with Image.open(image_path) as image:
                image = self.transform(image)

        return key, image
