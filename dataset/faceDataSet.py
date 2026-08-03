import os

import pandas as pd
from PIL import Image  # type: ignore[import]
from torch.utils.data import Dataset, DataLoader, Subset, random_split

class FaceDataSet(Dataset):
    def __init__(self, csv_file, image_dir, transform = None):
        self.csv_data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        self.index_to_name = self.csv_data.drop_duplicates('label').set_index('label')['name'].to_dict()

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, idx):
        row = self.csv_data.iloc[idx]
        img_path = os.path.join(self.image_dir, row["filename"])

        image = Image.open(img_path).convert("RGB")
        label = int(row["label"])

        if self.transform:
            image = self.transform(image)

        return image, label