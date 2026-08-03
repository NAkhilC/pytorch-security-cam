import torch
import sys
from config import RANDOM_STATE, TEST_SIZE
from dataset.faceDataSet import FaceDataSet
from torchvision import transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from models.faceCNN import FaceCNN
import torch.nn as nn
from prediction.predict import model_prediction
from train.trainModel import load_model, train_model

CSV_FILE = 'data/training/labels.csv'
IMAGE_DIR = 'data/training/images'
train_data = sys.argv[1]

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

full_dataset = FaceDataSet(CSV_FILE, IMAGE_DIR, transform)
num_classes = full_dataset.csv_data['label'].nunique()

label_list = full_dataset.csv_data['label'].tolist()
indices = list(range(len(full_dataset)))

train_index, value_index = train_test_split(
    indices, test_size=TEST_SIZE, stratify=label_list, random_state=RANDOM_STATE)

train_dataset = Subset(full_dataset, train_index)
value_dataset = Subset(full_dataset, value_index)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
value_loader = DataLoader(value_dataset, batch_size=32, shuffle=False)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaceCNN(num_classes=num_classes).to(device)
# standard loss function for multi-class classification problems
criterion = nn.CrossEntropyLoss()
# standard optimizer for training deep learning models
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

if (train_data == "train"):
    train_model(model, train_loader, value_loader, criterion,
                optimizer, device, num_classes, full_dataset)
else:
    model, index_to_name = load_model("face_cnn_model.pth", device)
    model_prediction(transform, model, device, full_dataset)
