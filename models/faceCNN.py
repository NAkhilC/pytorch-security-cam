import torch.nn as nn
class FaceCNN(nn.Module):
    def __init__(self, num_classes):
        print(f"Initializing FaceCNN with {num_classes} classes.")
        super(FaceCNN, self).__init__()
        self.features = nn.Sequential( 
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),  # replace negative numbers with 0
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),   # -> (32,64,64)
            nn.ReLU(),
            nn.MaxPool2d(2),                                # -> (32,32,32)

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),   # -> (64,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2),   
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),  # Adjusted for the new input size
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x