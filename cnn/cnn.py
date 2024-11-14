import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import logging
from sklearn.metrics import f1_score, roc_auc_score, roc_curve

# Configure logging
logging.basicConfig(filename='cnn_training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the metadata
metadata = pd.read_csv('./ProjectStorage/meta.csv')
label_lookup = {str(row['file']).split('.')[0]: 1 if row['label'] == 'spoof' else 0 for _, row in metadata.iterrows()}

# Custom dataset class
class SpectrogramDataset(Dataset):
    def __init__(self, data_dir, label_lookup, transform=None, target_length=128):
        self.data_dir = data_dir
        self.transform = transform
        self.target_length = target_length
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy')]
        self.label_lookup = label_lookup

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_id = ''.join(filter(str.isdigit, file_name))
        file_path = os.path.join(self.data_dir, file_name)
        spectrogram = np.load(file_path)
        
        if spectrogram.shape[1] < self.target_length:
            pad_width = self.target_length - spectrogram.shape[1]
            spectrogram = np.pad(spectrogram, ((0, 0), (0, pad_width)), mode='constant')
        elif spectrogram.shape[1] > self.target_length:
            spectrogram = spectrogram[:, :self.target_length]

        spectrogram = torch.tensor(spectrogram, dtype=torch.float32).unsqueeze(0)
        label = self.label_lookup.get(file_id, 0)
        label = torch.tensor(label, dtype=torch.float32)
        return spectrogram, label

# CNN model definition
class AudioCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

# Function to calculate EER
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fnr - fpr))
    eer = fpr[eer_index]
    return eer

# Evaluation function with additional metrics
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images).squeeze()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    # Calculate metrics
    accuracy = 100 * correct / total
    f1 = f1_score(all_labels, all_predictions)
    auc = roc_auc_score(all_labels, all_probs)
    eer = calculate_eer(all_labels, all_probs)

    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.2f}")
    print(f"AUC: {auc:.2f}")
    print(f"EER: {eer:.4f}")

    logging.info(f"Test Accuracy: {accuracy:.2f}%, F1-Score: {f1:.2f}, AUC: {auc:.2f}, EER: {eer:.4f}")

# Directories for dataset
train_dir = './output/cqt'
val_dir = './output/cqt'
test_dir = './output/cqt'

# Initialize datasets and DataLoaders
train_dataset = SpectrogramDataset(train_dir, label_lookup)
val_dataset = SpectrogramDataset(val_dir, label_lookup)
test_dataset = SpectrogramDataset(test_dir, label_lookup)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Training settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioCNN().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training")
    
    for spectrograms, labels in train_loader_tqdm:
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(spectrograms).squeeze()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        train_loader_tqdm.set_postfix({"Training Loss": running_loss / len(train_loader)})

    avg_train_loss = running_loss / len(train_loader)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Training Loss: {avg_train_loss:.4f}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
    
    with torch.no_grad():
        for spectrograms, labels in val_loader_tqdm:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            outputs = model(spectrograms).squeeze()
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_loader_tqdm.set_postfix({"Validation Loss": val_loss / len(val_loader)})

    avg_val_loss = val_loss / len(val_loader)
    logging.info(f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Validation Loss: {avg_val_loss:.4f}")

# Evaluate the model on the test set
evaluate_model(model, test_loader)
logging.info("Training completed successfully.")

# Save the trained model
torch.save(model.state_dict(), "audio_cnn_model.pth")
print("Model saved as 'audio_cnn_model.pth'")
