import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.mnist import MNIST
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd 
import os
from PIL import Image

class PatchEmbedding(nn.Module):
  def __init__(self, d_model, img_size, patch_size, n_channels):
    super().__init__()

    self.d_model = d_model # Dimensionality of Model
    self.img_size = img_size # Image Size
    self.patch_size = patch_size # Patch Size
    self.n_channels = n_channels # Number of Channels

    self.linear_project = nn.Conv2d(self.n_channels, self.d_model, kernel_size=self.patch_size, stride=self.patch_size)

  # B: Batch Size
  # C: Image Channels
  # H: Image Height
  # W: Image Width
  # P_col: Patch Column
  # P_row: Patch Row
  
  def forward(self, x):
    x = self.linear_project(x) # (B, C, H, W) -> (B, d_model, P_col, P_row)

    x = x.flatten(2) # (B, d_model, P_col, P_row) -> (B, d_model, P)

    x = x.transpose(-2, -1) # (B, d_model, P) -> (B, P, d_model)

    return x
  

class PositionalEncoding(nn.Module):
  def __init__(self, d_model, max_seq_length):
    super().__init__()

    self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token

    # Creating positional encoding
    pe = torch.zeros(max_seq_length, d_model)

    for pos in range(max_seq_length):
      for i in range(d_model):
        if i % 2 == 0:
          pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
        else:
          pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))

    self.register_buffer('pe', pe.unsqueeze(0))

  def forward(self, x):
    # Expand to have class token for every image in batch
    tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

    # Adding class tokens to the beginning of each embedding
    x = torch.cat((tokens_batch,x), dim=1)
    
    # Adjust the positional encoding size to match x's sequence length
    self.pe = nn.Parameter(torch.zeros(1, 65, 9).to(device)) 

    # Add positional encoding to embeddings
    x = x.to(self.pe.device)
    x = x + self.pe

    return x
  
class AttentionHead(nn.Module):
  def __init__(self, d_model, head_size):
    super().__init__()
    self.head_size = head_size

    self.query = nn.Linear(d_model, head_size)
    self.key = nn.Linear(d_model, head_size)
    self.value = nn.Linear(d_model, head_size)

  def forward(self, x):
    Q = self.query(x)
    K = self.key(x)
    V = self.value(x)

    # Dot Product of Queries and Keys
    attention = Q @ K.transpose(-2,-1)

    # Scaling
    attention = attention / (self.head_size ** 0.5)

    attention = torch.softmax(attention, dim=-1)

    attention = attention @ V

    return attention
  
class MultiHeadAttention(nn.Module):
  def __init__(self, d_model, n_heads):
    super().__init__()
    self.head_size = d_model // n_heads

    self.W_o = nn.Linear(d_model, d_model)

    self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

  def forward(self, x):
    # Combine attention heads
    out = torch.cat([head(x) for head in self.heads], dim=-1)

    out = self.W_o(out)

    return out

class TransformerEncoder(nn.Module):
  def __init__(self, d_model, n_heads, r_mlp=4):
    super().__init__()
    self.d_model = d_model
    self.n_heads = n_heads

    # Sub-Layer 1 Normalization
    self.ln1 = nn.LayerNorm(d_model)

    # Multi-Head Attention
    self.mha = MultiHeadAttention(d_model, n_heads)

    # Sub-Layer 2 Normalization
    self.ln2 = nn.LayerNorm(d_model)

    # Multilayer Perception
    self.mlp = nn.Sequential(
        nn.Linear(d_model, d_model*r_mlp),
        nn.GELU(),
        nn.Linear(d_model*r_mlp, d_model)
    )

  def forward(self, x):
    # Residual Connection After Sub-Layer 1
    out = x + self.mha(self.ln1(x))

    # Residual Connection After Sub-Layer 2
    out = out + self.mlp(self.ln2(out))

    return out

class VisionTransformer(nn.Module):
  def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers):
    super().__init__()

    assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
    assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

    self.d_model = d_model # Dimensionality of model
    self.n_classes = n_classes # Number of classes
    self.img_size = img_size # Image size
    self.patch_size = patch_size # Patch size
    self.n_channels = n_channels # Number of channels
    self.n_heads = n_heads # Number of attention heads

    self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
    self.max_seq_length = self.n_patches + 1

    self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
    self.positional_encoding = PositionalEncoding( self.d_model, self.max_seq_length)
    self.transformer_encoder = nn.Sequential(*[TransformerEncoder( self.d_model, self.n_heads) for _ in range(n_layers)])

    # Classification MLP
    self.classifier = nn.Sequential(
        nn.Linear(self.d_model, self.n_classes),
        nn.Softmax(dim=-1)
    )

  def forward(self, images):
    x = self.patch_embedding(images)

    x = self.positional_encoding(x)

    x = self.transformer_encoder(x)

    x = self.classifier(x[:,0])

    return x

d_model = 9
n_classes = 10
img_size = (32,32)
patch_size = (16,16)
n_channels = 1
n_heads = 3
n_layers = 3
batch_size = 128
epochs = 5
alpha = 0.005

# Load the metadata
metadata = pd.read_csv('/home/ubuntu/studies/ProjectStorage/meta.csv')

# Function to filter metadata for a specific folder
def meta_path(folder, metadata):
    # Create an empty DataFrame to store the new metadata
    new_metadata = pd.DataFrame()
    
    # Iterate over each file in the folder
    for file in os.listdir(folder):
        # Check if the file is a .npy file
        if file.endswith('.npy'):
            # Replace .npy with .wav to get the corresponding index
            wav_index = file.replace('.npy', '.wav')
            
            # Look for the index in the metadata DataFrame
            matching_row = metadata[metadata['file'] == wav_index]
            
            # Append the matching row to new_metadata
            if not matching_row.empty:
                new_metadata = pd.concat([new_metadata, matching_row], ignore_index=True)
    
    return new_metadata

# Paths to the directories containing the .npy spectrograms
train_data_dir = '/home/ubuntu/studies/ProjectStorage/trainset/mfcc'
val_data_dir = '/home/ubuntu/studies/ProjectStorage/valset/mfcc'
test_data_dir = '/home/ubuntu/studies/ProjectStorage/testset/mfcc'

# Use the meta_path function to filter the metadata for the training dataset
train_metadata = meta_path(train_data_dir, metadata)
test_metadata = meta_path(test_data_dir, metadata)
val_metadata = meta_path(val_data_dir, metadata)

# Set image transformations
transform = T.Compose([
    T.Resize((128, 128)),  # Resize images to a uniform size
    T.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
])

class NpySpectrogramDataset(Dataset):
    def __init__(self, metadata, data_dir, transform=None):
        self.metadata = metadata
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Get the file name from the metadata, replacing '.wav' with '.npy'
        file_name = self.metadata.iloc[idx]['file'].replace('.wav', '.npy')
        file_path = os.path.join(self.data_dir, file_name)

        # Load the spectrogram stored in .npy format
        spectrogram = np.load(file_path)

        # If the spectrogram is 1D, reshape it to 2D
        if len(spectrogram.shape) == 1:
            spectrogram = spectrogram.reshape(1, -1)  # Convert to shape (1, 145)
        
        # Ensure the spectrogram has a channel dimension
        if spectrogram.ndim == 2:
            spectrogram = spectrogram[np.newaxis, ...]  # Convert to (1, 1, 145)

        # Convert to a tensor
        spectrogram = torch.tensor(spectrogram, dtype=torch.float32)

        # Apply transformations 
        if self.transform:
            spectrogram = self.transform(spectrogram)

        # Get the label (1 = spoof, 0 = bona-fide)
        label = 1 if self.metadata.iloc[idx]['label'] == 'spoof' else 0
        return spectrogram, label

# Create dataset instances for training, validation, and test data
train_dataset = NpySpectrogramDataset(metadata=train_metadata, data_dir=train_data_dir, transform=transform)
val_dataset = NpySpectrogramDataset(metadata=val_metadata, data_dir=val_data_dir, transform=transform)
test_dataset = NpySpectrogramDataset(metadata=test_metadata, data_dir=test_data_dir, transform=transform)

# Check sample 
sample_spectrogram, sample_label = train_dataset[0]
print(f"Spectrogram Shape: {sample_spectrogram.shape}")
print(f"Label: {sample_label}")

# Create DataLoader instances for each dataset
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

# weighted loss function
class_counts = Counter(labels) 
total_samples = len(labels)
class_weights = {label: total_samples / count for label, count in class_counts.items()}
weights = torch.tensor([class_weights[i] for i in range(n_classes)], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# initialize model, optimizer, and training setup
transformer = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
optimizer = Adam(transformer.parameters(), lr=alpha)

# calculate metrics
def calculate_metrics(loader, model, device):
    y_true = []
    y_pred = []
    losses = []
    
    model.eval()  
    with torch.no_grad():
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            # Calculate predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Store true labels and predictions for metrics calculation
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Calculate evaluation metrics
    accuracy = 100 * np.sum(np.array(y_true) == np.array(y_pred)) / len(y_true)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    return {
        "loss": np.mean(losses),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    }

# training process with validation
for epoch in range(epochs):
    transformer.train()  
    training_loss = 0.0

    # Training loop
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() 

        # Forward pass
        outputs = transformer(inputs)
        
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        training_loss += loss.item()
    val_metrics = calculate_metrics(val_loader, transformer, device)

    print(f'Epoch {epoch + 1}/{epochs} - '
          f'Training Loss: {training_loss / len(train_loader):.3f} - '
          f'Validation Loss: {val_metrics["loss"]:.3f} - '
          f'Validation Accuracy: {val_metrics["accuracy"]:.2f}% - '
          f'Precision: {val_metrics["precision"]:.2f} - '
          f'Recall: {val_metrics["recall"]:.2f} - '
          f'F1 Score: {val_metrics["f1_score"]:.2f}' +
          (f' - ROC AUC: {val_metrics["roc_auc"]:.2f}' if val_metrics["roc_auc"] is not None else ''))

# Test the model after training
test_metrics = calculate_metrics(test_loader, transformer, device)
print(f'\nFinal Test Metrics:\n'
      f'Loss: {test_metrics["loss"]:.3f}\n'
      f'Accuracy: {test_metrics["accuracy"]:.2f}%\n'
      f'Precision: {test_metrics["precision"]:.2f}\n'
      f'Recall: {test_metrics["recall"]:.2f}\n'
      f'F1 Score: {test_metrics["f1_score"]:.2f}' +
      (f'\nROC AUC: {test_metrics["roc_auc"]:.2f}' if test_metrics["roc_auc"] is not None else ''))
