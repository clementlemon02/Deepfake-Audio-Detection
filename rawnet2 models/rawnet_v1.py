import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import os
import torchaudio
import pandas as pd
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import roc_curve
import random

# Define the pad function to ensure fixed input size
def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

# Normalize the waveform
def normalize(waveform):
    max_val = np.max(np.abs(waveform))  # Find the maximum absolute value
    if max_val > 0:
        return waveform / max_val  # Scale to the range [-1, 1]
    return waveform

# Define a class to load original data
class AudioDataset(Dataset):
    def __init__(self, audio_dir, meta_file, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform

        # Load the meta CSV file containing labels
        self.meta_data = pd.read_csv(meta_file)

        # Create a dictionary of filenames to labels
        self.audio_labels = dict(zip(self.meta_data['file'], self.meta_data['label']))
        
        # Get the list of audio files in the directory
        self.audio_files = [f for f in os.listdir(audio_dir) if f in self.audio_labels]  # Only files with labels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Get the filename and corresponding label
        audio_filename = self.audio_files[idx]
        label = self.audio_labels[audio_filename]
        
        # Load the audio waveform
        audio_path = os.path.join(self.audio_dir, audio_filename)
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to numpy array for padding
        waveform = waveform.squeeze().numpy()

        if self.transform:
            waveform = self.transform(waveform)

        # Convert label to a numerical format: 0 for 'spoof', 1 for 'bona-fide'
        label = 1 if label == 'spoof' else 0

        return waveform, label

# Class to load augmented data
class AugmentedAudioDataset(Dataset):
    def __init__(self, audio_dir, meta_file, transform=None):
        self.audio_dir = audio_dir
        self.transform = transform

        # Load the metadata CSV and map file numbers to labels
        self.meta_data = pd.read_csv(meta_file)
        # Extract numeric part and create a mapping for labels
        self.audio_labels = dict(zip(self.meta_data['file'], self.meta_data['label']))

        self.audio_files = [f for f in os.listdir(audio_dir) if '.wav' in f]

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        # Get the augmented file name
        audio_filename = self.audio_files[idx]
        
        # Extract the original file number from the augmented file name
        file_name = audio_filename.split('_')[-1]

        # Retrieve label using the extracted file number
        label = self.audio_labels.get(file_name, 'unknown')
        
        # Load the audio waveform
        audio_path = os.path.join(self.audio_dir, audio_filename)
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert to numpy array for padding
        waveform = waveform.squeeze().numpy()

        
        if self.transform:
            waveform = self.transform(waveform)

        # Convert label to numerical format: 0 for 'bonafide', 1 for 'spoof'
        label = 1 if label == 'spoof' else 0

        return waveform, label

# Define the transformation pipeline
audio_transforms = transforms.Compose([
    lambda x: pad(x),  
    lambda x: normalize(x),
    lambda x: Tensor(x)   
])



# Path to the directory containing the audio files and the CSV file
train_audio_dir = 'ProjectStorage/wav_train_set'
val_audio_dir = 'ProjectStorage/wav_val_set'
test_audio_dir = 'ProjectStorage/wav_test_set'
augmented_dir = 'ProjectStorage/augmented_audio_files'
meta_file = 'ProjectStorage/meta.csv'

# Initialize the dataset with labels
train_dataset = AudioDataset(audio_dir=train_audio_dir, meta_file=meta_file, transform=audio_transforms)
val_dataset = AudioDataset(audio_dir=val_audio_dir, meta_file=meta_file, transform=audio_transforms)
test_dataset = AudioDataset(audio_dir=test_audio_dir, meta_file=meta_file, transform=audio_transforms)
augmented_dataset = AugmentedAudioDataset(audio_dir=augmented_dir, meta_file=meta_file, transform=audio_transforms)
combined_train_dataset = ConcatDataset([train_dataset, augmented_dataset])

# Create DataLoaders for each dataset
train_loader = DataLoader(combined_train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Define the RawNet2 Model
class SincConv(nn.Module):
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(self, device, out_channels, kernel_size, in_channels=1, sample_rate=16000,
                 stride=1, padding=0, dilation=1, bias=False, groups=1):
        super(SincConv, self).__init__()

        if in_channels != 1:
            msg = "SincConv only supports one input channel (here, in_channels = {%i})" % (in_channels)
            raise ValueError(msg)

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate 

        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.device = device
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        if bias:
            raise ValueError('SincConv does not support bias.')
        if groups > 1:
            raise ValueError('SincConv does not support groups.')

        # initialize filterbanks using Mel scale
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)   # Hz to mel conversion
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)  # Mel to Hz conversion
        self.mel = filbandwidthsf
        self.hsupp = torch.arange(-(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1)
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

    def forward(self, x): # this method defines the forward pass of the SincConv layer 
        for i in range(len(self.mel) - 1):
            fmin = self.mel[i] # min frequency for each band-pass filter
            fmax = self.mel[i + 1] # max frequency
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(2 * fmax * self.hsupp / self.sample_rate)
            hLow = (2 * fmin / self.sample_rate) * np.sinc(2 * fmin * self.hsupp / self.sample_rate)
            hideal = hHigh - hLow
            
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(hideal) # Hamming window smooths the filters so they don’t have sharp edges, making the model perform better.

        band_pass_filter = self.band_pass.to(self.device)

        self.filters = (band_pass_filter).view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        bias=None, groups=1)


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super(Residual_block, self).__init__()
        self.first = first # indicates whether it is the first block in the model, which skips BN
        
        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features=nb_filts[0])

        self.lrelu = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels=nb_filts[0],
                               out_channels=nb_filts[1],
                               kernel_size=3,
                               padding=1,
                               stride=1)

        self.bn2 = nn.BatchNorm1d(num_features=nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels=nb_filts[1],
                               out_channels=nb_filts[1],
                               padding=1,
                               kernel_size=3,
                               stride=1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels=nb_filts[0],
                                              out_channels=nb_filts[1],
                                              padding=0,
                                              kernel_size=1,
                                              stride=1)
        else:
            self.downsample = False

        self.mp = nn.MaxPool1d(3)

    def forward(self, x): # Adds the input (identity) to the output of the convolutional layers
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu(out)
        else:
            out = x
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.lrelu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class RawNet(nn.Module):
    def __init__(self, d_args, device):
        super(RawNet, self).__init__()

        self.device = device

        self.Sinc_conv = SincConv(device=self.device,
                                   out_channels=d_args['filts'][0],
                                   kernel_size=d_args['first_conv'],
                                   in_channels=d_args['in_channels']
                                   )

        self.first_bn = nn.BatchNorm1d(num_features=d_args['filts'][0])
        self.selu = nn.SELU(inplace=True) # Scaled Exponential Linear Unit (activation function)
        # Next, sequential blocks of residual layers, which consist of convolutional layers with skip connections to prevent vanishing gradients
        self.block0 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1], first=True)) #[20,20]
        self.block1 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][1])) #[20,20]
        self.block2 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2])) #[20,128]
        d_args['filts'][2][0] = d_args['filts'][2][1]
        self.block3 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2])) #[128,128]
        self.block4 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2])) #[128,128]
        self.block5 = nn.Sequential(Residual_block(nb_filts=d_args['filts'][2])) #[128,128]
        self.avgpool = nn.AdaptiveAvgPool1d(1) # Average over time, create a summary of the feature map

        # Attention FC layers compute attention weights for each residual block. Highlight important parts of the feature maps
        self.fc_attention0 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                     l_out_features=d_args['filts'][1][-1])
        self.fc_attention1 = self._make_attention_fc(in_features=d_args['filts'][1][-1],
                                                     l_out_features=d_args['filts'][1][-1])
        self.fc_attention2 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention3 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention4 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])
        self.fc_attention5 = self._make_attention_fc(in_features=d_args['filts'][2][-1],
                                                     l_out_features=d_args['filts'][2][-1])

        self.bn_before_gru = nn.BatchNorm1d(num_features=d_args['filts'][2][-1])
        
        self.gru = nn.GRU(input_size=d_args['filts'][2][-1],
                          hidden_size=d_args['gru_node'],
                          num_layers=d_args['nb_gru_layer'],
                          batch_first=True)

        self.fc1_gru = nn.Linear(in_features=d_args['gru_node'],
                                  out_features=d_args['nb_fc_node'])

        self.fc2_gru = nn.Linear(in_features=d_args['nb_fc_node'],
                                  out_features=d_args['nb_classes'], bias=True)

        self.sig = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, y=None):
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)

        x = self.Sinc_conv(x)
        x = F.max_pool1d(torch.abs(x), 3)
        x = self.first_bn(x)
        x = self.selu(x)

        x0 = self.block0(x)
        y0 = self.avgpool(x0).view(x0.size(0), -1)  # torch.Size([batch, filter])
        y0 = self.fc_attention0(y0)
        y0 = self.sig(y0).view(y0.size(0), y0.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x0 * y0 + y0  # (batch, filter, time) x (batch, filter, 1)

        x1 = self.block1(x)
        y1 = self.avgpool(x1).view(x1.size(0), -1)  # torch.Size([batch, filter])
        y1 = self.fc_attention1(y1)
        y1 = self.sig(y1).view(y1.size(0), y1.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x1 * y1 + y1  # (batch, filter, time) x (batch, filter, 1)

        x2 = self.block2(x)
        y2 = self.avgpool(x2).view(x2.size(0), -1)  # torch.Size([batch, filter])
        y2 = self.fc_attention2(y2)
        y2 = self.sig(y2).view(y2.size(0), y2.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x2 * y2 + y2  # (batch, filter, time) x (batch, filter, 1)

        x3 = self.block3(x)
        y3 = self.avgpool(x3).view(x3.size(0), -1)  # torch.Size([batch, filter])
        y3 = self.fc_attention3(y3)
        y3 = self.sig(y3).view(y3.size(0), y3.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x3 * y3 + y3  # (batch, filter, time) x (batch, filter, 1)

        x4 = self.block4(x)
        y4 = self.avgpool(x4).view(x4.size(0), -1)  # torch.Size([batch, filter])
        y4 = self.fc_attention4(y4)
        y4 = self.sig(y4).view(y4.size(0), y4.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x4 * y4 + y4  # (batch, filter, time) x (batch, filter, 1)

        x5 = self.block5(x)
        y5 = self.avgpool(x5).view(x5.size(0), -1)  # torch.Size([batch, filter])
        y5 = self.fc_attention5(y5)
        y5 = self.sig(y5).view(y5.size(0), y5.size(1), -1)  # torch.Size([batch, filter, 1])
        x = x5 * y5 + y5  # (batch, filter, time) x (batch, filter, 1)

        x = self.bn_before_gru(x)

        x = x.transpose(1, 2)
        x, _ = self.gru(x)  # return only hidden states of last layer
        x = x[:, -1, :]  # (batch, seq_len, hidden_size)

        x = self.fc1_gru(x)
        x = self.fc2_gru(x)
        x = self.logsoftmax(x)

        return x

    def _make_attention_fc(self, in_features, l_out_features):
        l_fc = []
        l_fc.append(nn.Linear(in_features = in_features,
                    out_features = l_out_features))
        return nn.Sequential(*l_fc)
    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
                first = first))
            if i == 0: nb_filts[0] = nb_filts[1]
            
        return nn.Sequential(*layers)

    def summary(self, input_size, batch_size=-1, device="cuda", print_fn = None):
        if print_fn == None: printfn = print
        model = self
        
        def register_hook(module):
            def hook(module, input, output):
                class_name = str(module.__class__).split(".")[-1].split("'")[0]
                module_idx = len(summary)
                
                m_key = "%s-%i" % (class_name, module_idx + 1)
                summary[m_key] = OrderedDict()
                summary[m_key]["input_shape"] = list(input[0].size())
                summary[m_key]["input_shape"][0] = batch_size
                if isinstance(output, (list, tuple)):
                    summary[m_key]["output_shape"] = [
						[-1] + list(o.size())[1:] for o in output
					]
                else:
                    summary[m_key]["output_shape"] = list(output.size())
                    if len(summary[m_key]["output_shape"]) != 0:
                        summary[m_key]["output_shape"][0] = batch_size
                        
                params = 0
                if hasattr(module, "weight") and hasattr(module.weight, "size"):
                    params += torch.prod(torch.LongTensor(list(module.weight.size())))
                    summary[m_key]["trainable"] = module.weight.requires_grad
                if hasattr(module, "bias") and hasattr(module.bias, "size"):
                    params += torch.prod(torch.LongTensor(list(module.bias.size())))
                summary[m_key]["nb_params"] = params
                
            if (
				not isinstance(module, nn.Sequential)
				and not isinstance(module, nn.ModuleList)
				and not (module == model)
			):
                hooks.append(module.register_forward_hook(hook))
                
        device = device.lower()
        assert device in [
			"cuda",
			"cpu",
		], "Input device is not valid, please specify 'cuda' or 'cpu'"
        
        if device == "cuda" and torch.cuda.is_available():
            dtype = torch.cuda.FloatTensor
        else:
            dtype = torch.FloatTensor
        if isinstance(input_size, tuple):
            input_size = [input_size]
        x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
        summary = OrderedDict()
        hooks = []
        model.apply(register_hook)
        model(*x)
        for h in hooks:
            h.remove()
            
        print_fn("----------------------------------------------------------------")
        line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
        print_fn(line_new)
        print_fn("================================================================")
        total_params = 0
        total_output = 0
        trainable_params = 0
        for layer in summary:
            line_new = "{:>20}  {:>25} {:>15}".format(
				layer,
				str(summary[layer]["output_shape"]),
				"{0:,}".format(summary[layer]["nb_params"]),
			)
            total_params += summary[layer]["nb_params"]
            total_output += np.prod(summary[layer]["output_shape"])
            if "trainable" in summary[layer]:
                if summary[layer]["trainable"] == True:
                    trainable_params += summary[layer]["nb_params"]
            print_fn(line_new)



device = 'cuda' if torch.cuda.is_available() else 'cpu'
d_args = {
    'in_channels': 1,    
    'first_conv': 1024,
    'filts': [20, [20,20], [20,128],[128,128]], 
    'blocks': [2,4],
    'gru_node': 1024,
    'nb_gru_layer': 3,  
    'nb_fc_node': 1024, 
    'nb_classes': 2}

model = RawNet(d_args, device).to(device)

weights = torch.tensor([1/13930, 1/8306], device=device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.Adam(model.parameters(), lr=0.0015, weight_decay=1e-4)
num_epochs = 20
early_stopping_patience = 3
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=4)
best_val_loss = float('inf')

# EER Calculation Function
def calculate_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fnr[np.nanargmin(np.abs(fnr - fpr))]
    return eer, eer_threshold

# Training Loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_correct = 0
    total_samples = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(inputs)  # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights

        # Accumulate loss and calculate accuracy
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1) 
        total_correct += (predicted == labels).sum().item() 
        total_samples += labels.size(0)

    # Calculate average loss and accuracy for the epoch
    avg_loss = epoch_loss / len(train_loader)
    accuracy = total_correct / total_samples * 100  # Convert to percentage
    print(f'Epoch [{epoch + 1}/{num_epochs}], Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # Validation Loop
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    val_correct = 0
    val_samples = 0
    all_labels = []
    all_scores = []

    with torch.no_grad():  # Disable gradient calculation for validation
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, labels)  # Calculate loss

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_samples += labels.size(0)
            all_labels.extend(labels.cpu().numpy())
            all_scores.extend(outputs[:, 1].cpu().numpy()) 

    # Calculate EER using collected labels and scores
    eer, eer_threshold = calculate_eer(np.array(all_labels), np.array(all_scores))

    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = val_correct / val_samples * 100 
    print(f'Validation - Average Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.2f}%, EER: {eer:.4f}')

    # Adjust learning rate based on validation loss
    scheduler.step(avg_val_loss)

    # Early stopping logic
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        early_stopping_counter = 0          early_stopping_counter += 1
        if early_stopping_counter >= early_stopping_patience:
            print("Early stopping triggered")
            break  # Stop training if no improvement after `patience` epochs


import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import numpy as np
import matplotlib.pyplot as plt

def evaluate_model(model, test_loader, device):
    # Set the model to evaluation mode
    model.eval()

    # Initialize variables to track performance metrics
    all_labels = []
    all_predictions = []
    all_probs = []
    
    # Disable gradient computation
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)

            # Forward pass: Get model predictions
            outputs = model(data)
            _, predicted = torch.max(outputs, dim=1)
            
            # Store labels and predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            probs = torch.softmax(outputs, dim=1) 
            all_probs.extend(probs[:, 1].cpu().numpy()) 
            
    # Convert lists to numpy arrays for metric calculation
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probs = np.array(all_probs)

    # Calculate Accuracy
    accuracy = np.mean(all_predictions == all_labels) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    
    # Calculate Precision, Recall, and F1-Score (binary or multi-class)
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    
    # Confusion Matrix
    conf_matrix = confusion_matrix(all_labels, all_predictions)
    print(f"Confusion Matrix:\n{conf_matrix}")
    
    # Calculate EER
    fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
    fnr = 1 - tpr  # False Negative Rate
    eer_index = np.nanargmin(np.abs(fnr - fpr))  # Find index where FNR = FPR
    eer = fpr[eer_index]  # EER value
    eer_threshold = thresholds[eer_index]  # EER threshold
    print(f"EER: {eer:.2f} at threshold: {eer_threshold:.2f}")
    
    # ROC-AUC (for binary classification)
    if len(np.unique(all_labels)) == 2:  # Only for binary classification
        auc_roc = roc_auc_score(all_labels, all_probs)
        print(f"AUC-ROC: {auc_roc:.2f}")
        
        # Optionally, plot the ROC curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.plot(fpr, tpr, label='ROC Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        
        

        
# test with test_loader ( 20% original data )
evaluate_model(model, test_loader, device='cuda' if torch.cuda.is_available() else 'cpu')



class TestAudioDataset(Dataset):
    def __init__(self, bonafide_dir, spoof_dir, num_samples = 3000, transform=None):
        """
        Initialize the dataset with bonafide and spoof directories.
        
        Args:
            bonafide_dir (str): Path to the directory with bonafide audio files.
            spoof_dir (str): Path to the directory with spoofed audio files.
            transform (callable, optional): Optional transform to be applied to the audio data.
        """
        self.bonafide_dir = bonafide_dir
        self.spoof_dir = spoof_dir
        self.transform = transform
        
        # Get a list of bonafide files and spoofed files
        self.bonafide_files = [os.path.join(bonafide_dir, f) for f in os.listdir(bonafide_dir) if f.endswith('.wav')]
        self.spoof_files = [os.path.join(spoof_dir, f) for f in os.listdir(spoof_dir) if f.endswith('.wav')]

        # Randomly sample files
        self.bonafide_files = random.sample(self.bonafide_files, min(num_samples, len(self.bonafide_files)))
        self.spoof_files = random.sample(self.spoof_files, min(num_samples, len(self.spoof_files)))

        # Combine the lists and create labels
        self.audio_files = self.bonafide_files + self.spoof_files
        self.labels = [0] * len(self.bonafide_files) + [1] * len(self.spoof_files)  # 0 for bonafide, 1 for spoof

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        """
        Get a single audio sample and its corresponding label.
        
        Args:
            idx (int): Index of the sample.
        
        Returns:
            waveform (torch.Tensor): The loaded audio waveform.
            label (int): The label (1 for bonafide, 0 for spoof).
        """
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        # Load the audio waveform
        waveform, sample_rate = torchaudio.load(audio_path)

        # Convert waveform to numpy for padding or processing if needed
        waveform = waveform.squeeze().numpy()

        # Apply any transforms if specified
        if self.transform:
            waveform = self.transform(waveform)

        return waveform, label
    

## Test with fmcc dataset
bonafide_dir = "ProjectStorage/test/fmcc_bonafide_cut"
spoof_dir = "ProjectStorage/test/fmcc_spoofed_cut" 
fmcc_dataset = TestAudioDataset(bonafide_dir, spoof_dir, transform=audio_transforms)
# Create the DataLoader
test_fmcc_loader = DataLoader(fmcc_dataset, batch_size=32, shuffle= False)
evaluate_model(model, test_fmcc_loader, device='cuda' if torch.cuda.is_available() else 'cpu')


## Test with jstu dataset
bonafide_dir = "ProjectStorage/test/jsut_bonafide"  
spoof_dir = "ProjectStorage/test/jsut_spoof"
jsut_dataset = TestAudioDataset(bonafide_dir, spoof_dir, transform=audio_transforms)
# Create the DataLoader
test_jsut_loader = DataLoader(jsut_dataset, batch_size=32, shuffle= False)
evaluate_model(model, test_jsut_loader, device='cuda' if torch.cuda.is_available() else 'cpu')


## Test with jstu dataset
bonafide_dir = "ProjectStorage/test/ljspeech_bonafide_cut"
spoof_dir = "ProjectStorage/test/ljspeech_spoof_cut"
ljspeech_dataset = TestAudioDataset(bonafide_dir, spoof_dir, transform=audio_transforms)
# Create the DataLoader
test_lj_loader = DataLoader(ljspeech_dataset, batch_size=32, shuffle= False)
evaluate_model(model, test_lj_loader, device='cuda' if torch.cuda.is_available() else 'cpu')




