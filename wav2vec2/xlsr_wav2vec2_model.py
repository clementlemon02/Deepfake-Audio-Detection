# -*- coding: utf-8 -*-
"""xlsr_wav2vec2.ipynb"""

# Standard libraries
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union

# Data manipulation and numerical libraries
import numpy as np
import pandas as pd

# PyTorch and neural network-related libraries
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from packaging import version
from tqdm import tqdm

# Transformers
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    EvalPrediction,
    AutoConfig,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    is_apex_available,
)
from transformers.file_utils import ModelOutput
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Model
)

# Machine learning metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc
)

# Visualization
import matplotlib.pyplot as plt

# AMP (Automatic Mixed Precision) setup
if is_apex_available():
    from apex import amp
if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast


"""## Model"""

# Initialize processor and model
model_name_or_path = "facebook/wav2vec2-large-xlsr-53"
pooling_mode = "mean"

@dataclass
class SpeechClassifierOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

class Wav2Vec2ClassificationHead(nn.Module):
    """Head for wav2vec classification task."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Wav2Vec2ForSpeechClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.pooling_mode = config.pooling_mode
        self.config = config

        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = Wav2Vec2ClassificationHead(config)

        self.init_weights()

    def freeze_feature_extractor(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    def merged_strategy(
            self,
            hidden_states,
            mode="mean"
    ):
        if mode == "mean":
            outputs = torch.mean(hidden_states, dim=1)
        elif mode == "sum":
            outputs = torch.sum(hidden_states, dim=1)
        elif mode == "max":
            outputs = torch.max(hidden_states, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        hidden_states = self.merged_strategy(hidden_states, mode=self.pooling_mode)
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels == 2:  # Explicitly handle binary classification
                    self.config.problem_type = "single_label_classification"
                elif self.num_labels > 2 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SpeechClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# Initialize feature extractor (instead of processor)
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        feature_extractor (:class:`~transformers.Wav2Vec2FeatureExtractor`)
            The feature extractor used for processing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index).
        max_length (:obj:`int`, `optional`):
            Maximum length of the ``input_values`` of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set, will pad the sequence to a multiple of the provided value.
    """

    feature_extractor: Wav2Vec2FeatureExtractor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        # Ensure that labels are treated as integers for classification
        d_type = torch.long  # Use torch.long for classification labels

        batch = self.feature_extractor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        batch["labels"] = torch.tensor(label_features, dtype=d_type)

        return batch

data_collator = DataCollatorCTCWithPadding(feature_extractor=feature_extractor, padding=True)
is_regression = False

"""## Training"""

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

model.freeze_feature_extractor()


# Specify the output directory
output_dir = "./w2v2models/my_model_6"

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=1,
    evaluation_strategy="epoch",  # Ensure evaluation happens after each epoch
    save_strategy="epoch",        # Ensure the model is saved after each epoch
    num_train_epochs=20,           # Set higher, but with early stopping in mind
    fp16=False,
    logging_steps=10,              # Log more frequently
    learning_rate=1e-6,
    save_total_limit=5,
    load_best_model_at_end=True,   # Ensures that the best model is loaded after training
    metric_for_best_model="eval_loss",  # Can be adjusted based on your evaluation metric
    greater_is_better=False,       # Since lower validation loss is better
)


class CTCTrainer(Trainer):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     # Initialize use_amp based on the fp16 setting from training_args
    #     self.use_amp = self.args.fp16

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize use_amp based on the fp16 setting from training_args
        self.use_amp = self.args.fp16 and _is_native_amp_available
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (:obj:`nn.Module`):
                The model to train.
            inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument :obj:`labels`. Check your model's documentation for all accepted arguments.

        Return:
            :obj:`torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        # Reduce the loss to a scalar by averaging the batch losses
        loss = loss.mean()

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()


# Define early stopping callback
early_stopping_callback = EarlyStoppingCallback(
    early_stopping_patience=3,  # Number of evaluation steps with no improvement before stopping
    early_stopping_threshold=0.01  # Minimum change to consider as an improvement
)

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=feature_extractor,
    callbacks=[early_stopping_callback]
)

trainer.train()


"""## Evaluation"""

# Specify the checkpoint directory
checkpoint_dir = "./w2v2models/my_model_6/checkpoint-25080"

# Load the model configuration and model from the checkpoint
config = AutoConfig.from_pretrained(checkpoint_dir)
model = Wav2Vec2ForSpeechClassification.from_pretrained(checkpoint_dir, config=config)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load the feature extractor from the checkpoint
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(checkpoint_dir)

# Check if a GPU is available and use it; otherwise, fall back to the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Helper function to calculate the EER
def calculate_eer(fpr, tpr):
    eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
    return eer

# Updated compute_metrics function to include EER and ROC AUC
def compute_metrics(p: EvalPrediction, predictions_proba):
    preds = p.predictions  # Directly use the predictions as class labels
    labels = p.label_ids

    # Compute basic classification metrics
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted')
    f1 = f1_score(labels, preds, average='weighted')

    # Compute ROC Curve and EER
    fpr, tpr, thresholds = roc_curve(labels, predictions_proba[:, 1])  # Assuming binary classification
    roc_auc = auc(fpr, tpr)
    eer = calculate_eer(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # Create a dictionary of the results
    metrics = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "EER": eer,
        "ROC AUC": roc_auc
    }

    return metrics

# Function to evaluate the test set
def evaluate_test_set(test_dataset, model, device):
    test_dataloader = DataLoader(test_dataset, batch_size=8)
    model.eval()  # Set the model to evaluation mode

    all_preds = []
    all_labels = []
    all_probs = []

    for batch in test_dataloader:
        input_values = batch['input_values'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.no_grad():
            outputs = model(input_values, attention_mask=attention_mask)
            logits = outputs.logits

        # Calculate probabilities and get the predicted class index
        probs = torch.softmax(logits, dim=-1)  # Get probabilities
        preds = torch.argmax(probs, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    # Compute the metrics using the updated function
    metrics = compute_metrics(EvalPrediction(predictions=np.array(all_preds),
                                             label_ids=np.array(all_labels)),
                              predictions_proba=np.array(all_probs))

    # Convert the metrics dictionary to a DataFrame for a nice display
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
    metrics_df.index.name = 'Metric'
    return metrics_df

"""### Test 1"""

# Evaluate the test set
test_results_df = evaluate_test_set(test_dataset, model, device)
print("Test set results:\n", test_results_df)

"""### Test 2"""

def load_test_files_from_wav(label, wav_folder):
    audio_data = []

    print("Loading audio files with the provided label...")
    # Loop through all .wav files in the directory
    for wav_file in tqdm(os.listdir(wav_folder), desc="Processing .wav files"):
        if wav_file.endswith('.wav'):
            audio_path = os.path.join(wav_folder, wav_file)
            audio_segments, attention_masks = speech_file_to_array_fn(audio_path)

            # For each audio file, assign the provided label
            for segment, attention_mask in zip(audio_segments, attention_masks):
                audio_data.append((segment, attention_mask, label))

    print(f"Loaded {len(audio_data)} audio segments with the label '{label}' successfully.")
    return audio_data

# Load Test Data
label_list = ["spoof", "bona-fide"]
audio_cc_bf = load_test_files_from_wav("bona-fide", './ProjectStorage/test/fmcc_bonafide_cut')
audio_cc_sp = load_test_files_from_wav("spoof", './ProjectStorage/test/fmcc_spoofed_cut')

audio_df_ccbf = create_audio_dataframe(audio_cc_bf)
audio_df_ccsp = create_audio_dataframe(audio_cc_sp)

cc_df = pd.concat([audio_df_ccbf, audio_df_ccsp], axis=0, ignore_index=True)
cc_dataset = AudioDataset(cc_df)

# Randomly select a sample from the audio_data list
idx = np.random.randint(0, len(audio_cc_bf))
sample_audio, attention_mask, label = audio_cc_bf[idx]

print(f"ID Location: {idx}")
print(f"      Label: {label}")
print()

# Convert the audio signal to a PyTorch tensor with a specified dtype
speech_tensor = torch.tensor(sample_audio, dtype=torch.float32).unsqueeze(0)

# Resample the audio signal to ensure it matches the required sampling rate
speech_resampled = librosa.resample(sample_audio, orig_sr=16000, target_sr=16000)

# Play the audio using IPython display
ipd.Audio(data=speech_resampled, rate=16000)

# Evaluate the test set
test_results_df_2 = evaluate_test_set(cc_dataset, model, device)
print("Test set results:\n", test_results_df_2)

"""### Test 3"""

# Load Test Data
audio_js_bf = load_test_files_from_wav("bona-fide", './ProjectStorage/test/jsut_bonafide')
audio_js_sp = load_test_files_from_wav("spoof", './ProjectStorage/test/jsut_spoof')

audio_df_jsbf = create_audio_dataframe(audio_js_bf)
audio_df_jssp = create_audio_dataframe(audio_js_sp)

js_df = pd.concat([audio_df_jsbf, audio_df_jssp], axis=0, ignore_index=True)
js_dataset = AudioDataset(js_df)

# Randomly select a sample from the audio_data list
idx = np.random.randint(0, len(audio_js_bf))
sample_audio, attention_mask, label = audio_js_bf[idx]

print(f"ID Location: {idx}")
print(f"      Label: {label}")
print()

# Convert the audio signal to a PyTorch tensor with a specified dtype
speech_tensor = torch.tensor(sample_audio, dtype=torch.float32).unsqueeze(0)

# Resample the audio signal to ensure it matches the required sampling rate
speech_resampled = librosa.resample(sample_audio, orig_sr=16000, target_sr=16000)

# Play the audio using IPython display
ipd.Audio(data=speech_resampled, rate=16000)

# Evaluate the test set
test_results_df_3 = evaluate_test_set(js_dataset, model, device)
print("Test set results:\n", test_results_df_3)

"""### Test 4"""

# Load Test Data
audio_lj_bf = load_test_files_from_wav("bona-fide", './ProjectStorage/test/ljspeech_bonafide_cut')
audio_lj_sp = load_test_files_from_wav("spoof", './ProjectStorage/test/ljspeech_spoof_cut')

audio_df_ljbf = create_audio_dataframe(audio_lj_bf)
audio_df_ljsp = create_audio_dataframe(audio_lj_sp)

lj_df = pd.concat([audio_df_ljbf, audio_df_ljsp], axis=0, ignore_index=True)
lj_dataset = AudioDataset(lj_df)

# Randomly select a sample from the audio_data list
idx = np.random.randint(0, len(audio_lj_bf))
sample_audio, attention_mask, label = audio_lj_bf[idx]

print(f"ID Location: {idx}")
print(f"      Label: {label}")
print()

# Convert the audio signal to a PyTorch tensor with a specified dtype
speech_tensor = torch.tensor(sample_audio, dtype=torch.float32).unsqueeze(0)

# Resample the audio signal to ensure it matches the required sampling rate
speech_resampled = librosa.resample(sample_audio, orig_sr=16000, target_sr=16000)

# Play the audio using IPython display
ipd.Audio(data=speech_resampled, rate=16000)

# Evaluate the test set
test_results_df_4 = evaluate_test_set(lj_dataset, model, device)
print("Test set results:\n", test_results_df_4)

