# Deepfake Audio Detection Project

## Overview
This repository contains a comprehensive deepfake audio detection system that evaluates the effectiveness of five distinct machine learning models. These models are designed to detect and identify manipulated audio recordings to provide robust and accurate results. The models implemented in this project are:


## Features

### Models
- **RNN**: Effective for time-series analysis by learning temporal patterns in audio data.
- **CNN**: Ideal for extracting spatial features from audio spectrograms for classification.
- **Transformer**: Employs self-attention for capturing global context in audio sequences.
- **wav2vec**: A pre-trained model that provides rich audio embeddings, enhancing feature representation.
- **RawNet2**: Processes raw waveforms for direct end-to-end learning without additional feature extraction.

### Audio Representations
The models are trained and evaluated using a variety of audio representations:
- **Mel Spectrograms**: Represent audio with a frequency scale that aligns closely with human auditory perception.
- **MFCC (Mel-Frequency Cepstral Coefficients)**: Extracts features that closely mimic the human ear's response.
- **VQT (Variable-Q Transform)**: Provides adaptive resolution for analyzing audio at different frequency bands.
- **CQT (Constant-Q Transform)**: Represents audio with a resolution that matches the perception of musical pitch.
- **Log Spectrograms**: Logarithmic scaling highlights frequency components to help models identify distinctive audio features.

### Comprehensive Evaluation Metrics
The project evaluates the models using:
- **Accuracy**: Overall performance metric.
- **F1-Score**: Balance between precision and recall.
- **AUC (Area Under the Curve)**: Measures the ability to distinguish between classes.
- **EER (Equal Error Rate)**: Indicates the point where the false acceptance rate equals the false rejection rate.


### Dataset Used
In The Wild Dataset: https://deepfake-total.com/in_the_wild
