import os
import librosa
import numpy as np
from tqdm import tqdm

# Define all input and output directories
input_dirs = {
    "train": "./wav_train_set",
    "val": "./wav_val_set",
    "test": "./wav_test_set",
    "fmcc_bonafide": "./test/fmcc_bonafide_cut",
    "fmcc_spoofed": "./test/fmcc_spoofed_cut",
    "jsut_bonafide": "./test/jsut_bonafide",
    "jsut_spoofed": "./test/jsut_spoof",
    "ljspeech_bonafide": "./test/ljspeech_bonafide_cut",
    "ljspeech_spoofed": "./test/ljspeech_spoof_cut"
}

output_dirs = {
    "train": "./trainset/mel",
    "val": "./valset/mel",
    "test": "./testset/mel",
    "fmcc_bonafide": "./test/mel_fmcc_bonafide_cut",
    "fmcc_spoofed": "./test/mel_fmcc_spoofed_cut",
    "jsut_bonafide": "./test/mel_jsut_bonafide",
    "jsut_spoofed": "./test/mel_jsut_spoof",
    "ljspeech_bonafide": "./test/mel_ljspeech_bonafide_cut",
    "ljspeech_spoofed": "./test/mel_ljspeech_spoof_cut",
}

# Create output directories if they don't exist
for output_dir in output_dirs.values():
    os.makedirs(output_dir, exist_ok=True)

# Function to ensure the audio length is exactly 4 seconds
def ensure_four_seconds(y, sr):
    target_length = sr * 4
    current_length = len(y)
    if current_length > target_length:
        start = np.random.randint(0, current_length - target_length)
        y = y[start:start + target_length]
    elif current_length < target_length:
        repeat_factor = int(np.ceil(target_length / current_length))
        y = np.tile(y, repeat_factor)[:target_length]
    return y

# Function to generate and save Mel spectrogram as .npy file
def generate_mel(wav_file, output_file):
    y, sr = librosa.load(wav_file, sr=None)
    y = ensure_four_seconds(y, sr)
    
    # Generate Mel spectrogram with 84 Mel bands
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=84)
    
    # Convert to decibels (log scale)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Save the Mel spectrogram as a .npy file
    np.save(output_file, log_mel_spec)

# Process each input directory and save to corresponding output directory
for set_type, input_dir in input_dirs.items():
    output_dir = output_dirs[set_type]
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    print(f"Processing {set_type} set for Mel spectrogram generation:")
    for wav_file in tqdm(wav_files, desc=f"Generating Mel spectrogram for {set_type}", unit="file"):
        # Full path to the .wav file
        full_wav_path = os.path.join(input_dir, wav_file)
        
        # Output file path (same name as the .wav file but with _mel.npy suffix)
        output_file = os.path.join(output_dir, os.path.splitext(wav_file)[0] + '_mel.npy')
        
        # Generate and save the Mel spectrogram as a .npy file
        generate_mel(full_wav_path, output_file)

print("Mel spectrogram generation completed.")
