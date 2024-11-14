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
    "train": "./trainset/vqt",
    "val": "./valset/vqt",
    "test": "./testset/vqt",
    "fmcc_bonafide": "./test/vqt_fmcc_bonafide_cut",
    "fmcc_spoofed": "./test/vqt_fmcc_spoofed_cut",
    "jsut_bonafide": "./test/vqt_jsut_bonafide",
    "jsut_spoofed": "./test/vqt_jsut_spoof",
    "ljspeech_bonafide": "./test/vqt_ljspeech_bonafide_cut",
    "ljspeech_spoofed": "./test/vqt_ljspeech_spoof_cut",
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

# Function to generate and save VQT spectrogram as .npy file
def generate_vqt(wav_file, output_file):
    y, sr = librosa.load(wav_file, sr=None)
    y = ensure_four_seconds(y, sr)
    
    # Generate VQT spectrogram with 84 bins
    vqt = np.abs(librosa.vqt(y, sr=sr, n_bins=84, bins_per_octave=12, fmin=librosa.note_to_hz('C1')))
    
    # Convert to decibels (log scale)
    vqt_dB = librosa.amplitude_to_db(vqt, ref=np.max)
    
    # Save the VQT spectrogram as a .npy file
    np.save(output_file, vqt_dB)

# Process each input directory and save to corresponding output directory
for set_type, input_dir in input_dirs.items():
    output_dir = output_dirs[set_type]
    wav_files = [f for f in os.listdir(input_dir) if f.endswith('.wav')]
    
    print(f"Processing {set_type} set for VQT generation:")
    for wav_file in tqdm(wav_files, desc=f"Generating VQT for {set_type}", unit="file"):
        # Full path to the .wav file
        full_wav_path = os.path.join(input_dir, wav_file)
        
        # Output file path (same name as the .wav file but with _vqt.npy suffix)
        output_file = os.path.join(output_dir, os.path.splitext(wav_file)[0] + '_vqt.npy')
        
        # Skip if the output file already exists
        if os.path.exists(output_file):
            continue
        
        # Generate and save the VQT spectrogram as a .npy file
        generate_vqt(full_wav_path, output_file)

print("VQT spectrogram generation completed.")
