import concurrent.futures
import os
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

def process_file(base_name, cqt_dir, vqt_dir, mel_dir, mfcc_dir, logspec_dir, stack_dir, target_shape):
    try:
        # Load and resize each spectrogram type
        cqt = resize(np.load(os.path.join(cqt_dir, f"{base_name}_cqt.npy")), target_shape, anti_aliasing=True)
        vqt = resize(np.load(os.path.join(vqt_dir, f"{base_name}_vqt.npy")), target_shape, anti_aliasing=True)
        mel = resize(np.load(os.path.join(mel_dir, f"{base_name}_mel.npy")), target_shape, anti_aliasing=True)
        mfcc = resize(np.load(os.path.join(mfcc_dir, f"{base_name}_mfcc.npy")), target_shape, anti_aliasing=True)
        logspec = resize(np.load(os.path.join(logspec_dir, f"{base_name}_logspec.npy")), target_shape, anti_aliasing=True)

        # Stack and save
        stacked_spectrogram = np.stack([cqt, vqt, mel, mfcc, logspec], axis=0)
        np.save(os.path.join(stack_dir, f"{base_name}_stacked.npy"), stacked_spectrogram)

    except FileNotFoundError as e:
        print(f"Missing file for {base_name}: {e}")

def preprocess_and_save_parallel(base_dir, stack_dir, target_shape=(128, 128), max_workers=4):
    os.makedirs(stack_dir, exist_ok=True)
    
    # Define directories for each spectrogram type
    cqt_dir = os.path.join(base_dir, 'cqt')
    vqt_dir = os.path.join(base_dir, 'vqt')
    mel_dir = os.path.join(base_dir, 'mel')
    mfcc_dir = os.path.join(base_dir, 'mfcc')
    logspec_dir = os.path.join(base_dir, 'logspec')
    
    # Use cqt folder as the main reference and parallelize processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_file, filename.rsplit('_', 1)[0], cqt_dir, vqt_dir, mel_dir, mfcc_dir, logspec_dir, stack_dir, target_shape)
            for filename in os.listdir(cqt_dir) if filename.endswith('.npy')
        ]
        
        # Show progress
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc=f"Processing {base_dir}"):
            pass
    print(f"Preprocessing complete. Stacked spectrograms saved to {stack_dir}")

# Define input directories and corresponding output 'stack' directories to match desired structure
input_dirs = {
    "trainset": "./trainset",
    "valset": "./valset",
    "testset": "./testset",
}

# Preprocess each directory and save output in its corresponding 'stack' folder
for key, base_dir in input_dirs.items():
    output_dir = os.path.join(base_dir, "stack")
    preprocess_and_save_parallel(base_dir=base_dir, stack_dir=output_dir)
