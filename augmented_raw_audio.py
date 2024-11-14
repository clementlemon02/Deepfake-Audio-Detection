
# # Augmented Raw Audio

# ## 1. import library and read folder

import librosa
import numpy as np
import os
import soundfile as sf
import scipy.signal as signal
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*bottleneck.*")


input_folder = "../release_in_the_wild"
output_folder = "../augmented_audio_files"





## 2. Augmentation functions
def add_noise(audio, noise_factor=0.005):
    noise = np.random.randn(len(audio))
    augmented_audio = audio + noise_factor * noise
    return augmented_audio

def time_stretch(audio, rate=1.1):
    return librosa.effects.time_stretch(audio, rate=rate)

def pitch_shift(audio, sample_rate, n_steps=2):
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=n_steps)


# ## 3. augment audio file
# 1. split into thirds <br>
# 2.1 noisy, stretched, shifted pitch <br>
# 2.2 louder vol, reverb, shifted time <br>
# 2.3 smaller vol, lower frequency components, noisy <br>



# count number of files in folder
def count_wav_files(folder_path):
    # Get the list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter the files that end with .wav
    wav_files = [file for file in files if file.endswith('.wav')]
    
    # Return the count of .wav files
    return len(wav_files)


def augment_audio_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print("output_folder doesn't exist")
        
    total_file_count = count_wav_files(folder_path)
    print(f"Number of .wav files: {total_file_count}")
    
    quartered_file_count = total_file_count//3
    count = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # Assuming WAV files, change if needed
            file_path = os.path.join(folder_path, filename)
            audio, sr = librosa.load(file_path)
        
######### Apply augmentations ###########
        
            # noisy, stretched, shifted pitch
            if count < quartered_file_count: #1/3
                noisy_audio = add_noise(audio)
                stretched_audio = time_stretch(noisy_audio)
                noisy_stretch_pitch_audio = pitch_shift(stretched_audio, sr)
                sf.write(os.path.join(output_folder, f"noisy_stretch_pitch_audio_{filename}"), noisy_stretch_pitch_audio, sr)
            
            # louder vol, reverb, shifted time
            elif count > quartered_file_count and count < quartered_file_count*2: #2/3
                # Shift the audio by rolling (circular shift)
                shifted_audio = np.roll(audio, sr // 10)  # Shift by 0.1 second
                reverb_audio = librosa.effects.preemphasis(shifted_audio)
                # Increase volume by a factor of 1.5
                timeShifted_reverb_louder_audio = audio * 1.5
                sf.write(os.path.join(output_folder, f"timeShifted_reverb_louder_audio_{filename}"), timeShifted_reverb_louder_audio, sr)
                
            # smaller vol, lower frequency components, noisy
            else: #3/3
                # Decrease volume by a factor of 0.5
                quieter_audio = audio * 0.5
                # Apply a simple low-pass filter (keeping low frequencies, removing highs)
                sos = signal.butter(10, 1000, 'low', fs=sr, output='sos')  # Cutoff frequency at 1kHz
                lower_freq_audio = signal.sosfilt(sos, quieter_audio)
                noisy_quieter_lowerFreq_audio = add_noise(lower_freq_audio)
                sf.write(os.path.join(output_folder, f"noisy_quieter_lowerFreq_audio_{filename}"), noisy_quieter_lowerFreq_audio, sr)
                
            count+=1
            
            #print(f"Processed {filename}")
            print(f"Current process of .wav files: {count}", " / ", f"{total_file_count}")


augment_audio_in_folder(input_folder, output_folder)
