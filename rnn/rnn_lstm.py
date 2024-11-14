
import numpy as np
import pandas as pd
#import librosa
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

import os
from glob import glob
import matplotlib.pyplot as plt
import re
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses info and warnings
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.callbacks import EarlyStopping

import joblib
from sklearn.utils.class_weight import compute_class_weight


'''#  "**128 rows, 57 cols**\n",
#     "\n",
#     "128: (y-axis) the **frequency bins aka (the number of frequency components)**.<br> The y-axis of a spectrogram corresponds to frequencies, and this number represents how many frequency bands the spectrogram has divided the signal into. So in this case, you have 128 different frequencies. <br><br>\n",
#     "57: (x-axis) the **time steps (or time frames)**.<br> The x-axis of the spectrogram corresponds to time, and 57 indicates how many time slices or frames were taken from your audio signal. Each time step represents a slice of the audio signal over a short duration."
#    ]'''


# # 0. run previously saved variables and models
### RUN THIS ONLY IF YOU HAVE THE FILES ####

# X_train_padded = np.load("X_train_padded.npy")
# X_test_padded = np.load("X_test_padded.npy")
# X_val_padded = np.load("X_val_padded.npy")

# trng_labels_index = joblib.load('trng_labels_index.joblib')
# test_labels_index = joblib.load('test_labels_index.joblib')
# val_labels_index = joblib.load('val_labels_index.joblib')

'''# 1.2 load all data and compile
# 1. find max row and col from all spectrograms
# 2. set as target shape (max_row, max_col)
# 3. pad each spectrogram so that all can be of same shape'''


# # Directory containing spectrogram .npy files
# spectrogram_dir = '/home/ubuntu/studies/ProjectStorage/new_spectrograms'
# spectrogram_folder = os.listdir(spectrogram_dir)

# aug_spectrogram_dir = '/home/ubuntu/studies/ProjectStorage/new_augmented_spectrograms'
# aug_spectrogram_folder = os.listdir(aug_spectrogram_dir)


# # Step 1: Find the maximum number of rows and columns across all og spectrograms
# og_max_rows = max(np.load(os.path.join(spectrogram_dir,file)).shape[0] for file in spectrogram_folder)
# og_max_cols = max(np.load(os.path.join(spectrogram_dir,file)).shape[1] for file in spectrogram_folder)

# # Step 2: Find the maximum number of rows and columns across all aug spectrograms
# aug_max_rows = max(np.load(os.path.join(aug_spectrogram_dir,file)).shape[0] for file in aug_spectrogram_folder)
# aug_max_cols = max(np.load(os.path.join(aug_spectrogram_dir,file)).shape[1] for file in aug_spectrogram_folder)

# # Step 3: Find the maximum number of rows and columns across all spectrograms
# max_cols = max(og_max_cols, aug_max_cols)
# max_rows = max(og_max_rows, aug_max_rows)

# print(f"Max Rows: {max_rows}, Max Columns: {max_cols}")
max_rows, max_cols = 128, 1076


# ### 1.2.1 load training set


spectrogram_dir = '/home/ubuntu/studies/ProjectStorage/train_spectrograms'
spectrogram_folder = os.listdir(spectrogram_dir)

# trng_padded_spectrograms = joblib.load('trng_padded_spectrograms.joblib')
# trng_labels_index = joblib.load('trng_labels_index.joblib')
# count = len(trng_labels_index)

# save the index of the file to find labels for each file later
trng_labels_index = []

# Count only .npy files
# total_file_count = len([file for file in spectrogram_folder if file.endswith('.npy')])
total_file_count = 54015
print(f"Total number of spectrograms: {total_file_count}")
count = 0

# Load and compile all spectrograms into a single dataset
# padded_spectrograms = []
spectrograms = []


for file in spectrogram_folder:
    if file.endswith('.npy'):
        spectrogram_path = os.path.join(spectrogram_dir, file)
        spectrogram = np.load(spectrogram_path)

        # # amount of padding to use
        # pad_rows = max_rows - spectrogram.shape[0]
        # pad_cols = max_cols - spectrogram.shape[1]

        # # Pad with zeros at the end of rows and columns to match the max shape
        # padded_spectrogram = np.pad(spectrogram, ((0, pad_rows), (0, pad_cols)), mode='constant')
        # padded_spectrograms.append(padded_spectrogram)

        ####### to ensure that every file in folder has been appended ###########
        # Use regex to extract the first sequence of digits in the file name
        match = re.search(r'\d+', file)
        if match:
            # Extract the integer part and print it
            file_number = int(match.group())
            print(f"File Loaded and appended: #{file_number}")
            trng_labels_index.append(file_number)
        spectrograms.append(spectrogram)
        count += 1
        print(f"Total files loaded and appended: {count} / {total_file_count}")

        if count % 100 == 0:
            # joblib.dump(padded_spectrograms, 'trng_padded_spectrograms.joblib')
            joblib.dump(trng_labels_index, 'trng_labels_index.joblib')
            joblib.dump(spectrograms, 'trng_spectrograms.joblib')


# # Convert the list of spectrograms to a numpy array
# trng_spectrogram_data = np.array(padded_spectrograms)
trng_spectrogram_data = np.array(spectrograms)


joblib.dump(trng_labels_index, 'trng_labels_index.joblib')
joblib.dump(trng_spectrograms, 'trng_spectrograms.joblib')




### load validation set

spectrogram_dir = '/home/ubuntu/studies/ProjectStorage/val_spectrograms'
spectrogram_folder = os.listdir(spectrogram_dir)

count = 0
# save the index of the file to find labels for each file later
val_labels_index = []

# Count only .npy files
total_file_count = len([file for file in spectrogram_folder if file.endswith('.npy')])
print(f"Total number of spectrograms: {total_file_count}")

# Load and compile all spectrograms into a single dataset
val_spectrograms = []

for file in spectrogram_folder:
    if file.endswith('.npy'):
        spectrogram_path = os.path.join(spectrogram_dir, file)
        spectrogram = np.load(spectrogram_path)

        # # amount of padding to use
        # pad_rows = max_rows - spectrogram.shape[0]
        # pad_cols = max_cols - spectrogram.shape[1]

        # # Pad with zeros at the end of rows and columns to match the max shape
        # padded_spectrogram = np.pad(spectrogram, ((0, pad_rows), (0, pad_cols)), mode='constant')
        # padded_spectrograms.append(padded_spectrogram)
        val_spectrograms.append(spectrogram)

        ####### to ensure that every file in folder has been appended ###########
        # Use regex to extract the first sequence of digits in the file name
        match = re.search(r'\d+', file)
        if match:
            # Extract the integer part and print it
            file_number = int(match.group())
            print(f"File Loaded and appended: #{file_number}")
            val_labels_index.append(file_number)
        count += 1
        print(f"Total files loaded and appended: {count} / {total_file_count}")

        if count % 100 == 0:
            # joblib.dump(padded_spectrograms, 'val_padded_spectrograms.joblib')
            joblib.dump(val_spectrograms, 'val_spectrograms.joblib')
            joblib.dump(val_labels_index, 'val_labels_index.joblib')


# Convert the list of spectrograms to a numpy array
# val_spectrogram_data = np.array(padded_spectrograms)

# save validation
joblib.dump(val_spectrograms, 'val_spectrograms.joblib')
joblib.dump(val_labels_index, 'val_labels_index.joblib')


# ### 1.2.3 load test set


spectrogram_dir = '/home/ubuntu/studies/ProjectStorage/test_spectrograms'
spectrogram_folder = os.listdir(spectrogram_dir)

count = 0
# save the index of the file to find labels for each file later
test_labels_index = []

# Count only .npy files
total_file_count = len([file for file in spectrogram_folder if file.endswith('.npy')])
print(f"Total number of spectrograms: {total_file_count}")

# Load and compile all spectrograms into a single dataset
#padded_spectrograms = []
test_spectrograms = []
count = 5600
for file in spectrogram_folder[5600:]:
    if file.endswith('.npy'):
        spectrogram_path = os.path.join(spectrogram_dir, file)
        spectrogram = np.load(spectrogram_path)
        test_spectrograms.append(spectrogram)

        # # # amount of padding to use
        # pad_rows = max_rows - spectrogram.shape[0]
        # pad_cols = max_cols - spectrogram.shape[1]

        # # Pad with zeros at the end of rows and columns to match the max shape
        # padded_spectrogram = np.pad(spectrogram, ((0, pad_rows), (0, pad_cols)), mode='constant')
        # padded_spectrograms.append(padded_spectrogram)

        ####### to ensure that every file in folder has been appended ###########
        # Use regex to extract the first sequence of digits in the file name
        match = re.search(r'\d+', file)
        if match:
            # Extract the integer part and print it
            file_number = int(match.group())
            print(f"File Loaded and appended: #{file_number}")
            test_labels_index.append(file_number)
        count += 1
        print(f"Total files loaded and appended: {count} / {total_file_count}")

        if count % 100 == 0:
            # np.save('test_padded_spectrograms.npy', padded_spectrograms)
            joblib.dump(test_spectrograms, 'test_spectrograms.joblib')
            joblib.dump(test_labels_index, 'test_labels_index.joblib')


# Convert the list of spectrograms to a numpy array
# test_spectrogram_data = np.array(padded_spectrograms)

# Check the final shape of the data
print(f"Max Rows: {max_rows}, Max Columns: {max_cols}")
#print(f"Compiled Spectrogram Data Shape: {test_spectrogram_data.shape}")
joblib.dump(test_spectrograms, 'test_spectrograms.joblib')
joblib.dump(test_labels_index, 'test_labels_index.joblib')




# # 2. extract labels and insert to y train, validation and test set

meta_csv = pd.read_csv('/home/ubuntu/studies/ProjectStorage/meta.csv')
labels = meta_csv['label'].map({'spoof':0, 'bona-fide':1})
len(labels)
X_train = trng_spectrograms
X_val = val_spectrograms
X_test = test_spectrograms

y_train = np.array([labels[i] for i in trng_labels_index])
y_val = np.array([labels[i] for i in val_labels_index])
y_test = np.array([labels[i] for i in test_labels_index])
np.save("y_train.npy", y_train)
np.save("y_val.npy", y_val)
np.save("y_test.npy", y_test)


# ## 2.1 padding

X_train_padded = []
count = 0
for spectrogram in X_train:
    # amount of padding to use
    pad_rows = max_rows - spectrogram.shape[0]
    pad_cols = max_cols - spectrogram.shape[1]

    # Pad with zeros at the end of rows and columns to match the max shape
    padded_spectrogram = np.pad(spectrogram, ((0, pad_rows), (0, pad_cols)), mode='constant')
    X_train_padded.append(padded_spectrogram)
    count +=1

    print(f"{count}/{len(X_train)}")



X_train_padded = np.array(X_train_padded)
X_train_padded.shape



X_val_padded = []
count = 0
for spectrogram in X_val:
    # amount of padding to use
    pad_rows = max_rows - spectrogram.shape[0]
    pad_cols = max_cols - spectrogram.shape[1]

    # Pad with zeros at the end of rows and columns to match the max shape
    padded_spectrogram = np.pad(spectrogram, ((0, pad_rows), (0, pad_cols)), mode='constant')
    X_val_padded.append(padded_spectrogram)
    count +=1

    print(f"{count}/{len(X_val)}")

X_val_padded = np.array(X_val_padded)
X_val_padded.shape



X_test_padded = []
count = 0
for spectrogram in X_test:
    # amount of padding to use
    pad_rows = max_rows - spectrogram.shape[0]
    pad_cols = max_cols - spectrogram.shape[1]

    # Pad with zeros at the end of rows and columns to match the max shape
    padded_spectrogram = np.pad(spectrogram, ((0, pad_rows), (0, pad_cols)), mode='constant')
    X_test_padded.append(padded_spectrogram)
    count +=1

    print(f"{count}/{len(X_test)}")

X_test_padded = np.array(X_test_padded)
X_test_padded.shape


np.save("X_test_padded.npy", X_test_padded)
np.save("X_train_padded.npy", X_train_padded)
np.save("X_val_padded.npy", X_val_padded)


# # 4. LSTM

# ## 4.1 training
# Calculate class weights based on the training labels
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# Define the RNN model with LSTM layers
def build_lstm_model(input_shape):
    model = Sequential()
    # LSTM layer with 64 units, input shape is (timesteps, features)
    model.add(LSTM(64, return_sequences=False, input_shape=input_shape))

    # Dropout for regularization
    model.add(Dropout(0.3))

    # Fully connected layer
    model.add(Dense(64, activation='relu'))

    # Another Dropout layer to prevent overfitting
    model.add(Dropout(0.3))

    # Output layer with sigmoid for binary classification
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
early_stopping = EarlyStopping(
    monitor='val_accuracy',   # Monitor validation accuracy
    patience=3,               # Number of epochs with no improvement after which training will be stopped
    verbose=1,                # Verbosity mode
    mode='max',               # 'max' means we want the largest value
    restore_best_weights=True # Restore model weights from the epoch with the best value of the monitored quantity
)

input_shape = (X_train_padded.shape[2], X_train_padded.shape[1])
lstm_model = build_lstm_model(input_shape)
lstm_model.summary()

# Train the model
lstm_model.fit(X_train_padded, y_train, epochs=30, batch_size=16, validation_data= (X_val_padded, y_val),class_weight=class_weights, callbacks=[early_stopping])


# # 4.2 testing

# Get predictions
lstm_y_pred_proba = lstm_model.predict(X_test_padded).flatten()
lstm_y_pred = (lstm_y_pred_proba > 0.5).astype(int)

# Function to calculate Equal Error Rate (EER)
def calculate_eer(y_true, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.abs(fnr - fpr))]
    eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
    return eer
# LSTM Model Metrics
lstm_accuracy = accuracy_score(y_test, lstm_y_pred)
lstm_f1 = f1_score(y_test, lstm_y_pred)
lstm_auc = roc_auc_score(y_test, lstm_y_pred_proba)
lstm_eer = calculate_eer(y_test, lstm_y_pred_proba)

# Print Results
print("LSTM Model Performance:")
print(f"  - Accuracy: {lstm_accuracy:.4f}")
print(f"  - F1 Score: {lstm_f1:.4f}")
print(f"  - AUC: {lstm_auc:.4f}")
print(f"  - EER: {lstm_eer:.4f}")