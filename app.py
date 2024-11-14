import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image, ImageDraw
import io
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import lime
from lime import lime_image
from skimage.segmentation import slic
import cv2

from cnnarc import AudioCNN

st.set_page_config(page_title="Deepfake Audio Detection with XAI", page_icon="")

class_names = ['bona-fide', 'spoof']

# Load the PyTorch model
@st.cache_resource()
def load_model():
    """Load the PyTorch model from the .pth file"""
    model = AudioCNN()  # Initialize the model architecture

    # Run a dummy forward pass to initialize fc1
    dummy_input = torch.zeros(1, 1, 128, 128)  # Assuming input shape is (1, 1, 128, 128)
    model(dummy_input)

    # Now load the saved weights
    model.load_state_dict(torch.load('audio_cnn_model.pth', map_location=torch.device('cpu'), weights_only=True))
    model.eval()  # Set the model to evaluation mode
    return model

# Image preprocessing for model prediction (grayscale)
def preprocess_image_for_model(image):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # Resize image to 128x128
        transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
        transforms.ToTensor(),          # Convert to tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize pixel values to [-1, 1]
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    return image_tensor

# Create spectrogram from uploaded .wav file
def create_spectrogram(sound_bytes):
    """Create a spectrogram from in-memory audio bytes."""
    y, sr = librosa.load(sound_bytes, sr=16000)  # Resample to 16kHz
    ms = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)  # Fewer Mel bands for faster processing
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # Create a figure and save the spectrogram to an in-memory buffer
    fig = plt.figure(figsize=(6, 6))
    librosa.display.specshow(log_ms, sr=sr)
    
    # Use io.BytesIO to save the image in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure to free memory

    # Convert the image buffer to a PIL Image for further processing
    buf.seek(0)
    img = Image.open(buf).convert("RGB")  # Keep it as RGB for XAI
    return img

# Make predictions using the PyTorch model (grayscale input)
def make_prediction(image_tensor, model):
    with torch.no_grad():
        # Forward pass through the model
        output = model(image_tensor)  # This output is the sigmoid probability

        # Since it's binary classification, the confidence score is the output itself
        confidence_score = output.item()  # Convert to a scalar

        # If confidence score > 0.5, classify as positive class (1), else negative class (0)
        predicted_class = 1 if confidence_score > 0.5 else 0

        # Return the predicted class and the confidence score
        return predicted_class, confidence_score

# LIME explanation for the spectrogram (use grayscale for the model, but RGB for XAI)
def model_predict_for_lime(image, model):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale for model
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Convert the RGB image to grayscale before passing to the model
    image_tensor = torch.stack([
        transform(Image.fromarray(np.uint8(np.squeeze(img)))) for img in image
    ])
    
    with torch.no_grad():
        outputs = model(image_tensor)
        return outputs.numpy()

# LIME explanation
def lime_explain(image, model):
    img_array = np.array(image) / 255.0
    explainer = lime_image.LimeImageExplainer()
    segmentation_fn = lambda x: slic(x, n_segments=100, compactness=10, sigma=1)
    
    explanation = explainer.explain_instance(
        img_array.astype('double'),
        classifier_fn=lambda x: model_predict_for_lime(x, model), 
        segmentation_fn=segmentation_fn, 
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(np.argmax(explanation.local_pred), positive_only=False, num_features=10, hide_rest=False)

    # Convert temp back to uint8 format for OpenCV processing
    temp_uint8 = np.uint8(temp * 255)
    
    # Convert back to RGB for rendering in Streamlit
    lime_image_display = Image.fromarray(temp_uint8)

    lime_text_explanation = (
        "The LIME explanation highlights regions of the spectrogram that influenced the model's decision. Green Region highlights the regions that did not influence the model's decision, as they have been masked out by LIME. "
    )

    return lime_image_display, lime_text_explanation

# Grad-CAM explanation
def gradcam_explain(image, model, class_idx):
    gradients = []

    def save_gradients(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hook for the last convolutional layer
    target_layer = model.conv3  # Modify this to match your architecture
    hook = target_layer.register_full_backward_hook(save_gradients)

    # Forward pass through the model
    model_output = model(image)
    class_output = model_output
    model.zero_grad()
    class_output.backward()

    pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
    activations = gradients[0].detach()

    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]

    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = torch.maximum(heatmap, torch.tensor(0.0))
    heatmap /= torch.max(heatmap)

    # Resize the heatmap to match the original image size
    heatmap = cv2.resize(heatmap.cpu().numpy(), (image.shape[2], image.shape[3]))

    # Normalize heatmap to [0, 255] and convert to uint8
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Extract the first image from the batch
    if image.dim() == 4:
        original_image = image[0]  # Extract the first image in the batch
    else:
        original_image = image

    # Ensure original_image is RGB
    original_image = original_image.permute(1, 2, 0).cpu().numpy()  # Change to H x W x C format
    if original_image.shape[2] == 1:  # If grayscale, convert to RGB
        original_image = cv2.cvtColor(np.uint8(255 * original_image), cv2.COLOR_GRAY2RGB)
    else:
        original_image = np.uint8(255 * original_image)

    # Ensure both heatmap and original image have the same number of channels
    if heatmap.shape[2] == 3 and original_image.shape[2] == 3:
        # Combine the heatmap with the original image
        superimposed_img = cv2.addWeighted(original_image, 0.6, heatmap, 0.4, 0)

    hook.remove()

    gradcam_text_explanation = (
        "The Grad-CAM heatmap highlights regions of the spectrogram where the model focused the most to classify the audio."
    )

    return Image.fromarray(superimposed_img), gradcam_text_explanation

# Generate both Grad-CAM and LIME explanations in one function
def generate_xai_explanations(image, preprocessed_image, model, class_idx):
    gradcam_image, gradcam_text = gradcam_explain(preprocessed_image, model, class_idx)
    lime_image, lime_text = lime_explain(image, model)

    return gradcam_image, gradcam_text, lime_image, lime_text

def homepage():
    st.subheader("Upload a .wav file")
    uploaded_file = st.file_uploader(' ', type='wav')

    if uploaded_file is not None:  
        st.write('### Play audio')
        audio_bytes = uploaded_file.read()
        st.audio(audio_bytes, format='audio/wav')

        st.write('### Spectrogram Image:')
        sound_bytes = io.BytesIO(audio_bytes)
        spectrogram_image = create_spectrogram(sound_bytes)
        st.image(spectrogram_image, caption="Generated Spectrogram", use_column_width=True)

        # Use the correct preprocessing function for model prediction (grayscale)
        preprocessed_image = preprocess_image_for_model(spectrogram_image)
        model = load_model()

        with st.spinner('Fetching Results...'):
            class_label, confidence = make_prediction(preprocessed_image, model)

        st.write(f"#### Confidence Score: {confidence:.4f}")
        st.write(f"#### The uploaded audio file is classified as: **{class_names[class_label]}**")
        

        # Button to toggle showing explanations
        if st.button('Show XAI Explanations'):
            with st.spinner('Generating explanations...'):
                gradcam_image, gradcam_text, lime_image, lime_text = generate_xai_explanations(
                    spectrogram_image, preprocessed_image, model, class_label
                )
            st.image(gradcam_image, caption="Grad-CAM Explanation", use_column_width=True)
            st.write(gradcam_text)

            st.image(lime_image, caption="LIME Explanation", use_column_width=True)
            st.write(lime_text)

    else:
        st.info("Please upload a .wav file")


# Main function to handle page selection
def main():
    page = st.sidebar.selectbox("App Selections", ["Homepage", "About"])
    if page == "Homepage":
        st.title("Deepfake Audio Detection using CNN with XAI")
        homepage()
    elif page == "About":
        st.title("About the Project")
        st.markdown("""
            This project focuses on detecting deepfake audio using a CNN model and explainable AI (XAI) techniques.
        """)

if __name__ == "__main__":
    main()
