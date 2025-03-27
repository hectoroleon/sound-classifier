import streamlit as st
import torch
import librosa
import numpy as np
import torchvision.transforms as T
from PIL import Image
from src.model_transfer_convnext import ConvNeXtAudioClassifier

# Page setup
st.set_page_config(page_title="🌍 What’s That Sound? (ESC-50 Classifier)", layout="centered")
st.title("🌍 What’s That Sound? (ESC-50 Classifier)")
st.markdown("Upload a `.wav` file and get the predicted sound category using a ConvNeXt model.")

# ESC-50 class labels
LABELS = [
    'airplane', 'breathing', 'brushing_teeth', 'can_opening', 'car_horn',
    'cat', 'chainsaw', 'chirping_birds', 'church_bells', 'clapping',
    'clock_alarm', 'clock_tick', 'coughing', 'cow', 'crackling_fire',
    'crickets', 'crow', 'crying_baby', 'dog', 'door_wood_creaks',
    'door_wood_knock', 'drinking_sipping', 'engine', 'fireworks', 'footsteps',
    'frog', 'glass_breaking', 'hand_saw', 'helicopter', 'hen', 'insects',
    'keyboard_typing', 'laughing', 'mouse_click', 'pig', 'pouring_water',
    'rain', 'rooster', 'sea_waves', 'sheep', 'siren', 'sneezing', 'snoring',
    'thunderstorm', 'toilet_flush', 'train', 'vacuum_cleaner',
    'washing_machine', 'water_drops', 'wind'
]

# Emoji mapping
label_to_emoji = {
    'airplane': '✈️', 'breathing': '😮‍💨', 'brushing_teeth': '🪥', 'can_opening': '🥫', 'car_horn': '🚗📣',
    'cat': '🐱', 'chainsaw': '🪓', 'chirping_birds': '🐦', 'church_bells': '🔔', 'clapping': '👏',
    'clock_alarm': '⏰', 'clock_tick': '🕰️', 'coughing': '🤧', 'cow': '🐄', 'crackling_fire': '🔥',
    'crickets': '🦗', 'crow': '🐦', 'crying_baby': '👶😭', 'dog': '🐶', 'door_wood_creaks': '🚪',
    'door_wood_knock': '🚪👊', 'drinking_sipping': '🥤', 'engine': '⚙️', 'fireworks': '🎆', 'footsteps': '👣',
    'frog': '🐸', 'glass_breaking': '🪟💥', 'hand_saw': '🪚', 'helicopter': '🚁', 'hen': '🐔', 'insects': '🐛',
    'keyboard_typing': '⌨️', 'laughing': '😂', 'mouse_click': '🖱️', 'pig': '🐷', 'pouring_water': '🚿',
    'rain': '🌧️', 'rooster': '🐓', 'sea_waves': '🌊', 'sheep': '🐑', 'siren': '🚨', 'sneezing': '🤧',
    'snoring': '💤', 'thunderstorm': '⛈️', 'toilet_flush': '🚽', 'train': '🚆', 'vacuum_cleaner': '🧹',
    'washing_machine': '🧺', 'water_drops': '💧', 'wind': '🌬️'
}

# 🔽 Dropdown to show all classes
with st.expander("📚 See all 50 available sound classes"):
    selected = st.selectbox("ESC-50 Categories", LABELS, index=0, label_visibility="collapsed")
    st.markdown("You can upload a sample that belongs to one of these categories.")

# Load model once
@st.cache_resource
def load_model():
    model = ConvNeXtAudioClassifier(num_classes=50)
    model.load_state_dict(torch.load("models/convnext_audio_classifier.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Upload audio
audio_file = st.file_uploader("Upload a WAV file", type=["wav"])

if audio_file:
    y, sr = librosa.load(audio_file, sr=22050, duration=5.0)

    st.subheader("🔊 Playback")
    st.audio(audio_file, format="audio/wav")

    # Mel-spectrogram for model input
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())
    mel_img = Image.fromarray(np.uint8(mel_norm * 255))

    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Lambda(lambda x: x.expand(3, -1, -1)),
        T.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = transform(mel_img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred_index = torch.argmax(output, dim=1).item()
        pred_label = LABELS[pred_index]
        confidence = torch.softmax(output, dim=1)[0][pred_index].item()
        emoji = label_to_emoji.get(pred_label, "🎧")

    st.subheader("🧠 Prediction")
    st.write(f"{emoji} **{pred_label}** — {confidence:.1%} confidence")

# Attribution
st.markdown("---")
st.markdown("👨‍💻 Developed by **Héctor Ornelas**")

