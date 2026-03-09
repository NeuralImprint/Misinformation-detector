import streamlit as st
import feedparser
import urllib.parse
from sentence_transformers import SentenceTransformer, util
import whisper
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import tempfile
import librosa

# --- CONFIGURATION & MODEL LOADING ---
ST_DEVICE = "cpu"  # Streamlit Cloud does not have GPU

@st.cache_resource
def load_deepfake_model():
    """Load the EfficientNetV2-S classifier using timm with custom 5-class head."""
    try:
        import timm
        model = timm.create_model('tf_efficientnetv2_s', num_classes=5)
        model.classifier = nn.Sequential(
            nn.Linear(model.classifier.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 5)
        )
        weights_path = os.path.join(os.path.dirname(__file__), "weight.pth")
        if os.path.exists(weights_path):
            state_dict = torch.load(weights_path, map_location=ST_DEVICE)
            new_state_dict = {k.replace("base_model.", ""): v for k, v in state_dict.items()}
            model.load_state_dict(new_state_dict)
            model.eval().to(ST_DEVICE)
            return model, True
        return None, False
    except Exception as e:
        st.error(f"Error loading Deepfake weights: {e}")
        return None, False

@st.cache_resource
def load_audio_deepfake_model():
    """Load a pre-trained audio deepfake detector."""
    try:
        from transformers import pipeline
        return pipeline("audio-classification", model="MIT/ast-finetuned-audioset-10-10-0.4593")
    except Exception as e:
        st.error(f"Error loading Audio model: {e}")
        return None

@st.cache_resource
def load_whisper_model():
    """Load Whisper speech-to-text model."""
    try:
        return whisper.load_model("base", device=ST_DEVICE)
    except Exception as e:
        st.warning(f"Whisper load issue: {e}")
        return None

@st.cache_resource
def load_sentence_transformer():
    """Load sentence similarity model for fact-checking."""
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_face_detector():
    """Initialize MediaPipe Face Detector using the Tasks API."""
    model_path = os.path.join(os.path.dirname(__file__), "blaze_face_short_range.tflite")
    if not os.path.exists(model_path):
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
        urllib.request.urlretrieve(url, model_path)
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceDetectorOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_detection_confidence=0.5
    )
    return vision.FaceDetector.create_from_options(options)

# --- UTILITY FUNCTIONS ---

def verify_claim(claim, sentence_model):
    """Search Google News RSS and compare semantic similarity."""
    encoded_query = urllib.parse.quote(claim[:200])
    url = f"https://news.google.com/rss/search?q={encoded_query}"
    feed = feedparser.parse(url)
    articles = [entry.title for entry in feed.entries[:5]]
    if not articles:
        return 0, []
    claim_map = sentence_model.encode([claim] + articles)
    scores = util.cos_sim(claim_map[0], claim_map[1:])
    return float(torch.max(scores)), articles

def preprocess_face(face_img):
    """Preprocess a face crop for EfficientNetV2-S inference."""
    preprocess = transforms.Compose([
        transforms.Resize(384),
        transforms.CenterCrop(384),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    return preprocess(pil_img).unsqueeze(0).to(ST_DEVICE)

def analyze_visual(frame, model, face_detector):
    """Detect faces in a frame and classify each as real/fake."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = face_detector.detect(mp_image)
    if not results.detections:
        return None

    scores = []
    for d in results.detections:
        bbox = d.bounding_box
        x, y, w, h = int(bbox.origin_x), int(bbox.origin_y), int(bbox.width), int(bbox.height)
        pad = int(min(w, h) * 0.3)
        face = frame[max(0, y - pad):y + h + pad, max(0, x - pad):x + w + pad]
        if face.size == 0:
            continue
        input_t = preprocess_face(face)
        with torch.no_grad():
            probs = F.softmax(model(input_t), dim=1)
            scores.append(probs[0][0].item())  # Class 0 = Real (FF++ convention)
    return np.mean(scores) if scores else 0.5

# --- STREAMLIT UI ---

st.set_page_config(page_title="Shodh AI - Unified Verifier", layout="wide", page_icon="🛡️")
st.title("🛡️ Shodh AI: All-in-One Verifier")
st.markdown("Upload any **Video**, **Audio**, or **Image** to scan for Deepfakes and Fact-Check claims.")

# Unified Uploader
uploaded_file = st.file_uploader(
    "Drop your media here",
    type=["mp4", "wav", "mp3", "jpg", "png", "jpeg", "avi", "mkv", "webm", "ogg", "flac"]
)

with st.sidebar:
    st.header("⚙️ System Status")
    df_model, df_ok = load_deepfake_model()
    st.write(f"Visual AI: {'✅ Loaded' if df_ok else '❌ Error'}")
    audio_df_model = load_audio_deepfake_model()
    st.write(f"Audio AI: {'✅ Loaded' if audio_df_model else '❌ Error'}")
    whisper_model = load_whisper_model()
    st.write(f"Transcription: {'✅ Loaded' if whisper_model else '❌ Error'}")
    sentence_model = load_sentence_transformer()
    st.write("Fact-Checker: ✅ Loaded")
    st.divider()
    st.caption("Built with ❤️ by Shodh AI")

if uploaded_file:
    file_type = uploaded_file.type.split('/')[0]  # 'image', 'video', or 'audio'
    ext = uploaded_file.name.split('.')[-1].lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.divider()
    col_pre, col_res = st.columns([1, 1])

    with col_pre:
        st.subheader("📂 Media Preview")
        if file_type == 'image':
            st.image(uploaded_file, use_container_width=True)
        elif file_type == 'video':
            st.video(uploaded_file)
        elif file_type == 'audio':
            st.audio(uploaded_file)

    with col_res:
        st.subheader("🔬 AI Analysis Report")

        # --- Run Full Scan ---
        if st.button("🚀 Run Full Security Scan", type="primary"):
            vis_score, aud_score, transcript = None, None, None

            # 1. Visual Analysis (Image or Video)
            if file_type in ['image', 'video'] and df_ok:
                with st.spinner("🔍 Analyzing Visual Integrity..."):
                    if file_type == 'image':
                        img = cv2.imread(tmp_path)
                        if img is not None:
                            vis_score = analyze_visual(img, df_model, get_face_detector())
                    else:
                        cap = cv2.VideoCapture(tmp_path)
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        frame_scores = []
                        sample_count = min(10, total)
                        for i in range(0, total, max(1, total // sample_count)):
                            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                            ret, frame = cap.read()
                            if ret:
                                s = analyze_visual(frame, df_model, get_face_detector())
                                if s is not None:
                                    frame_scores.append(s)
                        cap.release()
                        vis_score = np.mean(frame_scores) if frame_scores else None

            # 2. Audio & Speech Analysis (Video or Audio)
            if file_type in ['video', 'audio']:
                with st.spinner("🎧 Processing Audio & Speech..."):
                    audio_path = tmp_path
                    if file_type == 'video':
                        try:
                            from moviepy import VideoFileClip
                            with VideoFileClip(tmp_path) as video:
                                if video.audio:
                                    audio_path = tmp_path.rsplit('.', 1)[0] + ".wav"
                                    video.audio.write_audiofile(audio_path, logger=None)
                                else:
                                    audio_path = None
                        except Exception:
                            audio_path = None

                    if audio_path and os.path.exists(audio_path):
                        try:
                            audio_arr, _ = librosa.load(audio_path, sr=16000)

                            # Audio authenticity check
                            if audio_df_model:
                                ads = audio_df_model(audio_arr)
                                for r in ads:
                                    if r['label'] == "Speech synthesizer":
                                        aud_score = r['score']

                            # Speech-to-text
                            if whisper_model:
                                transcript = whisper_model.transcribe(audio_arr)["text"]
                        except Exception:
                            pass

            # Save results to session state
            st.session_state['v_score'] = vis_score
            st.session_state['a_score'] = aud_score
            st.session_state['script'] = transcript
            # Clear old fact-check results
            st.session_state.pop('fact_score', None)
            st.session_state.pop('fact_articles', None)

        # --- Display Results (persisted via session state) ---
        if 'v_score' in st.session_state and st.session_state['v_score'] is not None:
            st.metric("Visual Trust Score", f"{st.session_state['v_score']:.2%}")
            if st.session_state['v_score'] < 0.5:
                st.error("🚩 Visual Manipulation Detected!")
            else:
                st.success("✅ Real-Source Video Match")

        if 'a_score' in st.session_state and st.session_state['a_score'] is not None:
            st.metric("Audio Synthetic Probability", f"{st.session_state['a_score']:.2%}")
            if st.session_state['a_score'] > 0.3:
                st.error("🚩 AI-Generated Voice Detected!")
            else:
                st.success("✅ Natural Human Audio")

        if 'script' in st.session_state and st.session_state['script']:
            st.subheader("📝 Speech-to-Text Verification")
            edited_script = st.text_area(
                "Verify and edit the detected speech before fact-checking:",
                value=st.session_state['script'],
                height=150,
                help="The AI transcription may have errors. Edit before running fact-check for best results."
            )

            if st.button("🔍 Fact-Check this Script"):
                with st.spinner("Searching Global News Sources..."):
                    score, articles = verify_claim(edited_script, sentence_model)
                    st.session_state['fact_score'] = score
                    st.session_state['fact_articles'] = articles
                    st.session_state['script'] = edited_script

        if 'fact_score' in st.session_state:
            st.metric("News Correlation", f"{st.session_state['fact_score']:.2%}")
            if st.session_state['fact_score'] > 0.7:
                st.success("✅ High correlation with verified news sources.")
            elif st.session_state['fact_score'] > 0.3:
                st.warning("⚠️ Partial match found. Verify manually.")
            else:
                st.error("🚩 No reliable news match found. Possible misinformation.")
            if st.session_state.get('fact_articles'):
                st.write("**Related Articles:**")
                for a in st.session_state['fact_articles']:
                    st.write(f"- {a}")

    # Cleanup temp file
    try:
        os.unlink(tmp_path)
    except Exception:
        pass

st.divider()
st.subheader("🔍 Quick Claim Fact-Checker")
manual_claim = st.text_input("Type a claim to verify manually:")
if manual_claim:
    with st.spinner("Checking..."):
        s, a = verify_claim(manual_claim, sentence_model)
        st.metric("Truth Score", f"{s:.2%}")
        if a:
            st.write("**Related Articles:**")
            for item in a:
                st.write(f"• {item}")
        else:
            st.info("No matching news articles found.")
