# 🛡️ Shodh AI: Misinformation & Deepfake Detector

Shodh AI is a unified security dashboard that detects digital manipulation in Video, Audio, and Images, and verifies spoken content against global news sources.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

## 🚀 Features

- **Visual Deepfake Detection** — EfficientNetV2-S model (fine-tuned on FaceForensics++) identifies face manipulation in images and videos.
- **Audio Authenticity Scan** — Detects synthetic/AI-generated voices using Audio Spectrogram Transformers.
- **Speech-to-Text Transcription** — Powered by OpenAI Whisper for high-accuracy script extraction.
- **Editable Transcript Verification** — Review and correct the AI transcript before fact-checking.
- **News Fact-Checker** — Cross-references transcripts with Google News RSS to calculate a truth/correlation score.
- **Unified Interface** — One upload. One click. Full analysis.

## 🛠️ Local Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/Misinformation-detector.git
cd Misinformation-detector
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Dashboard
```bash
streamlit run shodh_dashboard.py
```

## ☁️ Streamlit Cloud Deployment

This project is ready for one-click deployment on [Streamlit Cloud](https://share.streamlit.io/):

1. Push your repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app** → Select your repo → Set **Main file** to `shodh_dashboard.py`.
4. Click **Deploy**.

> The `packages.txt` file automatically installs system dependencies (ffmpeg, libsndfile) on the cloud server.

## 📦 Project Structure

| File | Description |
|------|-------------|
| `shodh_dashboard.py` | Main Streamlit application (cloud-ready) |
| `scanner_app.py` | Desktop auto-scanner (PyQt6, local use only) |
| `weight.pth` | Trained deepfake detection weights (~80 MB) |
| `blaze_face_short_range.tflite` | MediaPipe face detection model |
| `requirements.txt` | Python dependencies |
| `packages.txt` | System-level apt dependencies for Streamlit Cloud |
| `.streamlit/config.toml` | Streamlit theme and server configuration |

## 🛡️ Ethics & Disclaimer

This tool is intended for **research and educational purposes**. Always verify AI results with official sources.
