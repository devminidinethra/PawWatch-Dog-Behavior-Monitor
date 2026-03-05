# 🐾 PawWatch — Dog Behavior & Emotion Monitoring System

> **BSc Computing Final Year Project** · University of Greenwich

A real-time dog behavior and emotion monitoring application built with Streamlit, YOLOv8, and MobileNetV2. PawWatch detects your dog on camera, classifies their emotional state, tracks behavioral signals like pacing and tail movement, and sends instant WhatsApp alerts when distress is detected.

---

## 📸 Screenshots

| Live Camera | Demo Upload | Alert Status |
|---|---|---|
| Real-time detection with bounding box | Single image or video analysis | WhatsApp alerts log |

---

## ✨ Features

- **Real-time emotion detection** — classifies dog emotions into 4 categories: `angry`, `happy`, `relaxed`, `sad`
- **YOLOv8 dog detection** — automatically locates the dog in every frame before classifying
- **Behavioral signals** — tracks pacing score and tail movement alongside emotion
- **Live camera support** — connect a webcam or IP/RTSP camera stream
- **Image & video upload** — analyse still photos or full video files frame by frame
- **Behavior history** — session log with emotion distribution, filtering, and CSV export
- **WhatsApp alerts** — instant push notifications via Twilio when angry or sad is detected
- **Rolling window smoothing** — live camera uses a 10-frame majority vote to reduce flicker (disabled for single image/video so emotion always matches confidence)
- **Light theme UI** — clean, responsive web dashboard designed for readability

---

## 🧠 Model Architecture

| Component | Details |
|---|---|
| **Dog Detection** | YOLOv8n (Ultralytics) — COCO class 16 |
| **Emotion Classification** | MobileNetV2 fine-tuned on dog emotion dataset |
| **Input Size** | 96 × 96 px |
| **Preprocessing** | `keras.applications.mobilenet_v2.preprocess_input` |
| **Classes** | `angry`, `happy`, `relaxed`, `sad` |
| **Framework** | TensorFlow / Keras (`TF_USE_LEGACY_KERAS=1`) |

---

## 🗂️ Project Structure

```
pawwatch/
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit theme (forces light mode)
├── models/
│   └── final_model.h5         # Downloaded automatically on first run
└── README.md
```

> **Note:** `models/final_model.h5` is not committed to the repo. It is downloaded automatically at startup from the URL stored in Streamlit Secrets.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- A webcam (for live camera mode) — optional
- A Twilio account (for WhatsApp alerts) — optional

### 1. Clone the repository

```bash
git clone https://github.com/devminidinethra/PawWatch-Dog-Behavior-Monitor.git
cd pawwatch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up Streamlit Secrets

Create a file at `.streamlit/secrets.toml`:

```toml
MODEL_URL = "https://huggingface.co/devmini-gamage/PawWatch-Model/resolve/main/final_model.h5"
```

This is the direct download URL to the trained `.h5` model file. The app downloads it automatically on first launch and caches it locally.

> **Hosting the model:** Uploaded the `final_model.h5` to Hugging Face that provides a direct download link.

### 4. Run the app

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Cloud

1. Push your repo to GitHub (do **not** commit `models/` or `.streamlit/secrets.toml`)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Under **Advanced settings → Secrets**, add:
   ```toml
   MODEL_URL = "https://huggingface.co/devmini-gamage/PawWatch-Model/resolve/main/final_model.h5"
   ```
4. Deploy — the model downloads automatically on first boot

> **Live camera** does not work on Streamlit Cloud (no webcam access in the browser). Use the **Demo Upload** tab on cloud deployments.

---

## 📱 WhatsApp Alerts Setup

PawWatch sends WhatsApp messages (via the Twilio Sandbox) when your dog is detected as **angry** or **sad**.

### Step 1 — Join the Twilio WhatsApp Sandbox

1. Log in to [console.twilio.com](https://console.twilio.com)
2. Navigate to **Messaging → Try it out → Send a WhatsApp message**
3. You will see a sandbox number (`+14155238886`) and a unique keyword, e.g. `join silver-tiger`
4. On your phone, open WhatsApp and send that exact message to **+1 415 523 8886**
5. Wait for the confirmation reply from Twilio

### Step 2 — Fill in the sidebar

Once the app is running, expand the sidebar and enable **WhatsApp Alerts**:

| Field | Value |
|---|---|
| Your WhatsApp number | Your phone number with country code, e.g. `+94761234567` |
| Account SID | From your Twilio Console dashboard |
| Auth Token | From your Twilio Console dashboard (click to reveal) |
| Twilio sandbox number | Always `+14155238886` for the sandbox |

### Alert behaviour

- Triggers on: **ANGRY** or **SAD** emotions only
- Cooldown: **60 seconds** between alerts (prevents spam)
- All alerts are logged in the **Alert Status** tab

---

## 📦 Requirements

```
streamlit
opencv-python-headless
numpy
pandas
Pillow
ultralytics
tensorflow
keras
twilio
```

Full pinned versions in `requirements.txt`.

---

## ⚙️ Configuration

All camera and alert settings are accessible in the sidebar at runtime — no code changes needed.

| Setting | Default | Description |
|---|---|---|
| Camera index | `0` | Webcam index (0 = default camera) |
| RTSP URL | *(blank)* | IP camera stream, e.g. `rtsp://192.168.1.x/stream` |
| Detection confidence | `0.35` | YOLOv8 confidence threshold |
| Alert cooldown | `60s` | Minimum time between WhatsApp alerts |

---

## 🔧 Known Issues & Notes

- **`TF_USE_LEGACY_KERAS=1`** is set at startup to ensure compatibility with Keras 3 and the saved `.h5` model format
- **Emotion/confidence mismatch bug (fixed):** The rolling window smoother is disabled for single image and video uploads (`smooth=False`), so the displayed emotion label always matches the confidence score shown
- **Live camera on cloud:** Browser-based webcam access is not supported on Streamlit Cloud. Use the Demo Upload tab instead
- **Twilio trial accounts** can only send to verified numbers via SMS. WhatsApp sandbox bypasses this restriction and works internationally

---

## 📊 How It Works

```
Camera / Upload
      │
      ▼
 YOLOv8n Detection
 (finds dog bounding box)
      │
      ▼
 Crop + Resize to 96×96
      │
      ▼
 MobileNetV2 Classification
 (angry / happy / relaxed / sad)
      │
      ├── Live camera → Rolling 10-frame majority vote (smoothing)
      └── Image/Video → Raw prediction (no smoothing)
      │
      ▼
 Behavioral Metrics
 (pacing score, tail movement delta)
      │
      ▼
 Session History + Alert Check
      │
      └── If angry/sad + cooldown elapsed → WhatsApp alert via Twilio
```

---

## 🎓 Academic Context

This project was developed as part of the **BSc Computing** final year dissertation at the **University of Greenwich**.

**Research focus:** Applying computer vision and deep learning to animal welfare monitoring — specifically, whether a lightweight CNN (MobileNetV2) can reliably infer canine emotional states from facial and postural cues in real-time conditions.

---

## 📄 License

This project is submitted for academic assessment. All rights reserved © University of Greenwich, 2026.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) for real-time object detection
- [TensorFlow / Keras](https://keras.io) for model training and inference
- [Streamlit](https://streamlit.io) for the web application framework
- [Twilio](https://twilio.com) for WhatsApp messaging API
