"""
PawWatch — Dog Behavior & Emotion Monitoring System
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import io, time, tempfile, urllib.request, hashlib
from collections import deque, Counter
from datetime import datetime

import cv2, numpy as np, pandas as pd
import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="PawWatch — Pet Behavior Monitor",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════
IMG_SIZE  = 96
CLASSES   = ["angry", "happy", "relaxed", "sad"]
EMOJI     = {"angry":"😠","happy":"😊","relaxed":"😌","sad":"😢","no_dog":"🐕"}
C_HEX     = {"angry":"#dc2626","happy":"#16a34a","relaxed":"#2563eb","sad":"#7c3aed"}
C_BG      = {"angry":"#fee2e2","happy":"#dcfce7","relaxed":"#dbeafe","sad":"#ede9fe"}
C_BD      = {"angry":"#fca5a5","happy":"#86efac","relaxed":"#93c5fd","sad":"#c4b5fd"}
C_BGT     = {"angry":(60,75,220),"happy":(60,180,80),"relaxed":(200,130,50),"sad":(160,80,200)}
DOG_CLASS        = 16
ALERT_EMOTIONS   = {"angry","sad"}
ALERT_COOLDOWN   = 60
HISTORY_MAX      = 300
MODEL_LOCAL_PATH = "models/final_model.h5"

LOGIN_USERNAME = "admin"
LOGIN_PASSWORD = "pawwatch2024"

# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════
_defaults = dict(
    model=None, yolo=None,
    history=[], alerts=[], last_alert_ts=0,
    camera_running=False,
    pos_history=deque(maxlen=15),
    beh_window=deque(maxlen=10),
    prev_frame=None,
    phone_number="", twilio_sid="", twilio_token="", twilio_from="",
    alerts_enabled=False,
    last_upload_hash=None,
    image_result=None,
    video_results=None,
    authenticated=False,
)
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
#  CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Sora:wght@600;700&display=swap');

:root {
  --bg:      #f0f4f8;
  --card:    #ffffff;
  --border:  #dde3ec;
  --text:    #1e293b;
  --sub:     #64748b;
  --muted:   #94a3b8;
  --green:   #16a34a;
  --green-d: #15803d;
  --green-l: #dcfce7;
  --green-b: #86efac;
  --red:     #dc2626;
  --red-l:   #fee2e2;
  --red-b:   #fca5a5;
  --blue:    #2563eb;
  --blue-l:  #dbeafe;
  --blue-b:  #93c5fd;
  --amber-l: #fef3c7;
  --amber-b: #fcd34d;
  --purp:    #7c3aed;
  --purp-l:  #ede9fe;
  --purp-b:  #c4b5fd;
}

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"] {
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  background: var(--bg) !important;
}
[data-testid="stHeader"]             { background: transparent !important; }
[data-testid="stMainBlockContainer"] { padding-top: 0.7rem !important; }

.stButton > button {
  background: var(--green) !important; color: #fff !important;
  border: none !important; border-radius: 8px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important; font-size: 0.85rem !important;
  padding: .52rem 1.4rem !important; transition: all .18s !important;
  box-shadow: 0 1px 4px rgba(22,163,74,.22) !important;
}
.stButton > button:hover {
  background: var(--green-d) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(22,163,74,.3) !important;
}
[data-testid="stDownloadButton"] button {
  background: #fff !important; color: var(--green) !important;
  border: 1.5px solid var(--green) !important;
}
[data-testid="stDownloadButton"] button:hover { background: var(--green-l) !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: #fff !important; border-radius: 10px !important;
  padding: 4px !important; gap: 3px !important;
  border: 1px solid var(--border) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important; border-radius: 7px !important;
  color: var(--sub) !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important; font-size: .82rem !important;
  padding: .42rem 1.1rem !important; transition: all .15s !important;
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  background: var(--bg) !important; color: var(--text) !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
  background: var(--green) !important; color: #fff !important;
  box-shadow: 0 2px 8px rgba(22,163,74,.25) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-panel"]  { padding-top: 18px !important; }

[data-testid="stSidebar"] {
  background: #fff !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stFileUploader"] {
  background: #fff !important;
  border: 2px dashed var(--border) !important;
  border-radius: 10px !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--green) !important; }
[data-testid="stProgressBar"] > div > div { background: var(--green) !important; }

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

.pw-nav {
  display: flex; align-items: center; gap: 16px; padding: 14px 24px;
  background: linear-gradient(135deg, #15803d 0%, #16a34a 60%, #22c55e 100%);
  border-radius: 14px; margin-bottom: 18px;
  box-shadow: 0 4px 20px rgba(22,163,74,.25);
}
.pw-nav-icon  { font-size: 2.6rem; line-height: 1; }
.pw-nav-title { font-family: 'Sora', sans-serif; font-size: 1.7rem; font-weight: 700; color: #fff; line-height: 1; }
.pw-nav-sub   { font-size: .78rem; color: rgba(255,255,255,.8); margin-top: 4px; }
.pw-nav-badge {
  margin-left: auto; display: flex; align-items: center; gap: 8px;
  background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.4);
  color: #fff; font-size: .76rem; font-weight: 700;
  padding: 5px 16px; border-radius: 20px; white-space: nowrap;
}
.pw-pulse { width:7px; height:7px; background:#bbf7d0; border-radius:50%; animation:pw-p 2s infinite; }
@keyframes pw-p { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

.kpi-card {
  background: #fff; border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 18px; text-align: center;
  box-shadow: 0 1px 6px rgba(0,0,0,.04); transition: transform .15s, box-shadow .15s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 5px 18px rgba(0,0,0,.08); }
.kpi-icon { font-size: 1.5rem; margin-bottom: 6px; }
.kpi-num  { font-family: 'Sora', sans-serif; font-size: 2rem; font-weight: 700; color: #1e293b; line-height: 1; }
.kpi-unit { font-size: .9rem; color: #64748b; }
.kpi-lbl  { font-size: .68rem; color: #64748b; text-transform: uppercase; letter-spacing: .09em; font-weight: 700; margin-top: 5px; }

.sec-title {
  font-family: 'Sora', sans-serif; font-size: .95rem; font-weight: 700; color: #1e293b;
  display: flex; align-items: center; gap: 8px;
  padding-bottom: 9px; margin-bottom: 13px; border-bottom: 2px solid var(--border);
}

.callout { display:flex; align-items:flex-start; gap:12px; padding:12px 15px; border-radius:10px; margin-bottom:16px; font-size:.83rem; line-height:1.55; }
.callout.info    { background:var(--blue-l);  border:1px solid var(--blue-b);  color:#1e3a5f; }
.callout.success { background:var(--green-l); border:1px solid var(--green-b); color:#14532d; }
.callout.warn    { background:var(--amber-l); border:1px solid var(--amber-b); color:#78350f; }
.callout-ico { font-size:1.1rem; flex-shrink:0; margin-top:2px; }

.emo-card { border-radius:14px; padding:22px 18px; text-align:center; border:1.5px solid; margin-bottom:12px; }
.emo-card .ec-emoji { font-size:3.2rem; line-height:1; margin-bottom:9px; }
.emo-card .ec-name  { font-family:'Sora',sans-serif; font-size:1.5rem; font-weight:700; margin-bottom:6px; }
.emo-card .ec-conf  { font-size:.83rem; color:#64748b; }

.stat-row { display:flex; justify-content:space-between; align-items:center; padding:10px 14px; background:var(--bg); border-radius:8px; margin-bottom:6px; border:1px solid var(--border); font-size:.84rem; }
.stat-lbl { color:#64748b; font-weight:500; }
.stat-val { color:#1e293b; font-weight:700; }

.prob-wrap { margin-bottom:11px; }
.prob-head { display:flex; justify-content:space-between; font-size:.8rem; margin-bottom:5px; font-weight:700; color:#1e293b; }
.prob-track { background:var(--bg); border-radius:5px; height:9px; border:1px solid var(--border); overflow:hidden; }
.prob-fill  { height:9px; border-radius:5px; transition:width .4s ease; }

.dist-wrap { margin-bottom:13px; }
.dist-head { display:flex; justify-content:space-between; font-size:.82rem; margin-bottom:5px; font-weight:700; color:#1e293b; }
.dist-sub  { color:#64748b; font-weight:500; font-size:.78rem; }
.dist-track { background:var(--bg); border-radius:6px; height:12px; border:1px solid var(--border); overflow:hidden; }
.dist-fill  { height:12px; border-radius:6px; }

.hist-row { display:flex; align-items:center; gap:10px; padding:9px 13px; background:#fff; border:1px solid var(--border); border-radius:9px; margin-bottom:5px; font-size:.81rem; transition:box-shadow .15s; }
.hist-row:hover { box-shadow:0 2px 10px rgba(0,0,0,.07); }
.hist-time { color:#94a3b8; font-size:.7rem; min-width:50px; }
.hist-conf { color:#64748b; font-size:.73rem; margin-left:3px; }
.hist-meta { color:#94a3b8; font-size:.7rem; margin-left:auto; }

.emo-badge { display:inline-flex; align-items:center; gap:4px; padding:3px 10px; border-radius:20px; font-size:.71rem; font-weight:800; text-transform:uppercase; letter-spacing:.06em; border:1.5px solid; min-width:86px; justify-content:center; }

.alert-card { display:flex; gap:14px; padding:14px 16px; border-radius:12px; border:1.5px solid; margin-bottom:10px; }
.alert-ico  { font-size:1.5rem; flex-shrink:0; padding-top:2px; }
.alert-body { flex:1; }
.alert-ttl  { font-weight:800; font-size:.88rem; margin-bottom:4px; }
.alert-msg  { font-size:.79rem; color:#64748b; line-height:1.5; }
.sms-ok  { font-size:.72rem; color:#16a34a; font-weight:700; margin-top:5px; }
.sms-err { font-size:.72rem; color:#dc2626; font-weight:700; margin-top:5px; }
.sms-off { font-size:.72rem; color:#94a3b8; margin-top:5px; }

.cfg-box { background:var(--bg); border:1px solid var(--border); border-radius:12px; overflow:hidden; }
.cfg-row { display:flex; justify-content:space-between; align-items:center; padding:10px 16px; border-bottom:1px solid var(--border); font-size:.83rem; }
.cfg-row:last-child { border-bottom:none; }
.cfg-k { color:#64748b; font-weight:500; }
.cfg-v { color:#1e293b; font-weight:700; }

.how-box { background:#fff; border:1px solid var(--border); border-radius:12px; padding:14px 16px; }
.how-step { display:flex; gap:12px; align-items:flex-start; padding:8px 0; border-bottom:1px solid var(--border); font-size:.83rem; line-height:1.55; color:#1e293b; }
.how-step:last-child { border-bottom:none; }
.how-num { background:var(--green-l); color:#15803d; border:1px solid var(--green-b); font-weight:800; font-size:.72rem; width:22px; height:22px; border-radius:50%; display:flex; align-items:center; justify-content:center; flex-shrink:0; margin-top:2px; }

.empty-state { background:#fff; border:1.5px dashed #cbd5e1; border-radius:14px; padding:38px 26px; text-align:center; }
.e-ico { font-size:2.8rem; margin-bottom:10px; }
.e-ttl { font-family:'Sora',sans-serif; font-size:1rem; font-weight:700; color:#1e293b; margin-bottom:7px; }
.e-sub { font-size:.82rem; color:#64748b; line-height:1.6; }

.live-badge { display:inline-flex; align-items:center; gap:7px; background:#fee2e2; color:#dc2626; border:1.5px solid #fca5a5; font-size:.74rem; font-weight:800; padding:5px 12px; border-radius:20px; letter-spacing:.07em; }
.live-dot   { width:7px; height:7px; background:#dc2626; border-radius:50%; animation:blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

.result-card { background:#fff; border:1px solid var(--border); border-radius:14px; padding:20px; box-shadow:0 2px 10px rgba(0,0,0,.05); }

.dom-box { border-radius:12px; border:1.5px solid; padding:14px 18px; margin-top:12px; display:flex; align-items:center; gap:14px; }
.dom-emo  { font-size:2rem; }
.dom-name { font-family:'Sora',sans-serif; font-weight:700; font-size:1rem; }
.dom-sub  { font-size:.76rem; color:#64748b; margin-top:2px; }

.sb-brand { padding:10px 0 6px; display:flex; align-items:center; gap:12px; }
.sb-logo  { font-size:2rem; }
.sb-name  { font-family:'Sora',sans-serif; font-size:1.1rem; font-weight:700; color:#1e293b; }
.sb-sub   { font-size:.68rem; color:#64748b; margin-top:1px; }
.sb-section { font-size:.68rem; font-weight:800; text-transform:uppercase; letter-spacing:.1em; color:#64748b; margin:14px 0 6px; padding-bottom:4px; border-bottom:1px solid var(--border); }

.pw-footer { margin-top:50px; padding:16px 24px; border-top:1px solid var(--border); display:flex; justify-content:space-between; align-items:center; font-size:.7rem; color:#94a3b8; }

.login-wrap {
  max-width: 420px; margin: 60px auto 0;
  background: #fff; border: 1px solid var(--border);
  border-radius: 18px; padding: 40px 36px;
  box-shadow: 0 8px 40px rgba(0,0,0,.09);
}
.login-logo  { font-size: 3rem; text-align: center; margin-bottom: 6px; }
.login-title {
  font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 700;
  color: #1e293b; text-align: center; margin-bottom: 4px;
}
.login-sub { font-size: .8rem; color: #64748b; text-align: center; margin-bottom: 28px; }

.graphs-section {
  background: #fff; border: 1px solid var(--border);
  border-radius: 14px; padding: 24px 22px; margin-top: 24px;
  box-shadow: 0 2px 10px rgba(0,0,0,.04);
}
.graphs-title {
  font-family: 'Sora', sans-serif; font-size: 1rem; font-weight: 700;
  color: #1e293b; display: flex; align-items: center; gap: 9px;
  padding-bottom: 12px; margin-bottom: 2px;
  border-bottom: 2px solid var(--border);
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  MODEL DOWNLOAD + LOAD
# ══════════════════════════════════════════════════════════════════════════════
def download_model_if_needed():
    if os.path.exists(MODEL_LOCAL_PATH):
        return MODEL_LOCAL_PATH
    url = st.secrets.get("MODEL_URL", "")
    if not url:
        st.error("MODEL_URL not set in Streamlit Secrets.")
        st.stop()
    os.makedirs("models", exist_ok=True)
    bar = st.progress(0, text="Downloading model weights…")
    def hook(c, bs, tot):
        if tot > 0:
            bar.progress(min(int(c*bs*100/tot),100)/100,
                         text=f"Downloading… {min(int(c*bs*100/tot),100)}%")
    try:
        urllib.request.urlretrieve(url, MODEL_LOCAL_PATH, hook)
        bar.empty()
    except Exception as e:
        bar.empty(); st.error(f"Download failed: {e}"); st.stop()
    return MODEL_LOCAL_PATH

@st.cache_resource(show_spinner="Loading emotion model…")
def load_cnn(path):
    import keras
    return keras.models.load_model(path, compile=False)

@st.cache_resource(show_spinner="Loading YOLOv8…")
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

model_path = download_model_if_needed()
if st.session_state.model is None: st.session_state.model = load_cnn(model_path)
if st.session_state.yolo  is None: st.session_state.yolo  = load_yolo()

# ══════════════════════════════════════════════════════════════════════════════
#  LOGIN GATE
# ══════════════════════════════════════════════════════════════════════════════
if not st.session_state.authenticated:
    _, mid, _ = st.columns([1, 1.6, 1])
    with mid:
        st.markdown("""
        <div class="login-logo">🐾</div>
        <div class="login-title">PawWatch</div>
        <div class="login-sub">Dog Behavior &amp; Emotion Monitoring<br>
        Please sign in to continue</div>
        """, unsafe_allow_html=True)

        username_input = st.text_input("Username", placeholder="Enter username",
                                       key="login_user")
        password_input = st.text_input("Password", placeholder="Enter password",
                                       type="password", key="login_pass")
        login_btn = st.button("🔐  Sign In", use_container_width=True)

        if login_btn:
            if username_input == LOGIN_USERNAME and password_input == LOGIN_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.markdown("""
                <div class="callout warn" style="margin-top:12px">
                  <span class="callout-ico">⚠️</span>
                  <span><strong>Invalid credentials.</strong> Please check your
                  username and password and try again.</span>
                </div>""", unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:center;margin-top:20px;font-size:.72rem;color:#94a3b8">
          University of Greenwich · BSc Computing · 001512468
        </div>""", unsafe_allow_html=True)

    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def preprocess(crop_bgr):
    from keras.applications.mobilenet_v2 import preprocess_input
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    arr = np.array(Image.fromarray(rgb).resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
    return np.expand_dims(preprocess_input(arr), 0)

def detect_dog(frame, yolo, conf=0.35):
    best = None
    for r in yolo(frame, classes=[DOG_CLASS], conf=conf, verbose=False):
        if r.boxes is None: continue
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            c = float(box.conf[0])
            if best is None or c > best["conf"]:
                best = {"bbox":(x1,y1,x2,y2), "conf":c}
    return best

def classify(crop_bgr, model):
    p   = model.predict(preprocess(crop_bgr), verbose=0)[0]
    idx = int(np.argmax(p))
    return CLASSES[idx], float(p[idx]), {CLASSES[i]: float(p[i]) for i in range(4)}

def calc_pacing(bbox, ph):
    ph.append(((bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2))
    if len(ph) < 5: return 0.0
    return float(np.var([p[0] for p in ph]) + np.var([p[1] for p in ph]))

def calc_tail(frame, bbox, prev):
    if prev is None: return 0.0
    x1,y1,x2,y2 = bbox
    ty = min(y1+int(.75*(y2-y1)), frame.shape[0]-1)
    a = frame[ty:y2,x1:x2].astype(float)
    b = prev[ty:y2,x1:x2].astype(float)
    return 0.0 if a.shape!=b.shape or a.size==0 else float(np.mean(np.abs(a-b)))

def process_frame(frame, model, yolo, smooth=True):
    h, w = frame.shape[:2]
    det  = detect_dog(frame, yolo)
    res  = {"dog_found":False,"emotion":"no_dog","confidence":0.0,
            "probs":{},"pacing":0.0,"tail":0.0,"bbox":None}
    if det:
        x1,y1,x2,y2 = det["bbox"]; pad = 20
        crop = frame[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
        if crop.size > 0:
            raw_emo, conf, probs = classify(crop, model)
            if smooth:
                st.session_state.beh_window.append(raw_emo)
                emotion = Counter(st.session_state.beh_window).most_common(1)[0][0]
            else:
                emotion = raw_emo
            pacing = calc_pacing(det["bbox"], st.session_state.pos_history)
            tail   = calc_tail(frame, det["bbox"], st.session_state.prev_frame)
            res.update({"dog_found":True,"emotion":emotion,"confidence":conf,
                        "probs":probs,"pacing":round(pacing,1),"tail":round(tail,1),
                        "bbox":det["bbox"]})
            col = C_BGT.get(emotion,(120,120,120))
            cv2.rectangle(frame,(x1,y1),(x2,y2),col,2)
            lbl = f"{emotion.upper()}  {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, .65, 2)
            cv2.rectangle(frame,(x1,y1-th-12),(x1+tw+10,y1),col,-1)
            cv2.putText(frame,lbl,(x1+5,y1-4),cv2.FONT_HERSHEY_SIMPLEX,.65,(255,255,255),2)
    st.session_state.prev_frame = frame.copy()
    return frame, res

# ══════════════════════════════════════════════════════════════════════════════
#  HISTORY & ALERTS
# ══════════════════════════════════════════════════════════════════════════════
def maybe_alert(emotion):
    now = time.time()
    if emotion not in ALERT_EMOTIONS: return
    if now - st.session_state.last_alert_ts < ALERT_COOLDOWN: return
    emap = {"angry":"😠","sad":"😢"}
    msg  = (f"🐾 *PawWatch Alert* [{datetime.now().strftime('%H:%M:%S')}]\n\n"
            f"{emap.get(emotion,'')} Your dog appears *{emotion.upper()}*!\n"
            f"Please check in on them.")
    entry = {"ts":datetime.now().strftime("%H:%M:%S"),
             "emotion":emotion,"message":msg,"sms_sent":False}
    st.session_state.alerts.append(entry)
    st.session_state.last_alert_ts = now
    if (st.session_state.alerts_enabled
            and st.session_state.twilio_sid
            and st.session_state.phone_number):
        try:
            from twilio.rest import Client
            Client(st.session_state.twilio_sid,
                   st.session_state.twilio_token).messages.create(
                body=msg,
                from_=f"whatsapp:{st.session_state.twilio_from}",
                to=f"whatsapp:{st.session_state.phone_number}")
            entry["sms_sent"] = True
        except Exception as e:
            entry["sms_error"] = str(e)

def record(res):
    if not res["dog_found"]: return
    st.session_state.history.append({
        "ts":datetime.now().strftime("%H:%M:%S"),
        "emotion":res["emotion"],
        "confidence":round(res["confidence"]*100,1),
        "pacing":res["pacing"],"tail":res["tail"],
    })
    if len(st.session_state.history) > HISTORY_MAX:
        st.session_state.history.pop(0)
    maybe_alert(res["emotion"])

# ══════════════════════════════════════════════════════════════════════════════
#  HTML BUILDERS
# ══════════════════════════════════════════════════════════════════════════════
def _badge(emo):
    c=C_HEX.get(emo,"#64748b"); bg=C_BG.get(emo,"#f1f5f9"); bd=C_BD.get(emo,"#cbd5e1")
    return (f'<span class="emo-badge" style="color:{c};background:{bg};border-color:{bd}">'
            f'{EMOJI.get(emo,"")} {emo.upper()}</span>')

def _prob_bar(cls, p):
    c = C_HEX.get(cls,"#64748b")
    return (f'<div class="prob-wrap">'
            f'<div class="prob-head">'
            f'<span style="color:#1e293b">{EMOJI[cls]} {cls.capitalize()}</span>'
            f'<span style="color:{c}">{p:.0%}</span></div>'
            f'<div class="prob-track">'
            f'<div class="prob-fill" style="width:{int(p*100)}%;background:{c}"></div>'
            f'</div></div>')

def _dist_bar(cls, pct, cnt):
    c = C_HEX.get(cls,"#64748b")
    return (f'<div class="dist-wrap">'
            f'<div class="dist-head">'
            f'<span style="color:{c}">{EMOJI[cls]} {cls.upper()}</span>'
            f'<span class="dist-sub">{pct:.1%} · {cnt} frames</span></div>'
            f'<div class="dist-track">'
            f'<div class="dist-fill" style="width:{int(pct*100)}%;background:{c}"></div>'
            f'</div></div>')

def _emo_result(emo, conf, pacing=None, tail=None):
    c=C_HEX.get(emo,"#64748b"); bg=C_BG.get(emo,"#f1f5f9"); bd=C_BD.get(emo,"#cbd5e1")
    stats=""
    if pacing is not None:
        stats += (f'<div class="stat-row"><span class="stat-lbl">Pacing Score</span>'
                  f'<span class="stat-val">{pacing:.0f}</span></div>')
    if tail is not None:
        stats += (f'<div class="stat-row"><span class="stat-lbl">Tail Movement</span>'
                  f'<span class="stat-val">{tail:.1f}</span></div>')
    return (f'<div class="result-card">'
            f'<div class="emo-card" style="background:{bg};border-color:{bd}">'
            f'<div class="ec-emoji">{EMOJI.get(emo,"")}</div>'
            f'<div class="ec-name" style="color:{c}">{emo.upper()}</div>'
            f'<div class="ec-conf">Confidence: '
            f'<strong style="color:{c};font-size:1rem">{conf:.1%}</strong></div>'
            f'</div>{stats}</div>')

# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSIS GRAPHS
# ══════════════════════════════════════════════════════════════════════════════
def render_analysis_graphs(history_data):
    import plotly.graph_objects as go

    if not history_data:
        st.markdown(
            '<div class="empty-state" style="margin:0"><div class="e-ico">📊</div>'
            '<div class="e-ttl">No data for graphs yet</div>'
            '<div class="e-sub">Analyse an image or video first to populate these charts.</div></div>',
            unsafe_allow_html=True)
        return

    df = pd.DataFrame(history_data).reset_index()
    df.rename(columns={"index": "frame_idx"}, inplace=True)

    layout_base = dict(
        font=dict(family="Plus Jakarta Sans, sans-serif", size=12, color="#1e293b"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=14, r=14, t=38, b=14),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)", bordercolor="#dde3ec", borderwidth=1,
        ),
    )
    axis_style = dict(gridcolor="#f0f4f8", linecolor="#dde3ec",
                      tickfont=dict(size=11, color="#64748b"))

    st.markdown('<div class="sec-title" style="margin-top:16px">📊 Emotion Distribution</div>',
                unsafe_allow_html=True)

    tally  = Counter(df["emotion"])
    labels = [c for c in CLASSES if c in tally]
    counts = [tally[c] for c in labels]
    colors = [C_HEX[c] for c in labels]
    bgs    = [C_BG[c]  for c in labels]

    fig1 = go.Figure(go.Bar(
        x=labels, y=counts,
        marker=dict(color=colors, line=dict(color=bgs, width=2)),
        text=counts, textposition="outside",
        textfont=dict(size=14, color="#1e293b",
                      family="Plus Jakarta Sans, sans-serif"),
        hovertemplate="<b>%{x}</b><br>Detections: %{y}<extra></extra>",
    ))
    fig1.update_layout(
        **layout_base,
        xaxis=dict(title=None, **axis_style),
        yaxis=dict(title="Detections", **axis_style),
        showlegend=False, bargap=0.38, height=300,
    )
    st.plotly_chart(fig1, use_container_width=True)

    st.markdown('<div class="sec-title">📈 Model Confidence Over Time</div>',
                unsafe_allow_html=True)

    fig2 = go.Figure()
    for emo in CLASSES:
        sub = df[df["emotion"] == emo]
        if sub.empty: continue
        fig2.add_trace(go.Scatter(
            x=sub["frame_idx"], y=sub["confidence"],
            mode="lines+markers",
            name=emo.capitalize(),
            line=dict(color=C_HEX[emo], width=2),
            marker=dict(size=7, color=C_HEX[emo],
                        line=dict(width=1.5, color="#fff")),
            hovertemplate=(
                f"<b>{emo.capitalize()}</b><br>"
                "Frame: %{x}<br>Confidence: %{y:.1f}%<extra></extra>"
            ),
        ))

    df["conf_ma"] = df["confidence"].rolling(window=5, min_periods=1).mean()
    fig2.add_trace(go.Scatter(
        x=df["frame_idx"], y=df["conf_ma"],
        mode="lines", name="Rolling avg (5)",
        line=dict(color="#94a3b8", width=2, dash="dot"),
        hovertemplate="Rolling avg: %{y:.1f}%<extra></extra>",
    ))
    fig2.update_layout(
        **layout_base,
        xaxis=dict(title="Frame #", **axis_style),
        yaxis=dict(title="Confidence (%)", range=[0, 105], **axis_style),
        height=320,
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="sec-title">🕐 Emotion Timeline</div>',
                unsafe_allow_html=True)

    emo_order = {c: i for i, c in enumerate(CLASSES)}
    df["emo_num"] = df["emotion"].map(emo_order)

    fig3 = go.Figure()
    for emo in CLASSES:
        sub = df[df["emotion"] == emo]
        if sub.empty: continue
        fig3.add_trace(go.Scatter(
            x=sub["frame_idx"], y=sub["emo_num"],
            mode="markers",
            name=emo.capitalize(),
            marker=dict(
                size=13, color=C_HEX[emo], symbol="circle",
                line=dict(width=2, color="#fff"),
            ),
            customdata=sub["confidence"],
            hovertemplate=(
                f"<b>{emo.capitalize()}</b><br>"
                "Frame: %{x}<br>"
                "Confidence: %{customdata:.1f}%<extra></extra>"
            ),
        ))
    fig3.update_layout(
        **layout_base,
        xaxis=dict(title="Frame #", **axis_style),
        yaxis=dict(
            title=None,
            tickvals=list(range(len(CLASSES))),
            ticktext=[c.capitalize() for c in CLASSES],
            **axis_style,
        ),
        height=300,
    )
    st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class="sb-brand">
        <span class="sb-logo">🐾</span>
        <div><div class="sb-name">PawWatch</div>
        <div class="sb-sub">Pet Behavior Monitor</div></div>
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="sb-section">📷 Camera</div>', unsafe_allow_html=True)
    cam_index   = st.number_input("Camera index", 0, 10, 0)
    rtsp_url    = st.text_input("RTSP stream URL", placeholder="rtsp://…",
                                 help="Leave blank to use webcam")
    conf_thresh = st.slider("Detection confidence", 0.2, 0.9, 0.35, 0.05)

    st.divider()
    st.markdown('<div class="sb-section">💬 WhatsApp Alerts</div>', unsafe_allow_html=True)
    st.session_state.alerts_enabled = st.toggle(
        "Enable WhatsApp alerts via Twilio", st.session_state.alerts_enabled)
    if st.session_state.alerts_enabled:
        st.markdown("""
        <div style="background:#fffbeb;border:1px solid #fcd34d;border-radius:10px;
                    padding:11px 13px;margin-bottom:10px;font-size:.75rem;
                    color:#78350f;line-height:1.6">
          <div style="font-weight:800;margin-bottom:5px;font-size:.78rem">
            📋 Twilio Setup Guide
          </div>
          <div style="margin-bottom:4px">
            <strong>1.</strong> Go to
            <a href="https://www.twilio.com/try-twilio" target="_blank"
               style="color:#92400e;font-weight:700">twilio.com/try-twilio</a>
            and create a free account.
          </div>
          <div style="margin-bottom:4px">
            <strong>2.</strong> In the Console, copy your
            <em>Account SID</em> and <em>Auth Token</em>.
          </div>
          <div style="margin-bottom:4px">
            <strong>3.</strong> Go to <em>Messaging → Try it out →
            Send a WhatsApp message</em> and join the sandbox by sending
            the join code to <strong>+14155238886</strong>.
          </div>
          <div>
            <strong>4.</strong> Enter your details below and your WhatsApp
            number (with country code, e.g. +94761234567).
          </div>
        </div>
        """, unsafe_allow_html=True)
        st.session_state.phone_number = st.text_input(
            "Your WhatsApp number", st.session_state.phone_number,
            placeholder="+94761234567", help="Include country code")
        st.session_state.twilio_sid   = st.text_input(
            "Account SID", st.session_state.twilio_sid, type="password")
        st.session_state.twilio_token = st.text_input(
            "Auth Token", st.session_state.twilio_token, type="password")
        st.session_state.twilio_from  = st.text_input(
            "Twilio sandbox number", st.session_state.twilio_from,
            placeholder="+14155238886",
            help="The Twilio WhatsApp sandbox number is always +14155238886")

    st.divider()
    if st.button("🗑️  Clear Session Data", use_container_width=True):
        st.session_state.history          = []
        st.session_state.alerts           = []
        st.session_state.image_result     = None
        st.session_state.video_results    = None
        st.session_state.last_upload_hash = None
        st.session_state.camera_running   = False
        st.session_state.last_alert_ts    = 0
        st.session_state.prev_frame       = None
        st.session_state.beh_window       = deque(maxlen=10)
        st.session_state.pos_history      = deque(maxlen=15)
        st.rerun()

    if st.button("🚪  Sign Out", use_container_width=True):
        st.session_state.authenticated    = False
        st.session_state.camera_running   = False
        st.session_state.history          = []
        st.session_state.alerts           = []
        st.session_state.image_result     = None
        st.session_state.video_results    = None
        st.session_state.last_upload_hash = None
        st.session_state.last_alert_ts    = 0
        st.session_state.prev_frame       = None
        st.session_state.beh_window       = deque(maxlen=10)
        st.session_state.pos_history      = deque(maxlen=15)
        st.rerun()

    st.markdown("""
    <div style="margin-top:20px;padding-top:12px;border-top:1px solid #dde3ec;
                font-size:.67rem;color:#94a3b8;text-align:center;line-height:1.65">
        University of Greenwich<br>BSc Computing · 001512468<br>MobileNetV2 + YOLOv8n
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  NAVBAR + KPIs
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="pw-nav">
    <div class="pw-nav-icon">🐾</div>
    <div>
        <div class="pw-nav-title">PawWatch</div>
        <div class="pw-nav-sub">Dog Behavior &amp; Emotion Monitoring · University of Greenwich</div>
    </div>
    <div class="pw-nav-badge"><div class="pw-pulse"></div>System Online</div>
</div>""", unsafe_allow_html=True)

hist     = st.session_state.history
dominant = Counter(h["emotion"] for h in hist).most_common(1)[0][0] if hist else None
avg_conf = round(np.mean([h["confidence"] for h in hist]),1) if hist else 0.0
n_alerts = len(st.session_state.alerts)

c1,c2,c3,c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="kpi-card"><div class="kpi-icon">📸</div>'
                f'<div class="kpi-num">{len(hist)}</div>'
                f'<div class="kpi-lbl">Frames Analysed</div></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="kpi-card"><div class="kpi-icon">🎯</div>'
                f'<div class="kpi-num">{avg_conf:.1f}<span class="kpi-unit">%</span></div>'
                f'<div class="kpi-lbl">Avg Confidence</div></div>',
                unsafe_allow_html=True)
with c3:
    ac = "#dc2626" if n_alerts else "#1e293b"
    st.markdown(f'<div class="kpi-card"><div class="kpi-icon">🔔</div>'
                f'<div class="kpi-num" style="color:{ac}">{n_alerts}</div>'
                f'<div class="kpi-lbl">Alerts Fired</div></div>',
                unsafe_allow_html=True)
with c4:
    if dominant:
        dc=C_HEX[dominant]; dbg=C_BG[dominant]; dbd=C_BD[dominant]
        st.markdown(f'<div class="kpi-card" style="background:{dbg};border-color:{dbd}">'
                    f'<div class="kpi-icon">{EMOJI[dominant]}</div>'
                    f'<div class="kpi-num" style="color:{dc};font-size:1.45rem">{dominant.upper()}</div>'
                    f'<div class="kpi-lbl">Dominant Emotion</div></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div class="kpi-card"><div class="kpi-icon">🐕</div>'
                    '<div class="kpi-num" style="font-size:1.45rem;color:#94a3b8">—</div>'
                    '<div class="kpi-lbl">Dominant Emotion</div></div>',
                    unsafe_allow_html=True)

st.markdown("<div style='margin-bottom:4px'></div>", unsafe_allow_html=True)

t_live,t_demo,t_hist,t_alrt = st.tabs([
    "📹  Live Camera","🖼️  Demo Upload",
    "📊  Behavior History","🔔  Alert Status",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — LIVE CAMERA
# ══════════════════════════════════════════════════════════════════════════════
with t_live:
    st.markdown(
        '<div class="callout info"><span class="callout-ico">💡</span>'
        '<span><strong>Live camera</strong> works when running locally. '
        'On Streamlit Cloud use <strong>Demo Upload</strong> instead.</span></div>',
        unsafe_allow_html=True)

    feed_col,panel_col = st.columns([3,1],gap="large")
    with panel_col:
        st.markdown('<div class="sec-title">🐕 Current Behavior</div>',unsafe_allow_html=True)
        emo_ph=st.empty(); conf_ph=st.empty(); pace_ph=st.empty(); tail_ph=st.empty()
        st.markdown('<div class="sec-title" style="margin-top:16px">📊 Probabilities</div>',
                    unsafe_allow_html=True)
        bar_phs={c: st.empty() for c in CLASSES}
        emo_ph.markdown(
            '<div class="empty-state" style="padding:22px 14px">'
            '<div class="e-ico">🐕</div><div class="e-ttl">Waiting…</div>'
            '<div class="e-sub">Press Start Camera</div></div>',
            unsafe_allow_html=True)

    with feed_col:
        bc1,bc2,bc3 = st.columns([1,1,2])
        start_btn = bc1.button("▶️  Start Camera",use_container_width=True)
        stop_btn  = bc2.button("⏹️  Stop",use_container_width=True)
        live_ph   = bc3.empty()
        frm_ph=st.empty(); sts_ph=st.empty()

    if start_btn:
        st.session_state.camera_running=True
        st.session_state.beh_window.clear(); st.session_state.pos_history.clear()
    if stop_btn:
        st.session_state.camera_running=False

    if st.session_state.camera_running:
        src = rtsp_url.strip() if rtsp_url.strip() else int(cam_index)
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            sts_ph.error(f"Cannot open camera: {src}")
            st.session_state.camera_running=False
        else:
            live_ph.markdown(
                '<div class="live-badge"><div class="live-dot"></div>LIVE</div>',
                unsafe_allow_html=True)
            while st.session_state.camera_running:
                ret,frame=cap.read()
                if not ret: time.sleep(0.1); continue
                frame=cv2.resize(frame,(640,480))
                ann,res=process_frame(frame,st.session_state.model,
                                      st.session_state.yolo,smooth=True)
                frm_ph.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),use_container_width=True)
                if res["dog_found"]:
                    emo=res["emotion"]; c_=C_HEX.get(emo,"#64748b")
                    bg_=C_BG.get(emo,"#f1f5f9"); bd_=C_BD.get(emo,"#cbd5e1")
                    emo_ph.markdown(
                        f'<div class="emo-card" style="background:{bg_};border-color:{bd_}">'
                        f'<div class="ec-emoji">{EMOJI[emo]}</div>'
                        f'<div class="ec-name" style="color:{c_}">{emo.upper()}</div>'
                        f'<div class="ec-conf">Confidence: {res["confidence"]:.1%}</div></div>',
                        unsafe_allow_html=True)
                    conf_ph.markdown(
                        f'<div class="stat-row"><span class="stat-lbl">Confidence</span>'
                        f'<span class="stat-val" style="color:{c_}">{res["confidence"]:.1%}</span></div>',
                        unsafe_allow_html=True)
                    pace_ph.markdown(
                        f'<div class="stat-row"><span class="stat-lbl">Pacing Score</span>'
                        f'<span class="stat-val">{res["pacing"]:.0f}</span></div>',
                        unsafe_allow_html=True)
                    tail_ph.markdown(
                        f'<div class="stat-row"><span class="stat-lbl">Tail Movement</span>'
                        f'<span class="stat-val">{res["tail"]:.1f}</span></div>',
                        unsafe_allow_html=True)
                    for cls in CLASSES:
                        bar_phs[cls].markdown(
                            _prob_bar(cls,res["probs"].get(cls,0.0)),
                            unsafe_allow_html=True)
                    record(res)
                time.sleep(0.06)
            cap.release(); live_ph.empty()
            sts_ph.info("Camera stopped.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — DEMO UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
with t_demo:
    mode = st.radio("What would you like to analyse?",
                    ["📷  Single Image","🎬  Video File"],horizontal=True)
    st.markdown("<div style='margin-top:6px'></div>",unsafe_allow_html=True)

    if "Image" in mode:
        st.session_state.video_results = None

        up = st.file_uploader("Upload a photo of your dog",
                               type=["jpg","jpeg","png","webp"],
                               help="Clear, well-lit photos give best results")

        if up:
            file_hash = hashlib.md5(up.getvalue()).hexdigest()

            if file_hash != st.session_state.last_upload_hash:
                pil   = Image.open(up).convert("RGB")
                frame = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

                with st.spinner("Analysing…"):
                    ann, res = process_frame(
                        cv2.resize(frame,(640,480)),
                        st.session_state.model,
                        st.session_state.yolo,
                        smooth=False)

                buf_pil = io.BytesIO(); pil.save(buf_pil, format="PNG")
                buf_ann = io.BytesIO()
                Image.fromarray(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB)).save(buf_ann, format="PNG")

                st.session_state.image_result = {
                    "pil_bytes": buf_pil.getvalue(),
                    "ann_bytes": buf_ann.getvalue(),
                    "res":       res,
                }
                st.session_state.last_upload_hash = file_hash
                if res["dog_found"]:
                    record(res)
                st.rerun()

        if st.session_state.image_result is not None:
            ir       = st.session_state.image_result
            res      = ir["res"]
            pil_disp = Image.open(io.BytesIO(ir["pil_bytes"]))
            ann_disp = Image.open(io.BytesIO(ir["ann_bytes"]))

            img_col,res_col = st.columns(2,gap="large")
            with img_col:
                st.markdown('<div class="sec-title">📷 Your Photo</div>',
                            unsafe_allow_html=True)
                st.image(pil_disp,use_container_width=True)
                st.markdown('<div class="sec-title" style="margin-top:14px">🔍 Detected Region</div>',
                            unsafe_allow_html=True)
                st.image(ann_disp,use_container_width=True)

            with res_col:
                st.markdown('<div class="sec-title">🧠 Analysis Result</div>',
                            unsafe_allow_html=True)
                if res["dog_found"]:
                    st.markdown(
                        _emo_result(res["emotion"],res["confidence"],
                                    res["pacing"],res["tail"]),
                        unsafe_allow_html=True)
                    st.markdown('<div style="margin-top:14px"></div>',unsafe_allow_html=True)
                    st.markdown('<div class="sec-title">📊 Emotion Probabilities</div>',
                                unsafe_allow_html=True)
                    st.markdown(
                        "".join(_prob_bar(cls,res["probs"].get(cls,0.0))
                                for cls in CLASSES),
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        '<div class="empty-state"><div class="e-ico">🔍</div>'
                        '<div class="e-ttl">No dog detected</div>'
                        '<div class="e-sub">Try a clearer photo or lower the '
                        'detection confidence in the sidebar.</div></div>',
                        unsafe_allow_html=True)

            if res["dog_found"] and st.session_state.history:
                st.markdown(
                    '<div class="graphs-section">'
                    '<div class="graphs-title">📉 Analysis Graphs</div>',
                    unsafe_allow_html=True)
                render_analysis_graphs(st.session_state.history)
                st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.session_state.image_result     = None
        st.session_state.last_upload_hash = None

        up = st.file_uploader("Upload a video of your dog",
                               type=["mp4","avi","mov","mkv"],
                               help="MP4 recommended")
        vc1,vc2 = st.columns(2)
        every_n = vc1.slider("Sample every N frames",1,30,5)
        max_fr  = vc2.slider("Max frames to process",10,300,80)

        if up and st.button("▶️  Start Video Analysis",use_container_width=True):
            st.session_state.video_results = None

            with tempfile.NamedTemporaryFile(delete=False,suffix=".mp4") as tmp:
                tmp.write(up.read()); tmp_path=tmp.name

            prog=st.progress(0,text="Starting…"); preview=st.empty()
            rows=[]; idx=processed=0
            cap=cv2.VideoCapture(tmp_path)

            while True:
                ret,frame=cap.read()
                if not ret or processed>=max_fr: break
                if idx % every_n == 0:
                    frame=cv2.resize(frame,(640,480))
                    ann,res=process_frame(frame,st.session_state.model,
                                          st.session_state.yolo,smooth=False)
                    preview.image(cv2.cvtColor(ann,cv2.COLOR_BGR2RGB),
                                  caption=f"Frame {idx}",use_container_width=True)
                    if res["dog_found"]:
                        rows.append({"frame":idx,**res})
                        record(res)
                    processed+=1
                    prog.progress(min(processed/max_fr,1.0),
                                  text=f"Analysing frame {idx}…")
                idx+=1

            cap.release(); os.unlink(tmp_path)
            prog.empty(); preview.empty()

            st.session_state.video_results = {"rows":rows,"processed":processed}
            st.rerun()

        if st.session_state.video_results is not None:
            rows      = st.session_state.video_results["rows"]
            processed = st.session_state.video_results["processed"]

            if rows:
                tally = Counter(r["emotion"] for r in rows)
                total = sum(tally.values())

                st.markdown(
                    f'<div class="callout success"><span class="callout-ico">✅</span>'
                    f'<span><strong>{processed} frames processed</strong> · '
                    f'{len(rows)} dog detections · {total} emotion readings</span></div>',
                    unsafe_allow_html=True)

                left_v,right_v = st.columns(2,gap="large")

                with left_v:
                    st.markdown('<div class="sec-title">📊 Emotion Distribution</div>',
                                unsafe_allow_html=True)
                    st.markdown(
                        "".join(_dist_bar(cls,
                                          tally.get(cls,0)/total if total else 0,
                                          tally.get(cls,0)) for cls in CLASSES),
                        unsafe_allow_html=True)
                    dv=max(tally,key=tally.get)
                    dc=C_HEX[dv]; dbg=C_BG[dv]; dbd=C_BD[dv]
                    st.markdown(
                        f'<div class="dom-box" style="background:{dbg};border-color:{dbd}">'
                        f'<div class="dom-emo">{EMOJI[dv]}</div>'
                        f'<div><div class="dom-name" style="color:{dc}">Dominant: {dv.upper()}</div>'
                        f'<div class="dom-sub">{tally[dv]} of {total} frames</div>'
                        f'</div></div>',unsafe_allow_html=True)

                with right_v:
                    st.markdown('<div class="sec-title">📊 Average Emotion Probabilities</div>',
                                unsafe_allow_html=True)
                    for cls in CLASSES:
                        avg_p = float(np.mean([r["probs"].get(cls,0.0)
                                               for r in rows if r.get("probs")]))
                        st.markdown(_prob_bar(cls,avg_p),unsafe_allow_html=True)

                video_hist = [
                    {
                        "ts": f"f{r['frame']}",
                        "emotion": r["emotion"],
                        "confidence": round(r["confidence"] * 100, 1),
                        "pacing": r["pacing"],
                        "tail": r["tail"],
                    }
                    for r in rows
                ]
                st.markdown(
                    '<div class="graphs-section">'
                    '<div class="graphs-title">📉 Analysis Graphs</div>',
                    unsafe_allow_html=True)
                render_analysis_graphs(video_hist)
                st.markdown('</div>', unsafe_allow_html=True)

            else:
                st.markdown(
                    '<div class="empty-state"><div class="e-ico">🎬</div>'
                    '<div class="e-ttl">No dog detected</div>'
                    '<div class="e-sub">No dog found in any sampled frame. '
                    'Try lowering the detection confidence threshold in the sidebar.</div></div>',
                    unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — BEHAVIOR HISTORY
# ══════════════════════════════════════════════════════════════════════════════
with t_hist:
    hist=st.session_state.history
    if not hist:
        st.markdown(
            '<div class="empty-state" style="margin-top:10px">'
            '<div class="e-ico">📊</div><div class="e-ttl">No data yet</div>'
            '<div class="e-sub">Start the live camera or upload images/videos to '
            'begin collecting behavior data.</div></div>',
            unsafe_allow_html=True)
    else:
        left,right=st.columns([5,7],gap="large")
        with left:
            st.markdown('<div class="sec-title">📊 Emotion Distribution</div>',
                        unsafe_allow_html=True)
            tally=Counter(h["emotion"] for h in hist); total_h=len(hist)
            st.markdown(
                "".join(_dist_bar(cls,tally.get(cls,0)/total_h if total_h else 0,
                                   tally.get(cls,0)) for cls in CLASSES),
                unsafe_allow_html=True)
            st.markdown('<div class="sec-title" style="margin-top:20px">📈 Session Stats</div>',
                        unsafe_allow_html=True)
            dom=max(tally,key=tally.get) if tally else "—"
            avg_p=round(np.mean([h["pacing"] for h in hist]),1)
            avg_t=round(np.mean([h["tail"]   for h in hist]),1)
            st.markdown(
                f'<div class="cfg-box">'
                f'<div class="cfg-row"><span class="cfg-k">Total frames</span>'
                f'<span class="cfg-v">{total_h}</span></div>'
                f'<div class="cfg-row"><span class="cfg-k">Avg confidence</span>'
                f'<span class="cfg-v">{avg_conf:.1f}%</span></div>'
                f'<div class="cfg-row"><span class="cfg-k">Avg pacing</span>'
                f'<span class="cfg-v">{avg_p}</span></div>'
                f'<div class="cfg-row"><span class="cfg-k">Avg tail movement</span>'
                f'<span class="cfg-v">{avg_t}</span></div>'
                f'<div class="cfg-row"><span class="cfg-k">Dominant emotion</span>'
                f'<span class="cfg-v" style="color:{C_HEX.get(dom,"#1e293b")}">'
                f'{EMOJI.get(dom,"")} {dom.upper()}</span></div>'
                f'</div>',unsafe_allow_html=True)

        with right:
            fc1,fc2=st.columns(2)
            sort_asc   = fc1.toggle("Show oldest first",False)
            emo_filter = fc2.multiselect("Filter by emotion",CLASSES,default=CLASSES)
            display=[h for h in hist if h["emotion"] in emo_filter]
            if not sort_asc: display=list(reversed(display))
            st.markdown(
                f'<div class="sec-title">🕐 Detection Log '
                f'<span style="font-size:.72rem;color:#64748b;font-weight:500">'
                f'— {len(display)} records</span></div>',
                unsafe_allow_html=True)
            if display:
                st.markdown(
                    "".join(
                        f'<div class="hist-row">'
                        f'<span class="hist-time">{e["ts"]}</span>'
                        f'{_badge(e["emotion"])}'
                        f'<span class="hist-conf">{e["confidence"]:.1f}%</span>'
                        f'<span class="hist-meta">Pace {e["pacing"]:.0f} · Tail {e["tail"]:.1f}</span>'
                        f'</div>'
                        for e in display[:100]),
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="empty-state" style="padding:20px">'
                    '<div class="e-sub">No records match the selected filter.</div></div>',
                    unsafe_allow_html=True)

        st.markdown("<div style='margin-top:12px'></div>",unsafe_allow_html=True)
        exp_col,_=st.columns([2,5])
        if exp_col.button("📥  Export History to CSV",use_container_width=True):
            csv=pd.DataFrame(hist).to_csv(index=False)
            st.download_button("⬇️  Download pawwatch_history.csv",
                               csv,"pawwatch_history.csv","text/csv")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — ALERT STATUS
# ══════════════════════════════════════════════════════════════════════════════
with t_alrt:
    log_col,cfg_col=st.columns([3,2],gap="large")
    with log_col:
        st.markdown('<div class="sec-title">🚨 Alert Log</div>',unsafe_allow_html=True)
        alerts=st.session_state.alerts
        if not alerts:
            st.markdown(
                '<div class="empty-state"><div class="e-ico">✅</div>'
                '<div class="e-ttl">No alerts this session</div>'
                '<div class="e-sub">Alerts fire when your dog shows '
                '<strong>ANGRY</strong> or <strong>SAD</strong> emotions.</div></div>',
                unsafe_allow_html=True)
        else:
            for a in reversed(alerts):
                emo=a["emotion"]
                c_=C_HEX.get(emo,"#dc2626"); bg_=C_BG.get(emo,"#fee2e2"); bd_=C_BD.get(emo,"#fca5a5")
                if a.get("sms_sent"):
                    sms='<div class="sms-ok">✓ WhatsApp delivered</div>'
                elif "sms_error" in a:
                    sms=f'<div class="sms-err">✗ Failed — {a["sms_error"]}</div>'
                elif st.session_state.alerts_enabled:
                    sms='<div class="sms-ok">✓ WhatsApp sent</div>'
                else:
                    sms='<div class="sms-off">WhatsApp not configured</div>'
                st.markdown(
                    f'<div class="alert-card" style="background:{bg_};border-color:{bd_}">'
                    f'<div class="alert-ico">{EMOJI.get(emo,"🚨")}</div>'
                    f'<div class="alert-body">'
                    f'<div class="alert-ttl" style="color:{c_}">{emo.upper()} detected · {a["ts"]}</div>'
                    f'<div class="alert-msg">{a["message"]}</div>'
                    f'{sms}</div></div>',unsafe_allow_html=True)

    with cfg_col:
        st.markdown('<div class="sec-title">⚙️ Configuration</div>',unsafe_allow_html=True)
        sms_s="🟢 Enabled" if st.session_state.alerts_enabled else "🔴 Disabled"
        ph_disp=("••••"+st.session_state.phone_number[-4:]
                 if len(st.session_state.phone_number)>4 else "—")
        st.markdown(
            f'<div class="cfg-box">'
            f'<div class="cfg-row"><span class="cfg-k">WhatsApp alerts</span>'
            f'<span class="cfg-v">{sms_s}</span></div>'
            f'<div class="cfg-row"><span class="cfg-k">Trigger emotions</span>'
            f'<span class="cfg-v">😠 Angry · 😢 Sad</span></div>'
            f'<div class="cfg-row"><span class="cfg-k">Cooldown</span>'
            f'<span class="cfg-v">{ALERT_COOLDOWN}s</span></div>'
            f'<div class="cfg-row"><span class="cfg-k">Alerts this session</span>'
            f'<span class="cfg-v" style="color:{"#dc2626" if alerts else "#1e293b"}">'
            f'{len(alerts)}</span></div>'
            f'<div class="cfg-row"><span class="cfg-k">Owner phone</span>'
            f'<span class="cfg-v" style="font-size:.79rem">{ph_disp}</span></div>'
            f'</div>',unsafe_allow_html=True)

        st.markdown('<div class="sec-title" style="margin-top:20px">📖 How Alerts Work</div>',
                    unsafe_allow_html=True)
        st.markdown(
            '<div class="how-box">'
            '<div class="how-step"><div class="how-num">1</div>'
            '<div>PawWatch monitors your dog using YOLOv8 + MobileNetV2.</div></div>'
            '<div class="how-step"><div class="how-num">2</div>'
            '<div><strong>ANGRY</strong> or <strong>SAD</strong> triggers an alert immediately.</div></div>'
            '<div class="how-step"><div class="how-num">3</div>'
            '<div>A 60-second cooldown prevents repeated notifications.</div></div>'
            '<div class="how-step"><div class="how-num">4</div>'
            '<div>A <strong>WhatsApp message</strong> is sent via Twilio if configured.</div></div>'
            '<div class="how-step"><div class="how-num">5</div>'
            '<div>All alerts are logged here for review during the session.</div></div>'
            '</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="pw-footer">'
    '<span>🐾 <strong style="color:#64748b">PawWatch</strong> — Dog Behavior &amp; Emotion Monitoring</span>'
    '<span>MobileNetV2 · YOLOv8n · University of Greenwich · BSc Computing · 001512468</span>'
    '</div>',unsafe_allow_html=True)
