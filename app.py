"""
PawWatch — Dog Behavior & Emotion Monitoring System
"""

import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import time, tempfile, urllib.request
from collections import deque, Counter
from datetime import datetime

import cv2, numpy as np, pandas as pd
import streamlit as st
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
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
  --amber:   #d97706;
  --amber-l: #fef3c7;
  --amber-b: #fcd34d;
  --purp:    #7c3aed;
  --purp-l:  #ede9fe;
  --purp-b:  #c4b5fd;

  --c-angry:  #dc2626; --bg-angry:  #fee2e2; --bd-angry:  #fca5a5;
  --c-happy:  #16a34a; --bg-happy:  #dcfce7; --bd-happy:  #86efac;
  --c-relax:  #2563eb; --bg-relax:  #dbeafe; --bd-relax:  #93c5fd;
  --c-sad:    #7c3aed; --bg-sad:    #ede9fe; --bd-sad:    #c4b5fd;
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
  background: var(--green) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 8px !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  font-weight: 700 !important;
  font-size: 0.85rem !important;
  padding: .52rem 1.4rem !important;
  transition: all .18s !important;
  box-shadow: 0 1px 4px rgba(22,163,74,.22) !important;
}
.stButton > button:hover {
  background: var(--green-d) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 4px 14px rgba(22,163,74,.3) !important;
}
[data-testid="stDownloadButton"] button {
  background: #fff !important;
  color: var(--green) !important;
  border: 1.5px solid var(--green) !important;
}
[data-testid="stDownloadButton"] button:hover { background: var(--green-l) !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
  background: #fff !important;
  border-radius: 10px !important;
  padding: 4px !important; gap: 3px !important;
  border: 1px solid var(--border) !important;
  box-shadow: 0 1px 4px rgba(0,0,0,.05) !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  background: transparent !important;
  border-radius: 7px !important;
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
  transition: border-color .15s !important;
}
[data-testid="stFileUploader"]:hover { border-color: var(--green) !important; }

[data-testid="stProgressBar"] > div > div { background: var(--green) !important; }

::-webkit-scrollbar       { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 4px; }

/* ── component classes (navbar, cards, badges, bars, etc.) ── */
.pw-nav {
  display: flex; align-items: center; gap: 16px;
  padding: 14px 24px;
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
  padding: 5px 16px; border-radius: 20px; letter-spacing: .04em; white-space: nowrap;
}
.pw-pulse { width: 7px; height: 7px; background: #bbf7d0; border-radius: 50%; animation: pw-p 2s infinite; }
@keyframes pw-p { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:.5;transform:scale(.8)} }

.kpi-card {
  background: #fff; border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 18px; text-align: center;
  box-shadow: 0 1px 6px rgba(0,0,0,.04);
  transition: transform .15s, box-shadow .15s;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 5px 18px rgba(0,0,0,.08); }
.kpi-icon  { font-size: 1.5rem; margin-bottom: 6px; }
.kpi-num   { font-family: 'Sora', sans-serif; font-size: 2rem; font-weight: 700; color: #1e293b; line-height: 1; }
.kpi-unit  { font-size: .9rem; color: #64748b; }
.kpi-lbl   { font-size: .68rem; color: #64748b; text-transform: uppercase; letter-spacing: .09em; font-weight: 700; margin-top: 5px; }

.sec-title {
  font-family: 'Sora', sans-serif; font-size: .95rem; font-weight: 700;
  color: #1e293b; display: flex; align-items: center; gap: 8px;
  padding-bottom: 9px; margin-bottom: 13px; border-bottom: 2px solid var(--border);
}

.callout { display: flex; align-items: flex-start; gap: 12px; padding: 12px 15px; border-radius: 10px; margin-bottom: 16px; font-size: .83rem; line-height: 1.55; }
.callout.info    { background: var(--blue-l);  border: 1px solid var(--blue-b);  color: #1e3a5f; }
.callout.success { background: var(--green-l); border: 1px solid var(--green-b); color: #14532d; }
.callout.warn    { background: var(--amber-l); border: 1px solid var(--amber-b); color: #78350f; }
.callout-ico { font-size: 1.1rem; flex-shrink: 0; margin-top: 2px; }

.emo-card { border-radius: 14px; padding: 22px 18px; text-align: center; border: 1.5px solid; margin-bottom: 12px; }
.emo-card .ec-emoji { font-size: 3.2rem; line-height: 1; margin-bottom: 9px; }
.emo-card .ec-name  { font-family: 'Sora', sans-serif; font-size: 1.5rem; font-weight: 700; margin-bottom: 6px; }
.emo-card .ec-conf  { font-size: .83rem; color: #64748b; }

.stat-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 14px; background: var(--bg); border-radius: 8px; margin-bottom: 6px; border: 1px solid var(--border); font-size: .84rem; }
.stat-lbl { color: #64748b; font-weight: 500; }
.stat-val { color: #1e293b; font-weight: 700; }

.prob-wrap { margin-bottom: 11px; }
.prob-head { display: flex; justify-content: space-between; font-size: .8rem; margin-bottom: 5px; font-weight: 700; color: #1e293b; }
.prob-track { background: var(--bg); border-radius: 5px; height: 9px; border: 1px solid var(--border); overflow: hidden; }
.prob-fill  { height: 9px; border-radius: 5px; transition: width .4s ease; }

.dist-wrap { margin-bottom: 13px; }
.dist-head { display: flex; justify-content: space-between; font-size: .82rem; margin-bottom: 5px; font-weight: 700; color: #1e293b; }
.dist-sub  { color: #64748b; font-weight: 500; font-size: .78rem; }
.dist-track { background: var(--bg); border-radius: 6px; height: 12px; border: 1px solid var(--border); overflow: hidden; }
.dist-fill  { height: 12px; border-radius: 6px; }

.hist-row { display: flex; align-items: center; gap: 10px; padding: 9px 13px; background: #fff; border: 1px solid var(--border); border-radius: 9px; margin-bottom: 5px; font-size: .81rem; transition: box-shadow .15s; }
.hist-row:hover { box-shadow: 0 2px 10px rgba(0,0,0,.07); }
.hist-time { color: #94a3b8; font-size: .7rem; min-width: 50px; }
.hist-conf { color: #64748b; font-size: .73rem; margin-left: 3px; }
.hist-meta { color: #94a3b8; font-size: .7rem; margin-left: auto; }

.emo-badge { display: inline-flex; align-items: center; gap: 4px; padding: 3px 10px; border-radius: 20px; font-size: .71rem; font-weight: 800; text-transform: uppercase; letter-spacing: .06em; border: 1.5px solid; min-width: 86px; justify-content: center; }

.alert-card { display: flex; gap: 14px; padding: 14px 16px; border-radius: 12px; border: 1.5px solid; margin-bottom: 10px; }
.alert-ico  { font-size: 1.5rem; flex-shrink: 0; padding-top: 2px; }
.alert-body { flex: 1; }
.alert-ttl  { font-weight: 800; font-size: .88rem; margin-bottom: 4px; }
.alert-msg  { font-size: .79rem; color: #64748b; line-height: 1.5; }
.sms-ok  { font-size: .72rem; color: #16a34a; font-weight: 700; margin-top: 5px; }
.sms-err { font-size: .72rem; color: #dc2626; font-weight: 700; margin-top: 5px; }
.sms-off { font-size: .72rem; color: #94a3b8; margin-top: 5px; }

.cfg-box { background: var(--bg); border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
.cfg-row { display: flex; justify-content: space-between; align-items: center; padding: 10px 16px; border-bottom: 1px solid var(--border); font-size: .83rem; }
.cfg-row:last-child { border-bottom: none; }
.cfg-k { color: #64748b; font-weight: 500; }
.cfg-v { color: #1e293b; font-weight: 700; }

.how-box { background: #fff; border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; }
.how-step { display: flex; gap: 12px; align-items: flex-start; padding: 8px 0; border-bottom: 1px solid var(--border); font-size: .83rem; line-height: 1.55; color: #1e293b; }
.how-step:last-child { border-bottom: none; }
.how-num { background: var(--green-l); color: #15803d; border: 1px solid var(--green-b); font-weight: 800; font-size: .72rem; width: 22px; height: 22px; border-radius: 50%; display: flex; align-items: center; justify-content: center; flex-shrink: 0; margin-top: 2px; }

.empty-state { background: #fff; border: 1.5px dashed #cbd5e1; border-radius: 14px; padding: 38px 26px; text-align: center; }
.e-ico { font-size: 2.8rem; margin-bottom: 10px; }
.e-ttl { font-family: 'Sora', sans-serif; font-size: 1rem; font-weight: 700; color: #1e293b; margin-bottom: 7px; }
.e-sub { font-size: .82rem; color: #64748b; line-height: 1.6; }

.live-badge { display: inline-flex; align-items: center; gap: 7px; background: #fee2e2; color: #dc2626; border: 1.5px solid #fca5a5; font-size: .74rem; font-weight: 800; padding: 5px 12px; border-radius: 20px; letter-spacing: .07em; }
.live-dot   { width: 7px; height: 7px; background: #dc2626; border-radius: 50%; animation: blink 1s infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.2} }

.result-card { background: #fff; border: 1px solid var(--border); border-radius: 14px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,.05); }

.dom-box { border-radius: 12px; border: 1.5px solid; padding: 14px 18px; margin-top: 12px; display: flex; align-items: center; gap: 14px; }
.dom-emo  { font-size: 2rem; }
.dom-name { font-family: 'Sora', sans-serif; font-weight: 700; font-size: 1rem; }
.dom-sub  { font-size: .76rem; color: #64748b; margin-top: 2px; }

.sb-brand { padding: 10px 0 6px; display: flex; align-items: center; gap: 12px; }
.sb-logo  { font-size: 2rem; }
.sb-name  { font-family: 'Sora', sans-serif; font-size: 1.1rem; font-weight: 700; color: #1e293b; }
.sb-sub   { font-size: .68rem; color: #64748b; margin-top: 1px; }
.sb-section { font-size: .68rem; font-weight: 800; text-transform: uppercase; letter-spacing: .1em; color: #64748b; margin: 14px 0 6px; padding-bottom: 4px; border-bottom: 1px solid var(--border); }

.pw-footer { margin-top: 50px; padding: 16px 24px; border-top: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; font-size: .7rem; color: #94a3b8; }
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
        st.error("❌  MODEL_URL not set in Streamlit Secrets.")
        st.stop()
    os.makedirs("models", exist_ok=True)
    bar = st.progress(0, text="⬇️  Downloading model weights — first load only…")
    def hook(c, bs, tot):
        if tot > 0:
            bar.progress(min(int(c*bs*100/tot), 100)/100,
                         text=f"⬇️  Downloading… {min(int(c*bs*100/tot),100)}%")
    try:
        urllib.request.urlretrieve(url, MODEL_LOCAL_PATH, hook)
        bar.empty()
    except Exception as e:
        bar.empty(); st.error(f"❌  Download failed: {e}"); st.stop()
    return MODEL_LOCAL_PATH

@st.cache_resource(show_spinner="🧠  Loading emotion model…")
def load_cnn(path):
    import keras
    return keras.models.load_model(path, compile=False)

@st.cache_resource(show_spinner="🔍  Loading YOLOv8…")
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

model_path = download_model_if_needed()
if st.session_state.model is None: st.session_state.model = load_cnn(model_path)
if st.session_state.yolo  is None: st.session_state.yolo  = load_yolo()

# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
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
            lbl = f"{EMOJI.get(emotion,'')} {emotion.upper()}  {conf:.0%}"
            (tw,th),_ = cv2.getTextSize(lbl,cv2.FONT_HERSHEY_SIMPLEX,.65,2)
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
    emoji_map = {"angry":"😠","sad":"😢"}
    msg = (f"🐾 *PawWatch Alert* [{datetime.now().strftime('%H:%M:%S')}]\n\n"
           f"{emoji_map.get(emotion,'')} Your dog appears *{emotion.upper()}*!\n"
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
        "pacing":res["pacing"], "tail":res["tail"],
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
            f'<span class="dist-name" style="color:{c}">{EMOJI[cls]} {cls.upper()}</span>'
            f'<span class="dist-sub">{pct:.1%} · {cnt} frames</span></div>'
            f'<div class="dist-track">'
            f'<div class="dist-fill" style="width:{int(pct*100)}%;background:{c}"></div>'
            f'</div></div>')

def _emo_result(emo, conf, pacing=None, tail=None):
    c=C_HEX.get(emo,"#64748b"); bg=C_BG.get(emo,"#f1f5f9"); bd=C_BD.get(emo,"#cbd5e1")
    stats = ""
    if pacing is not None:
        stats += (f'<div class="stat-row">'
                  f'<span class="stat-lbl">Pacing Score</span>'
                  f'<span class="stat-val">{pacing:.0f}</span></div>')
    if tail is not None:
        stats += (f'<div class="stat-row">'
                  f'<span class="stat-lbl">Tail Movement</span>'
                  f'<span class="stat-val">{tail:.1f}</span></div>')
    return (f'<div class="result-card">'
            f'<div class="emo-card" style="background:{bg};border-color:{bd}">'
            f'<div class="ec-emoji">{EMOJI.get(emo,"")}</div>'
            f'<div class="ec-name" style="color:{c}">{emo.upper()}</div>'
            f'<div class="ec-conf">Confidence: '
            f'<strong style="color:{c};font-size:1rem">{conf:.1%}</strong></div>'
            f'</div>{stats}</div>')
