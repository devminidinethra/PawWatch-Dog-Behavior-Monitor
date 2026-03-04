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
