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
