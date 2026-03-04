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
