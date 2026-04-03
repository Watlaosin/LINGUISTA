import base64
import math
import threading
import time
from pathlib import Path

import av
import cv2
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from streamlit_webrtc import (
    RTCConfiguration,
    VideoProcessorBase,
    webrtc_streamer,
)

from sign_predictor import SignLanguagePredictor


st.set_page_config(
    page_title="Linguista",
    page_icon="🙌",
    layout="centered",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).parent
LOGO_PATH = BASE_DIR / "assets" / "logo.png"

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
        ]
    }
)


@st.cache_resource
def load_predictor():
    return SignLanguagePredictor(confidence_threshold=0.2)


@st.cache_data
def base64_logo():
    if LOGO_PATH.exists():
        return base64.b64encode(LOGO_PATH.read_bytes()).decode()
    return ""


# =========================
# Query-param page sync
# =========================
query_page = st.query_params.get("page", "home")
if query_page not in {"home", "demo", "stage"}:
    query_page = "home"


# =========================
# Session State
# =========================
def init_session_state():
    defaults = {
        "page": query_page,
        "stage_index": 0,
        "stage_started": False,
        "stage_start_time": None,
        "stage_status": "idle",
        "stage_feedback": "",
        "stage_balloons_shown": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

st.session_state.page = st.query_params.get("page", st.session_state.page)
if st.session_state.page not in {"home", "demo", "stage"}:
    st.session_state.page = "home"


# =========================
# Data
# =========================
STAGES = [
    {
        "target": "Hungry",
        "label": "Hungry",
        "video": BASE_DIR / "assets" / "demo_videos" / "hungry.mp4",
    },
    {
        "target": "Sleepy",
        "label": "Sleepy",
        "video": BASE_DIR / "assets" / "demo_videos" / "sleepy.mp4",
    },
    {
        "target": "Drink",
        "label": "Drink",
        "video": BASE_DIR / "assets" / "demo_videos" / "drink.mp4",
    },
    {
        "target": "Yes",
        "label": "Yes",
        "video": BASE_DIR / "assets" / "demo_videos" / "yes.mp4",
    },
]

SUCCESS_MESSAGES = [
    "Amazing! 🌟",
    "Nice one! 🎉",
    "So good! 💖",
    "You got it! ✨",
]

FAIL_MESSAGES = [
    "Almost there! 🌈",
    "Try again! 💪",
    "One more time! 🫶",
    "Keep going! ✨",
]


# =========================
# Navigation helpers
# =========================
def set_page(page_name: str):
    st.session_state.page = page_name
    st.query_params["page"] = page_name


def go_home():
    set_page("home")


def go_demo():
    set_page("demo")


def go_stage():
    set_page("stage")


def start_stage():
    st.session_state.stage_started = True
    st.session_state.stage_start_time = time.time()
    st.session_state.stage_status = "running"
    st.session_state.stage_feedback = ""
    st.session_state.stage_balloons_shown = False


def reset_stage():
    st.session_state.stage_started = False
    st.session_state.stage_start_time = None
    st.session_state.stage_status = "idle"
    st.session_state.stage_feedback = ""
    st.session_state.stage_balloons_shown = False


def next_stage():
    if st.session_state.stage_index < len(STAGES) - 1:
        st.session_state.stage_index += 1
    reset_stage()


# =========================
# Reusable UI helpers
# =========================
def card(title: str, body: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def pill(text: str, cls: str = ""):
    st.markdown(
        f"""<div class="pill {cls}">{text}</div>""",
        unsafe_allow_html=True,
    )


def app_link(label: str, page: str, variant: str = ""):
    st.markdown(
        f"""
        <a class="app-link {variant}" href="/?page={page}" target="_self">
            {label}
        </a>
        """,
        unsafe_allow_html=True,
    )


def bottom_nav_item(icon: str, page: str, variant: str = "", active: bool = False):
    active_class = "active" if active else ""
    return (
        f'<a class="bottom-nav-link {variant} {active_class}" '
        f'href="/?page={page}" target="_self" aria-label="{page}">'
        f'<span class="bottom-nav-icon">{icon}</span>'
        f"</a>"
    )


def create_camera_stream(key: str):
    return webrtc_streamer(
        key=key,
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 480},
                "height": {"ideal": 640},
                "frameRate": {"ideal": 12},
                "facingMode": "user",
            },
            "audio": False,
        },
        async_processing=True,
    )


def show_stage_demo(video_path, target_sign: str):
    if video_path and Path(video_path).exists():
        st.markdown(
            f"""
            <div class="card" style="padding:12px 16px;">
                <div class="card-title">Demo Clip</div>
                <div class="card-body">Watch how to sign <b>{target_sign}</b> before starting.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, center, right = st.columns([1, 2, 1])
        with center:
            st.video(str(video_path))
    else:
        card(
            "Demo clip missing 🎬",
            f"Add a video file for {target_sign} in assets/demo_videos.",
        )


# =========================
# Styling
# =========================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    :root {
        /* Primary palette - preserved */
        --chocolate-kisses: #45151B;
        --mauvelous: #EA9DAE;
        --royal-orange: #F99256;
        --bittersweet-shimmer: #C74E51;
        --caramel: #FBDE9C;
        
        /* Refined role assignments */
        --primary: var(--chocolate-kisses);
        --accent: var(--bittersweet-shimmer);
        --highlight: var(--royal-orange);
        --surface: rgba(255, 255, 255, 0.92);
        --surface-elevated: rgba(255, 255, 255, 0.97);
        --text-primary: var(--chocolate-kisses);
        --text-secondary: #5C2A32;
        --text-tertiary: #8B5A62;
        --border-subtle: rgba(69, 21, 27, 0.06);
        --border-light: rgba(255, 255, 255, 0.5);
        --shadow-sm: 0 1px 2px rgba(69, 21, 27, 0.04);
        --shadow-md: 0 4px 12px rgba(69, 21, 27, 0.08);
        --shadow-lg: 0 8px 24px rgba(69, 21, 27, 0.12);
        --shadow-xl: 0 16px 40px rgba(69, 21, 27, 0.14);
        --nav-surface: rgba(45, 15, 20, 0.95);
        
        /* Spacing scale (iOS-inspired 4pt grid) */
        --space-1: 4px;
        --space-2: 8px;
        --space-3: 12px;
        --space-4: 16px;
        --space-5: 20px;
        --space-6: 24px;
        --space-8: 32px;
        
        /* Radius scale */
        --radius-sm: 10px;
        --radius-md: 14px;
        --radius-lg: 20px;
        --radius-xl: 24px;
        --radius-full: 9999px;
    }

    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display:none !important;}
    [data-testid="collapsedControl"] {display:none !important;}

    html, body, [class*="css"] {
        font-family: "Inter", -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
        letter-spacing: -0.01em;
    }

    a {
        text-decoration: none !important;
    }

    .stApp {
        background: linear-gradient(175deg, 
            #FDF6E8 0%, 
            #FAECD4 25%,
            #F8E0C8 50%,
            #F5D4C0 75%,
            var(--mauvelous) 100%
        );
        color: var(--text-primary);
        min-height: 100vh;
    }

    .main .block-container {
        max-width: 420px;
        padding: var(--space-5) var(--space-4) 100px var(--space-4);
    }

    /* Logo */
    .logo-image-wrap {
        width: 100%;
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 0 auto var(--space-4) auto;
    }

    .logo-image-box {
        width: 80px;
        height: 80px;
        border-radius: 50%;
        background: var(--accent);
        display: flex;
        align-items: center;
        justify-content: center;
        animation: logoFloat 3s ease-in-out infinite;
        box-shadow: 
            var(--shadow-lg),
            0 0 0 4px rgba(255, 255, 255, 0.3);
    }

    .logo-image-box img {
        width: 130%;
        height: 130%;
        object-fit: cover;
        display: block;
        transform: translate(-1px, 8px);
    }

    /* Brand Header */
    .top-brand {
        background: var(--primary);
        color: var(--mauvelous);
        border-radius: var(--radius-xl);
        padding: var(--space-5);
        box-shadow: var(--shadow-xl);
        margin-bottom: var(--space-4);
        text-align: center;
    }

    .brand-title {
        font-size: 1.625rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.03em;
        color: var(--mauvelous);
        margin-top: var(--space-1);
    }

    .brand-sub {
        margin-top: var(--space-2);
        color: rgba(234, 157, 174, 0.85);
        font-size: 0.875rem;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Hero Card */
    .hero-card {
        background: var(--surface-elevated);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: var(--radius-xl);
        padding: var(--space-5);
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-md);
        margin-bottom: var(--space-4);
    }

    .hero-title {
        font-size: 1.75rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.15;
        letter-spacing: -0.03em;
        margin-bottom: var(--space-2);
    }

    .hero-text {
        color: var(--text-secondary);
        font-size: 0.9375rem;
        font-weight: 450;
        line-height: 1.55;
    }

    /* Cards */
    .card {
        background: var(--surface);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-subtle);
        margin-bottom: var(--space-3);
    }

    .card-title {
        font-size: 0.8125rem;
        font-weight: 700;
        color: var(--text-tertiary);
        text-transform: uppercase;
        letter-spacing: 0.04em;
        margin-bottom: var(--space-2);
    }

    .card-body {
        color: var(--text-secondary);
        font-size: 0.9375rem;
        font-weight: 450;
        line-height: 1.5;
    }

    /* Pills */
    .pill {
        display: inline-flex;
        align-items: center;
        background: var(--highlight);
        color: white;
        border-radius: var(--radius-full);
        padding: var(--space-2) var(--space-3);
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.02em;
        margin-bottom: var(--space-3);
    }

    .pill.soft {
        background: rgba(234, 157, 174, 0.2);
        color: var(--accent);
    }

    .pill.yellow {
        background: rgba(251, 222, 156, 0.4);
        color: var(--text-primary);
    }

    /* Prediction Card */
    .prediction-card {
        background: var(--primary);
        color: var(--caramel);
        border-radius: var(--radius-xl);
        padding: var(--space-5);
        box-shadow: var(--shadow-xl);
        margin-bottom: var(--space-4);
    }

    .prediction-label {
        color: rgba(234, 157, 174, 0.8);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: var(--space-2);
    }

    .prediction-value {
        font-size: 1.875rem;
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -0.02em;
        word-break: break-word;
    }

    /* Stats Grid */
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: var(--space-3);
        margin-bottom: var(--space-4);
    }

    .stat-box {
        background: var(--surface);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-subtle);
    }

    .stat-title {
        font-size: 0.6875rem;
        color: var(--text-tertiary);
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stat-value {
        font-size: 1.25rem;
        color: var(--text-primary);
        font-weight: 800;
        margin-top: var(--space-1);
        letter-spacing: -0.02em;
    }

    /* Top Items */
    .top-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: rgba(251, 222, 156, 0.25);
        border-radius: var(--radius-md);
        padding: var(--space-3) var(--space-4);
        margin-bottom: var(--space-2);
        border: 1px solid rgba(251, 222, 156, 0.3);
    }

    .top-name {
        font-weight: 700;
        color: var(--text-primary);
        font-size: 0.9375rem;
    }

    .top-score {
        font-weight: 800;
        color: var(--accent);
        font-size: 0.9375rem;
    }

    /* Stage Target */
    .stage-target {
        background: linear-gradient(180deg, 
            rgba(234, 157, 174, 0.95) 0%, 
            rgba(234, 157, 174, 0.8) 100%
        );
        backdrop-filter: blur(12px);
        border-radius: var(--radius-xl);
        padding: var(--space-5);
        text-align: center;
        box-shadow: var(--shadow-md);
        margin-bottom: var(--space-4);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .stage-kicker {
        font-size: 0.75rem;
        font-weight: 700;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: var(--space-2);
    }

    .stage-word {
        font-size: 1.875rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.1;
        letter-spacing: -0.02em;
    }

    /* Timer */
    .timer-box {
        background: var(--surface);
        border-radius: var(--radius-lg);
        text-align: center;
        padding: var(--space-4);
        margin-bottom: var(--space-3);
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-subtle);
    }

    .timer-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--accent);
        letter-spacing: -0.02em;
        font-variant-numeric: tabular-nums;
    }

    /* Feedback States */
    .feedback-success {
        background: var(--accent);
        color: white;
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        font-weight: 700;
        font-size: 0.9375rem;
        text-align: center;
        margin-bottom: var(--space-3);
        box-shadow: var(--shadow-md);
    }

    .feedback-fail {
        background: var(--highlight);
        color: white;
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        font-weight: 700;
        font-size: 0.9375rem;
        text-align: center;
        margin-bottom: var(--space-3);
        box-shadow: var(--shadow-md);
    }

    .tiny-note {
        text-align: center;
        color: var(--text-tertiary);
        font-size: 0.8125rem;
        font-weight: 500;
        margin: var(--space-3) 0;
    }

    /* Video Player */
    video {
        width: 80% !important;
        max-width: 300px !important;
        height: auto !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: var(--radius-lg) !important;
        object-fit: cover !important;
        background: var(--primary) !important;
        box-shadow: var(--shadow-lg) !important;
        border: none !important;
    }

    /* Buttons */
    .stButton > button {
        width: 100%;
        border: none !important;
        border-radius: var(--radius-md) !important;
        height: 52px !important;
        padding: 0 var(--space-5) !important;
        font-size: 0.9375rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.01em !important;
        background: var(--primary) !important;
        color: var(--mauvelous) !important;
        box-shadow: var(--shadow-md) !important;
        transition: all 0.15s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg) !important;
    }

    .stButton > button:active {
        transform: translateY(0) scale(0.99);
    }

    .stButton > button:focus,
    .stButton > button:focus-visible {
        outline: none !important;
        box-shadow: 
            0 0 0 3px rgba(199, 78, 81, 0.25),
            var(--shadow-md) !important;
    }

    /* App Links */
    .app-link {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 52px;
        border-radius: var(--radius-md);
        font-size: 0.9375rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        text-decoration: none !important;
        box-shadow: var(--shadow-md);
        transition: all 0.15s ease;
        user-select: none;
        -webkit-tap-highlight-color: transparent;
    }

    .app-link:hover {
        transform: translateY(-1px);
        box-shadow: var(--shadow-lg);
    }

    .app-link:active {
        transform: translateY(0) scale(0.99);
    }

    .app-link.freestyle-link {
        background: var(--highlight);
        color: white !important;
    }

    .app-link.stage-link {
        background: var(--accent);
        color: white !important;
    }

    /* Bottom Navigation */
    .nav-wrap {
        position: fixed;
        left: 50%;
        transform: translateX(-50%);
        bottom: var(--space-3);
        width: min(400px, calc(100vw - var(--space-6)));
        background: var(--nav-surface);
        backdrop-filter: blur(24px);
        -webkit-backdrop-filter: blur(24px);
        border-radius: var(--radius-xl);
        padding: var(--space-3) var(--space-4);
        box-shadow: 
            0 8px 32px rgba(0, 0, 0, 0.2),
            0 0 0 1px rgba(255, 255, 255, 0.05);
        z-index: 999999;
    }

    .bottom-nav-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: var(--space-2);
        align-items: center;
    }

    .bottom-nav-link {
        width: 52px;
        height: 52px;
        margin: 0 auto;
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none !important;
        background: transparent;
        border: none;
        transition: all 0.2s ease;
        user-select: none;
        -webkit-tap-highlight-color: transparent;
    }

    .bottom-nav-link:hover {
        background: rgba(255, 255, 255, 0.08);
    }

    .bottom-nav-link:active {
        transform: scale(0.95);
    }

    .bottom-nav-icon {
        font-size: 1.625rem;
        line-height: 1;
        opacity: 0.7;
        transition: opacity 0.2s ease;
    }

    .bottom-nav-link.home-link .bottom-nav-icon { color: var(--caramel); }
    .bottom-nav-link.demo-link .bottom-nav-icon { color: #7CC4D9; }
    .bottom-nav-link.stage-link-nav .bottom-nav-icon { color: var(--mauvelous); }

    .bottom-nav-link.active {
        background: rgba(255, 255, 255, 0.12);
    }

    .bottom-nav-link.active .bottom-nav-icon {
        opacity: 1;
    }

    @keyframes logoFloat {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-3px); }
    }

    /* Responsive */
    @media (max-width: 480px) {
        .main .block-container {
            padding: var(--space-4) var(--space-3) 96px var(--space-3);
        }

        .hero-title {
            font-size: 1.5rem;
        }

        .prediction-value,
        .stage-word {
            font-size: 1.625rem;
        }

        .timer-value {
            font-size: 1.75rem;
        }

        .nav-wrap {
            width: min(400px, calc(100vw - var(--space-4)));
            bottom: var(--space-2);
            padding: var(--space-2) var(--space-3);
            border-radius: var(--radius-lg);
        }

        .bottom-nav-link {
            width: 48px;
            height: 48px;
        }

        .bottom-nav-icon {
            font-size: 1.5rem;
        }

        .logo-image-box {
            width: 72px;
            height: 72px;
        }

        video {
            width: 85% !important;
            max-width: 280px !important;
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Video Processor
# =========================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.predictor = load_predictor()
        self.result = {
            "frames_collected": 0,
            "frames_needed": 45,
            "prediction": "Collecting...",
            "raw_label": None,
            "display_label": None,
            "confidence": 0.0,
            "is_confident": False,
            "top3": [],
        }
        self.lock = threading.Lock()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        try:
            processed_frame, result = self.predictor.process_frame(
                img,
                draw_landmarks=True,
            )
            with self.lock:
                self.result = result.copy()
        except Exception:
            processed_frame = img

        return av.VideoFrame.from_ndarray(processed_frame, format="bgr24")


# =========================
# Header
# =========================
if LOGO_PATH.exists():
    st.markdown(
        f"""
        <div class="logo-image-wrap">
            <div class="logo-image-box">
                <img src="data:image/png;base64,{base64_logo()}" alt="Linguista logo">
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="top-brand">
        <div class="brand-title">Linguista 🤲</div>
        <div class="brand-sub">Practice sign language in a playful, mobile-friendly way.</div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================
# Home
# =========================
if st.session_state.page == "home":
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Learn sign language<br>the fun way ✨</div>
            <div class="hero-text">
                Practice with live AI detection and playful mini challenges.
                It’s friendly, simple, and designed to feel good on your phone.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card(
        "How it works 🌈",
        "Open Freestyle to explore your sign prediction in real time, or try Stage Mode to complete mini sign challenges one by one.",
    )

    c1, c2 = st.columns(2)

    with c1:
        app_link("Freestyle", "demo", "freestyle-link")

    with c2:
        app_link("Stage Mode", "stage", "stage-link")

    st.markdown(
        '<div class="tiny-note">Tip: try to center yourself for an accurate prediction!</div>',
        unsafe_allow_html=True,
    )


# =========================
# Freestyle
# =========================
elif st.session_state.page == "demo":
    st_autorefresh(interval=1800, key="prediction_refresh")

    pill("Freestyle 🎥")

    back_col, title_col = st.columns([0.28, 0.72])

    with back_col:
        if st.button("← Home", use_container_width=True):
            go_home()
            st.rerun()

    with title_col:
        st.markdown(
            """
            <div class="card" style="padding:12px 16px;">
                <div class="card-title">Freestyle Detection</div>
                <div class="card-body">Show your sign to the camera and let the model guess it ✨</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ctx = create_camera_stream("sign-detection")

    if ctx.video_processor:
        with ctx.video_processor.lock:
            result = ctx.video_processor.result.copy()

        current_sign = result.get("display_label") or result.get("prediction", "Collecting...")
        confidence = result.get("confidence", 0.0)
        frames = f"{result.get('frames_collected', 0)}/{result.get('frames_needed', 0)}"

        st.markdown(
            f"""
            <div class="prediction-card">
                <div class="prediction-label">Current prediction</div>
                <div class="prediction-value">{current_sign}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-title">Confidence</div>
                    <div class="stat-value">{confidence:.3f}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-title">Frames</div>
                    <div class="stat-value">{frames}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        card("Top 3 guesses 💫", "The model’s favorite guesses right now.")

        if result.get("top3"):
            for item in result["top3"]:
                st.markdown(
                    f"""
                    <div class="top-item">
                        <span class="top-name">{item['label']}</span>
                        <span class="top-score">{item['confidence']:.3f}</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            card("Warming up ⏳", "Hold your sign a little longer so the model can collect enough frames.")

    else:
        card("Camera not started yet 📷", "Tap Start above to begin freestyle mode.")


# =========================
# Stage Mode
# =========================
elif st.session_state.page == "stage":
    if st.session_state.stage_status == "running":
        st_autorefresh(interval=1000, key="stage_refresh")

    stage = STAGES[st.session_state.stage_index]
    target_sign = stage["target"]
    demo_video = stage.get("video")
    progress_text = f"Stage {st.session_state.stage_index + 1} / {len(STAGES)}"
    camera_key = f"stage-detection-{st.session_state.stage_index}"

    pill("Stage Mode 🎮", "yellow")

    nav1, nav2 = st.columns([0.28, 0.72])

    with nav1:
        if st.button("← Home", use_container_width=True):
            st.session_state.stage_index = 0
            reset_stage()
            go_home()
            st.rerun()

    with nav2:
        st.markdown(
            f"""
            <div class="card" style="padding:12px 16px;">
                <div class="card-title">{progress_text}</div>
                <div class="card-body">Match the target sign before time runs out 💪</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="stage-target">
            <div class="stage-kicker">Target sign</div>
            <div class="stage-word">{target_sign}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.stage_status == "idle":
        show_stage_demo(demo_video, target_sign)
        card("Ready? 🌟", "Watch the demo clip, then press Start and make the sign before the timer hits zero.")

        if st.button("Start Stage", use_container_width=True):
            start_stage()
            st.rerun()

    elif st.session_state.stage_status == "running":
        ctx = create_camera_stream(camera_key)

        if ctx.video_processor:
            with ctx.video_processor.lock:
                result = ctx.video_processor.result.copy()

            detected_sign = result.get("display_label") or result.get("prediction", "Collecting...")
            confidence = result.get("confidence", 0.0)
            is_confident = result.get("is_confident", False)

            elapsed = time.time() - st.session_state.stage_start_time
            time_left = max(0, math.ceil(30 - elapsed))

            if (
                str(detected_sign).strip().lower() == str(target_sign).strip().lower()
                and is_confident
                and confidence >= 0.70
            ):
                st.session_state.stage_status = "passed"
                st.session_state.stage_feedback = SUCCESS_MESSAGES[
                    st.session_state.stage_index % len(SUCCESS_MESSAGES)
                ]
                st.rerun()

            elif elapsed >= 30:
                st.session_state.stage_status = "failed"
                st.session_state.stage_feedback = FAIL_MESSAGES[
                    st.session_state.stage_index % len(FAIL_MESSAGES)
                ]
                st.rerun()

            st.markdown(
                f"""
                <div class="timer-box">
                    <div class="card-title">Time left ⏰</div>
                    <div class="timer-value">{time_left}s</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="prediction-card">
                    <div class="prediction-label">Your current sign</div>
                    <div class="prediction-value">{detected_sign}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="stats-grid">
                    <div class="stat-box">
                        <div class="stat-title">Confidence</div>
                        <div class="stat-value">{confidence:.3f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Target</div>
                        <div class="stat-value">{target_sign}</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            pill("Go go go! 💨", "soft")

        else:
            card("Camera not started yet 📷", "Tap Start on the camera box above if your browser asks for permission.")

    elif st.session_state.stage_status == "passed":
        st.markdown(
            """
            <div class="timer-box">
                <div class="card-title">Status ✅</div>
                <div class="timer-value" style="font-size:1.35rem;">Correct hand sign ✅</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""<div class="feedback-success">{st.session_state.stage_feedback}</div>""",
            unsafe_allow_html=True,
        )

        if not st.session_state.stage_balloons_shown:
            st.balloons()
            st.session_state.stage_balloons_shown = True

        if st.session_state.stage_index < len(STAGES) - 1:
            if st.button("Next Stage", use_container_width=True):
                next_stage()
                st.rerun()
        else:
            st.markdown(
                """<div class="feedback-success">You cleared all stages! 🎉💖</div>""",
                unsafe_allow_html=True,
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Play Again", use_container_width=True):
                    st.session_state.stage_index = 0
                    reset_stage()
                    st.rerun()

            with c2:
                if st.button("Home", use_container_width=True):
                    st.session_state.stage_index = 0
                    reset_stage()
                    go_home()
                    st.rerun()

    elif st.session_state.stage_status == "failed":
        show_stage_demo(demo_video, target_sign)

        st.markdown(
            f"""<div class="feedback-fail">{st.session_state.stage_feedback}</div>""",
            unsafe_allow_html=True,
        )

        if st.button("Try Again", use_container_width=True):
            reset_stage()
            st.rerun()


# =========================
# Bottom nav
# =========================
nav_html = f"""
<div class="nav-wrap">
    <div class="bottom-nav-grid">
        {bottom_nav_item("🏠", "home", "home-link", active=(st.session_state.page == "home"))}
        {bottom_nav_item("🎥", "demo", "demo-link", active=(st.session_state.page == "demo"))}
        {bottom_nav_item("🎮", "stage", "stage-link-nav", active=(st.session_state.page == "stage"))}
    </div>
</div>
"""

st.markdown(nav_html, unsafe_allow_html=True)
