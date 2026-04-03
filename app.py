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
    "Amazing!",
    "Nice one!",
    "Perfect!",
    "You got it!",
]

FAIL_MESSAGES = [
    "Almost there!",
    "Try again!",
    "One more time!",
    "Keep going!",
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


NAV_ICONS = {
    "home": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>',
    "demo": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"></path><circle cx="12" cy="13" r="4"></circle></svg>',
    "stage": '<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>',
}


def bottom_nav_item(icon_key: str, page: str, variant: str = "", active: bool = False):
    active_class = "active" if active else ""
    icon_svg = NAV_ICONS.get(icon_key, icon_key)
    return (
        f'<a class="bottom-nav-link {variant} {active_class}" '
        f'href="/?page={page}" target="_self" aria-label="{page}">'
        f'<span class="bottom-nav-icon">{icon_svg}</span>'
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
                <div class="card-title">Watch the demo</div>
                <div class="card-body">Learn how to sign <b>{target_sign}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        left, center, right = st.columns([1, 2, 1])
        with center:
            st.video(str(video_path))
    else:
        card(
            "Demo clip missing",
            f"Add a video file for {target_sign} in assets/demo_videos.",
        )


# =========================
# Styling
# =========================
st.markdown(
    """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800;900&display=swap');

    :root {
        /* Your existing color palette - KEPT EXACTLY */
        --chocolate-kisses: #45151B;
        --mauvelous: #EA9DAE;
        --royal-orange: #F99256;
        --bittersweet-shimmer: #C74E51;
        --caramel: #FBDE9C;
        --soft-rose: #F8D9E0;
        
        /* Semantic color roles */
        --bg-primary: #FDF8F4;
        --bg-card: #FFFFFF;
        --text-primary: #45151B;
        --text-secondary: #7A4A52;
        --text-muted: #A67580;
        --border-light: rgba(69, 21, 27, 0.08);
        --shadow-sm: 0 2px 8px rgba(69, 21, 27, 0.06);
        --shadow-md: 0 4px 16px rgba(69, 21, 27, 0.08);
        --shadow-lg: 0 8px 24px rgba(69, 21, 27, 0.10);
    }

    /* Hide Streamlit defaults */
    #MainMenu, header, footer,
    [data-testid="stToolbar"],
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
    }

    /* Base typography - Duolingo uses rounded, friendly fonts */
    html, body, [class*="css"] {
        font-family: "Nunito", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }

    a {
        text-decoration: none !important;
    }

    /* Clean, solid background - no gradients */
    .stApp {
        background: var(--bg-primary);
        color: var(--text-primary);
    }

    /* Container - mobile-first */
    .main .block-container {
        max-width: 420px;
        padding: 16px 20px 100px 20px;
    }

    /* Logo - simplified, no heavy effects */
    .logo-image-wrap {
        width: 100%;
        display: flex;
        justify-content: center;
        margin: 8px auto 16px auto;
    }

    .logo-image-box {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: var(--bittersweet-shimmer);
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: var(--shadow-md);
    }

    .logo-image-box img {
        width: 130%;
        height: 130%;
        object-fit: cover;
        display: block;
    }

    /* Top brand header - cleaner, solid */
    .top-brand {
        background: var(--chocolate-kisses);
        color: var(--mauvelous);
        border-radius: 20px;
        padding: 20px;
        box-shadow: var(--shadow-lg);
        margin-bottom: 20px;
        text-align: center;
    }

    .brand-title {
        font-size: 1.5rem;
        font-weight: 800;
        line-height: 1.2;
        letter-spacing: -0.3px;
        color: var(--mauvelous);
    }

    .brand-sub {
        margin-top: 6px;
        color: var(--soft-rose);
        font-size: 0.9rem;
        font-weight: 600;
        line-height: 1.5;
        opacity: 0.9;
    }

    /* Hero card - clean, solid white */
    .hero-card {
        background: var(--bg-card);
        border-radius: 20px;
        padding: 24px;
        border: 1px solid var(--border-light);
        box-shadow: var(--shadow-sm);
        margin-bottom: 16px;
    }

    .hero-title {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text-primary);
        line-height: 1.2;
        margin-bottom: 12px;
    }

    .hero-text {
        color: var(--text-secondary);
        font-size: 0.95rem;
        font-weight: 500;
        line-height: 1.6;
    }

    /* Standard card - solid, minimal */
    .card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 16px 20px;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
        margin-bottom: 16px;
    }

    .card-title {
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--text-primary);
        margin-bottom: 6px;
    }

    .card-body {
        color: var(--text-secondary);
        font-size: 0.9rem;
        font-weight: 500;
        line-height: 1.5;
    }

    /* Pills - solid colors, no heavy shadows */
    .pill {
        display: inline-block;
        background: var(--royal-orange);
        color: white;
        border-radius: 20px;
        padding: 8px 16px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 12px;
        letter-spacing: 0.3px;
    }

    .pill.soft {
        background: var(--soft-rose);
        color: var(--chocolate-kisses);
    }

    .pill.yellow {
        background: var(--caramel);
        color: var(--chocolate-kisses);
    }

    /* Prediction card - solid dark background */
    .prediction-card {
        background: var(--chocolate-kisses);
        color: var(--caramel);
        border-radius: 20px;
        padding: 20px;
        box-shadow: var(--shadow-lg);
        margin-bottom: 16px;
    }

    .prediction-label {
        color: var(--soft-rose);
        font-size: 0.85rem;
        font-weight: 700;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .prediction-value {
        font-size: 1.75rem;
        font-weight: 800;
        line-height: 1.1;
        word-break: break-word;
    }

    /* Stats grid - clean boxes */
    .stats-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 16px;
    }

    .stat-box {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 16px;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
    }

    .stat-title {
        font-size: 0.75rem;
        color: var(--text-muted);
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stat-value {
        font-size: 1.25rem;
        color: var(--text-primary);
        font-weight: 800;
        margin-top: 4px;
    }

    /* Top items list - clean rows */
    .top-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        background: var(--soft-rose);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 8px;
    }

    .top-name {
        font-weight: 700;
        color: var(--text-primary);
        font-size: 0.95rem;
    }

    .top-score {
        font-weight: 800;
        color: var(--bittersweet-shimmer);
        font-size: 0.95rem;
    }

    /* Stage target - solid background */
    .stage-target {
        background: var(--mauvelous);
        border-radius: 20px;
        padding: 24px;
        text-align: center;
        box-shadow: var(--shadow-md);
        margin-bottom: 16px;
    }

    .stage-kicker {
        font-size: 0.8rem;
        font-weight: 700;
        color: var(--chocolate-kisses);
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        opacity: 0.8;
    }

    .stage-word {
        font-size: 2rem;
        font-weight: 800;
        color: var(--chocolate-kisses);
        line-height: 1.1;
    }

    /* Timer box - clean white */
    .timer-box {
        background: var(--bg-card);
        border-radius: 16px;
        text-align: center;
        padding: 16px;
        margin-bottom: 16px;
        box-shadow: var(--shadow-sm);
        border: 1px solid var(--border-light);
    }

    .timer-value {
        font-size: 2rem;
        font-weight: 800;
        color: var(--bittersweet-shimmer);
    }

    /* Feedback messages - solid colors */
    .feedback-success {
        background: var(--bittersweet-shimmer);
        color: white;
        border-radius: 16px;
        padding: 16px 20px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 16px;
    }

    .feedback-fail {
        background: var(--royal-orange);
        color: white;
        border-radius: 16px;
        padding: 16px 20px;
        font-weight: 700;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 16px;
    }

    /* Tiny note */
    .tiny-note {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 12px;
        margin-bottom: 8px;
    }

    /* Video styling - cleaner */
    video {
        width: 100% !important;
        max-width: 280px !important;
        height: auto !important;
        margin: 0 auto !important;
        display: block !important;
        border-radius: 16px !important;
        object-fit: cover !important;
        background: var(--chocolate-kisses) !important;
        box-shadow: var(--shadow-md) !important;
        border: none !important;
    }

    /* Buttons - Duolingo-style pill buttons */
    .stButton > button {
        width: 100%;
        border: none !important;
        border-radius: 16px !important;
        height: 56px !important;
        padding: 0 24px !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.2px !important;
        background: var(--chocolate-kisses) !important;
        color: var(--mauvelous) !important;
        box-shadow: 0 4px 0 #2D0E12 !important;
        transition: all 0.1s ease !important;
        position: relative !important;
        top: 0 !important;
    }

    .stButton > button:hover {
        top: 2px !important;
        box-shadow: 0 2px 0 #2D0E12 !important;
    }

    .stButton > button:active {
        top: 4px !important;
        box-shadow: 0 0 0 #2D0E12 !important;
    }

    .stButton > button:focus,
    .stButton > button:focus-visible {
        outline: none !important;
        box-shadow: 0 4px 0 #2D0E12, 0 0 0 3px rgba(234, 157, 174, 0.4) !important;
    }

    /* App link buttons - Duolingo-style */
    .app-link {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 56px;
        border-radius: 16px;
        font-size: 1rem;
        font-weight: 700;
        letter-spacing: 0.2px;
        text-decoration: none !important;
        transition: all 0.1s ease;
        user-select: none;
        -webkit-tap-highlight-color: transparent;
        position: relative;
        top: 0;
    }

    .app-link:hover {
        top: 2px;
    }

    .app-link:active {
        top: 4px;
    }

    .app-link.freestyle-link {
        background: var(--royal-orange);
        color: white !important;
        box-shadow: 0 4px 0 #C76A3A;
    }

    .app-link.freestyle-link:hover {
        box-shadow: 0 2px 0 #C76A3A;
    }

    .app-link.freestyle-link:active {
        box-shadow: 0 0 0 #C76A3A;
    }

    .app-link.stage-link {
        background: var(--bittersweet-shimmer);
        color: white !important;
        box-shadow: 0 4px 0 #9A3A3C;
    }

    .app-link.stage-link:hover {
        box-shadow: 0 2px 0 #9A3A3C;
    }

    .app-link.stage-link:active {
        box-shadow: 0 0 0 #9A3A3C;
    }

    /* Bottom navigation - clean iOS-style tab bar */
    .nav-wrap {
        position: fixed;
        left: 50%;
        transform: translateX(-50%);
        bottom: 16px;
        width: min(380px, calc(100vw - 32px));
        background: var(--chocolate-kisses);
        border-radius: 24px;
        padding: 8px 12px;
        box-shadow: var(--shadow-lg);
        z-index: 999999;
    }

    .bottom-nav-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 8px;
        align-items: center;
    }

    .bottom-nav-link {
        height: 52px;
        margin: 0 auto;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none !important;
        background: transparent;
        transition: all 0.15s ease;
        user-select: none;
        -webkit-tap-highlight-color: transparent;
        width: 100%;
    }

    .bottom-nav-link:hover {
        background: rgba(255, 255, 255, 0.08);
    }

    .bottom-nav-link:active {
        transform: scale(0.95);
    }

    .bottom-nav-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        opacity: 0.5;
        transition: opacity 0.15s ease;
        color: var(--mauvelous);
    }

    .bottom-nav-icon svg {
        width: 24px;
        height: 24px;
    }

    .bottom-nav-link.active {
        background: var(--mauvelous);
    }

    .bottom-nav-link.active .bottom-nav-icon {
        opacity: 1;
        color: var(--chocolate-kisses);
    }

    /* Responsive adjustments */
    @media (max-width: 480px) {
        .main .block-container {
            padding: 12px 16px 96px 16px;
        }

        .hero-title {
            font-size: 1.4rem;
        }

        .prediction-value,
        .stage-word {
            font-size: 1.5rem;
        }

        .timer-value {
            font-size: 1.75rem;
        }

        .nav-wrap {
            width: calc(100vw - 24px);
            bottom: 12px;
            padding: 6px 10px;
            border-radius: 20px;
        }

        .bottom-nav-link {
            height: 48px;
            border-radius: 14px;
        }

        .bottom-nav-icon svg {
            width: 20px;
            height: 20px;
        }

        video {
            max-width: 260px !important;
        }

        .logo-image-box {
            width: 64px;
            height: 64px;
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
        <div class="brand-title">Linguista</div>
        <div class="brand-sub">Learn sign language the fun way</div>
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
            <div class="hero-title">Practice with AI-powered detection</div>
            <div class="hero-text">
                Get instant feedback as you learn. Start with Freestyle mode to explore, 
                or challenge yourself in Stage Mode.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    card(
        "Choose your mode",
        "Freestyle lets you practice freely. Stage Mode gives you challenges to complete.",
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
        '<div class="tiny-note">Tip: Center yourself in the frame for best results</div>',
        unsafe_allow_html=True,
    )


# =========================
# Freestyle
# =========================
elif st.session_state.page == "demo":
    st_autorefresh(interval=1800, key="prediction_refresh")

    pill("Freestyle")

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
                <div class="card-body">Show your sign to the camera</div>
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

        card("Top predictions", "Most likely signs detected")

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
            card("Warming up", "Hold your sign a bit longer")

    else:
        card("Camera not started", "Tap Start above to begin")


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

    pill("Stage Mode", "yellow")

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
                <div class="card-body">Match the target sign before time runs out</div>
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
        card("Ready?", "Watch the demo, then press Start to begin")

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
                    <div class="card-title">Time left</div>
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

            pill("Keep going", "soft")

        else:
            card("Camera not started", "Allow camera access to begin")

    elif st.session_state.stage_status == "passed":
        st.markdown(
            """
            <div class="timer-box">
                <div class="card-title">Status</div>
                <div class="timer-value" style="font-size:1.35rem;">Correct!</div>
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
                """<div class="feedback-success">You cleared all stages!</div>""",
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
        {bottom_nav_item("home", "home", "home-link", active=(st.session_state.page == "home"))}
        {bottom_nav_item("demo", "demo", "demo-link", active=(st.session_state.page == "demo"))}
        {bottom_nav_item("stage", "stage", "stage-link-nav", active=(st.session_state.page == "stage"))}
    </div>
</div>
"""

st.markdown(nav_html, unsafe_allow_html=True)
