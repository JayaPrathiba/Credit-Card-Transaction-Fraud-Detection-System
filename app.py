import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
import random
import time
import plotly.express as px
import plotly.graph_objects as go

# Optional SHAP — falls back gracefully if not installed
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="JP Neural Defense Network",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# SESSION STATE
# =========================================================
if "page" not in st.session_state:
    st.session_state.page = "landing"          # landing -> features -> results
if "last_input" not in st.session_state:
    st.session_state.last_input = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "fraud_history" not in st.session_state:
    st.session_state.fraud_history = pd.DataFrame(
        columns=["time", "amount", "risk_score", "verdict"]
    )
if "detecting" not in st.session_state:
    st.session_state.detecting = False


def go_to(page_name: str):
    st.session_state.page = page_name


# =========================================================
# GLOBAL CSS — cyber theme, fonts, fade transitions, glow box
# =========================================================
def inject_base_css():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Rajdhani:wght@400;500;600;700&family=Share+Tech+Mono&family=JetBrains+Mono:wght@400;600;800&display=swap');

        :root{
            --void:#04050a;
            --void-2:#0a0d14;
            --danger:#ff3b4e;
            --danger-glow:rgba(255,59,78,0.55);
            --cash:#00e676;
            --cash-glow:rgba(0,230,118,0.45);
            --cyan:#00f6ff;
            --cyan-glow:rgba(0,246,255,0.55);
            --gold:#ffd24c;
            --glass:rgba(255,255,255,0.055);
            --glass-border:rgba(0,246,255,0.30);
        }

        #MainMenu, header, footer {visibility:hidden;}
        .stApp{
            background:radial-gradient(circle at 20% 0%, #0b0f18 0%, var(--void) 55%, #010102 100%);
        }
        .block-container{padding-top:1.0rem; padding-bottom:2rem;}

        .fade-wrap{ animation: pageFade 0.65s ease both; }
        @keyframes pageFade{
            0%{opacity:0; filter:blur(6px);}
            100%{opacity:1; filter:blur(0);}
        }

        .binary-rain{position:fixed; inset:0; overflow:hidden; z-index:0; pointer-events:none;}
        .bin-col{
            position:absolute; top:-100%;
            font-family:'JetBrains Mono', monospace; font-size:0.9rem; line-height:1.1;
            white-space:nowrap; animation-name:drop; animation-timing-function:linear; animation-iteration-count:infinite;
        }
        @keyframes drop{
            0%{transform:translateY(-100%);}
            100%{transform:translateY(220%);}
        }

        .circuit-corner{position:fixed; width:340px; height:340px; z-index:0; pointer-events:none; opacity:0.55;}
        .circuit-tl{top:-40px; left:-40px;}
        .circuit-tr{top:-40px; right:-40px; transform:scaleX(-1);}
        .circuit-bl{bottom:-40px; left:-40px; transform:scaleY(-1);}
        .circuit-br{bottom:-40px; right:-40px; transform:scale(-1,-1);}
        .circuit-corner path{ filter: drop-shadow(0 0 6px var(--cyan-glow)); }
        .circuit-pulse{ stroke-dasharray: 6 14; animation: circuitFlow 3.5s linear infinite; }
        @keyframes circuitFlow{ to{ stroke-dashoffset:-200; } }

        .topbar{position:relative; z-index:2; display:flex; align-items:center; justify-content:space-between; padding:0.4rem 0.2rem 0.8rem 0.2rem;}
        .jp-logo{display:flex; align-items:center; gap:0.6rem;}
        .jp-logo-badge{
            width:46px; height:46px; border-radius:12px;
            display:flex; align-items:center; justify-content:center;
            background:linear-gradient(135deg, rgba(0,246,255,0.18), rgba(255,59,78,0.18));
            border:1px solid var(--cyan); box-shadow:0 0 18px var(--cyan-glow), inset 0 0 12px rgba(0,246,255,0.25);
            font-family:'Orbitron',sans-serif; font-weight:900; color:#fff; font-size:1.1rem;
            animation: logoPulse 2.6s ease-in-out infinite;
        }
        @keyframes logoPulse{ 0%,100%{box-shadow:0 0 18px var(--cyan-glow), inset 0 0 12px rgba(0,246,255,0.25);} 50%{box-shadow:0 0 32px var(--cyan-glow), inset 0 0 20px rgba(0,246,255,0.4);} }
        .jp-logo-text{font-family:'Orbitron',sans-serif; font-weight:700; letter-spacing:2px; color:#dffcff; font-size:0.95rem;}
        .jp-logo-sub{font-family:'Share Tech Mono',monospace; font-size:0.65rem; color:var(--cyan); opacity:0.75; letter-spacing:1px;}

        .lightning-box{
            position:relative; z-index:1; overflow:hidden; max-width:820px; margin:0 auto;
            padding:3rem 2.6rem; border-radius:26px;
            background:var(--glass); border:1.5px solid var(--glass-border);
            backdrop-filter:blur(22px) saturate(150%);
            box-shadow:0 0 0 1px rgba(0,246,255,0.08), 0 0 40px rgba(0,246,255,0.25), 0 25px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08);
            text-align:center;
            animation: boxBreathe 4s ease-in-out infinite;
        }
        @keyframes boxBreathe{
            0%,100%{ box-shadow:0 0 0 1px rgba(0,246,255,0.08), 0 0 40px rgba(0,246,255,0.22), 0 25px 80px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08); }
            50%{ box-shadow:0 0 0 1px rgba(0,246,255,0.18), 0 0 65px rgba(0,246,255,0.4), 0 25px 90px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.14); }
        }
        .lightning-box::before{
            content:""; position:absolute; inset:-2px; border-radius:26px; padding:2px;
            background:linear-gradient(120deg, var(--cyan), transparent 30%, transparent 70%, var(--danger));
            -webkit-mask:linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor; mask-composite: exclude;
            opacity:0.8;
        }

        .eyebrow{font-family:'JetBrains Mono',monospace; letter-spacing:3px; color:var(--cyan); font-size:0.78rem; margin-bottom:1rem; text-shadow:0 0 10px var(--cyan-glow);}
        .title{
            font-family:'Orbitron',sans-serif; font-weight:900; font-size:2.5rem; line-height:1.18; letter-spacing:1.5px;
            background:linear-gradient(90deg,#ffffff 0%, var(--cyan) 45%, var(--danger) 100%);
            -webkit-background-clip:text; background-clip:text; color:transparent;
            margin:0 0 1rem 0; text-shadow:0 0 50px rgba(0,246,255,0.25);
        }
        .subtitle{font-family:'Rajdhani',sans-serif; color:rgba(255,255,255,0.78); font-size:1.05rem; letter-spacing:0.5px;}

        .ai-monitor-tag{
            font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:var(--cash);
            border:1px solid rgba(0,230,118,0.4); border-radius:30px; padding:0.4rem 1rem;
            display:inline-flex; align-items:center; gap:0.5rem; background:rgba(0,230,118,0.06);
            box-shadow:0 0 14px rgba(0,230,118,0.25);
        }
        .ai-monitor-tag .pip{width:7px; height:7px; border-radius:50%; background:var(--cash); box-shadow:0 0 8px var(--cash-glow); animation:pulse 1.3s ease-in-out infinite;}
        @keyframes pulse{0%,100%{opacity:1;} 50%{opacity:0.25;}}

        div[data-testid="stButton"] button{
            background:linear-gradient(90deg, var(--cyan), var(--danger));
            border:none; color:#03060a; font-family:'Orbitron',sans-serif; font-weight:800;
            letter-spacing:1.2px; padding:0.9rem 1.3rem; border-radius:14px;
            box-shadow:0 0 22px var(--cyan-glow);
            transition:transform .2s ease, box-shadow .2s ease;
        }
        div[data-testid="stButton"] button:hover{
            transform:translateY(-2px) scale(1.02);
            box-shadow:0 0 36px var(--cyan-glow), 0 0 18px var(--danger-glow);
        }

        .glass-panel{
            position:relative; z-index:1; background:var(--glass); border:1px solid var(--glass-border);
            border-radius:20px; padding:1rem 1.1rem 0.3rem 1.1rem; backdrop-filter:blur(16px);
            box-shadow:0 0 24px rgba(0,246,255,0.12), 0 15px 40px rgba(0,0,0,0.4); margin-bottom:1.1rem;
            transition: transform .25s ease, box-shadow .25s ease;
        }
        .glass-panel:hover{
            transform: translateY(-4px) scale(1.012);
            box-shadow:0 0 42px rgba(0,246,255,0.30), 0 20px 50px rgba(0,0,0,0.5);
        }
        .panel-title{font-family:'Orbitron',sans-serif; font-size:0.9rem; letter-spacing:1.5px; color:var(--cyan); margin-bottom:0.3rem; text-shadow:0 0 8px var(--cyan-glow);}

        section[data-testid="stSidebar"]{
            background:linear-gradient(180deg,#070a10 0%, #05070b 100%);
            border-right:1px solid rgba(0,246,255,0.18);
        }
        section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] label{
            font-family:'JetBrains Mono',monospace; color:#dffcff !important;
        }
        div[data-baseweb="slider"] > div > div{background:linear-gradient(90deg, var(--cash), var(--cyan)) !important;}
        div[data-testid="stSlider"] [role="slider"]{background:var(--gold) !important; box-shadow:0 0 12px rgba(255,210,76,0.7) !important;}

        .legit-pill{color:var(--cash); font-weight:700; text-shadow:0 0 8px var(--cash-glow);}
        .fraud-pill{color:var(--danger); font-weight:700; text-shadow:0 0 8px var(--danger-glow);}

        .fingerprint-wrap{display:flex; justify-content:center; align-items:center; margin:1.2rem 0;}
        .fp-ring{ animation: fpSpin 2.4s linear infinite; transform-origin:center; }
        @keyframes fpSpin{ to{ transform:rotate(360deg); } }
        .fp-scanline{ animation: fpScan 1.6s ease-in-out infinite; }
        @keyframes fpScan{ 0%{ transform:translateY(-40px); opacity:0;} 50%{opacity:1;} 100%{ transform:translateY(40px); opacity:0;} }

        .lock-row{display:flex; justify-content:center; gap:2rem; margin:0.6rem 0 1.4rem 0;}
        .lock-icon{ animation: lockShake 0.9s ease-in-out infinite; }
        @keyframes lockShake{ 0%,100%{transform:rotate(0deg);} 25%{transform:rotate(-6deg);} 75%{transform:rotate(6deg);} }

        .step-track{display:flex; justify-content:center; gap:0.6rem; margin-bottom:1rem;}
        .step-dot{width:34px; height:6px; border-radius:4px; background:rgba(255,255,255,0.15);}
        .step-dot.active{background:linear-gradient(90deg,var(--cyan),var(--danger)); box-shadow:0 0 10px var(--cyan-glow);}

        /* ---- landing page extra UI polish ---- */
        .stat-strip{
            display:flex; justify-content:center; gap:1.4rem; flex-wrap:wrap;
            margin:1.6rem 0 0.4rem 0; position:relative; z-index:1;
        }
        .stat-chip{
            background:var(--glass); border:1px solid var(--glass-border); border-radius:16px;
            padding:0.8rem 1.3rem; min-width:150px; text-align:center; backdrop-filter:blur(14px);
            box-shadow:0 0 18px rgba(0,246,255,0.10);
            transition:transform .25s ease, box-shadow .25s ease;
        }
        .stat-chip:hover{ transform:translateY(-3px); box-shadow:0 0 30px rgba(0,246,255,0.25); }
        .stat-num{font-family:'Orbitron',sans-serif; font-weight:800; font-size:1.5rem; color:#fff; text-shadow:0 0 14px var(--cyan-glow);}
        .stat-label{font-family:'JetBrains Mono',monospace; font-size:0.66rem; letter-spacing:1.5px; color:rgba(255,255,255,0.55); margin-top:0.15rem;}

        .scan-line-overlay{
            position:fixed; inset:0; z-index:1; pointer-events:none;
            background:linear-gradient(180deg, transparent 0%, rgba(0,246,255,0.04) 50%, transparent 100%);
            background-size:100% 220px; animation: scanMove 5.5s linear infinite;
            mix-blend-mode:screen;
        }
        @keyframes scanMove{ 0%{ background-position-y:-220px;} 100%{ background-position-y:120vh;} }

        .corner-tag{
            font-family:'JetBrains Mono',monospace; font-size:0.66rem; letter-spacing:1.5px;
            color:rgba(0,246,255,0.55); position:relative; z-index:1; text-align:center; margin-top:0.4rem;
        }
        .vignette-overlay{
            position:fixed; inset:0; z-index:1; pointer-events:none;
            background:radial-gradient(circle at 50% 38%, transparent 35%, rgba(2,3,6,0.55) 100%);
        }

        /* ---- cyber background image layer behind hooded man ---- */
        .cyber-bg-image{
            position:fixed; inset:0; z-index:-2; pointer-events:none;
            background-size:cover; background-position:center 30%;
            opacity:0.34; filter:saturate(1.3) contrast(1.1);
        }
        .cyber-bg-fade{
            position:fixed; inset:0; z-index:-1; pointer-events:none;
            background:radial-gradient(circle at 50% 35%, rgba(4,5,10,0.25) 0%, rgba(4,5,10,0.88) 70%, #04050a 100%);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# REUSABLE COMPONENTS
# =========================================================
def jp_topbar(subtitle="NEURAL DEFENSE NET"):
    st.markdown(
        f"""
        <div class="topbar">
            <div class="jp-logo">
                <div class="jp-logo-badge">JP</div>
                <div>
                    <div class="jp-logo-text">JP {subtitle}</div>
                    <div class="jp-logo-sub">// ENCRYPTED DEEP-LEARNING FRAUD INTEL SYSTEM</div>
                </div>
            </div>
            <div class="ai-monitor-tag"><span class="pip"></span>AI NEURAL SENTINEL ACTIVE — MONITORING</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def circuit_corners():
    """Root-like glowing circuit pathways in all 4 corners."""
    def path_svg(seed):
        random.seed(seed)
        d = "M10,10 "
        x, y = 10, 10
        for _ in range(5):
            x += random.randint(20, 60)
            y += random.randint(20, 60)
            d += f"L{x},{y} "
        return d

    svg = f"""
    <svg class="circuit-corner circuit-tl" viewBox="0 0 340 340" xmlns="http://www.w3.org/2000/svg">
        <path d="{path_svg(1)}" stroke="#00f6ff" stroke-width="2" fill="none" class="circuit-pulse" opacity="0.8"/>
        <path d="{path_svg(2)}" stroke="#00e676" stroke-width="1.5" fill="none" class="circuit-pulse" opacity="0.5"/>
        <circle cx="10" cy="10" r="4" fill="#00f6ff"/>
    </svg>
    <svg class="circuit-corner circuit-tr" viewBox="0 0 340 340" xmlns="http://www.w3.org/2000/svg">
        <path d="{path_svg(3)}" stroke="#00f6ff" stroke-width="2" fill="none" class="circuit-pulse" opacity="0.8"/>
        <path d="{path_svg(4)}" stroke="#ff3b4e" stroke-width="1.5" fill="none" class="circuit-pulse" opacity="0.5"/>
        <circle cx="10" cy="10" r="4" fill="#00f6ff"/>
    </svg>
    <svg class="circuit-corner circuit-bl" viewBox="0 0 340 340" xmlns="http://www.w3.org/2000/svg">
        <path d="{path_svg(5)}" stroke="#00f6ff" stroke-width="2" fill="none" class="circuit-pulse" opacity="0.8"/>
        <path d="{path_svg(6)}" stroke="#00e676" stroke-width="1.5" fill="none" class="circuit-pulse" opacity="0.5"/>
        <circle cx="10" cy="10" r="4" fill="#00f6ff"/>
    </svg>
    <svg class="circuit-corner circuit-br" viewBox="0 0 340 340" xmlns="http://www.w3.org/2000/svg">
        <path d="{path_svg(7)}" stroke="#00f6ff" stroke-width="2" fill="none" class="circuit-pulse" opacity="0.8"/>
        <path d="{path_svg(8)}" stroke="#ff3b4e" stroke-width="1.5" fill="none" class="circuit-pulse" opacity="0.5"/>
        <circle cx="10" cy="10" r="4" fill="#00f6ff"/>
    </svg>
    """
    st.markdown(svg, unsafe_allow_html=True)


def binary_rain(n=46, color="cash"):
    cols = ""
    palette = ["#00e676", "#00f6ff"] if color == "cash" else ["#00f6ff", "#00e676"]
    for _ in range(n):
        left = random.randint(0, 100)
        delay = round(random.uniform(0, 6), 2)
        duration = round(random.uniform(4, 11), 2)
        text = "".join(random.choice("01") for _ in range(36))
        text_html = "<br>".join(text)
        c = random.choice(palette)
        cols += (
            f'<div class="bin-col" style="left:{left}%; animation-delay:{delay}s; '
            f'animation-duration:{duration}s; color:{c}; opacity:{round(random.uniform(0.18,0.42),2)}; '
            f'text-shadow:0 0 8px {c};">{text_html}</div>'
        )
    st.markdown(f'<div class="binary-rain">{cols}</div>', unsafe_allow_html=True)


def cyber_bg_image(url):
    """Static cyber-security themed background image layer (fixed, full-screen, behind everything)."""
    st.markdown(
        f"""
        <div class="cyber-bg-image" style="background-image:url('{url}');"></div>
        <div class="cyber-bg-fade"></div>
        """,
        unsafe_allow_html=True,
    )


def step_track(active_idx):
    dots = ""
    for i in range(3):
        cls = "step-dot active" if i == active_idx else "step-dot"
        dots += f'<div class="{cls}"></div>'
    st.markdown(f'<div class="step-track">{dots}</div>', unsafe_allow_html=True)


def hooded_man_svg():
    st.markdown(
        """
        <div style="display:flex; justify-content:center; margin:-2rem 0 -1rem 0; opacity:0.9; position:relative; z-index:1;">
        <svg width="260" height="300" viewBox="0 0 260 300" xmlns="http://www.w3.org/2000/svg">
            <defs>
                <radialGradient id="hoodGlow" cx="50%" cy="40%" r="60%">
                    <stop offset="0%" stop-color="#0a3a44"/>
                    <stop offset="100%" stop-color="#020305"/>
                </radialGradient>
            </defs>
            <path d="M40 300 Q40 150 130 130 Q220 150 220 300 Z" fill="#070b10" stroke="#00f6ff" stroke-width="1.2" opacity="0.9"/>
            <path d="M55 230 Q60 110 130 95 Q200 110 205 230 Q205 280 130 290 Q55 280 55 230 Z" fill="url(#hoodGlow)" stroke="#00f6ff" stroke-width="1.3"/>
            <ellipse cx="108" cy="190" rx="6" ry="9" fill="#00f6ff" opacity="0.85">
                <animate attributeName="opacity" values="0.85;0.2;0.85" dur="2.4s" repeatCount="indefinite"/>
            </ellipse>
            <ellipse cx="152" cy="190" rx="6" ry="9" fill="#00f6ff" opacity="0.85">
                <animate attributeName="opacity" values="0.85;0.2;0.85" dur="2.4s" repeatCount="indefinite"/>
            </ellipse>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def fingerprint_animation():
    st.markdown(
        """
        <div class="fingerprint-wrap">
        <svg width="110" height="110" viewBox="0 0 110 110">
            <g class="fp-ring">
                <circle cx="55" cy="55" r="48" fill="none" stroke="#00f6ff" stroke-width="2" stroke-dasharray="6 10" opacity="0.7"/>
            </g>
            <path d="M55 25 C35 25 25 40 25 55 C25 75 35 88 40 92" fill="none" stroke="#00e676" stroke-width="3" stroke-linecap="round"/>
            <path d="M55 32 C40 32 32 44 32 55 C32 70 40 80 45 85" fill="none" stroke="#00e676" stroke-width="2.4" stroke-linecap="round"/>
            <path d="M55 39 C45 39 39 47 39 55 C39 66 45 74 50 78" fill="none" stroke="#00f6ff" stroke-width="2" stroke-linecap="round"/>
            <path d="M70 28 C82 35 85 46 83 58 C81 70 73 80 65 86" fill="none" stroke="#00e676" stroke-width="3" stroke-linecap="round"/>
            <path d="M68 36 C76 41 78 49 76 58 C74 67 68 74 62 79" fill="none" stroke="#00f6ff" stroke-width="2.2" stroke-linecap="round"/>
            <rect x="15" y="15" width="80" height="3" fill="#00f6ff" opacity="0.6" class="fp-scanline"/>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


def lock_animation():
    st.markdown(
        """
        <div class="lock-row">
        <svg width="46" height="46" viewBox="0 0 24 24" class="lock-icon" style="animation-delay:0s;">
            <rect x="4" y="11" width="16" height="10" rx="2" fill="none" stroke="#00f6ff" stroke-width="1.6"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4" fill="none" stroke="#00f6ff" stroke-width="1.6"/>
            <circle cx="12" cy="16" r="1.6" fill="#00f6ff"/>
        </svg>
        <svg width="46" height="46" viewBox="0 0 24 24" class="lock-icon" style="animation-delay:0.3s;">
            <rect x="4" y="11" width="16" height="10" rx="2" fill="none" stroke="#00e676" stroke-width="1.6"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4" fill="none" stroke="#00e676" stroke-width="1.6"/>
            <circle cx="12" cy="16" r="1.6" fill="#00e676"/>
        </svg>
        <svg width="46" height="46" viewBox="0 0 24 24" class="lock-icon" style="animation-delay:0.6s;">
            <rect x="4" y="11" width="16" height="10" rx="2" fill="none" stroke="#ff3b4e" stroke-width="1.6"/>
            <path d="M7 11V7a5 5 0 0 1 10 0v4" fill="none" stroke="#ff3b4e" stroke-width="1.6"/>
            <circle cx="12" cy="16" r="1.6" fill="#ff3b4e"/>
        </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# SPINNING GLOBE — FIXED IMPLEMENTATION
# =========================================================
# Previously this injected <script> tags via st.markdown(), but Streamlit's
# markdown sanitizer does not reliably execute <script> tags, so the globe
# never actually rendered (only the static CSS layers showed up).
#
# Fix: use streamlit.components.v1.html(), which renders inside a REAL
# iframe and DOES execute JavaScript. To make the globe behave like a true
# full-screen fixed background (sitting behind Streamlit's own content
# instead of being stuck inside its small iframe box), the script reaches
# into window.parent.document and attaches the canvas directly to the
# parent page's <body>. This is the standard trick for "fixed background"
# effects in Streamlit.
def spinning_globe_bg(height=0):
    components.html(
        """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
        <script>
        (function(){
            var parentDoc = window.parent.document;

            // avoid double-init across reruns
            if (parentDoc.getElementById('globe-bg-canvas')) {
                return;
            }

            var canvas = parentDoc.createElement('canvas');
            canvas.id = 'globe-bg-canvas';
            canvas.style.position = 'fixed';
            canvas.style.top = '0';
            canvas.style.left = '0';
            canvas.style.width = '100vw';
            canvas.style.height = '100vh';
            canvas.style.zIndex = '0';
            canvas.style.pointerEvents = 'none';
            parentDoc.body.appendChild(canvas);

            function init(){
                var THREE = window.THREE;
                var renderer = new THREE.WebGLRenderer({canvas:canvas, alpha:true, antialias:true});
                renderer.setPixelRatio(window.parent.devicePixelRatio || 1);
                renderer.setSize(window.parent.innerWidth, window.parent.innerHeight);

                var scene = new THREE.Scene();
                var camera = new THREE.PerspectiveCamera(45, window.parent.innerWidth/window.parent.innerHeight, 0.1, 1000);
                camera.position.z = 2.85;

                var globe = new THREE.Mesh(
                    new THREE.SphereGeometry(1, 64, 64),
                    new THREE.MeshPhongMaterial({color:0x050d1f, emissive:0x020810, transparent:true, opacity:0.93})
                );
                scene.add(globe);

                scene.add(new THREE.Mesh(
                    new THREE.SphereGeometry(1.002, 36, 36),
                    new THREE.MeshBasicMaterial({color:0x00f6ff, wireframe:true, transparent:true, opacity:0.07})
                ));
                scene.add(new THREE.Mesh(
                    new THREE.SphereGeometry(1.13, 32, 32),
                    new THREE.MeshBasicMaterial({color:0x0055ff, transparent:true, opacity:0.09, side:THREE.BackSide})
                ));
                scene.add(new THREE.Mesh(
                    new THREE.SphereGeometry(1.18, 32, 32),
                    new THREE.MeshBasicMaterial({color:0x00f6ff, transparent:true, opacity:0.04, side:THREE.BackSide})
                ));

                scene.add(new THREE.AmbientLight(0x111122, 1.6));
                var d = new THREE.DirectionalLight(0x4488ff, 2.2); d.position.set(3,2,5); scene.add(d);
                var rr = new THREE.DirectionalLight(0xff3b4e, 0.55); rr.position.set(-3,-1,-2); scene.add(rr);

                var cities = [
                    [40.71,-74.01],[51.51,-0.13],[35.68,139.65],[22.32,114.17],[1.35,103.82],
                    [48.86,2.35],[37.77,-122.42],[55.76,37.62],[39.90,116.41],[28.61,77.21],
                    [-23.55,-46.63],[19.08,72.88],[31.23,121.47],[-33.87,151.21],[25.20,55.27],
                    [41.01,28.98],[52.52,13.41],[40.42,-3.70],[45.47,9.19],[-26.20,28.05],
                    [30.04,31.24],[37.57,126.98],[13.76,100.50],[3.14,101.69],[6.52,3.38],
                    [43.65,-79.38],[19.43,-99.13],[34.05,-118.24],[41.88,-87.63],[47.61,-122.33],
                    [50.08,14.44],[59.33,18.07],[60.17,24.94],[53.35,-6.26],[47.38,8.54]
                ];

                function ll2v(lat,lon,r){
                    var p=(90-lat)*Math.PI/180, t=(lon+180)*Math.PI/180;
                    return new THREE.Vector3(-r*Math.sin(p)*Math.cos(t), r*Math.cos(p), r*Math.sin(p)*Math.sin(t));
                }

                var dotGeo = new THREE.SphereGeometry(0.013,8,8);
                var cityPos = [];
                var pulseRings = [];
                cities.forEach(function(c){
                    var pos = ll2v(c[0],c[1],1.005);
                    cityPos.push(pos);
                    var col = Math.random()>0.5 ? 0x00e676 : 0x00f6ff;
                    var dot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({color:col}));
                    dot.position.copy(pos);
                    globe.add(dot);

                    var halo = new THREE.Mesh(new THREE.SphereGeometry(0.022,8,8), new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.25}));
                    halo.position.copy(pos);
                    globe.add(halo);

                    var ring = new THREE.Mesh(
                        new THREE.RingGeometry(0.018,0.030,16),
                        new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.75,side:THREE.DoubleSide})
                    );
                    ring.position.copy(pos);
                    ring.lookAt(pos.clone().multiplyScalar(2));
                    ring.userData.phase = Math.random()*Math.PI*2;
                    pulseRings.push(ring);
                    globe.add(ring);
                });

                function makeArc(p1,p2,segs){
                    var pts=[];
                    for(var i=0;i<=segs;i++){
                        var t=i/segs;
                        var v=new THREE.Vector3().lerpVectors(p1,p2,t);
                        var lift=1.0+0.28*Math.sin(Math.PI*t);
                        v.normalize().multiplyScalar(lift);
                        pts.push(v);
                    }
                    return pts;
                }

                var pairs=[
                    [0,1],[0,6],[0,26],[0,28],[1,3],[1,4],[1,16],[1,33],
                    [2,3],[2,8],[2,12],[2,21],[3,4],[4,5],[4,11],[4,22],
                    [5,6],[5,15],[6,27],[7,9],[8,9],[9,11],[10,0],[10,26],
                    [11,12],[12,13],[13,14],[14,15],[16,17],[17,18],[18,19],
                    [19,20],[20,21],[21,22],[22,23],[23,24],[27,28],[28,29],
                    [29,30],[30,31],[31,32],[32,33],[33,34],[34,16],[0,10]
                ];

                pairs.forEach(function(p){
                    var i=p[0],j=p[1];
                    if(i>=cityPos.length||j>=cityPos.length) return;
                    var pts=makeArc(cityPos[i],cityPos[j],44);
                    var curve=new THREE.CatmullRomCurve3(pts);
                    var col=Math.random()>0.5?0x00f6ff:0x00e676;
                    globe.add(new THREE.Mesh(
                        new THREE.TubeGeometry(curve,44,0.0017,4,false),
                        new THREE.MeshBasicMaterial({color:col,transparent:true,opacity:0.55})
                    ));
                });

                var packets=[];
                pairs.slice(0,18).forEach(function(p){
                    var i=p[0],j=p[1];
                    if(i>=cityPos.length||j>=cityPos.length) return;
                    var pts=makeArc(cityPos[i],cityPos[j],44);
                    var dot=new THREE.Mesh(new THREE.SphereGeometry(0.008,6,6), new THREE.MeshBasicMaterial({color:0xffd24c,transparent:true,opacity:0.9}));
                    dot.userData={pts:pts,t:Math.random(),speed:0.004+Math.random()*0.006};
                    globe.add(dot);
                    packets.push(dot);
                });

                var sv=[];
                for(var s=0;s<2400;s++){sv.push((Math.random()-0.5)*80,(Math.random()-0.5)*80,(Math.random()-0.5)*80);}
                var starGeo=new THREE.BufferGeometry();
                starGeo.setAttribute('position',new THREE.Float32BufferAttribute(sv,3));
                scene.add(new THREE.Points(starGeo,new THREE.PointsMaterial({color:0xffffff,size:0.052,transparent:true,opacity:0.65})));

                window.parent.addEventListener('resize',function(){
                    camera.aspect=window.parent.innerWidth/window.parent.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.parent.innerWidth,window.parent.innerHeight);
                });

                var clock=new THREE.Clock();
                function animate(){
                    requestAnimationFrame(animate);
                    var t=clock.getElapsedTime();
                    globe.rotation.y=t*0.09;

                    pulseRings.forEach(function(ring){
                        var s=1.0+0.35*Math.abs(Math.sin(t*1.6+ring.userData.phase));
                        ring.scale.set(s,s,s);
                        ring.material.opacity=0.25+0.6*Math.abs(Math.sin(t*1.9+ring.userData.phase));
                    });

                    packets.forEach(function(pkt){
                        pkt.userData.t+=pkt.userData.speed;
                        if(pkt.userData.t>1) pkt.userData.t=0;
                        var idx=Math.floor(pkt.userData.t*(pkt.userData.pts.length-1));
                        pkt.position.copy(pkt.userData.pts[Math.min(idx,pkt.userData.pts.length-1)]);
                    });

                    renderer.render(scene,camera);
                }
                animate();
            }

            if (window.THREE) {
                init();
            } else {
                var check = setInterval(function(){
                    if (window.THREE) {
                        clearInterval(check);
                        init();
                    }
                }, 50);
            }
        })();
        </script>
        """,
        height=height,
    )


def remove_globe_bg():
    """Call when leaving a page that had the globe, so it doesn't persist underneath other pages."""
    components.html(
        """
        <script>
        (function(){
            var parentDoc = window.parent.document;
            var c = parentDoc.getElementById('globe-bg-canvas');
            if (c) { c.remove(); }
        })();
        </script>
        """,
        height=0,
    )


# =========================================================
# PAGE 1 — LANDING
# =========================================================
def render_landing():
    inject_base_css()
    cyber_bg_image(
        "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?q=80&w=1600&auto=format&fit=crop"
    )
    spinning_globe_bg()   # globe sits in front of the photo, behind the hooded man / panel
    binary_rain(n=50, color="cash")
    circuit_corners()

    st.markdown('<div class="vignette-overlay"></div>', unsafe_allow_html=True)
    st.markdown('<div class="scan-line-overlay"></div>', unsafe_allow_html=True)

    st.markdown('<div class="fade-wrap">', unsafe_allow_html=True)
    jp_topbar("CRYPTO-TRANSACTION NEURAL DEFENSE")
    step_track(0)
    hooded_man_svg()

    st.markdown(
        """
        <div class="landing-wrap" style="display:flex; justify-content:center;">
          <div class="lightning-box">
            <p class="eyebrow">// ENCRYPTED DEEP-LEARNING FRAUD INTEL SYSTEM //</p>
            <h1 class="title">CRYPTO-TRANSACTION<br>NEURAL DEFENSE NETWORK</h1>
            <p class="subtitle">AI Neural Sentinel · Real-time risk scoring · Predictive anomaly detection<br>
            Zero-trust transaction mitigation. Status: <span style="color:#00e676;">SECURED</span></p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div style='height:1.6rem;'></div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1.3, 1, 1.3])
    with c2:
        if st.button("🔓 ACCESS SECURE DASHBOARD", use_container_width=True):
            go_to("features")
            st.rerun()

    st.markdown(
        """
        <div class="stat-strip">
            <div class="stat-chip"><div class="stat-num">2.4M+</div><div class="stat-label">TXNs SCANNED</div></div>
            <div class="stat-chip"><div class="stat-num">99.2%</div><div class="stat-label">DETECTION ACCURACY</div></div>
            <div class="stat-chip"><div class="stat-num">35</div><div class="stat-label">GLOBAL NODES ONLINE</div></div>
            <div class="stat-chip"><div class="stat-num">&lt;40ms</div><div class="stat-label">AVG RESPONSE TIME</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        "<p style='text-align:center; color:rgba(0,246,255,0.45); font-family:\"JetBrains Mono\",monospace; "
        "font-size:0.72rem; margin-top:1.2rem; letter-spacing:1px;'>[SENTINEL_ALERT: CLOAKED OBSERVATION PROTOCOL INITIATED]</p>",
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# PAGE 2 — FEATURE SELECTION (with spinning globe bg)
# =========================================================
def render_features():
    inject_base_css()
    remove_globe_bg()   # globe now lives on the landing page only
    binary_rain(n=30, color="cyan")
    circuit_corners()

    st.markdown('<div class="fade-wrap">', unsafe_allow_html=True)
    jp_topbar("GLOBAL THREAT GRID")
    step_track(1)

    st.markdown(
        """
        <div style="text-align:center; margin-bottom:1rem;">
            <span class="ai-monitor-tag">🛰️ AI IS MONITORING GLOBAL TRANSACTION GRID — LIVE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="lightning-box" style="max-width:900px;">', unsafe_allow_html=True)
    st.markdown('<p class="eyebrow">// TRANSACTION FEATURE INPUT MATRIX //</p>', unsafe_allow_html=True)
    st.markdown('<h2 class="title" style="font-size:1.7rem;">CONFIGURE TRANSACTION VECTOR</h2>', unsafe_allow_html=True)

    payment_options = ["credit card", "debit card", "PayPal", "bank transfer"]
    category_options = ["electronics", "clothing", "toys & games", "home & garden"]
    device_options = ["desktop", "mobile", "tablet"]

    if "amount_input" not in st.session_state:
        st.session_state["amount_input"] = 100.0
        st.session_state["payment_input"] = payment_options[0]
        st.session_state["category_input"] = category_options[0]
        st.session_state["quantity_input"] = 1
        st.session_state["age_input"] = 30
        st.session_state["device_input"] = device_options[0]
        st.session_state["account_age_input"] = 100
        st.session_state["hour_input"] = 12

    rc1, rc2 = st.columns(2)
    with rc1:
        if st.button("🎲 GENERATE RANDOM TRANSACTION", use_container_width=True):
            st.session_state["amount_input"] = round(random.uniform(5, 9500), 2)
            st.session_state["payment_input"] = random.choice(payment_options)
            st.session_state["category_input"] = random.choice(category_options)
            st.session_state["quantity_input"] = random.randint(1, 10)
            st.session_state["age_input"] = random.randint(16, 80)
            st.session_state["device_input"] = random.choice(device_options)
            st.session_state["account_age_input"] = random.randint(1, 365)
            st.session_state["hour_input"] = random.randint(0, 23)
            st.session_state["just_randomized"] = True
            st.rerun()

    if st.session_state.pop("just_randomized", False):
        st.markdown('<p style="color:#00e676; font-family:JetBrains Mono, monospace; font-size:0.8rem;">⚡ Randomized vector generated — fields below updated. Edit freely or proceed.</p>', unsafe_allow_html=True)

    colA, colB = st.columns(2)
    with colA:
        amount = st.number_input("💰 Transaction Amount", 0.0, 10000.0, key="amount_input")
        payment = st.selectbox("💳 Payment Method", payment_options, key="payment_input")
        category = st.selectbox("📦 Product Category", category_options, key="category_input")
        quantity = st.slider("🔢 Quantity", 1, 10, key="quantity_input")
    with colB:
        age = st.number_input("🧑 Customer Age", 10, 100, key="age_input")
        device = st.selectbox("🖥️ Device Used", device_options, key="device_input")
        account_age = st.slider("📅 Account Age (Days)", 1, 365, key="account_age_input")
        hour = st.slider("⏱️ Transaction Hour", 0, 23, key="hour_input")

    st.markdown("<div style='height:0.6rem;'></div>", unsafe_allow_html=True)

    with rc2:
        if st.button("🛡️ PROCEED TO THREAT SCAN →", use_container_width=True, type="primary"):
            st.session_state.last_input = pd.DataFrame([{
                "Transaction Amount": amount,
                "Payment Method": payment,
                "Product Category": category,
                "Quantity": quantity,
                "Customer Age": age,
                "Device Used": device,
                "Account Age Days": account_age,
                "Transaction Hour": hour,
            }])
            go_to("results")
            st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    if st.button("← Back to title screen"):
        go_to("landing")
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# RESULTS PAGE HELPERS
# =========================================================
def risk_color(score):
    if score >= 70:
        return "#ff3b4e"
    if score >= 30:
        return "#ffd24c"
    return "#00e676"


def animated_gauge(target_score):
    placeholder = st.empty()
    steps = 22
    for i in range(steps + 1):
        val = round(target_score * i / steps, 1)
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=val,
                number={"suffix": "%", "font": {"color": "#ffffff", "family": "Orbitron"}},
                title={"text": "NEURAL RISK SCORE", "font": {"color": "#00f6ff", "family": "JetBrains Mono", "size": 15}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "rgba(255,255,255,0.3)"},
                    "bar": {"color": risk_color(target_score)},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(0,230,118,0.30)"},
                        {"range": [30, 70], "color": "rgba(255,210,76,0.30)"},
                        {"range": [70, 100], "color": "rgba(255,59,78,0.30)"},
                    ],
                },
            )
        )
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="white", height=290, margin=dict(l=20, r=20, t=50, b=10))
        placeholder.plotly_chart(fig, use_container_width=True)
        time.sleep(0.016)


def style_status(val):
    color = "#ff3b4e" if val == "Fraud" else "#00e676"
    return f"color:{color}; font-weight:700;"


def explain_prediction(model, input_data):
    """SHAP explanation if available, else simple rule-based fallback."""
    st.markdown('<p class="panel-title">🧠 EXPLAIN PREDICTION</p>', unsafe_allow_html=True)
    if SHAP_AVAILABLE:
        try:
            explainer = shap.Explainer(model.predict, input_data)
            sv = explainer(input_data)
            vals = sv.values[0]
            feats = input_data.columns.tolist()
            df = pd.DataFrame({"Feature": feats, "Impact": vals}).sort_values("Impact", key=abs, ascending=False)
            fig = px.bar(df, x="Impact", y="Feature", orientation="h", color="Impact",
                         color_continuous_scale=["#00e676", "#ffd24c", "#ff3b4e"])
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", height=280)
            st.plotly_chart(fig, use_container_width=True)
            return
        except Exception:
            pass

    amt = input_data["Transaction Amount"].iloc[0]
    acc_age = input_data["Account Age Days"].iloc[0]
    hr = input_data["Transaction Hour"].iloc[0]
    notes = []
    if amt > 2000:
        notes.append(f"⚠️ High transaction amount (${amt:,.0f}) increases risk.")
    if acc_age < 30:
        notes.append(f"⚠️ New account ({acc_age} days old) is a common fraud signal.")
    if hr < 5 or hr > 22:
        notes.append(f"⚠️ Unusual transaction hour ({hr}:00) detected.")
    if not notes:
        notes.append("✅ No strong individual risk signals — score driven by combined pattern.")
    for n in notes:
        st.markdown(f'<p style="font-family:JetBrains Mono, monospace; font-size:0.85rem; color:#dffcff;">{n}</p>', unsafe_allow_html=True)


# =========================================================
# PAGE 3 — RESULTS / DETECTION (binary rain heavy)
# =========================================================
def render_results():
    inject_base_css()
    remove_globe_bg()
    binary_rain(n=70, color="cash")
    circuit_corners()

    st.markdown('<div class="fade-wrap">', unsafe_allow_html=True)
    jp_topbar("THREAT SCAN RESULTS")
    step_track(2)

    h1, h2 = st.columns([2.4, 1])
    with h1:
        st.markdown('<h1 style="font-family:Orbitron,sans-serif; color:#fff; font-size:1.7rem;">🏦 Real-Time Fraud Detection Dashboard</h1>', unsafe_allow_html=True)
        st.markdown(
            "<p style='color:rgba(255,255,255,0.6); font-family:JetBrains Mono,monospace; font-size:0.85rem;'>"
            "Monitor transactions, detect fraud, and analyze risk instantly.</p>",
            unsafe_allow_html=True,
        )
    with h2:
        st.markdown(
            """
            <div style="text-align:right;">
                <span class="ai-monitor-tag">🛰️ CYBER CRIME UNIT — THREAT MONITORING</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    bcol1, bcol2 = st.columns(2)
    with bcol1:
        if st.button("← Back to feature input"):
            go_to("features")
            st.rerun()
    with bcol2:
        if st.button("🏠 Back to title screen"):
            go_to("landing")
            st.rerun()

    # ---- load model ----
    model = None
    model_loaded = True
    try:
        model = joblib.load("fraud_model.pkl")
    except Exception:
        model_loaded = False
        st.warning("⚠️ `fraud_model.pkl` was not found in this folder, so prediction is disabled. The analytics demo below still works.")

    input_data = st.session_state.last_input
    if input_data is None:
        st.info("No transaction vector configured yet — go back to feature input.")
        input_data = pd.DataFrame([{
            "Transaction Amount": 100.0, "Payment Method": "credit card", "Product Category": "electronics",
            "Quantity": 1, "Customer Age": 30, "Device Used": "desktop", "Account Age Days": 100, "Transaction Hour": 12,
        }])

    if st.button("🚨 RUN DETECTION SCAN", use_container_width=True, type="primary"):
        st.session_state.detecting = True

    if st.session_state.detecting:
        if not model_loaded:
            st.error("Cannot run a prediction without `fraud_model.pkl`. Add the file next to app.py and rerun.")
        else:
            try:
                scan_box = st.empty()
                with scan_box.container():
                    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                    st.markdown('<p class="panel-title">🔍 SCANNING TRANSACTION SIGNATURE...</p>', unsafe_allow_html=True)
                    fingerprint_animation()
                    lock_animation()
                    st.markdown('</div>', unsafe_allow_html=True)
                time.sleep(1.1)
                scan_box.empty()

                prediction = model.predict(input_data)[0]
                try:
                    prob = model.predict_proba(input_data)[0][1]
                except Exception:
                    prob = float(prediction)
                risk_score = int(round(prob * 100))
                verdict = "Fraud" if prediction == 1 else "Legit"

                st.session_state.last_result = {"risk_score": risk_score, "verdict": verdict}
                new_row = pd.DataFrame([{
                    "time": time.strftime("%H:%M:%S"),
                    "amount": input_data["Transaction Amount"].iloc[0],
                    "risk_score": risk_score,
                    "verdict": verdict,
                }])
                st.session_state.fraud_history = pd.concat([st.session_state.fraud_history, new_row], ignore_index=True)
                st.session_state.detecting = False

                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown('<p class="panel-title">🎯 FRAUD RISK METER</p>', unsafe_allow_html=True)
                animated_gauge(risk_score)
                st.markdown("</div>", unsafe_allow_html=True)

                if prediction == 1:
                    st.error("🚨 FRAUD DETECTED — TRANSACTION QUARANTINED BY NEURAL SENTINEL.")
                else:
                    st.success("✅ TRANSACTION VERIFIED LEGITIMATE — NEURAL SENTINEL CLEARED ACCESS.")

                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                st.markdown('<p class="panel-title">📊 TRANSACTION DETAILS</p>', unsafe_allow_html=True)
                st.dataframe(input_data, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
                explain_prediction(model, input_data)
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.session_state.detecting = False

    # ---- analytics overview ----
    st.divider()
    st.markdown('<p class="panel-title" style="font-size:1.1rem;">📈 FRAUD ANALYTICS OVERVIEW</p>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        labels = ["Legit", "Fraud"]
        values = [93, 7]
        fig1 = px.pie(values=values, names=labels, title="Fraud Distribution", hole=0.45,
                      color=labels, color_discrete_map={"Legit": "#00e676", "Fraud": "#ff3b4e"})
        fig1.update_traces(textfont_color="white", marker=dict(line=dict(color="#06070a", width=2)))
        fig1.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white", legend_font_color="white")
        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        hours = list(range(24))
        fraud_rate = np.random.rand(24) * 10
        fig2 = px.line(x=hours, y=fraud_rate, title="Fraud Activity by Hour", markers=True)
        fig2.update_traces(line_color="#00f6ff", line_shape="spline", fill="tozeroy",
                            fillcolor="rgba(0,246,255,0.15)", marker=dict(color="#ffd24c", size=6))
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                            xaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Hour"),
                            yaxis=dict(gridcolor="rgba(255,255,255,0.08)", title="Fraud rate (%)"))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ---- fraud history graph ----
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">📜 FRAUD HISTORY (THIS SESSION)</p>', unsafe_allow_html=True)
    if len(st.session_state.fraud_history) > 0:
        hist = st.session_state.fraud_history.copy()
        fig3 = px.scatter(hist, x="time", y="risk_score", color="verdict",
                          color_discrete_map={"Legit": "#00e676", "Fraud": "#ff3b4e"},
                          size=[14]*len(hist))
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font_color="white",
                            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"), yaxis=dict(gridcolor="rgba(255,255,255,0.08)", range=[0,100]))
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(hist, use_container_width=True)
    else:
        st.markdown('<p style="font-family:JetBrains Mono, monospace; color:rgba(255,255,255,0.5); font-size:0.85rem;">No scans run yet this session — run a detection to populate history.</p>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- live alert panel ----
    st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
    st.markdown('<p class="panel-title">🚨 LIVE ALERT PANEL</p>', unsafe_allow_html=True)
    alerts = pd.DataFrame({"Time": ["12:01", "12:10", "12:15"], "Status": ["Legit", "Fraud", "Fraud"], "Amount": [120, 900, 450]})
    try:
        styled = alerts.style.map(style_status, subset=["Status"])
    except AttributeError:
        styled = alerts.style.applymap(style_status, subset=["Status"])
    st.dataframe(styled, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# =========================================================
# ROUTER
# =========================================================
if st.session_state.page == "landing":
    render_landing()
elif st.session_state.page == "features":
    render_features()
else:
    render_results()