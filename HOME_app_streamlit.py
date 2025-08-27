# HOME_master.py
# HOME — Humanity Over Mission Etiquette (single-file, feature-rich demo)
# - Chatbot (multi-agent heuristics)
# - Personality onboarding & alignment
# - Task manager, prioritizer, Pomodoro timer
# - Mood detection simulator, notifications, crisis escalation
# - Memory (session), export/import
# - TTS (gTTS) safe integration; voice-clone placeholder via uploaded sample
# - Image-based food calorie estimator (placeholder stub)
# - Finance mock engine + suggestions (heuristic)
# - Black & white theme, responsive interactive UI
# ------------------------------------------------------------
# Run:
# pip install streamlit pydantic numpy pandas gTTS pillow
# streamlit run HOME_master.py
# ------------------------------------------------------------

from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Optional
import time
import random
import io
import json
import base64

import numpy as np
import pandas as pd
import streamlit as st
from pydantic import BaseModel

# Optional libraries
try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except Exception:
    GTTS_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except Exception:
    PIL_AVAILABLE = False

# ----------------------------
# App config (call once)
# ----------------------------
APP_NAME = "HOME – Humanity Over Mission Etiquette (Master)"
st.set_page_config(page_title=APP_NAME, page_icon="🏠", layout="wide")

# ----------------------------
# Black & white theme - custom CSS
# ----------------------------
st.markdown(
    """
    <style>
    /* Page background & text */
    .reportview-container, .main, .block-container {
        background: linear-gradient(#ffffff, #ffffff);
        color: #111111;
    }
    /* Card style */
    .stCard {
        background-color: #ffffff !important;
        color: #111111 !important;
    }
    /* Sidebar */
    .css-1d391kg .css-1lcbmhc { background-color: #ffffff; color:#111111; }
    /* Buttons and inputs (high-contrast B/W) */
    .stButton>button, .stCheckbox>div, .stTextInput>div>input, textarea, .stSelectbox>div {
        border-radius: 8px;
        border: 1px solid #111111 !important;
        background: #ffffff !important;
        color: #111111 !important;
    }
    /* Chat bubbles - simple */
    .bubble-user { background:#111111; color:#ffffff; padding:10px; border-radius:10px; display:inline-block; margin:6px 0;}
    .bubble-home { background:#f3f3f3; color:#111111; padding:10px; border-radius:10px; display:inline-block; margin:6px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Utilities & Session initialization
# ----------------------------
def init_state():
    if st.session_state.get("_HOME_inited"):
        return
    st.session_state["_HOME_inited"] = True

    # user & onboarding
    st.session_state.user = {"name": None, "joined": None, "personality": None}
    st.session_state.personality_done = False

    # chat & memory
    st.session_state.chat_history: List[Dict[str, Any]] = []  # {"role":"user"/"home","text":..,"ts":..}
    st.session_state.memory = {"notes": [], "preferences": {}}

    # tasks
    st.session_state.tasks: List[Dict[str,Any]] = []

    # pomodoro
    st.session_state.pomodoro = {"status":"idle","cycle":0,"end_time":None,"focus_min":25,"short_break_min":5,"long_break_min":15}

    # wallet mock
    st.session_state.wallet = {"cash": 200.0, "investments": {"S&P 500":0.0,"Gold":0.0,"Real Estate":0.0,"SIPs":0.0}, "txn_count":0, "history":[]}
    st.session_state.subscription_active = False

    # mood & notifications
    st.session_state.mood_log = []
    st.session_state.notifications = []
    st.session_state.last_notification_ts = None

    # voice sample & clone flag
    st.session_state.voice_sample = None
    st.session_state.voice_clone_ready = False

    # options
    st.session_state.voice_enabled = True
    st.session_state.parroting = False

init_state()

def fmt_money(x: float) -> str:
    return f"${x:,.2f}"

def safe_rerun():
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    except Exception:
        pass

# ----------------------------
# Simple agents (heuristics) — multi-agent architecture
# ----------------------------
def emotion_agent(text: str) -> Dict[str,Any]:
    """Return emotion label, score, signals (simple heuristic)."""
    t = text.lower()
    score = 0
    signals = []
    keywords = {
        "panic":3, "panic attack":3, "suicide":4, "kill myself":4,
        "anxious":2, "anxiety":2, "stressed":2, "depressed":3, "sad":2,
        "happy":-2, "great":-2, "good":-1, "excited":-2
    }
    for k,v in keywords.items():
        if k in t:
            score += v
            signals.append(k)
    # hr heuristics
    label = "Neutral"
    if score >= 4:
        label = "High Stress / Crisis"
    elif score >= 2:
        label = "Elevated"
    elif score <= -1:
        label = "Positive"
    return {"label":label, "score":score, "signals":signals}

def task_agent(text: str) -> Optional[Dict[str,Any]]:
    """Try to propose a tiny task from user text (very basic)."""
    s = text.strip()
    if len(s) < 6:
        return None
    title = s if len(s) <= 120 else s[:120] + "..."
    return {"title": title, "deadline": (datetime.now()+timedelta(days=3)).isoformat(), "est_min":25, "importance":3, "done":False, "created_ts": datetime.now().isoformat()}

def finance_agent(text: str) -> str:
    t = text.lower()
    if "invest" in t or "investment" in t or "save" in t or "savings" in t:
        return "Heuristic: Consider dollar-cost averaging into diversified ETFs (S&P 500). Start small and build habit."
    return "Heuristic: Build a small emergency fund first before taking risk."

# ----------------------------
# Memory & notification helpers
# ----------------------------
def remember(note: str):
    st.session_state.memory["notes"].append({"ts": datetime.now().isoformat(), "note": note})

def notify(title: str, body: str):
    ts = datetime.now().isoformat()
    st.session_state.notifications.append({"ts": ts, "title": title, "body": body})
    st.session_state.last_notification_ts = ts

def notification_check():
    """Check last mood log and push a gentle notification if required."""
    if not st.session_state.mood_log:
        return
    last = st.session_state.mood_log[-1]
    if last.get("mood") in ("Elevated","High Stress"):
        # throttle: 30 minutes
        if st.session_state.last_notification_ts:
            last_ts = datetime.fromisoformat(st.session_state.last_notification_ts)
            if datetime.now() - last_ts < timedelta(minutes=30):
                return
        notify("HOME checking in", f"I noticed elevated stress: {last['mood']}. Want to talk or try a breathing exercise?")

# ----------------------------
# TTS helpers (gTTS)
# ----------------------------
def tts_bytes(text: str) -> Optional[bytes]:
    if not GTTS_AVAILABLE:
        return None
    try:
        t = gTTS(text=text, lang="en")
        fp = io.BytesIO()
        t.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception:
        return None

# Voice clone placeholder
def create_voice_clone(sample_bytes: bytes) -> bool:
    """
    Placeholder: saving sample and setting a flag. Real voice cloning requires a model.
    - Replace this function to call an open-source voice cloning library (Coqui TTS, VITS, etc.)
    """
    if sample_bytes:
        st.session_state.voice_sample = sample_bytes
        st.session_state.voice_clone_ready = True
        return True
    return False

# ----------------------------
# Personality test onboarding
# ----------------------------
PERSONALITY_QUESTIONS = [
    {"k":"planning","q":"Prefer planning ahead or going with the flow?","opts":["Planning","Flow"]},
    {"k":"coping","q":"When stressed do you talk to others or withdraw?","opts":["Talk","Withdraw"]},
    {"k":"style","q":"Prefer concrete steps or big-picture ideas?","opts":["Steps","Big-picture"]},
    {"k":"social","q":"Energized by social or alone time?","opts":["Social","Alone"]},
]

# ----------------------------
# Task prioritizer and Pomodoro
# ----------------------------
def prioritize(tasks: List[Dict[str,Any]], max_work_min: int) -> List[Dict[str,Any]]:
    now = datetime.now()
    scored = []
    for t in tasks:
        if t.get("done"):
            continue
        dstr = t.get("deadline")
        try:
            dd = datetime.fromisoformat(dstr) if dstr else now + timedelta(days=7)
        except Exception:
            dd = now + timedelta(days=7)
        days_left = max((dd - now).total_seconds()/86400.0, 0.1)
        urgency = 1.0/days_left
        importance = float(t.get("importance",3))/5.0
        duration_fit = 1.0 if float(t.get("est_min",30)) <= max_work_min else 0.7
        score = 0.5*urgency + 0.4*importance + 0.1*duration_fit
        c = dict(t); c["score"] = score
        scored.append(c)
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored

def start_timer(kind: str):
    cfg = st.session_state.pomodoro
    now = datetime.now()
    if kind=="focus":
        minutes = cfg.get("focus_min",25)
        cfg["status"] = "focus"
    elif kind=="break":
        minutes = cfg.get("long_break_min",15) if (cfg.get("cycle",0)+1)%4==0 else cfg.get("short_break_min",5)
        cfg["status"] = "break"
    else:
        return
    cfg["end_time"] = (now + timedelta(minutes=minutes)).isoformat()

def remaining_seconds() -> int:
    cfg = st.session_state.pomodoro
    et = cfg.get("end_time")
    if not et:
        return 0
    try:
        rem = int((datetime.fromisoformat(et) - datetime.now()).total_seconds())
        return max(rem, 0)
    except Exception:
        return 0

# ----------------------------
# Image calorie estimator (placeholder)
# ----------------------------
def estimate_calories(img_bytes: bytes) -> Dict[str,Any]:
    """
    Placeholder: return a mocked estimation. Replace with a real model pipeline:
    1) object detection/classification (food items)
    2) portion size estimation (depth/known object)
    3) lookup calories per portion
    """
    items = [("sandwich", 350), ("salad", 180), ("pizza slice", 285), ("apple",95), ("rice bowl",420), ("banana",105)]
    it = random.choice(items)
    est = int(it[1] * random.uniform(0.75,1.25))
    return {"item": it[0], "calories": est, "confidence": f"{random.randint(60,94)}%"}

# ----------------------------
# Finance mock engine
# ----------------------------
FAKE_MARKETS = {"S&P 500":0.08, "Gold":0.04, "Real Estate":0.06, "SIPs":0.07}

def apply_growth_months(months: int):
    w = st.session_state.wallet
    for _ in range(months):
        new = {}
        profit = 0.0
        for k, amt in w["investments"].items():
            monthly = (1 + FAKE_MARKETS.get(k,0.05))**(1/12) - 1
            grown = amt * (1+monthly)
            profit += max(grown - amt, 0)
            new[k] = grown
        # fees: 1% AUM annual prorated + 10% profit share (demo)
        aum = sum(new.values())
        fee_aum = aum * (((1+0.01)**(1/12)) - 1)
        fee_profit = profit * 0.10
        total_fees = fee_aum + fee_profit
        if w["cash"] >= total_fees:
            w["cash"] -= total_fees
        else:
            shortfall = total_fees - w["cash"]
            w["cash"] = 0.0
            pv = sum(new.values())
            if pv > 0:
                for k in new.keys():
                    cut = shortfall * (new[k] / pv)
                    new[k] = max(new[k] - cut, 0.0)
        w["investments"] = new
        snapshot = {"ts": datetime.now().isoformat(), "cash": w["cash"], "total": w["cash"] + sum(new.values()), "alloc": dict(new)}
        w["history"].append(snapshot)

# ----------------------------
# Simple conversational reply generator (ai_reply)
# ----------------------------
CBT_SUGGESTIONS = [
    "Try a 4-7-8 breath: inhale 4, hold 7, exhale 8. Repeat 4 times.",
    "Name-it-to-tame-it: write the exact thought bothering you. Is it fact or fear?",
    "Try grounding: notice 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
    "Tiny step: what is the smallest next action under 2 minutes?"
]

POSITIVE = [
    "Progress over perfection — one small step at a time.",
    "You handled that better than you think.",
    "Your feelings make sense. I'm here with you."
]

def ai_reply(user_text: str) -> str:
    text = user_text.lower()
    blocks = []
    if any(k in text for k in ["panic","panic attack","help me","suicide","kill myself"]):
        blocks.append("I hear you're in serious distress. If you are in immediate danger, please contact local emergency services right now.")
    if any(k in text for k in ["anxious","anxiety","panic","overwhelmed","stressed"]):
        blocks.append("I hear a lot of stress in what you shared. That's hard — you're not alone.")
        blocks.append(random.choice(CBT_SUGGESTIONS))
    if any(k in text for k in ["sad","depressed","low","numb"]):
        blocks.append("Thanks for telling me. Low mood can be heavy. Let's do something gentle.")
        blocks.append(random.choice(CBT_SUGGESTIONS))
    if any(k in text for k in ["money","save","invest","debt","salary","broke"]):
        blocks.append(finance_agent(text))
    if any(k in text for k in ["task","deadline","work","study","project","exam"]):
        blocks.append("Let's break that task into a 25-minute focus sprint and a tiny next step.")
    if not blocks:
        blocks.append("Tell me more — what feels heaviest right now?")
    blocks.append(random.choice(POSITIVE))
    return " ".join(blocks)

# ----------------------------
# Sidebar (global controls)
# ----------------------------
with st.sidebar:
    st.header("HOME Controls")
    st.checkbox("Enable voice replies (gTTS)", value=st.session_state.voice_enabled, key="voice_enabled")
    st.checkbox("Parroting mode (parrot key phrases)", value=st.session_state.parroting, key="parroting")
    st.markdown("---")
    st.write("Memory & data")
    if st.button("Export memory JSON"):
        mem = {"memory": st.session_state.memory, "chat_history": st.session_state.chat_history, "tasks": st.session_state.tasks}
        b = io.BytesIO(json.dumps(mem, default=str).encode())
        st.download_button("Download memory.json", data=b, file_name="home_memory.json", mime="application/json")
    if st.button("Clear session (reset)"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()
    st.markdown("---")
    st.write("About")
    st.caption("Demo app. No medical/financial/legal advice. In crises contact local emergency services.")

# ----------------------------
# Main UI Layout
# ----------------------------
header_col, spacer = st.columns([0.75,0.25])
with header_col:
    st.title(APP_NAME)
    st.write("Mental health support • Productivity • Finance — local-first demo. Use menu to explore features.")

main, right = st.columns([0.72,0.28])

# ------------
# Home summary
# ------------
if st.session_state.get("page") is None:
    st.session_state["page"] = "home"

page = st.radio("Open section", ["Home Overview","Onboarding & Personality","Therapist Chat","Mood Detection","Tasks & Pomodoro","Wallet & Investing","Food Photo (Calories)","Human Therapist","Dev / Hooks"], index=0, horizontal=True)

# ---------- Home Overview ----------
if page == "Home Overview":
    with main:
        st.header("Welcome to HOME")
        if st.session_state.user.get("name"):
            st.write(f"Hello **{st.session_state.user['name']}** — joined {st.session_state.user.get('joined')}")
        else:
            st.write("Welcome — start by going to 'Onboarding & Personality' to introduce yourself.")
        st.markdown("### Quick snapshot")
        c1,c2,c3 = st.columns(3)
        c1.metric("Cash", fmt_money(st.session_state.wallet["cash"]))
        c2.metric("Invested", fmt_money(sum(st.session_state.wallet["investments"].values())))
        c3.metric("Notifications", len(st.session_state.notifications))
        st.markdown("---")
        st.subheader("Recent conversation")
        if st.session_state.chat_history:
            for m in st.session_state.chat_history[-6:]:
                if m["role"]=="user":
                    st.markdown(f"<div class='bubble-user'>You — {m['text']} <br><small>{m['ts'][:19]}</small></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='bubble-home'>HOME — {m['text']} <br><small>{m['ts'][:19]}</small></div>", unsafe_allow_html=True)
        else:
            st.info("No conversation yet. Visit Therapist Chat to begin.")
        st.markdown("---")
        st.write("This demo is full-featured and includes many placeholders for production-grade components (voice clone, image calorie estimation, news scanning, large LLMs). See Dev / Hooks to learn where to extend.")

# ---------- Onboarding ----------
elif page == "Onboarding & Personality":
    with main:
        st.header("Onboarding & Personality Alignment")
        if not st.session_state.user.get("name"):
            with st.form("onboard_form"):
                name = st.text_input("What should I call you?", value="You")
                about = st.text_area("Optional short note about yourself (one line)", value="", height=80)
                submitted = st.form_submit_button("Create profile")
                if submitted:
                    st.session_state.user["name"] = name
                    st.session_state.user["joined"] = datetime.now().isoformat()
                    if about:
                        remember(f"About: {about}")
                    st.success(f"Welcome, {name} — proceed to personality test.")
                    safe_rerun()
        elif not st.session_state.personality_done:
            st.subheader("Personality quick test (helps HOME align tone & suggestions)")
            with st.form("personality_form"):
                answers = {}
                for q in PERSONALITY_QUESTIONS:
                    ans = st.radio(q["q"], q["opts"], key=f"p_{q['k']}")
                    answers[q["k"]] = ans
                submitp = st.form_submit_button("Save personality")
                if submitp:
                    st.session_state.user["personality"] = answers
                    st.session_state.personality_done = True
                    st.success("Saved personality. HOME will align replies to this style.")
                    safe_rerun()
        else:
            st.write("Profile:")
            st.json(st.session_state.user)
            st.write("Memory notes:")
            st.json(st.session_state.memory)
            st.markdown("---")
            st.subheader("Voice sample (optional)")
            st.write("Upload a short recording (5-20s) if you want HOME to *later* create a voice clone. This is only a placeholder: actual cloning requires model integration in Dev / Hooks.")
            uploaded = st.file_uploader("Upload sample (wav/mp3)", type=["wav","mp3","m4a"])
            if uploaded:
                b = uploaded.read()
                st.session_state.voice_sample = b
                st.success("Sample saved in session. To actually build a voice clone go to Dev / Hooks and integrate a voice cloning model.")
            if st.session_state.voice_clone_ready:
                st.success("Voice-clone READY (placeholder flag). Replace with a real model to synthesize voice.")

# ---------- Therapist Chat ----------
elif page == "Therapist Chat":
    with main:
        st.header("AI Therapist Chat")
        st.write("Talk to HOME — multi-agent analysis runs (emotion, task extraction, finance hints). Parroting mode repeats part of your message before reply to show active listening.")
        col_input, col_action = st.columns([0.75,0.25])
        with col_input:
            user_msg = st.text_input("What's on your mind?", key="chatbox")
        with col_action:
            if st.button("Send", key="send_chat"):
                if not user_msg.strip():
                    st.warning("Type a message first.")
                else:
                    # save
                    st.session_state.chat_history.append({"role":"user","text":user_msg,"ts":datetime.now().isoformat()})
                    remember(f"chat: {user_msg[:180]}")
                    # agents
                    emo = emotion_agent(user_msg)
                    tiny = task_agent(user_msg)
                    if tiny:
                        # offer suggested task in memory
                        st.session_state.memory.setdefault("suggested_tasks",[]).append(tiny)
                    fin = finance_agent(user_msg)
                    # compose reply
                    reply = ai_reply(user_msg)
                    # include agent context
                    reply += f"\n\n[Emotion: {emo['label']}]"
                    if st.session_state.parroting:
                        short = " ".join(user_msg.split()[:12])
                        reply = f"I hear you say: \"{short}...\"\n\n{reply}"
                    st.session_state.chat_history.append({"role":"home","text":reply,"ts":datetime.now().isoformat()})
                    # TTS if available and enabled
                    if st.session_state.voice_enabled:
                        mp3 = tts_bytes(reply)
                        if mp3:
                            st.audio(mp3, format="audio/mp3")
                    # Notification check
                    notification_check()
                    safe_rerun()
        st.markdown("---")
        st.subheader("Conversation (recent)")
        for m in st.session_state.chat_history[-12:]:
            if m["role"]=="user":
                st.markdown(f"<div class='bubble-user'>You — {m['text']}<br><small>{m['ts'][:19]}</small></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bubble-home'>HOME — {m['text']}<br><small>{m['ts'][:19]}</small></div>", unsafe_allow_html=True)
        st.markdown("---")
        st.write("Memory suggestions from chat:")
        st.json(st.session_state.memory.get("suggested_tasks", []))

# ---------- Mood Detection ----------
elif page == "Mood Detection":
    with main:
        st.header("Mood Detection (simulator)")
        st.write("Simulate HRV, HR, and skin resistance (GSR). In production, feed wearable data stream.")
        c1,c2,c3 = st.columns(3)
        with c1:
            hrv = st.slider("HRV (ms)", 20, 120, 55, key="sim_hrv")
        with c2:
            hr = st.slider("Heart rate (bpm)", 45, 140, 78, key="sim_hr")
        with c3:
            gsr = st.slider("Skin resistance (kΩ)", 50, 500, 260, key="sim_gsr")
        # simple detector
        def detect_mood(hrv_ms:int, hr_bpm:int, gsr_kohm:float)->str:
            s=0
            if hrv_ms<40: s+=2
            if hr_bpm>95: s+=2
            if gsr_kohm<200: s+=1
            if s>=4: return "High Stress"
            if s>=2: return "Elevated"
            return "Calm"
        mood = detect_mood(hrv, hr, gsr)
        st.metric("Detected mood", mood)
        if st.button("Save reading"):
            st.session_state.mood_log.append({"ts":datetime.now().isoformat(),"hrv":hrv,"hr":hr,"gsr":gsr,"mood":mood})
            st.success("Saved reading.")
            notification_check()
        if st.session_state.mood_log:
            df = pd.DataFrame(st.session_state.mood_log)
            st.dataframe(df.tail(12))
        if mood in ("Elevated","High Stress"):
            st.warning("HOME: I detected elevated stress — consider talking or a breathing exercise.")
            if st.button("Start 4-7-8 breathing"):
                st.session_state.chat_history.append({"role":"home","text":CBT_SUGGESTIONS[0],"ts":datetime.now().isoformat()})
                safe_rerun()

# ---------- Tasks & Pomodoro ----------
elif page == "Tasks & Pomodoro":
    with main:
        st.header("Tasks & Pomodoro")
        st.subheader("Add a task")
        with st.form("task_add_form"):
            title = st.text_input("Title", key="t_title")
            deadline = st.date_input("Deadline", value=date.today()+timedelta(days=3), key="t_deadline")
            est = st.number_input("Est. minutes", 5, 240, 30, key="t_est")
            imp = st.slider("Importance 1-5", 1,5,3, key="t_imp")
            add = st.form_submit_button("Add Task")
            if add and title:
                st.session_state.tasks.append({"title":title,"deadline":datetime.combine(deadline, datetime.min.time()).isoformat(),"est_min":int(est),"importance":int(imp),"done":False,"created_ts":datetime.now().isoformat()})
                st.success("Task added.")
                safe_rerun()
        st.write("### Task list")
        if not st.session_state.tasks:
            st.info("No tasks yet.")
        else:
            for idx,t in enumerate(st.session_state.tasks):
                cols = st.columns([0.6,0.12,0.12,0.16])
                with cols[0]:
                    st.markdown(f"**{t['title']}** — est {t['est_min']} min — due {t['deadline'][:10]}")
                    if t.get("done"):
                        st.caption("✅ Done")
                with cols[1]:
                    st.caption(f"Imp: {t['importance']}/5")
                with cols[2]:
                    if not t.get("done"):
                        if st.button("Mark done", key=f"done_{idx}"):
                            st.session_state.tasks[idx]["done"]=True
                            st.success("Marked done.")
                            safe_rerun()
                with cols[3]:
                    if st.button("Pomodoro", key=f"p_{idx}"):
                        st.session_state.pomodoro["focus_min"] = t.get("est_min",25)
                        start_timer("focus")
                        st.success("Pomodoro started.")
                        safe_rerun()
        st.markdown("---")
        st.subheader("Prioritizer")
        max_work = st.number_input("Max work time (minutes)", 10, 240, 50, key="prio_max")
        suggested = prioritize(st.session_state.tasks, max_work)
        if suggested:
            for i,s in enumerate(suggested[:8],1):
                st.markdown(f"{i}. **{s['title']}** — score {s['score']:.3f} — est {s['est_min']} min — due {s['deadline'][:10]}")
        else:
            st.info("No suggestion (no tasks).")
        st.markdown("---")
        st.subheader("Pomodoro controls")
        pcfg = st.session_state.pomodoro
        a,b,c,d = st.columns(4)
        with a:
            pcfg["focus_min"] = st.number_input("Focus (min)", 5, 90, pcfg.get("focus_min",25), key="pc_focus")
        with b:
            pcfg["short_break_min"] = st.number_input("Short break (min)", 1, 30, pcfg.get("short_break_min",5), key="pc_short")
        with c:
            pcfg["long_break_min"] = st.number_input("Long break (min)", 5, 60, pcfg.get("long_break_min",15), key="pc_long")
        with d:
            st.write(f"Status: {pcfg['status']}")
        p1,p2,p3 = st.columns(3)
        with p1:
            if st.button("Start focus", key="pc_start_focus"):
                start_timer("focus"); safe_rerun()
        with p2:
            if st.button("Start break", key="pc_start_break"):
                start_timer("break"); safe_rerun()
        with p3:
            if st.button("Reset", key="pc_reset"):
                st.session_state.pomodoro.update({"status":"idle","end_time":None,"cycle":0}); safe_rerun()
        # show countdown
        rem = remaining_seconds()
        if st.session_state.pomodoro["status"] != "idle" and rem>0:
            st.metric("Remaining", f"{rem//60:02d}:{rem%60:02d}")
            total_secs = pcfg["focus_min"]*60 if pcfg["status"]=="focus" else (pcfg["long_break_min"]*60 if (pcfg["cycle"]+1)%4==0 else pcfg["short_break_min"]*60)
            progress = 0.0
            if total_secs>0:
                progress = min(max(1 - rem/total_secs, 0.0),1.0)
            st.progress(progress)
            time.sleep(1)
            safe_rerun()
        elif st.session_state.pomodoro["status"] != "idle" and rem==0:
            if pcfg["status"]=="focus":
                pcfg["cycle"] += 1
                start_timer("break")
                st.balloons()
                st.success("Focus complete. Break started.")
            else:
                start_timer("focus")
                st.info("Break complete. Focus started.")
            safe_rerun()

# ---------- Wallet & Investing ----------
elif page == "Wallet & Investing":
    with main:
        st.header("Wallet & Investing (mock)")
        w = st.session_state.wallet
        c1,c2,c3 = st.columns(3)
        c1.metric("Cash", fmt_money(w["cash"]))
        c2.metric("Invested", fmt_money(sum(w["investments"].values())))
        c3.metric("Transactions", w["txn_count"])
        st.markdown("---")
        st.subheader("Deposit & Invest")
        dep = st.number_input("Deposit amount", 0.0, 100000.0, 50.0, step=10.0, key="dep_amt")
        if st.button("Deposit"):
            if dep>0:
                w["cash"] += dep
                st.success(f"Deposited {fmt_money(dep)}")
                safe_rerun()
        st.write("Invest from cash:")
        asset = st.selectbox("Choose asset", list(w["investments"].keys()), key="invest_asset")
        amt = st.number_input("Amount to invest", 0.0, 100000.0, 10.0, step=10.0, key="invest_amt")
        if st.button("Invest now"):
            if amt <= w["cash"] and amt>0:
                fee = amt * 0.002
                net = amt - fee
                w["cash"] -= amt
                w["txn_count"] += 1
                w["investments"][asset] += net
                st.success(f"Invested {fmt_money(net)} in {asset} (fee {fmt_money(fee)})")
                safe_rerun()
            else:
                st.error("Insufficient cash.")
        st.markdown("---")
        st.subheader("Simulate market growth")
        months = st.slider("Months", 1, 60, 1)
        if st.button("Apply growth"):
            apply_growth_months(months)
            st.success(f"Applied {months} months of simulated growth.")
            safe_rerun()
        if w["history"]:
            df = pd.DataFrame(w["history"])
            st.line_chart(df.set_index("ts")["total"])
        st.markdown("---")
        st.subheader("Heuristic suggestions")
        note = st.text_input("Tell HOME about your financial goal (e.g., save for car)")
        if st.button("Get suggestion"):
            st.info(finance_agent(note or "save"))
            remember(f"finance_question: {note[:200]}")

# ---------- Food Photo (Calories) ----------
elif page == "Food Photo (Calories)":
    with main:
        st.header("Photo-based Food Calorie Estimator (placeholder)")
        st.write("Upload a photo. This demo returns a mocked estimate. To implement real estimates, integrate a food-detection + volume pipeline (e.g., open models or cloud vision + depth/scale).")
        uploaded = st.file_uploader("Upload photo (jpg/png)", type=["jpg","jpeg","png"])
        if uploaded:
            b = uploaded.read()
            st.image(b)
            est = estimate_calories(b)
            st.success(f"Detected: {est['item']} — est {est['calories']} kcal (conf {est['confidence']})")
            remember(f"food_estimate: {est}")

# ---------- Human Therapist ----------
elif page == "Human Therapist":
    with main:
        st.header("Escalation to Human Therapist (simulated)")
        st.write("This section simulates requesting a human callback. In production you'd connect to a provider network or telehealth service.")
        reason = st.selectbox("Reason for escalation", ["High stress readings","Panic attack","Depressed for 2+ weeks","User request","Other"])
        notes = st.text_area("Notes (optional)")
        if st.button("Request callback"):
            st.session_state.escalations.append({"ts":datetime.now().isoformat(), "reason":reason, "notes":notes})
            st.success("Callback request recorded (simulated). A human therapist will contact you per demo flow.")
        if st.session_state.escalations:
            st.write("Past requests")
            st.table(pd.DataFrame(st.session_state.escalations))

# ---------- Dev / Hooks ----------
elif page == "Dev / Hooks":
    with main:
        st.header("Dev / Hooks - extend HOME here")
        st.markdown("""
        This page lists the places to integrate real models or free/open APIs:
        - **Voice cloning**: replace `create_voice_clone(sample_bytes)` with integration to Coqui TTS / VITS / Real-Time-Voice-Cloning. Store model artifacts and create a function `synthesize_with_clone(text)` that returns mp3 bytes.
        - **Advanced NLP / Chat**: swap `ai_reply` with an LLM call (e.g., local LLaMA-based model hosted in your infra, or free hosted inference endpoints). Keep privacy rules in place.
        - **Image calories**: implement detection + portion estimation pipeline (open models exist for food classification; volume estimation needs reference objects or depth).
        - **News & Finance scanning**: to scan news you may integrate free news APIs, but respect rate limits; or run local scrapers and models.
        - **Human therapist integration**: hook to a scheduling API or provider network with user's consent.
        """)
        st.subheader("Voice cloning placeholder")
        uploaded = st.file_uploader("Upload voice sample to create placeholder clone", type=["wav","mp3","m4a"])
        if uploaded:
            b = uploaded.read()
            if st.button("Create placeholder clone"):
                ok = create_voice_clone(b)
                if ok:
                    st.success("Voice clone placeholder created (flag). Replace with real model in production.")
        st.markdown("---")
        st.subheader("Test notification check")
        if st.button("Run notification_check() now"):
            notification_check()
            st.success("Ran notification_check(); see notifications panel on right.")

# ----------------------------
# Right column: notifications, quick controls
# ----------------------------
with right:
    st.markdown("### Quick status")
    st.write(f"User: **{st.session_state.user.get('name') or '—'}**")
    st.write(f"Voice clone: {'ready' if st.session_state.voice_clone_ready else 'not ready'}")
    st.write("---")
    st.markdown("#### Notifications")
    if st.session_state.notifications:
        for n in reversed(st.session_state.notifications[-6:]):
            st.write(f"- **{n['title']}** — {n['body']}  <small>{n['ts'][:19]}</small>")
    else:
        st.write("No notifications")
    st.markdown("---")
    st.markdown("#### Quick memory")
    if st.button("Add memory: 'felt good today'"):
        remember("felt good today")
        st.success("Saved memory note.")
    if st.button("Export memory & chat"):
        payload = {"memory":st.session_state.memory, "chat":st.session_state.chat_history, "tasks":st.session_state.tasks}
        b = io.BytesIO(json.dumps(payload, default=str).encode())
        st.download_button("Download JSON", data=b, file_name="home_export.json", mime="application/json")

# End of app
