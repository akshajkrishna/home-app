# HOME – Humanity Over Mission Etiquette
# Streamlit demo app (single-file) for hackathon
# ------------------------------------------------------------
# Features:
# 1) AI Therapist (text + optional voice reply), memory per user session
# 2) Mood detection simulator (future wearable integration placeholder)
# 3) Task prioritization with Pomodoro workflow (deadline + duration + max work time)
# 4) Finance: smart wallet, mock investing (S&P 500, Gold, Real Estate, SIPs)
#    Fees: 10% of profits, 1% of assets (annualized, prorated monthly), 0.2% per transaction
#    Revenue allocation: 30% to investments pool, 10% bank, 10% marketing (modeled in dashboard)
# 5) Escalation to human therapist (simulated)
# 6) $10/month subscription banner
#
# How to run locally:
#   pip install streamlit pydantic python-dateutil numpy pandas
#   streamlit run HOME_app_streamlit.py
# Optional (for TTS voice replies):
#   pip install gTTS
# ------------------------------------------------------------

import time
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import random
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import streamlit as st
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from pydantic import BaseModel
from gtts import gTTS
import os
# -------------------------------
# Voice Function
# -------------------------------
def speak_text(text, filename="response.mp3"):
    """Convert text to speech and save as mp3"""
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# -------------------------------
# User Profile (Pydantic)
# -------------------------------
class UserProfile(BaseModel):
    name: str
    mood: str
    savings: float
    tasks: list

# -------------------------------
# Mock AI Core Logic
# -------------------------------
def ai_response(user: UserProfile, user_message: str):
    if "money" in user_message.lower():
        reply = f"Hey {user.name}, remember you’ve saved ${user.savings:.2f}. Let’s put a part of it into a safe SIP investment."
    elif "task" in user_message.lower():
        reply = f"{user.name}, let’s break your tasks into smaller chunks using the Pomodoro technique. Which one feels urgent today?"
    elif "hello" in user_message.lower():
        reply = f"Hello {user.name}! How are you feeling right now?"
    else:
        reply = f"I hear you, {user.name}. You’re not alone. Let’s work through this together."
    return reply

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="HOME - Humanity Over Mission Etiquette", layout="centered")

st.title("🏠 HOME: Humanity Over Mission Etiquette")
st.write("Your AI companion for **mental health**, **task management**, and **financial wellness**.")

# Voice toggle
voice_enabled = st.checkbox("🔊 Enable Voice Replies", value=True)

# User input form
with st.form("user_form"):
    name = st.text_input("What’s your name?", "Bob")
    mood = st.selectbox("How are you feeling?", ["😊 Happy", "😟 Anxious", "😔 Depressed", "😐 Neutral"])
    savings = st.number_input("Your current savings ($)", min_value=0.0, value=100.0, step=10.0)
    tasks = st.text_area("List your tasks (comma separated)", "Pay bills, Finish report, Buy groceries")
    submitted = st.form_submit_button("Start HOME")

# Session state to remember conversation history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if submitted:
    user = UserProfile(name=name, mood=mood, savings=savings, tasks=tasks.split(","))
    st.success(f"Welcome {user.name}! HOME is here for you. 💙")

    # Chat section
    user_message = st.text_input("💬 Talk to HOME:", "Hello")
    if st.button("Send"):
        # Add user message to history
        st.session_state.chat_history.append(("You", user_message))

        # Get AI reply
        reply = ai_response(user, user_message)
        st.session_state.chat_history.append(("HOME", reply))

        # Play voice if enabled
        if voice_enabled:
            audio_file = speak_text(reply)
            st.audio(audio_file, format="audio/mp3")

# Display chat history like WhatsApp
if st.session_state.chat_history:
    st.subheader("💭 Conversation")
    for sender, message in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"<div style='text-align:right; color:white; background:#25D366; padding:8px; border-radius:12px; margin:5px 0; display:inline-block;'>👤 {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align:left; color:white; background:#075E54; padding:8px; border-radius:12px; margin:5px 0; display:inline-block;'>🤖 {message}</div>", unsafe_allow_html=True)

# -------------------------------
# Voice Function
# -------------------------------
def speak_text(text, filename="response.mp3"):
    """Convert text to speech and save as mp3"""
    tts = gTTS(text=text, lang="en")
    tts.save(filename)
    return filename

# -------------------------------
# User Profile (Pydantic)
# -------------------------------
class UserProfile(BaseModel):
    name: str
    mood: str
    savings: float
    tasks: list

# -------------------------------
# Mock AI Core Logic
# -------------------------------
def ai_response(user: UserProfile, user_message: str):
    if "money" in user_message.lower():
        reply = f"Hey {user.name}, remember you’ve saved ${user.savings:.2f}. Let’s put a part of it into a safe SIP investment."
    elif "task" in user_message.lower():
        reply = f"{user.name}, let’s break your tasks into smaller chunks using the Pomodoro technique. Which one feels urgent today?"
    elif "hello" in user_message.lower():
        reply = f"Hello {user.name}! How are you feeling right now?"
    else:
        reply = f"I hear you, {user.name}. You’re not alone. Let’s work through this together."
    return reply

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="HOME - Humanity Over Mission Etiquette", layout="centered")

st.title("🏠 HOME: Humanity Over Mission Etiquette")
st.write("Your AI companion for **mental health**, **task management**, and **financial wellness**.")
# -----------------------------
# VOICE ENABLE TOGGLE
# -----------------------------
# Always store this in session_state
if "voice_enabled" not in st.session_state:
    st.session_state.voice_enabled = True

# Sidebar toggle (with unique key)
st.sidebar.header("⚙️ Settings")
st.session_state.voice_enabled = st.sidebar.checkbox(
    "🔊 Enable Voice Replies",
    value=st.session_state.voice_enabled,
    key="voice_checkbox_unique"
)

# User input form
with st.form("user_form"):
    name = st.text_input("What’s your name?", "Bob")
    mood = st.selectbox("How are you feeling?", ["😊 Happy", "😟 Anxious", "😔 Depressed", "😐 Neutral"])
    savings = st.number_input("Your current savings ($)", min_value=0.0, value=100.0, step=10.0)
    tasks = st.text_area("List your tasks (comma separated)", "Pay bills, Finish report, Buy groceries")
    submitted = st.form_submit_button("Start HOME")

if submitted:
    user = UserProfile(name=name, mood=mood, savings=savings, tasks=tasks.split(","))
    st.success(f"Welcome {user.name}! HOME is here for you. 💙")

    # Chat section
    user_message = st.text_input("💬 Talk to HOME:", "Hello")
    if st.button("Send"):
        reply = ai_response(user, user_message)

        # Show text reply
        st.markdown(f"**HOME:** {reply}")

        # Optional voice reply
        if voice_enabled:
            audio_file = speak_text(reply)
            st.audio(audio_file, format="audio/mp3")
# -----------------------------
# Utilities & Session Storage
# -----------------------------
APP_NAME = "HOME – Humanity Over Mission Etiquette"

DEFAULT_POMODORO_MIN = 25
DEFAULT_SHORT_BREAK_MIN = 5
DEFAULT_LONG_BREAK_MIN = 15

FAKE_MARKET_RETURNS_ANNUAL = {
    "S&P 500": 0.08,     # 8% avg annual
    "Gold": 0.04,        # 4%
    "Real Estate": 0.06, # 6%
    "SIPs": 0.07         # 7%
}

FEE_PROFIT_SHARE = 0.10   # 10% of profits
FEE_AUM_ANNUAL = 0.01     # 1% of assets per year
FEE_TXN = 0.002           # 0.2% per transaction

SUBSCRIPTION_USD = 10

# Revenue allocation from $1 subscription
REV_INVEST_POOL = 0.30
REV_BANK = 0.10
REV_MARKETING = 0.10
REV_NET = 0.60
REV_EQUITY_SHARES = 0.05  # from NET (for creating 1M shares)
REV_INVESTOR_PAYOUT = 0.10  # from NET
REV_RETAINED = 1.0 - (REV_EQUITY_SHARES + REV_INVESTOR_PAYOUT)  # remaining from NET


def init_state():
    if "user_name" not in st.session_state:
        st.session_state.user_name = "Bob"  # demo persona by default
    if "mood_log" not in st.session_state:
        st.session_state.mood_log = []  # list of dicts {ts, hrv, hr, gsr, mood}
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # list of (role, text)
    if "tasks" not in st.session_state:
        st.session_state.tasks = []  # each: {title, deadline, est_min, importance(1-5), done}
    if "pomodoro" not in st.session_state:
        st.session_state.pomodoro = {
            "status": "idle",  # idle|focus|break
            "cycle": 0,
            "end_time": None,
            "focus_min": DEFAULT_POMODORO_MIN,
            "short_break_min": DEFAULT_SHORT_BREAK_MIN,
            "long_break_min": DEFAULT_LONG_BREAK_MIN,
        }
    if "wallet" not in st.session_state:
        st.session_state.wallet = {
            "cash": 0.0,
            "investments": {k: 0.0 for k in FAKE_MARKET_RETURNS_ANNUAL.keys()},
            "last_valuation": 0.0,
            "txn_count": 0,
            "history": [],  # list of dicts {ts, cash, total, allocations}
            "weights": {"S&P 500": 0.4, "Gold": 0.2, "Real Estate": 0.2, "SIPs": 0.2}
        }
    if "subscription_active" not in st.session_state:
        st.session_state.subscription_active = True
    if "escalation_requests" not in st.session_state:
        st.session_state.escalation_requests = []
    if "business_metrics" not in st.session_state:
        st.session_state.business_metrics = {
            "users": 100000,  # demo assumption
            "arpu": SUBSCRIPTION_USD,
        }


def fmt_money(x: float) -> str:
    return f"${x:,.2f}"


# -----------------------------
# AI Therapist (rule-based demo)
# -----------------------------
CBT_SUGGESTIONS = [
    "Let’s try a 4-7-8 breath: inhale 4, hold 7, exhale 8. Repeat 4 times.",
    "Name-it-to-tame-it: write the exact thought that’s bothering you. Is it fact or fear?",
    "Try a 2‑minute grounding: notice 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
    "Reframe: what’s a kinder, realistic version of that thought?",
    "Tiny step: what’s the smallest next action that takes < 2 minutes?"
]

POSITIVE_REINFORCERS = [
    "You handled that better than you think.",
    "Progress > perfection. One small win at a time.",
    "Your feelings make sense. Let’s make them easier to carry.",
]


def ai_reply(user_text: str) -> str:
    """Simple heuristic therapist reply using memory context."""
    text = user_text.lower()
    reply_blocks = []

    # Reflect feelings
    if any(k in text for k in ["anxious", "anxiety", "panic", "overwhelmed", "stress", "stressed"]):
        reply_blocks.append("I hear a lot of stress in what you’re saying. That’s hard — and you’re not alone.")
        reply_blocks.append(random.choice(CBT_SUGGESTIONS))

    if any(k in text for k in ["sad", "low", "depressed", "numb"]):
        reply_blocks.append("Thanks for sharing that. Low mood can sap energy. Let’s keep things gentle.")
        reply_blocks.append(random.choice(CBT_SUGGESTIONS))

    if any(k in text for k in ["money", "bills", "debt", "salary", "broke"]):
        reply_blocks.append("Money anxiety hits hard. We can set up an automatic small saving rule today.")

    if any(k in text for k in ["deadline", "work", "task", "study", "exam", "project"]):
        reply_blocks.append("Let’s break the next task into a 25‑minute focus sprint. I’ll queue it up in your tasks.")

    if not reply_blocks:
        reply_blocks.append("Tell me more. What feels heaviest right now?")

    reply_blocks.append(random.choice(POSITIVE_REINFORCERS))
    return " " .join(reply_blocks)


# -----------------------------
# Mood Detection (simulator for wearable)
# -----------------------------

def detect_mood(hrv_ms: int, hr_bpm: int, gsr_kohm: float) -> str:
    """Very rough heuristic. In real product, replace with ML classifier over sensor time-series."""
    score = 0
    if hrv_ms < 40:  # low HRV
        score += 2
    if hr_bpm > 95:
        score += 2
    if gsr_kohm < 200:  # lower skin resistance -> higher arousal
        score += 1

    if score >= 4:
        return "High Stress"
    if score >= 2:
        return "Elevated"
    return "Calm"


# -----------------------------
# Task Prioritization & Pomodoro
# -----------------------------

def prioritize_tasks(tasks: List[Dict[str, Any]], max_work_min: int) -> List[Dict[str, Any]]:
    """Score tasks by urgency (deadline proximity), importance, and fit within max_work_min."""
    now = datetime.now()
    out = []
    for t in tasks:
        if t.get("done"):
            continue
        deadline = t.get("deadline")
        if isinstance(deadline, str) and deadline:
            try:
                deadline_dt = datetime.fromisoformat(deadline)
            except Exception:
                deadline_dt = now + timedelta(days=7)
        elif isinstance(deadline, datetime):
            deadline_dt = deadline
        else:
            deadline_dt = now + timedelta(days=7)
        days_left = max((deadline_dt - now).total_seconds() / 86400.0, 0.1)
        urgency = 1.0 / days_left
        importance = float(t.get("importance", 3)) / 5.0
        duration_fit = 1.0 if float(t.get("est_min", 30)) <= max_work_min else 0.7
        score = 0.5 * urgency + 0.4 * importance + 0.1 * duration_fit
        ot = dict(t)
        ot["score"] = score
        out.append(ot)
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


def start_timer(kind: str):
    cfg = st.session_state.pomodoro
    now = datetime.now()
    if kind == "focus":
        minutes = cfg["focus_min"]
        cfg["status"] = "focus"
    elif kind == "break":
        # every 4th break is long
        minutes = cfg["long_break_min"] if (cfg["cycle"] + 1) % 4 == 0 else cfg["short_break_min"]
        cfg["status"] = "break"
    else:
        return
    cfg["end_time"] = now + timedelta(minutes=minutes)


def timer_remaining() -> int:
    cfg = st.session_state.pomodoro
    if not cfg["end_time"]:
        return 0
    return max(int((cfg["end_time"] - datetime.now()).total_seconds()), 0)


# -----------------------------
# Finance Engine (mock)
# -----------------------------

def portfolio_value(investments: Dict[str, float]) -> float:
    return float(sum(investments.values()))


def apply_monthly_growth():
    """Apply one month of growth + fees to the portfolio. Append to history."""
    w = st.session_state.wallet
    if sum(w["investments"].values()) <= 0 and w["cash"] <= 0:
        return

    # Monthly growth per asset
    new_investments = {}
    profit = 0.0
    for asset, amt in w["investments"].items():
        monthly_rate = (1 + FAKE_MARKET_RETURNS_ANNUAL[asset]) ** (1/12) - 1
        grown = amt * (1 + monthly_rate)
        profit += max(grown - amt, 0.0)
        new_investments[asset] = grown

    # Fees
    aum = sum(new_investments.values())
    fee_aum_month = aum * ( (1 + FEE_AUM_ANNUAL) ** (1/12) - 1 )  # roughly 1% annual prorated
    fee_profit_share = profit * FEE_PROFIT_SHARE
    total_fees = fee_aum_month + fee_profit_share
    # Deduct fees from cash if possible, else proportionally from investments
    if w["cash"] >= total_fees:
        w["cash"] -= total_fees
    else:
        shortfall = total_fees - w["cash"]
        w["cash"] = 0.0
        # take from investments proportionally
        pv = sum(new_investments.values())
        if pv > 0:
            for asset in new_investments.keys():
                cut = shortfall * (new_investments[asset] / pv)
                new_investments[asset] = max(new_investments[asset] - cut, 0.0)

    w["investments"] = new_investments

    # Record history snapshot
    total = w["cash"] + sum(w["investments"].values())
    snapshot = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "cash": w["cash"],
        "total": total,
        "allocations": dict(w["investments"]) ,
    }
    w["history"].append(snapshot)


def deposit_cash(amount: float):
    if amount <= 0:
        return
    w = st.session_state.wallet
    w["cash"] += amount


def invest_from_cash(amount: float):
    if amount <= 0:
        return
    w = st.session_state.wallet
    if amount > w["cash"]:
        amount = w["cash"]
    if amount <= 0:
        return
    # transaction fee
    fee = amount * FEE_TXN
    net = max(amount - fee, 0)
    w["cash"] -= amount
    w["txn_count"] += 1
    # split by weights
    weights = w["weights"]
    total_weight = sum(weights.values()) or 1.0
    for asset, wt in weights.items():
        w["investments"][asset] += net * (wt / total_weight)


def rebalance(weights: Dict[str, float]):
    w = st.session_state.wallet
    total = sum(w["investments"].values())
    if total <= 0:
        w["weights"] = weights
        return
    # Simple sell-all and re-buy (with fees) for demo
    proceeds = total
    w["investments"] = {k: 0.0 for k in w["investments"].keys()}
    w["cash"] += proceeds
    invest_from_cash(w["cash"])  # invest all aligned to new weights
    w["weights"] = weights


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title=APP_NAME, page_icon="🏠", layout="wide")
init_state()

# Header / Subscription banner
colA, colB = st.columns([0.75, 0.25])
with colA:
    st.title(APP_NAME)
    st.caption("Therapist • Planner • Financial Coach — in one $10/month app. Built on GPT with memory. Future-ready for wearable mood detection.")
with colB:
    if st.session_state.subscription_active:
        st.success(f"Subscription active: ${SUBSCRIPTION_USD}/month")
    else:
        st.warning("Free trial (7 days)")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", [
    "👤 Therapist Chat",
    "📈 Mood Detection",
    "✅ Tasks & Pomodoro",
    "💰 Wallet & Investing",
    "📊 Business Dashboard",
    "🧑‍⚕️ Human Therapist",
    "ℹ️ About & Safety"
])

# -----------------------------
# Page: Therapist Chat
# -----------------------------
if page == "👤 Therapist Chat":
    st.subheader("AI Therapist (Text · Optional Voice)")
    st.write("HOME listens, learns, and supports. Conversations are stored locally in this demo to show memory.")

    # Conversation history
    for role, text in st.session_state.chat_history[-12:]:
        if role == "user":
            st.markdown(f"**You:** {text}")
        else:
            st.markdown(f"**HOME:** {text}")

    user_text = st.text_input("Share what’s on your mind…", key="chat_input")
    col1, col2, col3 = st.columns([0.4, 0.3, 0.3])
    with col1:
        if st.button("Send", use_container_width=True):
            if user_text.strip():
                st.session_state.chat_history.append(("user", user_text))
                reply = ai_reply(user_text)
                st.session_state.chat_history.append(("assistant", reply))
                st.experimental_rerun()
    with col2:
        if st.button("Quick Calm (4‑7‑8)"):
            st.session_state.chat_history.append(("assistant", CBT_SUGGESTIONS[0]))
            st.experimental_rerun()
    with col3:
        if st.button("Add Next Task from Chat"):
            # Auto-extract a tiny next action suggestion
            st.session_state.tasks.append({
                "title": f"Tiny next step from chat @ {datetime.now().strftime('%H:%M')}",
                "deadline": (datetime.now() + timedelta(days=2)).isoformat(),
                "est_min": 25,
                "importance": 3,
                "done": False,
            })
            st.success("Added a tiny next step to Tasks")

# -----------------------------
# Page: Mood Detection
# -----------------------------
elif page == "📈 Mood Detection":
    st.subheader("Wearable Mood Detection (Simulator)")
    st.caption("In production, HOME reads ring/bracelet sensors (HRV, heart rate, skin conductance) and classifies mood. Here we simulate inputs.")

    c1, c2, c3 = st.columns(3)
    with c1:
        hrv = st.slider("HRV (ms)", min_value=20, max_value=120, value=55)
    with c2:
        hr = st.slider("Heart Rate (bpm)", min_value=50, max_value=140, value=85)
    with c3:
        gsr = st.slider("Skin Resistance (kΩ)", min_value=50, max_value=500, value=260)

    mood = detect_mood(hrv, hr, gsr)
    st.metric("Detected Mood", mood)

    if st.button("Save Reading"):
        st.session_state.mood_log.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "hrv": hrv, "hr": hr, "gsr": gsr, "mood": mood
        })
        st.success("Saved. HOME will check in if stress is elevated.")

    if mood in ("Elevated", "High Stress"):
        st.info("HOME: I noticed signs of stress. Want to talk or start a 4‑7‑8 breath?")
        colx, coly = st.columns(2)
        with colx:
            if st.button("Open Chat"):
                st.session_state.chat_history.append(("assistant", "I noticed your stress markers rising. I’m here with you. What’s happening?"))
                st.experimental_rerun()
        with coly:
            if st.button("Start 4‑7‑8 Now"):
                st.session_state.chat_history.append(("assistant", CBT_SUGGESTIONS[0]))
                st.experimental_rerun()

    if st.session_state.mood_log:
        st.write("### Recent Readings")
        df = pd.DataFrame(st.session_state.mood_log)
        st.dataframe(df.tail(20), use_container_width=True)

# -----------------------------
# Page: Tasks & Pomodoro
# -----------------------------
elif page == "✅ Tasks & Pomodoro":
    st.subheader("Task Prioritization & Pomodoro")
    max_work = st.number_input("Max working time for next session (minutes)", min_value=10, max_value=180, value=50)

    with st.expander("Add Task"):
        t_title = st.text_input("Title")
        t_deadline = st.date_input("Deadline", value=date.today() + timedelta(days=3))
        t_est = st.number_input("Estimated minutes", 5, 600, 30)
        t_imp = st.slider("Importance", 1, 5, 3)
        if st.button("Add") and t_title:
            st.session_state.tasks.append({
                "title": t_title,
                "deadline": datetime.combine(t_deadline, datetime.min.time()).isoformat(),
                "est_min": int(t_est),
                "importance": int(t_imp),
                "done": False,
            })
            st.success("Task added")

    # Prioritized list
    prioritized = prioritize_tasks(st.session_state.tasks, max_work)
    st.write("### Suggested Order")
    if not prioritized:
        st.info("No tasks yet. Add one above.")
    for i, t in enumerate(prioritized, 1):
        col1, col2, col3, col4, col5 = st.columns([0.35, 0.2, 0.15, 0.15, 0.15])
        with col1:
            st.markdown(f"**{i}. {t['title']}**")
        with col2:
            st.caption(f"Due: {t['deadline'][:10]}")
        with col3:
            st.caption(f"Est: {t['est_min']} min")
        with col4:
            st.caption(f"Importance: {t['importance']}/5")
        with col5:
            idx = st.session_state.tasks.index(next(x for x in st.session_state.tasks if x['title']==t['title'] and x['deadline']==t['deadline']))
            if st.button("Mark Done", key=f"done_{i}"):
                st.session_state.tasks[idx]["done"] = True
                st.experimental_rerun()

    # Pomodoro controls
    st.write("---")
    st.write("### Pomodoro Timer")
    pcfg = st.session_state.pomodoro
    colp1, colp2, colp3, colp4 = st.columns(4)
    with colp1:
        pcfg["focus_min"] = st.number_input("Focus (min)", 10, 60, pcfg["focus_min"])
    with colp2:
        pcfg["short_break_min"] = st.number_input("Short Break (min)", 3, 20, pcfg["short_break_min"])
    with colp3:
        pcfg["long_break_min"] = st.number_input("Long Break (min)", 10, 45, pcfg["long_break_min"])
    with colp4:
        pass

    colpb1, colpb2, colpb3 = st.columns(3)
    with colpb1:
        if st.button("Start Focus"):
            start_timer("focus")
            st.experimental_rerun()
    with colpb2:
        if st.button("Start Break"):
            start_timer("break")
            st.experimental_rerun()
    with colpb3:
        if st.button("Reset"):
            st.session_state.pomodoro.update({"status":"idle", "end_time":None, "cycle":0})
            st.experimental_rerun()

    rem = timer_remaining()
    if pcfg["status"] != "idle" and rem > 0:
        st.metric(f"{pcfg['status'].capitalize()} ends in", f"{rem//60:02d}:{rem%60:02d}")
        st.progress(1 - rem / (pcfg['focus_min']*60 if pcfg['status']=='focus' else (pcfg['long_break_min']*60 if (pcfg['cycle']+1)%4==0 else pcfg['short_break_min']*60)))
        st.caption("Keep this tab open; timer updates each run.")
        time.sleep(1)
        st.experimental_rerun()
    elif pcfg["status"] != "idle" and rem == 0:
        if pcfg["status"] == "focus":
            pcfg["cycle"] += 1
            start_timer("break")
            st.balloons()
            st.success("Focus complete! Break started.")
        else:
            start_timer("focus")
            st.info("Break complete. Back to focus.")
        st.experimental_rerun()

# -----------------------------
# Page: Wallet & Investing
# -----------------------------
elif page == "💰 Wallet & Investing":
    st.subheader("Wallet, Savings & Investing (with JPMorgan-grade security model)")

    w = st.session_state.wallet
    cash = w["cash"]
    invested = portfolio_value(w["investments"])
    total = cash + invested

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cash", fmt_money(cash))
    c2.metric("Invested", fmt_money(invested))
    c3.metric("Total", fmt_money(total))
    c4.metric("Transactions", w["txn_count"])

    st.write("#### Deposit & Invest")
    dep = st.number_input("Deposit amount ($)", min_value=0.0, value=100.0, step=10.0)
    colw1, colw2 = st.columns(2)
    with colw1:
        if st.button("Deposit to Wallet"):
            deposit_cash(dep)
            st.success(f"Deposited {fmt_money(dep)}")
            st.experimental_rerun()
    with colw2:
        if st.button("Invest from Cash"):
            invest_from_cash(min(dep, st.session_state.wallet["cash"]))
            st.success("Invested from available cash (after 0.2% txn fee)")
            st.experimental_rerun()

    st.write("#### Target Allocation (Rebalance)")
    new_w = {}
    cols = st.columns(4)
    assets = list(FAKE_MARKET_RETURNS_ANNUAL.keys())
    for i, asset in enumerate(assets):
        with cols[i]:
            new_w[asset] = st.number_input(f"{asset} %", 0, 100, int(w["weights"].get(asset, 0)*100))
    total_pct = sum(new_w.values())
    if total_pct == 0:
        st.warning("Set at least one allocation > 0%.")
    else:
        new_w = {k: v/100 for k, v in new_w.items()}
        if st.button("Rebalance"):
            rebalance(new_w)
            st.success("Rebalanced (fees applied)")
            st.experimental_rerun()

    st.write("#### Simulate Monthly Growth")
    months = st.slider("Months to simulate", 1, 24, 1)
    if st.button("Apply Growth"):
        for _ in range(months):
            apply_monthly_growth()
        st.success(f"Applied {months} month(s) of market growth and fees")
        st.experimental_rerun()

    if w["history"]:
        st.write("#### Portfolio History")
        df = pd.DataFrame(w["history"])
        st.line_chart(df.set_index("ts")["total"])
        st.caption("Includes 10% profit share & ~1% AUM annualized fee; 0.2% transaction fees on buys.")

    st.write("---")
    st.write("#### Compliance & Partnerships")
    st.markdown("- Custody & execution modeled via **JPMorgan**‑grade security and compliance.\n- User permission required before any investment.\n- Data encrypted at rest and in transit. No sale of health/financial data.\n- Escalation to human therapist for safety or risk signals.")

# -----------------------------
# Page: Business Dashboard
# -----------------------------
elif page == "📊 Business Dashboard":
    st.subheader("Business Model & Projections")

    users = st.number_input("Active paying users", min_value=0, value=st.session_state.business_metrics["users"], step=1000)
    arpu = st.number_input("Monthly subscription ($)", min_value=1.0, value=float(st.session_state.business_metrics["arpu"]))

    monthly_rev = users * arpu
    yearly_rev = monthly_rev * 12

    # Allocation per $
    alloc_invest = monthly_rev * REV_INVEST_POOL
    alloc_bank = monthly_rev * REV_BANK
    alloc_marketing = monthly_rev * REV_MARKETING
    net = monthly_rev * REV_NET
    net_equity = net * REV_EQUITY_SHARES
    net_investors = net * REV_INVESTOR_PAYOUT
    net_retained = net * REV_RETAINED

    c1, c2, c3 = st.columns(3)
    c1.metric("Monthly Revenue", fmt_money(monthly_rev))
    c2.metric("Yearly Revenue", fmt_money(yearly_rev))
    c3.metric("Net Retained (monthly)", fmt_money(net_retained))

    st.write("#### Allocation per Month")
    alloc_df = pd.DataFrame({
        "Bucket": ["Investment Pool (30%)", "Banking (10%)", "Marketing (10%)", "Net Remaining (60%)", "→ Equity from Net (5%)", "→ Investors from Net (10%)", "→ Retained from Net (45%)"],
        "Amount": [alloc_invest, alloc_bank, alloc_marketing, net, net_equity, net_investors, net_retained]
    })
    st.dataframe(alloc_df, use_container_width=True)

    st.write("#### Finance-side Monetization (Illustrative)")
    deposits = st.number_input("User deposits under custody ($)", min_value=0.0, value=5_000_000.0, step=100_000.0)
    monthly_txn = st.number_input("Monthly transaction volume ($)", min_value=0.0, value=2_000_000.0, step=100_000.0)
    invest_profit = st.number_input("Monthly profit generated on managed assets ($)", min_value=0.0, value=200_000.0, step=10_000.0)

    rev_profit_share = invest_profit * 0.10
    rev_aum_1pct_year = deposits * 0.01
    rev_aum_month = rev_aum_1pct_year / 12
    rev_txn = monthly_txn * 0.002

    fin_rev_total = rev_profit_share + rev_aum_month + rev_txn

    fr_df = pd.DataFrame({
        "Source": ["10% of Profits", "1% of Deposits (annual) → monthly", "0.2% of Transactions"],
        "Monthly Revenue": [rev_profit_share, rev_aum_month, rev_txn]
    })
    st.dataframe(fr_df, use_container_width=True)
    st.metric("Finance-side Monthly Revenue", fmt_money(fin_rev_total))

    st.caption("Equity pool target: create 1,000,000 shares from 5% of NET. Investor payouts 10% of NET.")

# -----------------------------
# Page: Human Therapist
# -----------------------------
elif page == "🧑‍⚕️ Human Therapist":
    st.subheader("Escalate to a Human Therapist")
    st.write("If risk is detected (persistent high stress, self-harm signals, or user request), HOME routes to a licensed professional.")

    reason = st.selectbox("Reason", ["High stress readings", "Panic attack", "Depressed for 2+ weeks", "User request", "Other"])
    notes = st.text_area("Notes for therapist (optional)")
    if st.button("Request Callback"):
        st.session_state.escalation_requests.append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "reason": reason,
            "notes": notes
        })
        st.success("Request sent. A human therapist will contact you.")

    if st.session_state.escalation_requests:
        st.write("### Past Requests")
        st.table(pd.DataFrame(st.session_state.escalation_requests))

# -----------------------------
# Page: About & Safety
# -----------------------------
else:
    st.subheader("About HOME")
    st.markdown(
        """
        **HOME (Humanity Over Mission Etiquette)** combines three pillars:
        - **Mental Health:** AI therapist (text/voice), CBT‑based coping, memory of your context. Escalation to human care when needed.
        - **Productivity:** Priority planning, chunking, and Pomodoro.
        - **Finance:** Smart wallet & investing across S&P 500, Gold, Real Estate, SIPs (with explicit user consent). Fees: 10% of profits, ~1% AUM annually, 0.2% per transaction.

        **Privacy & Security**  
        Data encrypted at rest and in transit. Permissions required for any financial action. Partnering model with **JPMorgan**‑grade custody and compliance. We never sell health or financial data.

        **Disclaimer**  
        This hackathon demo is **not medical or financial advice**. In crises, contact local emergency services.
        """
    )

    st.write("\n\nBuilt with ❤️ using Streamlit.")

