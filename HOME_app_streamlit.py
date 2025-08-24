import streamlit as st
import pandas as pd
import numpy as np
import datetime
import time
import random
import io
from gtts import gTTS
import base64

# ---------------------------
# APP SETUP
# ---------------------------
st.set_page_config(page_title="HOME App", layout="wide")

# Session state
if "tasks" not in st.session_state:
    st.session_state["tasks"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "wallet_balance" not in st.session_state:
    st.session_state["wallet_balance"] = 0.0
if "investments" not in st.session_state:
    st.session_state["investments"] = {"S&P 500": 0, "Real Estate": 0, "Gold": 0}
if "pomodoro_state" not in st.session_state:
    st.session_state["pomodoro_state"] = "idle"
if "pomodoro_timer" not in st.session_state:
    st.session_state["pomodoro_timer"] = 0
if "users" not in st.session_state:
    st.session_state["users"] = 1000
if "arpu" not in st.session_state:
    st.session_state["arpu"] = 10
if "deposits" not in st.session_state:
    st.session_state["deposits"] = 5000
if "monthly_txn" not in st.session_state:
    st.session_state["monthly_txn"] = 20000
if "invest_profit" not in st.session_state:
    st.session_state["invest_profit"] = 1000

# ---------------------------
# FUNCTIONS
# ---------------------------
def generate_voice(text):
    tts = gTTS(text=text, lang="en")
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_bytes = fp.read()
    return base64.b64encode(audio_bytes).decode()

def add_task(title, deadline, est_time, importance):
    st.session_state["tasks"].append({
        "title": title,
        "deadline": deadline,
        "est_time": est_time,
        "importance": importance,
        "done": False
    })

def complete_task(idx):
    st.session_state["tasks"][idx]["done"] = True

def invest_money(asset, amount):
    if st.session_state["wallet_balance"] >= amount:
        st.session_state["wallet_balance"] -= amount
        st.session_state["investments"][asset] += amount
        return True
    return False

def simulate_investments(months):
    growth_rates = {"S&P 500": 0.07, "Real Estate": 0.04, "Gold": 0.02}
    for asset, invested in st.session_state["investments"].items():
        st.session_state["investments"][asset] *= (1 + growth_rates[asset])**(months/12)

# ---------------------------
# SIDEBAR NAV
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", [
    "Therapist Chat", "Mood Detection", "Tasks & Pomodoro",
    "Wallet & Investing", "Business Dashboard", "Human Therapist"
], key="nav_radio")

# ---------------------------
# PAGE: THERAPIST CHAT
# ---------------------------
if page == "Therapist Chat":
    st.title("🧠 AI Therapist Chat")

    voice_enabled = st.checkbox("🔊 Enable Voice Replies", value=True, key="voice_checkbox_chat")
    user_input = st.text_input("How are you feeling today?", key="chat_input")
    send = st.button("Send", key="chat_send")

    if send and user_input:
        reply = f"AI: I hear you saying '{user_input}'. You're not alone — tell me more."
        st.session_state["chat_history"].append(("You", user_input))
        st.session_state["chat_history"].append(("AI", reply))

    for speaker, msg in st.session_state["chat_history"]:
        st.write(f"**{speaker}:** {msg}")

    if voice_enabled and st.session_state["chat_history"]:
        last_msg = st.session_state["chat_history"][-1][1]
        audio_b64 = generate_voice(last_msg)
        st.audio(io.BytesIO(base64.b64decode(audio_b64)), format="audio/mp3")

    st.markdown("#### Quick Actions")
    st.button("Suggest Calm Activity", key="quick_calm")
    st.button("Add Task from Chat", key="add_task_from_chat")

# ---------------------------
# PAGE: MOOD DETECTION
# ---------------------------
elif page == "Mood Detection":
    st.title("📊 Mood Detection Sensors (Simulated)")

    hrv = st.slider("Heart Rate Variability (ms)", 20, 200, 75, key="hrv")
    hr = st.slider("Heart Rate (bpm)", 40, 180, 85, key="hr")
    gsr = st.slider("Galvanic Skin Response", 1, 100, 40, key="gsr")

    st.button("Save Reading", key="save_reading")
    st.button("Open Chat", key="open_chat_from_mood")
    st.button("Start Breathing Exercise", key="start_breath_now")

    st.write("📉 Over time, these readings will train HOME to better detect your stress patterns.")

# ---------------------------
# PAGE: TASKS & POMODORO
# ---------------------------
elif page == "Tasks & Pomodoro":
    st.title("✅ Task Manager with Pomodoro")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        title = st.text_input("Task Title", key="task_title")
    with col2:
        deadline = st.date_input("Deadline", key="task_deadline")
    with col3:
        est_time = st.number_input("Est. Time (min)", 5, 240, 30, key="task_est")
    with col4:
        importance = st.selectbox("Importance", ["Low", "Medium", "High"], key="task_imp")

    if st.button("Add Task", key="task_add"):
        add_task(title, deadline, est_time, importance)

    st.subheader("Your Tasks")
    for idx, task in enumerate(st.session_state["tasks"]):
        cols = st.columns([4,1])
        with cols[0]:
            st.write(f"**{task['title']}** | ⏳ {task['est_time']} min | 📅 {task['deadline']} | 🔥 {task['importance']} | ✅ {task['done']}")
        with cols[1]:
            if not task["done"]:
                # FIXED: use idx instead of i
                if st.button("Mark Done", key=f"done_{idx}"):
                    complete_task(idx)

    st.subheader("⏱ Pomodoro Timer")
    colA, colB, colC = st.columns(3)
    with colA:
        focus = st.number_input("Focus (min)", 10, 90, 25, key="focus_min")
    with colB:
        short_break = st.number_input("Short Break (min)", 1, 15, 5, key="short_break_min")
    with colC:
        long_break = st.number_input("Long Break (min)", 5, 60, 15, key="long_break_min")

    st.button("▶️ Start Focus", key="start_focus")
    st.button("☕ Short Break", key="start_break")
    st.button("🔄 Reset", key="pomodoro_reset")

# ---------------------------
# PAGE: WALLET & INVESTING
# ---------------------------
elif page == "Wallet & Investing":
    st.title("💰 Wallet & Smart Investing")

    st.metric("Wallet Balance", f"${st.session_state['wallet_balance']:.2f}")
    dep_amt = st.number_input("Deposit Amount", 0, 10000, 100, key="dep_amt")
    if st.button("Deposit", key="deposit_btn"):
        st.session_state["wallet_balance"] += dep_amt

    st.subheader("Investments")
    col1, col2 = st.columns(2)
    with col1:
        asset = st.selectbox("Choose Asset", ["S&P 500", "Real Estate", "Gold"], key="asset_choice")
    with col2:
        inv_amt = st.number_input("Amount to Invest", 0, 10000, 50, key="invest_amt")

    if st.button("Invest", key="invest_btn"):
        success = invest_money(asset, inv_amt)
        if success:
            st.success(f"Invested ${inv_amt} into {asset}!")
        else:
            st.error("Not enough balance!")

    st.bar_chart(pd.DataFrame.from_dict(st.session_state["investments"], orient="index"))

    months = st.slider("Simulate Growth (Months)", 1, 120, 12, key="months_sim")
    if st.button("Apply Growth", key="apply_growth"):
        simulate_investments(months)
        st.success("Growth applied!")

    st.write(st.session_state["investments"])

    st.subheader("Auto-Rebalance")
    new_w = {}
    for asset in st.session_state["investments"]:
        new_w[asset] = st.number_input(f"Target % for {asset}", 0, 100, 33, key=f"alloc_{asset}")
    if st.button("Rebalance", key="rebalance_btn"):
        st.info("Auto-rebalance not implemented in demo.")

# ---------------------------
# PAGE: BUSINESS DASHBOARD
# ---------------------------
elif page == "Business Dashboard":
    st.title("📈 Business Projections")

    st.session_state["users"] = st.number_input("Active Users", 100, 1000000, st.session_state["users"], key="users")
    st.session_state["arpu"] = st.number_input("ARPU ($)", 1, 100, st.session_state["arpu"], key="arpu")
    st.session_state["deposits"] = st.number_input("Monthly Deposits ($)", 100, 1000000, st.session_state["deposits"], key="deposits")
    st.session_state["monthly_txn"] = st.number_input("Monthly Transactions", 100, 1000000, st.session_state["monthly_txn"], key="monthly_txn")
    st.session_state["invest_profit"] = st.number_input("Investment Profit ($)", 0, 1000000, st.session_state["invest_profit"], key="invest_profit")

    revenue = st.session_state["users"] * st.session_state["arpu"]
    profit = revenue * 0.6
    st.metric("Revenue", f"${revenue:,.0f}")
    st.metric("Profit", f"${profit:,.0f}")

# ---------------------------
# PAGE: HUMAN THERAPIST
# ---------------------------
elif page == "Human Therapist":
    st.title("🧑‍⚕️ Escalation to Human Therapist")

    st.write("If AI detects you need urgent support, you can escalate here.")
    reason = st.text_area("Reason for escalation", key="esc_reason")
    notes = st.text_area("Additional Notes", key="esc_notes")
    st.button("Request Call", key="esc_request")
