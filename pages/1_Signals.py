import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Signals — Arxora", page_icon="assets/arxora_diamond.svg", layout="wide")

st.markdown('<link rel="preconnect" href="https://fonts.googleapis.com">', unsafe_allow_html=True)
st.markdown('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>', unsafe_allow_html=True)
st.markdown('<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)
st.markdown(f"<style>{Path('assets/style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.markdown("### Сигналы (демо)")


def signal_card(action, entry, tp1, tp2, sl, horizon, comment, confidence):
    color = {
        "LONG": "#22C55E",
        "SHORT": "#EF4444",
        "WAIT": "#A1A1AA",
        "CLOSE": "#F59E0B",
    }.get(action, "#A1A1AA")
    html = f"""
    <div class="card">
      <div class="card-head">
        <span class="pill" style="background:{color}">{action}</span>
        <span class="horizon">{horizon}</span>
        <span class="confidence">{confidence}%</span>
      </div>
      <div class="levels">
        <div><label>вход</label><b>{entry}</b></div>
        <div><label>цель 1</label><b>{tp1}</b></div>
        <div><label>цель 2</label><b>{tp2}</b></div>
        <div><label>стоп</label><b>{sl}</b></div>
      </div>
      <p class="comment">{comment}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

cols = st.columns(3)
with cols[0]:
    signal_card("WAIT", "—", "—", "—", "—", "ST", "Ждём реакции цены на уровне — вход не даёт преимущества.", 60)
with cols[1]:
    signal_card("LONG", "128.4", "132.0", "134.5", "125.9", "MID", "Покупатели контролируют импульс, риски ограничены.", 68)
with cols[2]:
    signal_card("SHORT", "151.2", "145.8", "142.3", "154.7", "LT", "Перекупленность в цене — базовый сценарий коррекции.", 64)
