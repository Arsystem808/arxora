import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Arxora",
    page_icon="assets/arxora_diamond.svg",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- Styles ----------
def inject_styles():
    css = Path("assets/style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    # Google font
    st.markdown('<link rel="preconnect" href="https://fonts.googleapis.com">', unsafe_allow_html=True)
    st.markdown('<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>', unsafe_allow_html=True)
    st.markdown('<link href="https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700&display=swap" rel="stylesheet">', unsafe_allow_html=True)

inject_styles()

# ---------- Hero block ----------
hero_html = """

<div class="hero">
  <div class="hero-top">
    <div class="wrap">
      <img class="logo-img" src="assets/arxora_logo.png" alt="Arxora" />
    </div>
  </div>
  <div class="hero-bottom">
    <div class="wrap">
      <span class="tagline">trade smarter.</span>
    </div>
  </div>
</div>
"""
st.markdown(hero_html, unsafe_allow_html=True)

# ---------- Value bullets ----------
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("### Решения, а не индикаторы")
    st.markdown("Чёткий статус (LONG/SHORT/CLOSE/WAIT), уровень входа, 1–2 цели и стоп.")
with col2:
    st.markdown("### Разные горизонты")
    st.markdown("Краткосрок, среднесрок и долгосрок — в одном экране.")
with col3:
    st.markdown("### Дисциплина > прогнозы")
    st.markdown("Главное — риск и исполнение. Минимум шума, максимум ясности.")

st.divider()

st.markdown("#### Пример карточек сигналов (демо)")

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

c1, c2, c3 = st.columns(3)
with c1:
    signal_card("WAIT", "—", "—", "—", "—", "ST", "Ситуация неустойчивая: ждём чёткой реакции цены от ключевой зоны.", 62)
with c2:
    signal_card("LONG", "128.4", "132.0", "134.5", "125.9", "MID", "Покупатели удерживают уровень: ожидаем импульс вверх с ограниченным риском.", 68)
with c3:
    signal_card("SHORT", "151.2", "145.8", "142.3", "154.7", "LT", "Рынок перегрет, приоритет коррекции от области предложения.", 64)

st.divider()
st.markdown("##### Как это работает (кратко)")
st.markdown(
    "- Смотрим **цену и поведение** — без перечисления индикаторов.\n"
    "- Даём **одно действие** и **конкретные уровни**.\n"
    "- Поддерживаем **три горизонта** с единым языком решений."
)

st.markdown(
    '<div class="footer">© Arxora. All rights reserved.</div>',
    unsafe_allow_html=True,
)

\
st.divider()

st.markdown("### Анализ тикера")
with st.form("analyze"):
    ccol1, ccol2, ccol3 = st.columns([2,1,1])
    ticker = ccol1.text_input("Тикер", value="AAPL")
    horizon = ccol2.selectbox("Горизонт", options=["ST","MID","LT"], index=0, help="ST — кратко, MID — средне, LT — долго")
    submitted = st.form_submit_button("Сгенерировать решение")
    if submitted:
        with st.spinner("Считаем..."):
            from services.data import load_ohlc
            from core.strategy import decide
            df = load_ohlc(ticker, horizon)
            dec = decide(df, horizon)
        st.markdown(f"##### {ticker} — {horizon}")
        # Render card via same HTML
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

        signal_card(dec.action, dec.entry, dec.tp1, dec.tp2, dec.sl, horizon, dec.comment, dec.confidence)
