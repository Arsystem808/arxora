# Arxora — Streamlit Brand Kit

Минимальный каркас Streamlit-приложения в стиле Arxora (фиолетовый/чёрный, шрифт Manrope).

## Быстрый старт
```bash
pip install -r requirements.txt
streamlit run app.py
```
Откроется домашняя страница с hero-блоком (**Arxora** + *trade smarter.*) и демо карточками сигналов.
Страница **pages/1_Signals.py** — отдельный лист с теми же карточками.

## Структура
```
arxora_streamlit/
  app.py
  pages/1_Signals.py
  assets/style.css
  assets/arxora_diamond.svg
  .streamlit/config.toml
  requirements.txt
```
## Цвета
- Фиолетовый: #5B5BF7
- Чёрный: #0B0D0E
- Белый: #FFFFFF

## Шрифт
- [Manrope](https://manropefont.com) через Google Fonts. При желании поменяй на Söhne/Graphik.

## Примечание
Это чистый фронт-каркас без логики выдачи сигналов. Встраивай свой backend в нужные секции.
