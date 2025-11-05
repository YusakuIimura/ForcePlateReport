import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# ===== 設定読み込み =====
SETTINGS_PATH = Path("./settings.json")
DEFAULT_SPORTS = ["野球", "ゴルフ", "CMJ", "歩行"]
DEFAULT_HOME_URL = "http://localhost:8502"

def load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

cfg = load_settings()
landing_cfg = cfg.get("landing", {})
SPORTS = landing_cfg.get("sports", DEFAULT_SPORTS)
HOME_URL = landing_cfg.get("home_base_url", DEFAULT_HOME_URL)

# ===== UI =====
st.set_page_config(page_title="競技を選択", layout="centered")
st.title("競技を選んで開始")
st.caption("選んだ競技に一致 or 空欄のデータだけがホーム画面に表示されます。")

sport = st.radio("競技", SPORTS, index=0, horizontal=True)
st.divider()
st.write("選んだらホーム画面へ進んでください。")

if st.button("➡ ホームへ（解析/管理）を開く", use_container_width=True):
    js = f"""
    <script>
      const sport = encodeURIComponent("{sport}");
      const url = "http://localhost:8502/?sport=" + sport;
      // あなたの例と同じパターンでJSを実行（確実に動く）
      window.open(url, "_blank"); // 同一タブにする場合は '_self'
    </script>
    """
    components.html(js, height=0, scrolling=False)