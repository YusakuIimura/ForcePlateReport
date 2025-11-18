import json
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

# ===== 設定読み込み =====
SETTINGS_PATH = Path("./settings.json")
DEFAULT_SPORTS = ["野球", "ゴルフ", "CMJ", "歩行"]

def get_server_address() -> str:
    """
    settings.json の launcher.server_address を読む。
    見つからなければ 'localhost' をデフォルトにする。
    """
    cfg_path = Path(__file__).resolve().parent / "settings.json"
    default = "localhost"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
        addr = cfg.get("launcher", {}).get("server_address", default)
        # 空文字などになっていたときの保険
        if not addr:
            return default
        return str(addr)
    except Exception:
        return default

SERVER_ADDR = get_server_address()


def load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

cfg = load_settings()
landing_cfg = cfg.get("landing", {})
SPORTS = landing_cfg.get("sports", DEFAULT_SPORTS)

# ===== UI =====
st.set_page_config(page_title="競技を選択", layout="centered")
st.title("競技を選んで開始")
st.caption("選んだ競技に一致 or 空欄のデータだけがホーム画面に表示されます。")

sport = st.radio("競技", SPORTS, index=0, horizontal=True)
st.divider()
st.write("選んだらホーム画面へ進んでください。")

# if st.button("➡ ホームへ（解析/管理）を開く", use_container_width=True):
#     js = f"""
#     <script>
#       const sport = encodeURIComponent("{sport}");
#       const url = "http://localhost:8502/?sport=" + sport;
#       // あなたの例と同じパターンでJSを実行（確実に動く）
#       window.open(url, "_blank"); // 同一タブにする場合は '_self'
#     </script>
#     """
#     components.html(js, height=0, scrolling=False)

if st.button("➡ ホームへ（解析/管理）を開く", use_container_width=True):
    js = f"""
    <script>
      const sport = encodeURIComponent("{sport}");
      const url = "http://{SERVER_ADDR}:8502/?sport=" + sport;
      window.open(url, "_blank"); // 同一タブにする場合は '_self'
    </script>
    """
    components.html(js, height=0, scrolling=False)