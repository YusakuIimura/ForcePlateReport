# launcher.py — settings.json を読み、Streamlit をその設定で起動
import os
import sys
import json
from pathlib import Path
import streamlit.web.cli as stcli

def base_dir() -> Path:
    # PyInstaller --onefile でも素の実行でもOKな基準ディレクトリ
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

def app_path() -> str:
    return str(base_dir() / "app.py")

def load_settings() -> dict:
    """settings.json を読み込み（なければデフォルト）。型/値の軽いバリデーションも実施。"""
    defaults = {
        "global": {"developmentMode": False},
        "server": {
            "address": "127.0.0.1",
            "port": 8501,
            "headless": False,
            "gatherUsageStats": False,
            "browserServerAddress": "",
            "enableXsrfProtection": True,
        },
    }
    path = base_dir() / "settings.json"
    if not path.exists():
        return defaults

    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[launcher] settings.json の読み込みに失敗: {e}. デフォルトを使用します。")
        return defaults

    # マージ（足りないキーはデフォルト）
    def merge(d, src):
        for k, v in src.items():
            if isinstance(v, dict):
                d.setdefault(k, {})
                merge(d[k], v)
            else:
                d.setdefault(k, v)
        return d

    cfg = merge(cfg, defaults)

    # 軽い型チェック＆補正
    g = cfg.get("global", {})
    s = cfg.get("server", {})
    g["developmentMode"] = bool(g.get("developmentMode", False))
    s["address"] = str(s.get("address", "127.0.0.1"))
    try:
        s["port"] = int(s.get("port", 8501))
    except Exception:
        s["port"] = 8501
    s["headless"] = bool(s.get("headless", False))
    s["gatherUsageStats"] = bool(s.get("gatherUsageStats", False))
    s["browserServerAddress"] = str(s.get("browserServerAddress", "") or "")
    s["enableXsrfProtection"] = bool(s.get("enableXsrfProtection", True))
    return cfg

if __name__ == "__main__":
    cfg = load_settings()
    g = cfg["global"]
    s = cfg["server"]

    # devMode と port の衝突回避（devMode=true の場合、port 指定はエラーになる版がある）
    if g.get("developmentMode", False):
        port_arg = None  # ポートは明示指定しない
    else:
        port_arg = f"--server.port={s['port']}"

    # Streamlit の起動引数を組み立て
    argv = [
        "streamlit", "run", app_path(),
        f"--global.developmentMode={'true' if g['developmentMode'] else 'false'}",
        f"--server.address={s['address']}",
        f"--server.headless={'true' if s['headless'] else 'false'}",
        f"--browser.gatherUsageStats={'true' if s['gatherUsageStats'] else 'false'}",
        f"--server.enableXsrfProtection={'true' if s['enableXsrfProtection'] else 'false'}",
    ]
    if s.get("browserServerAddress"):
        argv.append(f"--browser.serverAddress={s['browserServerAddress']}")
    if port_arg:
        argv.append(port_arg)

    # グローバル環境も明示（古い版への保険）
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENTMODE"] = "true" if g["developmentMode"] else "false"

    # 起動
    sys.argv = argv
    stcli.main()
