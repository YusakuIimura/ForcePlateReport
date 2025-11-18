# launcher.py - settings.json を読み込み、複数の Streamlit アプリを起動するランチャー
import os
import sys
import json
import time
import subprocess
from pathlib import Path


def base_dir() -> Path:
    """
    PyInstaller (--onefile) でも通常実行でも同じように
    「実行ファイルが置いてあるディレクトリ」を返す。
    """
    return Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))


def load_settings() -> dict:
    """
    settings.json を読み込んで launcher 用の設定を返す。
    ない場合や項目が足りない場合はデフォルトを補う。
    """
    defaults = {
        "launcher": {
            "server_address": "127.0.0.1",
            "headless": False,
            "apps": [
                # デフォルト：同じフォルダにある3つのアプリを起動
                {"name": "landing", "script": "Landing.py",    "port": 8501},
                {"name": "home",    "script": "Home.py",       "port": 8502},
                {"name": "player",  "script": "PlayerView.py", "port": 8503},
            ],
        }
    }

    path = base_dir() / "settings.json"
    if not path.exists():
        return defaults

    try:
        with path.open("r", encoding="utf-8") as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[launcher] settings.json の読み込みに失敗しました: {e}")
        print("[launcher] デフォルト設定で起動します。")
        return defaults

    # defaults で不足分を補完
    launcher_cfg = cfg.get("launcher", {})
    for k, v in defaults["launcher"].items():
        launcher_cfg.setdefault(k, v)

    # apps の中身もざっくりバリデーション
    apps = []
    for app in launcher_cfg.get("apps", []):
        try:
            name = str(app.get("name"))
            script = str(app.get("script"))
            port = int(app.get("port"))
            apps.append({"name": name, "script": script, "port": port})
        except Exception:
            # 変なエントリはスキップ
            continue

    if not apps:
        # 1件も有効なエントリがなければ defaults をそのまま使う
        apps = defaults["launcher"]["apps"]

    launcher_cfg["apps"] = apps
    return {"launcher": launcher_cfg}


def build_streamlit_command(script_path: Path, port: int,
                            address: str = "127.0.0.1",
                            headless: bool = False) -> list[str]:
    """
    1つの Streamlit アプリを起動するためのコマンドを組み立てる。
    - sys.executable を使って、「今動いている Python / exe」と同じ環境で起動。
    - 通常の Python 実行の場合：
        python -m streamlit run <script> --server.port=... --server.address=...
    """
    cmd = [
        sys.executable,
        "-m", "streamlit",
        "run",
        str(script_path),
        f"--server.port={port}",
        f"--server.address={address}",
        f"--server.headless={'true' if headless else 'false'}",
    ]
    return cmd


def main():
    cfg = load_settings()
    launcher_cfg = cfg["launcher"]

    base = base_dir()
    address = launcher_cfg.get("server_address", "127.0.0.1")
    headless = bool(launcher_cfg.get("headless", False))
    apps = launcher_cfg.get("apps", [])

    print("======================================")
    print("  ForcePlateReport launcher")
    print("======================================")
    print(f"  base_dir : {base}")
    print(f"  address  : {address}")
    print(f"  headless : {headless}")
    print("  apps:")
    for app in apps:
        print(f"    - {app['name']}: {app['script']} (port={app['port']})")
    print("======================================")

    procs: list[subprocess.Popen] = []

    # 各アプリを起動
    for app in apps:
        script_path = base / app["script"]
        if not script_path.exists():
            print(f"[launcher] WARNING: スクリプトが見つかりません: {script_path}")
            continue

        cmd = build_streamlit_command(
            script_path=script_path,
            port=app["port"],
            address=address,
            headless=headless,
        )

        print(f"[launcher] 起動: {app['name']} ({script_path}) port={app['port']}")
        print("          cmd:", " ".join(cmd))

        try:
            p = subprocess.Popen(cmd, cwd=str(base))
            procs.append(p)
        except Exception as e:
            print(f"[launcher] 起動に失敗しました ({app['name']}): {e}")

    if not procs:
        print("[launcher] 起動できたアプリがありませんでした。終了します。")
        return

    print("")
    print("すべてのアプリを起動しました。")
    print("ブラウザで以下のURLを開いてください：")
    for app in apps:
        print(f"  {app['name']}: http://{address}:{app['port']}/")
    print("")
    print("停止するには、このウィンドウで Ctrl+C を押してください。")

    try:
        # どれか1つでも終了したら他も止める、という簡単な監視ループ
        while True:
            alive = [p.poll() is None for p in procs]
            if not all(alive):
                print("[launcher] いずれかのアプリが終了したため、残りも終了させます。")
                break
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[launcher] Ctrl+C を検知しました。アプリを終了します。")
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.terminate()
                except Exception:
                    pass
        # 念のため待つ
        for p in procs:
            try:
                p.wait(timeout=5)
            except Exception:
                pass

        print("[launcher] すべてのプロセスを終了しました。")


if __name__ == "__main__":
    main()
