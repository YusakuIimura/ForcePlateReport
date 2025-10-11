import argparse, json, base64, io
from pathlib import Path
from typing import Dict, List, Any

from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np

# ============ 日本語フォント ============
def set_japanese_font():
    local_fonts = [
        "fonts/NotoSansJP-Regular.ttf",
        "fonts/NotoSansCJKjp-Regular.otf",
        "fonts/Meiryo.ttf",
    ]
    for p in local_fonts:
        if Path(p).exists():
            fm.addfont(p)
            plt.rcParams["font.family"] = fm.FontProperties(fname=p).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return
    candidates = ["Yu Gothic", "Yu Gothic UI", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "Noto Sans JP"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False

set_japanese_font()

# ============ ユーティリティ ============
def to_data_uri(png_bytes: bytes, mime="image/png") -> str:
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"

def ensure_dirs(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "assets").mkdir(parents=True, exist_ok=True)
    return out_dir / "assets"

def save_and_uri(assets_dir: Path, name: str, png_bytes: bytes) -> str:
    p = assets_dir / f"{name}.png"
    with open(p, "wb") as f:
        f.write(png_bytes)
    return to_data_uri(png_bytes)

# ============ 図（ピクセル固定） ============
def plot_lines_fixed(
    ts: List[float],
    series_dict: Dict[str, List[float]],
    px=(1000, 260),
    dpi=200,
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
) -> bytes:
    w, h = px
    fig, ax = plt.subplots(figsize=(w/dpi, h/dpi), dpi=dpi, layout="constrained")
    for label, y in series_dict.items():
        ax.plot(ts, y, label=label)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    if title:  ax.set_title(title, fontsize=10)
    if len(series_dict) > 1: ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25); ax.tick_params(labelsize=8)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi); plt.close(fig)
    return buf.getvalue()

def plot_radar_fixed(d: Dict[str, float], px=(1, 1), dpi=200) -> bytes:
    labels = list(d.keys())
    values = [float(v) for v in d.values()] if labels else [0.0]
    if not labels: labels = ["—"]
    N = len(labels)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values += values[:1]; angles += angles[:1]
    fig = plt.figure(figsize=(px[0]/dpi, px[1]/dpi), dpi=dpi, layout="constrained")
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values); ax.fill(angles, values, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
    ax.set_ylim(0,1)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=dpi); plt.close(fig)
    return buf.getvalue()

# ============ HTMLレンダリング ============
def render_html(
    data: Dict[str, Any],
    template_dir: str = ".",
    template_name: str = "report_template.html",
    out_dir: Path = Path("out"),
) -> str:
    env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)
    tmpl = env.get_template(template_name)

    assets_dir = ensure_dirs(out_dir)

    ts   = (data.get("timeseries") or {}).get("t") or []
    fz_l = (data.get("timeseries") or {}).get("fz_left") or []
    fz_r = (data.get("timeseries") or {}).get("fz_right") or []
    tz   = (data.get("timeseries") or {}).get("tz") or []

    fz_uri = ""
    if ts and (fz_l or fz_r):
        series = {}
        if fz_l: series["Fz Left"] = fz_l
        if fz_r: series["Fz Right"] = fz_r
        fz_uri = save_and_uri(assets_dir, "fz",
                              plot_lines_fixed(ts, series, px=(1000,260), title="Fz", xlabel="Time [s]"))

    tz_uri = ""
    if ts and tz:
        tz_uri = save_and_uri(assets_dir, "tz",
                              plot_lines_fixed(ts, {"Tz": tz}, px=(1000,260), title="Tz", xlabel="Time [s]"))

    radar_uri = save_and_uri(assets_dir, "radar",
                             plot_radar_fixed(data.get("radar") or {}, px=(440,440)))

    if data.get("cop_uri"):
        cop_uri = data["cop_uri"]
    else:
        cop_uri = save_and_uri(assets_dir, "cop",
                               plot_lines_fixed([0,1], {"CoP": [0,0]}, px=(440,440), title="CoP"))

    photo_uri = data.get("photo_uri", "")
    fz_uri    = data.get("fz_uri")    or fz_uri
    tz_uri    = data.get("tz_uri")    or tz_uri
    radar_uri = data.get("radar_uri") or radar_uri
    cop_uri   = data.get("cop_uri")   or cop_uri

    meta = data.get("meta") or {}
    player_name = (
        data.get("player_name")
        or meta.get("player_name")
        or meta.get("athlete_name")
        or meta.get("name")
        or ""
    )

    cog_html = data.get("cog_html", "")
    if not cog_html:
        cm = data.get("cog_metrics") or {}
        if cm:
            import pandas as _pd
            _df = _pd.DataFrame({"指標": list(cm.keys()), "値": list(cm.values())})
            cog_html = _df.to_html(index=False, border=0, classes="table table-sm")

    context = {
        **data,
        "fz_uri": fz_uri, "tz_uri": tz_uri,
        "radar_uri": radar_uri, "cop_uri": cop_uri, "photo_uri": photo_uri,
        "player_name": player_name,
        "cog_html": cog_html,
    }
    
    return tmpl.render(**context)

def main():
    p = argparse.ArgumentParser(description="Force Plate Report (HTML only)", allow_abbrev=False)
    p.add_argument("--data", default="sample_data.json", help="入力JSONのパス")
    p.add_argument("--template", default="report_template.html", help="テンプレHTMLのパス")
    p.add_argument("--out-dir", default="out", help="出力フォルダ")
    p.add_argument("--html-name", default="report.html", help="出力HTML名")
    args = p.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        data = json.load(f)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    html = render_html(
        data,
        template_dir=Path(args.template).parent.as_posix(),
        template_name=Path(args.template).name,
        out_dir=out_dir,
    )

    out_html = out_dir / args.html_name
    out_html.write_text(html, encoding="utf-8")
    print(f"[OK] HTML -> {out_html.as_posix()}")
    print(f"[OK] 画像 -> {(out_dir / 'assets').as_posix()}/*.png")
    print("印刷はブラウザで Ctrl+P（A4 横・余白14mmで出力）")

if __name__ == "__main__":
    main()
