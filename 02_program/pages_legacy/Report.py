# pages/Report.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import streamlit as st

from report_core import (
    log,
    load_csv_from_path,
    build_report_html_from_df,
    render_report_with_print_toolbar,
)

# ===== ページ設定 =====
st.set_page_config(page_title="Report Viewer", layout="wide")
st.title("Report Viewer")

# ===== セッションから選択データを取得 =====
records: List[Dict] = st.session_state.get("selected_records") or []
if not records:
    st.info("Home でデータ行を選択してください。")
    st.page_link("Home.py", label="← メインに戻る", icon="⏪")
    st.stop()

# GraphViewerで保持した開始/終了（index）
# 例: {"1. name (file.csv)": {"start": 120, "end": 260}, ...}
graph_ranges: Dict[str, Dict[str, int]] = st.session_state.get("graph_ranges", {}) or {}

def _label_for_record(rec: Dict, idx: int) -> str:
    """
    GraphViewer と同じラベル:
      '1. <name or title or csv名> (<csvファイル名>)'
    """
    row = rec.get("row", {}) or {}
    csv_path = rec.get("csv_path", "") or ""
    name = row.get("name") or row.get("title") or Path(csv_path).name
    return f"{idx}. {name} ({Path(csv_path).name})"

def _first_nonempty_player(df) -> Optional[str]:
    """CSVの 'player' 列（大小無視）から最初の非空ユニーク値を返す。無ければ None。"""
    cand_col = None
    for c in df.columns:
        if str(c).strip().lower() == "player":
            cand_col = c
            break
    if not cand_col:
        return None
    try:
        s = df[cand_col].astype(str).map(lambda x: x.strip())
        vals = [v for v in s.unique().tolist() if v]
        return vals[0] if vals else None
    except Exception:
        return None

# ===== データごとに「デバッグ表示 → レポート」をまとめて表示 =====
for i, rec in enumerate(records, start=1):
    label = _label_for_record(rec, i)
    csv_path = Path(rec.get("csv_path", "") or "")
    row_meta: Dict = dict(rec.get("row", {}) or {})
    data_dir = rec.get("data_dir", "") or ""

    with st.container(border=True):
        st.markdown(f"### データ {i}: {label}")

        # ---- 上段：デバッグ表示（Start/End と行メタ） ----
        c1, c2, c3 = st.columns([1.6, 2.2, 1.2])
        with c1:
            st.markdown("**CSV**")
            st.write(csv_path.as_posix())
            if data_dir:
                st.caption(f"data_dir: {data_dir}")
        with c2:
            st.markdown("**行のメタ情報（参考）**")
            st.json(row_meta, expanded=False)
        with c3:
            st.markdown("**Graph範囲（保持値）**")
            se = graph_ranges.get(label)
            if se:
                st.success(f"Start={int(se.get('start', 0))}, End={int(se.get('end', 0))}")
            else:
                st.info("未設定（GraphViewerで「💾 このデータの開始・終了位置を保持」を押してください）")

        st.divider()

        # ---- 下段：レポート表示 ----
        if not csv_path.exists():
            st.error(f"CSV が見つかりません: {csv_path.as_posix()}")
            continue

        # CSV 読み込み（report_core 既存関数）
        try:
            df, measured_at, date_str, time_str, duration_str = load_csv_from_path(csv_path)
        except Exception as e:
            st.error(f"CSV の読み込みに失敗しました: {csv_path}\n{e}")
            continue
        
        df_for_report = df  # 既定は全範囲
        if se:
            start_idx = int(se.get("start", 0))
            end_idx   = int(se.get("end", len(df) - 1))
            # 端を安全にクリップ
            start_idx = max(0, min(start_idx, len(df) - 1))
            end_idx   = max(start_idx, min(end_idx, len(df) - 1))
            df_for_report = df.iloc[start_idx:end_idx + 1].copy()
            # 必要ならデバッグ表示
            st.caption(f"DEBUG: apply range [{start_idx}, {end_idx}] to {csv_path.name}")
                
        

        # player_name を CSV から（無ければ row_meta からフォールバック）
        player_name = _first_nonempty_player(df) or \
                      (str(row_meta.get("Player") or row_meta.get("player") or "").strip() or None)

        # basic_meta: テンプレで使われる基本枠（report_core 側で meta に流し込まれる）
        basic_meta: Dict = {
            "filename": csv_path.name,
            "measured_at": measured_at,
            "date": date_str,
            "time": time_str,
            "duration_sec": duration_str,
            "player_name": player_name or "",  
        }
        basic_meta["handedness"] = str(row_meta.get("利き手", "") or row_meta.get("handedness", "")).strip()



        # GraphViewer の保持範囲（将来テンプレで使うならこちらから参照可能）
        if se:
            basic_meta["graph_range_start_idx"] = int(se.get("start", 0))
            basic_meta["graph_range_end_idx"] = int(se.get("end", 0))

        # file_meta: None を渡さず dict で（meta.* としてテンプレから参照される）
        # loader由来のメタを優先して row_meta をマージ
        file_meta_from_loader = {
            "filename": csv_path.name,
            "date": date_str,
            "time": time_str,
            "duration_sec": duration_str,
        }
        file_meta: Dict = {**row_meta, **file_meta_from_loader}
        if player_name:
            # 念のため player_name も meta 側で拾えるよう保険をかける
            file_meta.setdefault("player_name", player_name)
            # 互換用（テンプレが title/name を見る可能性）
            file_meta.setdefault("title", player_name)
            file_meta.setdefault("name", player_name)

        # レポートHTML生成 → 埋め込み
        try:
            html = build_report_html_from_df(df_for_report, basic_meta=basic_meta, file_meta=file_meta)
        except Exception as e:
            st.error(f"レポートHTMLの生成に失敗しました。\n{e}")
            continue

        wrapped = render_report_with_print_toolbar(html) if html else ""
        if wrapped:
            st.components.v1.html(wrapped, height=1000, scrolling=False)
        else:
            st.warning("レポートHTMLが空でした。テンプレや入力データをご確認ください。")

# ===== 戻るリンク =====
st.markdown("---")
st.page_link("Home.py", label="← メインに戻る", icon="⏪")
