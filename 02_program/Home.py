# Home.py
from pathlib import Path
import pandas as pd
import streamlit as st
import re
from pathlib import Path
from datetime import date

def _records_from_selected(df_sel, csv_col: str):
    recs = []
    for _, r in df_sel.iterrows():
        row_dict = r.to_dict()
        csv_value = str(row_dict.get(csv_col, "")).strip()
        csv_path = Path(csv_value)
        if not csv_path.is_absolute():
            csv_path = (DATA_DIR / csv_path).resolve()
        recs.append({
            "row": row_dict,
            "csv_col": csv_col,
            "csv_path": csv_path.as_posix(),
            "data_dir": DATA_DIR.as_posix(),
            "datalist_path": DATALIST_PATH.as_posix(),
        })
    return recs

st.set_page_config(page_title="メイン | 試験結果ビューア", layout="wide")

st.title("メイン画面（選手情報記入＆データ選択）")
st.caption("期間内のデータからレポート作成するデータを選択します。")
st.caption("選手情報は playerlist.csv に保存され、次回以降も利用されます。")


# ---- パス前提：このファイルと同階層に data/ フォルダ ----
APP_DIR = Path(__file__).parent.resolve()
DATA_DIR = APP_DIR / "data"
DATALIST_PATH = DATA_DIR / "Datalist.csv"
PLAYERLIST_PATH = (DATA_DIR / "playerlist.csv")


# ---- セッション初期化 ----
st.session_state.setdefault("logs", [])
st.session_state.setdefault("basic", {
    "player_name":"", "height_cm":"", "weight_kg":"",
    "foot_size_cm":"", "handedness":"右", "step_width_cm":""
})

# ---- data/ の存在確認 ----
if not DATA_DIR.exists():
    st.error(f"data フォルダが見つかりません: {DATA_DIR.as_posix()}")
    st.stop()

# ---- Datalist.csv 読み込み ----
if not DATALIST_PATH.exists():
    st.error(f"Datalist.csv が見つかりません: {DATALIST_PATH.as_posix()}")
    st.info("data/ フォルダに Datalist.csv と、各行が参照する CSV / MP4 を置いてください。")
    st.stop()

try:
    df_raw = pd.read_csv(DATALIST_PATH)
except Exception as e:
    st.error(f"Datalist.csv の読み込みに失敗: {e}")
    st.stop()


def guess_csv_col(df: pd.DataFrame) -> str:
    cand = [c for c in df.columns if "csv" in c.lower() or "path" in c.lower()]
    if cand:
        return cand[0]
    for c in df.columns:
        try:
            if df[c].astype(str).str.contains(r"\.csv$", case=False, regex=True).any():
                return c
        except Exception:
            pass
    return df.columns[0]

st.subheader("データ一覧")

# --- 日時列（Datalist.csv 固定: Date + Time） ---
dt_series = pd.to_datetime(
    df_raw["Date"].astype(str).str.strip() + " " + df_raw["Time"].astype(str).str.strip(),
    errors="coerce"
)
min_d = dt_series.dt.date.min()
max_d = dt_series.dt.date.max()
if pd.isna(min_d) or pd.isna(max_d):
    # もしCSVに日付が無い/壊れている場合のフェールセーフ
    today = date.today()
    min_d = max_d = today

# --- カレンダーUI（選択中の揺れを吸収） ---
raw_value = st.date_input(
    "Date 範囲を選択",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
    format="YYYY-MM-DD",
    key="date_range"
)

# raw_value が「単日」か「(start, end)」か、選択中で長さ1の可能性もある
if isinstance(raw_value, tuple):
    if len(raw_value) == 2:
        start_date, end_date = raw_value
    elif len(raw_value) == 1:
        start_date, end_date = raw_value[0], raw_value[0]  # 一時的に単日に丸める
    else:
        start_date, end_date = min_d, max_d
else:
    # 単日が返るケース
    start_date = end_date = raw_value

# どちらかが None の一時状態もケア（未確定の瞬間がある）
if start_date is None and end_date is None:
    start_date, end_date = min_d, max_d
elif start_date is None:
    start_date = end_date
elif end_date is None:
    end_date = start_date

# 万一 start > end になったら入れ替え
if start_date > end_date:
    start_date, end_date = end_date, start_date

# --- フィルタ適用（ここまで来れば常に安全） ---
mask = (dt_series.dt.date >= start_date) & (dt_series.dt.date <= end_date)
df_base = df_raw.loc[mask].copy()

try:
    if PLAYERLIST_PATH.exists():
        pl = pd.read_csv(PLAYERLIST_PATH, encoding="shift_jis")

        # 列名ゆれに少しだけ耐性（player は大小無視で一致、項目は候補から拾う）
        def _find_col(df, names):
            cmap = {str(c).strip().lower(): c for c in df.columns}
            for n in names:
                k = str(n).strip().lower()
                if k in cmap:
                    return cmap[k]
            return None

        p_d = _find_col(df_base, ["player"])              # datalist 側
        p_p = _find_col(pl,      ["player"])              # playerlist 側
        h_p = _find_col(pl,      ["利き手", "handedness", "dominant"])
        ht_p= _find_col(pl,      ["身長", "height", "height_cm"])
        wt_p= _find_col(pl,      ["体重", "weight", "weight_kg"])

        if p_d and p_p:
            use_cols = [p_p]
            if h_p:  use_cols.append(h_p)
            if ht_p: use_cols.append(ht_p)
            if wt_p: use_cols.append(wt_p)

            pl_small = pl[use_cols].copy()
            # 標準化した列名にそろえる
            ren = {}
            if h_p:  ren[h_p]  = "利き手"
            if ht_p: ren[ht_p] = "身長"
            if wt_p: ren[wt_p] = "体重"
            ren[p_p] = "player"
            pl_small = pl_small.rename(columns=ren)

            # 左外部結合（player キー）
        df_base = df_base.merge(pl_small.set_index("player"), how="left", left_on=p_d, right_index=True)

        # 欠損は空にしておく（存在しなければ列を作って空）
        for c in ["利き手", "身長", "体重"]:
            if c not in df_base.columns:
                df_base[c] = ""
            df_base[c] = df_base[c].fillna("")

        # 右端に並ぶように列の順序を最後に回す（既にある場合は pop→末尾追加）
        for c in ["利き手", "身長", "体重"]:
            if c in df_base.columns:
                col = df_base.pop(c)
                df_base[c] = col
    else:
        # ファイルが無い場合は空列を追加
        for c in ["利き手", "身長", "体重"]:
            if c not in df_base.columns:
                df_base[c] = ""
except Exception as e:
    st.warning(f"playerlist.csv の読み込み/マージに失敗しました: {e}")
    # 失敗しても空列で継続
    for c in ["利き手", "身長", "体重"]:
        if c not in df_base.columns:
            df_base[c] = ""


csv_col = guess_csv_col(df_base)

# 選択列を付与
SELECT_COL = "選択"
df_show = df_base.copy()
if SELECT_COL not in df_show.columns:
    df_show.insert(0, SELECT_COL, False)

edited = st.data_editor(
    df_show,
    use_container_width=True,
    hide_index=True,
    column_config={
        SELECT_COL: st.column_config.CheckboxColumn(required=False, help="複数選択できます"),
    },
    disabled=False,
    height=520,
    key="datalist_editor",
)

# ===== 不足項目の追加入力（画面上で編集）＆ 反映ボタン =====
st.subheader("不足しているプレイヤー情報の追加入力")
# 右端3列が欠損の行を編集対象に（全部編集したい場合は .any()→.notna() などに調整可）
need_fill_mask = (df_base[["利き手", "身長", "体重"]].isna() | (df_base[["利き手", "身長", "体重"]].astype(str) == "")).any(axis=1)
edit_src = df_base.loc[need_fill_mask, :].copy()

# 編集用の軽量ビュー（player と 3項目だけ）
edit_view_cols = []
# datalist 側の player 列名（p_d）を流用。なければ "player"
try:
    edit_player_col = p_d if (p_d in df_base.columns) else ("player" if "player" in df_base.columns else None)
except NameError:
    edit_player_col = "player" if "player" in df_base.columns else None

if edit_player_col is None:
    st.info("この表には player 列が見つからないため、画面上での追加入力は無効です。")
else:
    edit_src = edit_src[[edit_player_col, "利き手", "身長", "体重"]].copy()
    edit_src = edit_src.rename(columns={edit_player_col: "player"})  # 編集は "player" 名で統一
    edit_src = edit_src.drop_duplicates(subset=["player"])
    edit_src = edit_src.reset_index(drop=True)

    st.caption("※ 空欄になっているプレイヤーだけを抽出しています。必要事項を入力して『反映』を押してください。")
    editable = st.data_editor(
        edit_src,
        column_config={
            "player": st.column_config.TextColumn("player", help="キー（変更しないことを推奨）", disabled=True),
            "利き手": st.column_config.TextColumn("利き手", help="例：右 / 左"),
            "身長":   st.column_config.NumberColumn("身長", help="cm"),
            "体重":   st.column_config.NumberColumn("体重", help="kg"),
        },
        hide_index=True,
        use_container_width=True,
    )

    # === 反映ボタン ===
    do_apply = st.button("💾 反映（playerlist.csv を更新）")
    if do_apply:
        try:
            # 1) 現在の playerlist を Shift-JIS で読み込み
            pl = pd.read_csv(PLAYERLIST_PATH, encoding="shift_jis")

            # 列名ゆれ解決（既存の _find_col をそのまま使う）
            def _find_col(df, names):
                cmap = {str(c).strip().lower(): c for c in df.columns}
                for n in names:
                    k = str(n).strip().lower()
                    if k in cmap:
                        return cmap[k]
                return None

            p_p  = _find_col(pl, ["player"]) or "player"
            h_p  = _find_col(pl, ["利き手", "handedness", "dominant"]) or "利き手"
            ht_p = _find_col(pl, ["身長", "height", "height_cm"])       or "身長"
            wt_p = _find_col(pl, ["体重", "weight", "weight_kg"])       or "体重"

            # 足りない列は作っておく（既存カラムは保持）
            for c in [p_p, h_p, ht_p, wt_p]:
                if c not in pl.columns:
                    pl[c] = ""

            # 2) 編集結果を player キーで upsert（存在すれば更新、無ければ追加）
            #    キーは大小無視・前後空白無視で照合
            key_series = pl[p_p].astype(str).str.strip()
            key_lcase = key_series.str.lower()

            updates = 0
            inserts = 0
            for _, r in editable.iterrows():
                player = str(r["player"]).strip()
                if not player:
                    continue
                handed = str(r["利き手"]).strip() if pd.notna(r["利き手"]) else ""
                height = r["身長"]
                weight = r["体重"]

                # 既存行の位置（大小無視）
                match = key_lcase == player.lower()
                if match.any():
                    idx = match.idxmax()  # 最初の一致
                    # 入力が空でなければ更新（空はスキップ）
                    if handed:
                        pl.at[idx, h_p] = handed
                    if pd.notna(height) and str(height) != "":
                        pl.at[idx, ht_p] = height
                    if pd.notna(weight) and str(weight) != "":
                        pl.at[idx, wt_p] = weight
                    updates += 1
                else:
                    # 新規行を追加
                    row_new = {col: "" for col in pl.columns}
                    row_new[p_p]  = player
                    row_new[h_p]  = handed
                    row_new[ht_p] = height if pd.notna(height) and str(height) != "" else ""
                    row_new[wt_p] = weight if pd.notna(weight) and str(weight) != "" else ""
                    pl = pd.concat([pl, pd.DataFrame([row_new])], ignore_index=True)
                    inserts += 1

            # 3) Shift-JIS で上書き保存（バックアップが必要ならここで .bak を作成）
            #    例: PLAYERLIST_PATH.with_suffix(".bak.csv") に pl.to_csv(..., index=False)
            pl.to_csv(PLAYERLIST_PATH, index=False, encoding="shift_jis")

            # 4) 画面の df_base も即時反映したいので再マージ or rerun
            st.success(f"playerlist.csv を更新しました（更新 {updates} 件 / 追加 {inserts} 件）。")
            st.rerun()

        except FileNotFoundError:
            st.error(f"playerlist.csv が見つかりません: {PLAYERLIST_PATH}")
        except Exception as e:
            st.error(f"反映に失敗しました: {e}")

with st.expander("🔧 プレイヤー情報の修正（上書き）", expanded=False):
    try:
        # Shift-JIS で読込（型の揺れを避けるなら dtype=str）
        pl = pd.read_csv(PLAYERLIST_PATH, encoding="shift_jis")

        # ローカルな列名解決（大小文字・日本語にゆるく対応）
        def _find_col_local(df, names):
            cmap = {str(c).strip().lower(): c for c in df.columns}
            for n in names:
                k = str(n).strip().lower()
                if k in cmap:
                    return cmap[k]
            return None

        pcol = _find_col_local(pl, ["player"])
        hcol = _find_col_local(pl, ["利き手", "handedness", "dominant"])
        htcol = _find_col_local(pl, ["身長", "height", "height_cm"])
        wtcol = _find_col_local(pl, ["体重", "weight", "weight_kg"])

        if pcol is None:
            st.info("playerlist.csv に player 列がありません。先に列を追加してください。")
        else:
            players = (
                pl[pcol].astype(str).fillna("")
                .apply(lambda s: s.strip())
                .replace({"None": ""})
                .tolist()
            )
            players = sorted(set([p for p in players if p]))  # 空と重複を除去

            with st.form("overwrite_player_form"):
                target = st.selectbox("上書きする player を選択", players, index=0 if players else None)
                # 現在値を取得（大文字小文字無視で一意マッチ）
                handed_val = height_val = weight_val = ""
                if target:
                    m = pl[pcol].astype(str).str.strip().str.lower() == target.strip().lower()
                    if m.any():
                        row0 = pl.loc[m].iloc[0]
                        handed_val = str(row0.get(hcol, "")) if hcol else ""
                        height_val = str(row0.get(htcol, "")) if htcol else ""
                        weight_val = str(row0.get(wtcol, "")) if wtcol else ""

                col1, col2, col3 = st.columns(3)
                new_handed = col1.text_input("利き手（空欄は変更しない）", value=handed_val)
                new_height = col2.text_input("身長 cm（空欄は変更しない）", value=height_val)
                new_weight = col3.text_input("体重 kg（空欄は変更しない）", value=weight_val)

                submitted = st.form_submit_button("📝 上書き保存")
                if submitted and target:
                    # 既存列が無ければ作成（保守的）
                    if hcol is None:
                        hcol = "利き手"; pl[hcol] = ""
                    if htcol is None:
                        htcol = "身長";   pl[htcol] = ""
                    if wtcol is None:
                        wtcol = "体重";   pl[wtcol] = ""

                    m = pl[pcol].astype(str).str.strip().str.lower() == target.strip().lower()
                    if not m.any():
                        st.error("対象の player が見つかりませんでした。")
                    else:
                        idxs = pl.index[m]
                        # 空欄は変更しない
                        if str(new_handed).strip() != "":
                            pl.loc[idxs, hcol] = str(new_handed).strip()
                        if str(new_height).strip() != "":
                            pl.loc[idxs, htcol] = str(new_height).strip()
                        if str(new_weight).strip() != "":
                            pl.loc[idxs, wtcol] = str(new_weight).strip()

                        # Shift-JIS で保存
                        pl.to_csv(PLAYERLIST_PATH, index=False, encoding="shift_jis")
                        st.success(f"{target} の情報を上書きしました。")
                        st.rerun()

    except FileNotFoundError:
        st.info(f"playerlist.csv が見つかりません: {PLAYERLIST_PATH}")
    except Exception as e:
        st.error(f"上書き処理に失敗しました: {e}")


sel_mask = edited[SELECT_COL] == True
selected_rows = edited[sel_mask].drop(columns=[SELECT_COL], errors="ignore")

st.session_state["selected_records"] = _records_from_selected(selected_rows, csv_col)
st.session_state["selected_csv_paths"] = [r["csv_path"] for r in st.session_state["selected_records"]]


def _prepare_records(selected_rows):
    records = []
    for _, r in selected_rows.iterrows():
        row_dict = r.to_dict()
        csv_value = str(row_dict.get(csv_col, "")).strip()
        csv_path = Path(csv_value)
        if not csv_path.is_absolute():
            csv_path = (DATA_DIR / csv_path).resolve()
        records.append({
            "row": row_dict,
            "csv_col": csv_col,
            "csv_path": csv_path.as_posix(),
            "data_dir": DATA_DIR.as_posix(),
            "datalist_path": DATALIST_PATH.as_posix(),
        })
    return records

# 並列ボタン
col_btns = st.columns([0.01, 3, 6])  # 左の余白, ボタン群, 右の余白
with col_btns[1]:
    bcol1, bcol2 = st.columns([1, 1])
    with bcol1:
        go_graph = st.button("📈 グラフビュワーへ", type="primary")
    with bcol2:
        go_report = st.button("📝 レポートを開く")

# ↓ 以降は同じ処理
if go_graph or go_report:
    if selected_rows.empty:
        st.warning("1行以上選択してください。")
        st.stop()

    records = _prepare_records(selected_rows)
    st.session_state["selected_records"] = records
    st.session_state["selected_csv_paths"] = {
        f"{i+1}. {Path(rec['csv_path']).name}": rec["csv_path"]
        for i, rec in enumerate(records)
    }

    dest = "pages/GraphViewer.py" if go_graph else "pages/Report.py"
    st.switch_page(dest)

