import streamlit as st
import pandas as pd
from pathlib import Path
from urllib.parse import quote
import re
import datetime
import json

st.set_page_config(page_title="Home", layout="wide")

DATA_DIR = Path("data")
DATALIST_PATH = DATA_DIR / "datalist.csv"
USERLIST_PATH = DATA_DIR / "userlist.csv"

SELECT_COL = "_select_"
TS_COL = "_ts"
DISPLAY_COLS = ["csv_path", "Date", "Time", "user", "競技", "身長", "体重", "備考"]

SETTINGS_PATH = Path("./settings.json")
DEFAULT_SPORTS = ["野球", "ゴルフ", "CMJ", "歩行"]

def _load_settings():
    try:
        with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

_cfg = _load_settings()
_landing_cfg = _cfg.get("landing", {})
SPORTS = _landing_cfg.get("sports", DEFAULT_SPORTS)

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



# -----------------
# 基本的なI/O系
# -----------------

def list_fp_files(data_dir: Path) -> pd.DataFrame:
    records = []
    for p in data_dir.glob("*_FP.csv"):
        fname = p.name
        m = re.match(r"^(\d{8})_(\d{6})_FP\.csv$", fname)
        if not m:
            continue

        yyyymmdd = m.group(1)
        hhmmss = m.group(2)

        date_str = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
        time_str = f"{hhmmss[0:2]}:{hhmmss[2:4]}:{hhmmss[4:6]}"

        records.append({
            "csv_path": fname,
            "Date": date_str,
            "Time": time_str,
        })

    if not records:
        return pd.DataFrame(columns=["csv_path", "Date", "Time"])
    return pd.DataFrame(records)

def load_datalist(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(
            columns=["csv_path", "Date", "Time", "user", "競技", "身長", "体重","備考"]
        )

    df = pd.read_csv(path)
    for col in ["csv_path", "Date", "Time", "user", "競技", "身長", "体重","備考"]:
        if col not in df.columns:
            df[col] = ""
    return df[["csv_path", "Date", "Time", "user", "競技", "身長", "体重","備考"]].copy()

def load_userlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "競技", "身長", "体重"])

    df = pd.read_csv(path)
    for col in ["user", "競技", "身長", "体重"]:
        if col not in df.columns:
            df[col] = ""
    return df[["user", "競技", "身長", "体重"]].copy()

def _set_left_today():
    today = datetime.date.today()
    st.session_state.left_date_range = (today, today)

def _set_right_today():
    today = datetime.date.today()
    st.session_state.right_date_range = (today, today)

def _get_query_param(name: str, default: str = "") -> str:
    try:
        val = st.query_params.get(name, default)
        if isinstance(val, list):
            return val[0] if val else default
        return val
    except Exception:
        params = st.experimental_get_query_params()
        vals = params.get(name, [])
        return vals[0] if vals else default

# -----------------
# データ構築関連
# -----------------
def build_df_all() -> pd.DataFrame:
    """
    data/*.csv と datalist.csv と userlist.csv を統合して返す。
    常に csv_path は1行に潰して返す。
    競技は (datalistの競技) を優先し、空なら (userlistの競技) を採用。
    """
    base_df = list_fp_files(DATA_DIR)
    dl_df   = load_datalist(DATALIST_PATH)
    pl_df   = load_userlist(USERLIST_PATH)

    # datalist は csv_path ごとに1行だけ残す（user, 競技, 備考を持っておく）
    dl_df_unique = (
        dl_df.sort_values(["csv_path", "Date", "Time"])
             .drop_duplicates(subset=["csv_path"], keep="last")
             [["csv_path", "user", "競技", "備考"]]
             .copy()
    )
    dl_df_unique.rename(columns={"競技": "競技_dl"}, inplace=True)

    # userlist も user ごとに1行だけ（競技/身長/体重）
    pl_df_unique = (
        pl_df.sort_values(["user"])
             .drop_duplicates(subset=["user"], keep="last")
             [["user", "競技", "身長", "体重"]]
             .copy()
    )
    pl_df_unique.rename(columns={"競技": "競技_ul"}, inplace=True)

    # dataフォルダにあるcsvをベースに datalist をJOIN（user, 競技_dl, 備考）
    merged = pd.merge(
        base_df,
        dl_df_unique,
        on="csv_path",
        how="left",
    )

    # user情報から競技_ul/身長/体重をJOIN
    merged = pd.merge(
        merged,
        pl_df_unique,
        on="user",
        how="left",
    )

    # 競技は datalist優先 → 空なら userlist
    merged["競技"] = merged["競技_dl"].where(
        merged["競技_dl"].notna() & (merged["競技_dl"].astype(str).str.strip() != ""),
        merged["競技_ul"]
    )

    # 欠損補完
    for col in ["user", "競技", "身長", "体重", "備考"]:
        if col not in merged.columns:
            merged[col] = ""
    merged["user"]  = merged["user"].fillna("").astype(str)
    merged["競技"]   = merged["競技"].fillna("").astype(str)
    merged["身長"]    = merged["身長"].fillna("").astype(str)
    merged["体重"]    = merged["体重"].fillna("").astype(str)
    merged["備考"]    = merged["備考"].fillna("").astype(str)

    # タイムスタンプ列（フィルタ用）
    def to_ts(row):
        try:
            return pd.to_datetime(str(row["Date"]) + " " + str(row["Time"]), errors="coerce")
        except Exception:
            return pd.NaT
    merged[TS_COL] = merged.apply(to_ts, axis=1)

    # 念のため csv_path でユニーク化
    merged = (
        merged.sort_values(["csv_path", "Date", "Time"])
              .drop_duplicates(subset=["csv_path"], keep="last")
    )

    # 表示用
    merged = merged[DISPLAY_COLS + [TS_COL]].copy()
    merged[SELECT_COL] = False

    return merged

def get_user_choices(df_all: pd.DataFrame):
    vals = (
        df_all["user"]
        .astype(str)
        .fillna("")
        .str.strip()
        .replace("nan", "")
        .unique()
        .tolist()
    )
    vals = [v for v in vals if v]
    vals.sort()
    return ["すべて", "未登録"] + vals

def get_date_defaults(df_all: pd.DataFrame):
    if df_all[TS_COL].notna().any():
        min_ts = df_all[TS_COL].min()
        max_ts = df_all[TS_COL].max()
    else:
        now = datetime.datetime.now()
        min_ts, max_ts = now, now
    return min_ts.date(), max_ts.date()

def filter_df_for_display(df_all: pd.DataFrame,
                          chosen_user: str,
                          start_dt: datetime.datetime,
                          end_dt: datetime.datetime):
    df = df_all.copy()

    # user絞り
    if chosen_user == "未登録":
        # 空文字 or NaN を空欄扱い
        df = df[
            df["user"].isna()
            | (df["user"].astype(str).str.strip() == "")
            | (df["user"].astype(str).str.lower() == "nan")
        ]
    elif chosen_user != "すべて":
        df = df[df["user"].astype(str).str.strip() == chosen_user]

    # 日付範囲絞り
    mask_valid = df[TS_COL].notna()
    mask_range = (df[TS_COL] >= start_dt) & (df[TS_COL] <= end_dt)
    df = df[mask_valid & mask_range]

    return df[DISPLAY_COLS + [SELECT_COL]].copy()

def write_userlist(user: str, handed: str, height: str, weight: str):
    """
    userlist.csv を (user をキーに) 追加 or 上書き。
    """
    pl_df = load_userlist(USERLIST_PATH)
    mask = pl_df["user"].astype(str) == str(user)

    if mask.any():
        pl_df.loc[mask, ["競技", "身長", "体重"]] = [handed, height, weight]
    else:
        new_row = pd.DataFrame([{
            "user": user,
            "競技": handed,
            "身長": height,
            "体重": weight,
        }])
        pl_df = pd.concat([pl_df, new_row], ignore_index=True)

    pl_df.to_csv(USERLIST_PATH, index=False, encoding="utf-8-sig")

def rebuild_and_save_datalist(df_all_current: pd.DataFrame):
    """
    df_all_current から datalist.csv を作り直して保存。
    """
    # df_all_current: csv_path, Date, Time, user, ...
    out = df_all_current[["csv_path", "Date", "Time", "user", "競技", "身長", "体重","備考"]].copy()

    # 念のためユニーク化
    out = (
        out.sort_values(["csv_path", "Date", "Time"])
           .drop_duplicates(subset=["csv_path"], keep="last")
    )

    out.to_csv(DATALIST_PATH, index=False, encoding="utf-8-sig")

def assign_user_and_save_all(target_csv: str,
                               user: str,
                               handed: str,
                               height: str,
                               weight: str,
                               remarks: str = ""):
    """
    右カラム保存ボタン用。
    - userlist.csv を更新
    - 最新 df_all を再構築
    - その1件(target_csv)のuserを書き換え
    - datalist.csv を吐く
    """
    # 1. userlist を反映
    write_userlist(user, handed, height, weight)

    # 2. 最新の df_all を再構築
    df_all_current = build_df_all()

    # 3. 対象csvのuserを書き換え
    df_all_current.loc[
        df_all_current["csv_path"] == target_csv, "user"
    ] = user
    df_all_current.loc[
        df_all_current["csv_path"] == target_csv, "競技"
    ] = handed
    df_all_current.loc[
        df_all_current["csv_path"] == target_csv, "身長"
    ] = height
    df_all_current.loc[
        df_all_current["csv_path"] == target_csv, "体重"
    ] = weight
    if "備考" not in df_all_current.columns:
        df_all_current["備考"] = ""
    df_all_current.loc[df_all_current["csv_path"] == target_csv, "備考"] = remarks

    # 4. datalist.csv を再生成
    rebuild_and_save_datalist(df_all_current)

# -----------------
# UI
# -----------------


# st.title("(解析/レポート)ビュー")

# 最新ビュー
df_all = build_df_all()

valid_sports = {"野球", "ゴルフ", "CMJ", "歩行"}
selected_sport = _get_query_param("sport", "").strip()

if selected_sport in set(SPORTS):
    mask_empty = df_all["競技"].astype(str).str.strip().isin(["", "nan", "NaN"])
    mask_match = df_all["競技"].astype(str).str.strip() == selected_sport
    df_all = df_all[mask_empty | mask_match].copy()

    # st.info(f"ランディングで選択: **{selected_sport}**（競技が「{selected_sport}」または空欄のデータのみ表示）")

with st.container(border=True):
    st.subheader("ユーザーの登録")
    
    col1, col2 = st.columns([0.7, 0.3]) 
    with col1:
        # ==== 0. 左カラム用 日付範囲 ====
        default_start_l, default_end_l = get_date_defaults(df_all)
        cols_l = st.columns([1, 0.35])
        with cols_l[0]:
            picked_range_l = st.date_input(
                "表示する日付範囲",
                value=(default_start_l, default_end_l),
                help="この期間の計測だけを左の対象リストに出します",
                key="left_date_range",
            )
        with cols_l[1]:
            st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
            st.button("本日に設定", key="btn_left_today", on_click=_set_left_today)
    with col2:
        user_choices = get_user_choices(df_all)
        chosen_user = st.selectbox(
            "ユーザーで絞り込み",
            key="left_user_filter",
            options=user_choices,
            index=0,
        )
    
    # 日付型を日時へ
    if isinstance(picked_range_l, (list, tuple)) and len(picked_range_l) == 2:
        start_date_l, end_date_l = picked_range_l
    elif isinstance(picked_range_l, datetime.date):
        start_date_l, end_date_l = picked_range_l, picked_range_l
    else:
        start_date_l, end_date_l = default_start_l, default_end_l

    start_dt_l = datetime.datetime.combine(start_date_l, datetime.time.min)
    end_dt_l   = datetime.datetime.combine(end_date_l,   datetime.time.max)

    # 左カラムの対象候補を期間で絞る
    df_all_left = df_all[df_all[TS_COL].notna() & (df_all[TS_COL] >= start_dt_l) & (df_all[TS_COL] <= end_dt_l)]
    if chosen_user == "未登録":
        df_all_left = df_all_left[
            df_all_left["user"].isna()
            | (df_all_left["user"].astype(str).str.strip() == "")
            | (df_all_left["user"].astype(str).str.lower() == "nan")
        ]
    elif chosen_user != "すべて":
        df_all_left = df_all_left[df_all_left["user"].astype(str).str.strip() == chosen_user]

    # ==== 1. 対象CSV選択 ====
    all_csv_options = df_all_left["csv_path"].tolist()
    _label_map = {}
    if not df_all_left.empty:
        # csv_path単位で1つずつ代表行を取る
        tmp = df_all_left[["csv_path", "user"]].drop_duplicates(subset=["csv_path"])
        for _, r in tmp.iterrows():
            _csv = str(r["csv_path"])
            _user = str(r["user"]).strip() if pd.notna(r["user"]) else ""
            if (not _user) or (_user.lower() == "nan"):
                _user = "未登録"
            _label_map[_csv] = f"{_csv}（{_user}）"
    
    
    if not all_csv_options:
        st.info("dataフォルダに *_FP.csv がありません。")
    else:
        target_csv = st.selectbox(
            f"対象データ (csv)",
            options=all_csv_options,
            index=0,
            help="この計測を誰のものか決めます",
            key="target_csv_select",
            format_func=lambda p: _label_map.get(p, p),
        )

        # このcsvに現在割り当たってる値を取得
        row_now = df_all_left[df_all_left["csv_path"] == target_csv].head(1)
        current_user_val = str(row_now["user"].iloc[0]) if not row_now.empty and pd.notna(row_now["user"].iloc[0]) else ""
        current_handed_val = str(row_now["競技"].iloc[0]) if not row_now.empty and pd.notna(row_now["競技"].iloc[0]) else ""
        current_height_val = str(row_now["身長"].iloc[0]) if not row_now.empty and pd.notna(row_now["身長"].iloc[0]) else ""
        current_weight_val = str(row_now["体重"].iloc[0]) if not row_now.empty and pd.notna(row_now["体重"].iloc[0]) else ""
        current_remarks_val = str(row_now["備考"].iloc[0]) if not row_now.empty and pd.notna(row_now["備考"].iloc[0]) else ""

        # ==== 2. セッション初期化 ====
        for key in [
            "edit_user", "edit_handed", "edit_height", "edit_weight", "edit_remarks",
            "bound_csv",
            "pending_confirm",          # ← 確認待ちフラグ
            "pending_target_csv",       # ← 確認対象のcsv
            "pending_payload",          # ← 保存予定の内容
            "existing_user_select_prev",
        ]:
            if key not in st.session_state:
                st.session_state[key] = "" if key != "pending_confirm" else False

        # ==== 3 & 4. 動画プレビュー ＋ ユーザー情報 ====
        st.markdown("##### 動画 & ユーザー情報")

        video_col, info_col = st.columns([0.35, 0.65])

        # 左：動画
        with video_col:
            mp4_candidate = (DATA_DIR / target_csv).with_suffix(".mp4")
            if mp4_candidate.exists():
                st.video(str(mp4_candidate))
            else:
                st.info("対応する動画(.mp4)が見つかりませんでした。")

        # 右：ユーザー情報
        with info_col:
            st.markdown("###### ユーザー情報")
            st.caption(
                "既存ユーザーリストから読み込み  \n"
                "もしくは新規にユーザー情報を記入しデータベースを更新してください"
            )

            pl_df = load_userlist(USERLIST_PATH)
            existing_users = (
                pl_df["user"]
                .astype(str)
                .fillna("")
                .str.strip()
                .replace("nan", "")
                .tolist()
            )
            existing_users = sorted([u for u in existing_users if u])  # 空文字を除いてソート

            # ★ CSV切り替え時は、CSVの内容に合わせてドロップダウンとフォームを同期
            if st.session_state["bound_csv"] != target_csv:
                st.session_state["bound_csv"] = target_csv
                st.session_state["pending_confirm"] = False
                st.session_state["pending_target_csv"] = ""
                st.session_state["pending_payload"] = {}

                if current_user_val and current_user_val in existing_users:
                    # 既にこの計測にユーザー名が入っている → そのユーザーを選択状態に
                    st.session_state["existing_user_select"] = current_user_val
                    st.session_state["existing_user_select_prev"] = current_user_val

                    st.session_state["edit_user"] = current_user_val
                    st.session_state["edit_handed"] = current_handed_val
                    st.session_state["edit_height"] = current_height_val
                    st.session_state["edit_weight"] = current_weight_val
                    st.session_state["edit_remarks"] = current_remarks_val
                else:
                    # ユーザー未登録 → 新規登録モード
                    st.session_state["existing_user_select"] = "（新規登録）"
                    st.session_state["existing_user_select_prev"] = "（新規登録）"

                    st.session_state["edit_user"] = ""
                    st.session_state["edit_handed"] = ""
                    st.session_state["edit_height"] = ""
                    st.session_state["edit_weight"] = ""
                    # 備考だけは datalist.csv の値を初期表示にしておく
                    st.session_state["edit_remarks"] = current_remarks_val

            left_col_inner, right_col_inner = st.columns([0.25, 0.75])

            # 左：既存プレイヤー選択
            with left_col_inner:
                chosen_existing_user = st.selectbox(
                    "ユーザーリスト",
                    options=["（新規登録）"] + existing_users,
                    key="existing_user_select",
                    help="選ぶと右フォームに反映されます",
                )

                prev = st.session_state.get("existing_user_select_prev", None)
                if chosen_existing_user != prev:
                    if chosen_existing_user == "（新規登録）":
                        # 新規登録の場合はフォームをクリア
                        st.session_state["edit_user"] = ""
                        st.session_state["edit_handed"] = ""
                        st.session_state["edit_height"] = ""
                        st.session_state["edit_weight"] = ""
                        # 備考も新規としてクリア（ここは好みに応じて）
                        # st.session_state["edit_remarks"] = ""
                    else:
                        # 既存ユーザーの情報を userlist から読み込む
                        row_pl = pl_df[pl_df["user"] == chosen_existing_user].head(1)
                        if len(row_pl) > 0:
                            st.session_state["edit_user"] = chosen_existing_user
                            st.session_state["edit_handed"] = (
                                str(row_pl["競技"].iloc[0]) if pd.notna(row_pl["競技"].iloc[0]) else ""
                            )
                            st.session_state["edit_height"] = (
                                str(row_pl["身長"].iloc[0]) if pd.notna(row_pl["身長"].iloc[0]) else ""
                            )
                            st.session_state["edit_weight"] = (
                                str(row_pl["体重"].iloc[0]) if pd.notna(row_pl["体重"].iloc[0]) else ""
                            )
                            # userlist 側に備考を持つならここで反映してもよい

                    # 前回値を更新
                    st.session_state["existing_user_select_prev"] = chosen_existing_user

            # 右：フォーム（タイル配置）
            with right_col_inner:
                tile_cols = st.columns([0.3, 0.3, 0.2, 0.2])
                with tile_cols[0]:
                    st.text_input("ユーザー名", key="edit_user")
                with tile_cols[1]:
                    choices = [""] + list(SPORTS)
                    # セッションの値のみから初期選択を決める
                    current = (st.session_state.get("edit_handed") or "").strip()
                    default_idx = choices.index(current) if current in choices else 0
                    st.selectbox("競技", choices, index=default_idx, key="edit_handed")
                with tile_cols[2]:
                    st.text_input("身長", key="edit_height")
                with tile_cols[3]:
                    st.text_input("体重", key="edit_weight")
                st.text_area(
                    "備考",
                    key="edit_remarks",
                    height=90,
                    help="自由記述メモ（datalist.csv の備考列に保存されます）",
                )

            # ==== 5. 保存ボタン or 上書き確認 ====
            # フォームの内容
            form_user = st.session_state["edit_user"].strip()
            form_handed = (st.session_state["edit_handed"] or "").strip()
            form_height = (st.session_state["edit_height"] or "").strip()
            form_weight = (st.session_state["edit_weight"] or "").strip()
            form_remarks = (st.session_state["edit_remarks"] or "").strip()

            # userlist 上の既存プロファイルを取得
            def _norm(x): 
                return "" if pd.isna(x) else str(x).strip()

            row_exist = pl_df[pl_df["user"].astype(str).str.strip() == form_user]
            is_existing_user = bool(form_user) and not row_exist.empty
            
            # ★ メッセージ表示用の名前は、まずドロップダウンに揃える
            if chosen_existing_user == "（新規登録）":
                _display_user = form_user or "（新規登録）"
            else:
                _display_user = chosen_existing_user
                # 右フォームで名前を書き換えているならそちらを優先
                if form_user and form_user != chosen_existing_user:
                    _display_user = form_user

            st.markdown(
                f"下の「登録」ボタンでこのデータを**{_display_user}**選手のデータとして登録します"
            )

            if is_existing_user:
                # 既存プロファイル（現在の登録値）
                exist_handed = _norm(row_exist["競技"].iloc[0])
                exist_height = _norm(row_exist["身長"].iloc[0])
                exist_weight = _norm(row_exist["体重"].iloc[0])

                # フォーム値と差分があるか（= 上書きによって値が変わるか）
                profile_changed = (
                    (form_handed != exist_handed) or
                    (form_height != exist_height) or
                    (form_weight != exist_weight)
                )
            else:
                profile_changed = False  # 新規は差分の概念なし

            # まだ確認待ちでないときの表示
            if not st.session_state["pending_confirm"]:

                # 保存ボタン押下時の挙動
                def on_press_save():
                    if not form_user:
                        return  # 何もしない

                    if is_existing_user and profile_changed:
                        # 上書きになる場合のみ確認モードへ
                        st.session_state["pending_confirm"] = True
                        st.session_state["pending_target_csv"] = target_csv
                        st.session_state["pending_payload"] = {
                            "user": form_user,
                            "handed": form_handed,
                            "height": form_height,
                            "weight": form_weight,
                            "remarks": form_remarks,
                        }
                    else:
                        # 新規 or 既存だが値は同一 → そのまま保存
                        assign_user_and_save_all(
                            target_csv=target_csv,
                            user=form_user,
                            handed=form_handed,
                            height=form_height,
                            weight=form_weight,
                            remarks=form_remarks,
                        )
                        st.success("保存しました。")
                        st.rerun()

                st.button(
                    "💾　登録",
                    key="save_button",
                    on_click=on_press_save,
                )

            else:
                # 確認ダイアログ（上書き時のみ）
                pld = st.session_state["pending_payload"]
                old = pl_df[pl_df["user"].astype(str).str.strip() == pld["user"]].head(1)
                old_h = _norm(old["競技"].iloc[0]); old_ht = _norm(old["身長"].iloc[0]); old_w = _norm(old["体重"].iloc[0])

                st.error(
                    f"⚠️ 既存ユーザー『{pld['user']}』の登録値を上書きします。\n\n"
                    f"競技: {old_h} → {pld['handed']}\n"
                    f"身長:   {old_ht} → {pld['height']}\n"
                    f"体重:   {old_w} → {pld['weight']}"
                )

                c1, c2, _ = st.columns([0.3, 0.3, 0.4])

                with c1:
                    def do_confirm():
                        assign_user_and_save_all(
                            target_csv=st.session_state["pending_target_csv"],
                            user=pld["user"],
                            handed=pld["handed"],
                            height=pld["height"],
                            weight=pld["weight"],
                            remarks=pld.get("remarks", ""),
                        )
                        st.session_state["pending_confirm"] = False
                        st.session_state["pending_target_csv"] = ""
                        st.session_state["pending_payload"] = {}
                        st.success("上書き保存しました。")
                        st.rerun()
                    st.button("✅ 上書きする", key="confirm_overwrite", on_click=do_confirm)

                with c2:
                    def cancel_confirm():
                        st.session_state["pending_confirm"] = False
                        st.session_state["pending_target_csv"] = ""
                        st.session_state["pending_payload"] = {}
                    st.button("❌ キャンセル", key="cancel_overwrite", on_click=cancel_confirm)



# 下カラム（閲覧専用 + 絞り込み + 解析起動）
with st.container(border=True):
    st.subheader("(解析/レポート)ビュー")
    # フィルタUI
    col1, col2 = st.columns([0.7, 0.3]) 
    with col1:
        default_start_r, default_end_r = get_date_defaults(df_all)
        cols_r = st.columns([0.7, 0.3])
        with cols_r[0]:
            picked_range_r = st.date_input(
                "表示する日付範囲",
                value=(default_start_r, default_end_r),
                help="この期間の計測だけを右の対象リストに出します",
                key="right_date_range",
            )
        with cols_r[1]:
            st.markdown("<div style='height:30px'></div>", unsafe_allow_html=True)
            st.button("本日に設定", key="btn_right_today", on_click=_set_right_today)
    
    with col2:
        user_choices = get_user_choices(df_all)
        chosen_user = st.selectbox(
            "ユーザーで絞り込み",
            key = "right_user_filter",
            options=user_choices,
            index=0,
        )

    if isinstance(picked_range_r, (list, tuple)) and len(picked_range_r) == 2:
        start_date, end_date = picked_range_r
    elif isinstance(picked_range_r, datetime.date):
        start_date, end_date = picked_range_r, picked_range_r
    else:
        start_date, end_date = default_start_r, default_end_r

    start_dt = datetime.datetime.combine(start_date, datetime.time.min)
    end_dt   = datetime.datetime.combine(end_date, datetime.time.max)

    # フィルタ後データ
    df_for_view = filter_df_for_display(df_all, chosen_user, start_dt, end_dt)

    st.markdown("#### 計測データ一覧")
    column_cfg = {
        SELECT_COL: st.column_config.CheckboxColumn("選択", default=False),
        "csv_path": st.column_config.TextColumn("csv_path", disabled=True),
        "Date":     st.column_config.TextColumn("Date",     disabled=True),
        "Time":     st.column_config.TextColumn("Time",     disabled=True),
        "user":   st.column_config.TextColumn("user",   disabled=True),
        "競技":    st.column_config.TextColumn("競技",    disabled=True),
        "身長":     st.column_config.TextColumn("身長",     disabled=True),
        "体重":     st.column_config.TextColumn("体重",     disabled=True),
        "備考":     st.column_config.TextColumn("備考",     disabled=True),
    }

    view_cols = [SELECT_COL] + [c for c in DISPLAY_COLS] 
    edited = st.data_editor(
        df_for_view[view_cols],
        hide_index=True,
        key="datalist_editor",
        column_config=column_cfg,
    )


    # 解析ビュー起動
    st.markdown("#### 解析ビュー起動")
    if st.button("🚀 新規タブで解析ビューを開く"):
        sel_mask = edited[SELECT_COL] == True
        selected_rows = edited[sel_mask].copy()

        if selected_rows.empty:
            st.warning("先に一覧で1行以上チェックしてください。")
        else:
            base_url = f"http://{SERVER_ADDR}:8503"
            initial_tab = "graph"

            urls = []
            for _, r in selected_rows.iterrows():
                fname = str(r["csv_path"]).strip()
                if not fname:
                    continue
                abs_path = (DATA_DIR / fname).resolve()
                encoded_csv_path = quote(str(abs_path))
                url = f"{base_url}/?csv_path={encoded_csv_path}&tab={initial_tab}"
                urls.append(url)

            if not urls:
                st.warning("有効な csv_path がありませんでした。")
            else:
                js_lines = ["<script>", "const urls = ["]
                for u in urls:
                    js_lines.append(f'    "{u}",')
                js_lines.append("];")
                js_lines.append("for (const link of urls) {{ window.open(link, '_blank'); }}")
                js_lines.append("</script>")
                js_code = "\n".join(js_lines)

                st.components.v1.html(js_code, height=0, scrolling=False)



    # チェック済みプレビュー
    st.markdown("#### 現在チェックされている行（デバッグ用　最終的には削除）")
    sel_mask_prev = edited[SELECT_COL] == True
    sel_prev = edited[sel_mask_prev].copy()
    if sel_prev.empty:
        st.info("まだチェックされていません。")
    else:
        prev_list = []
        for _, r in sel_prev.iterrows():
            prev_list.append({
                "csv_path": r["csv_path"],
                "user": r["user"],
                "Date": r["Date"],
                "Time": r["Time"],
            })
        st.write(prev_list)



