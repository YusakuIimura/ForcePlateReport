import streamlit as st
import pandas as pd
from pathlib import Path
from urllib.parse import quote
import re
import datetime

st.set_page_config(page_title="Home", layout="wide")

DATA_DIR = Path("data")
DATALIST_PATH = DATA_DIR / "datalist.csv"
USERLIST_PATH = DATA_DIR / "userlist.csv"

SELECT_COL = "_select_"
TS_COL = "_ts"
DISPLAY_COLS = ["csv_path", "Date", "Time", "user", "利き手", "身長", "体重"]


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
            columns=["csv_path", "Date", "Time", "user", "利き手", "身長", "体重"]
        )

    df = pd.read_csv(path)
    for col in ["csv_path", "Date", "Time", "user", "利き手", "身長", "体重"]:
        if col not in df.columns:
            df[col] = ""
    return df[["csv_path", "Date", "Time", "user", "利き手", "身長", "体重"]].copy()


def load_userlist(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["user", "利き手", "身長", "体重"])

    df = pd.read_csv(path)
    for col in ["user", "利き手", "身長", "体重"]:
        if col not in df.columns:
            df[col] = ""
    return df[["user", "利き手", "身長", "体重"]].copy()


# -----------------
# データ構築関連
# -----------------

def build_df_all() -> pd.DataFrame:
    """
    data/*.csv と datalist.csv と userlist.csv を統合して返す。
    常に csv_path は1行に潰して返す。
    """
    base_df = list_fp_files(DATA_DIR)        # csv_path, Date, Time
    dl_df   = load_datalist(DATALIST_PATH)   # csv_path, Date, Time, user, 利き手, 身長, 体重
    pl_df   = load_userlist(USERLIST_PATH)  # user, 利き手, 身長, 体重

    # datalist は csv_path ごとに1行だけ残す
    dl_df_unique = (
        dl_df.sort_values(["csv_path", "Date", "Time"])
             .drop_duplicates(subset=["csv_path"], keep="last")
    )[["csv_path", "user"]].copy()

    # userlist も user ごとに1行だけ残す
    pl_df_unique = (
        pl_df.sort_values(["user"])
             .drop_duplicates(subset=["user"], keep="last")
    )[["user", "利き手", "身長", "体重"]].copy()

    # dataフォルダにあるcsvをベースにuserをJOIN
    merged = pd.merge(
        base_df,
        dl_df_unique,  # -> adds 'user'
        on="csv_path",
        how="left",
    )

    # user情報から利き手/身長/体重をJOIN
    merged = pd.merge(
        merged,
        pl_df_unique,
        on="user",
        how="left",
    )

    # 欠損補完
    for col in ["user", "利き手", "身長", "体重"]:
        if col not in merged.columns:
            merged[col] = ""
    merged["user"] = merged["user"].fillna("")
    merged["利き手"] = merged["利き手"].fillna("")
    merged["身長"] = merged["身長"].fillna("")
    merged["体重"] = merged["体重"].fillna("")

    # タイムスタンプ列（フィルタ用）
    def to_ts(row):
        try:
            return pd.to_datetime(
                str(row["Date"]) + " " + str(row["Time"]),
                errors="coerce"
            )
        except Exception:
            return pd.NaT
    merged[TS_COL] = merged.apply(to_ts, axis=1)

    # 念のためここでも csv_path でユニーク化
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
    return ["(すべて)"] + vals


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
    """
    左カラム表示用のフィルタ。
    編集はもうしないので型はそのままでOK。
    """
    df = df_all.copy()

    # user絞り
    if chosen_user != "(すべて)":
        df = df[df["user"].astype(str).str.strip() == chosen_user]

    # 日付範囲絞り
    mask_valid = df[TS_COL].notna()
    mask_range = (df[TS_COL] >= start_dt) & (df[TS_COL] <= end_dt)
    df = df[mask_valid & mask_range]

    # 画面表示用のみに整える
    df = df[DISPLAY_COLS + [SELECT_COL]].copy()

    return df


def write_userlist(user: str, handed: str, height: str, weight: str):
    """
    userlist.csv を (user をキーに) 追加 or 上書き。
    """
    pl_df = load_userlist(USERLIST_PATH)
    mask = pl_df["user"].astype(str) == str(user)

    if mask.any():
        pl_df.loc[mask, ["利き手", "身長", "体重"]] = [handed, height, weight]
    else:
        new_row = pd.DataFrame([{
            "user": user,
            "利き手": handed,
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
    out = df_all_current[["csv_path", "Date", "Time", "user", "利き手", "身長", "体重"]].copy()

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
                               weight: str):
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
        df_all_current["csv_path"] == target_csv, "利き手"
    ] = handed
    df_all_current.loc[
        df_all_current["csv_path"] == target_csv, "身長"
    ] = height
    df_all_current.loc[
        df_all_current["csv_path"] == target_csv, "体重"
    ] = weight

    # 4. datalist.csv を再生成
    rebuild_and_save_datalist(df_all_current)


# -----------------
# UI
# -----------------

st.title("ユーザー情報入力・解析データ選択画面")
st.caption("計測データにユーザー情報を追加し、解析ビューを起動してください")

# 最新ビュー
df_all = build_df_all()

left_col, right_col = st.columns([0.6, 0.4])

# 左カラム（閲覧専用 + 絞り込み + 解析起動）
with left_col:

    # フィルタUI
    col1, col2 = st.columns([0.5, 0.5]) 
    with col1:
        default_start, default_end = get_date_defaults(df_all)
        picked_range = st.date_input(
            "表示する日付範囲",
            value=(default_start, default_end),
            help="この期間の計測だけを表示します",
        )
    
    with col2:
        user_choices = get_user_choices(df_all)
        chosen_user = st.selectbox(
            "ユーザーで絞り込み",
            options=user_choices,
            index=0,
        )

    if isinstance(picked_range, (list, tuple)) and len(picked_range) == 2:
        start_date, end_date = picked_range
    elif isinstance(picked_range, datetime.date):
        start_date, end_date = picked_range, picked_range
    else:
        start_date, end_date = default_start, default_end

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
        "利き手":    st.column_config.TextColumn("利き手",    disabled=True),
        "身長":     st.column_config.TextColumn("身長",     disabled=True),
        "体重":     st.column_config.TextColumn("体重",     disabled=True),
    }

    # 編集不可にしたいが、チェックボックスは使いたいなら:
    # -> user/利き手/身長/体重 も disabled=True、CheckboxColumn はそのまま
    # data_editorは返り値を受け取れるので後で解析起動に使える
    edited = st.data_editor(
        df_for_view,
        hide_index=True,
        key="datalist_editor",
        column_config=column_cfg,
    )

    # datalist.csvの更新UIはもう置かない（削除済み）

    # 解析ビュー起動
    st.markdown("#### 解析ビュー起動")
    if st.button("🚀 新規タブで解析ビューを開く"):
        sel_mask = edited[SELECT_COL] == True
        selected_rows = edited[sel_mask].copy()

        if selected_rows.empty:
            st.warning("先に一覧で1行以上チェックしてください。")
        else:
            base_url = "http://localhost:8502"
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
                js_lines.append("for (const link of urls) { window.open(link, '_blank'); }")
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

# 右カラム（動画見ながら1件更新）
with right_col:
    st.subheader("動画を参照し、ユーザー情報を入力")

    # ==== 1. 対象CSV選択 ====
    all_csv_options = df_all["csv_path"].tolist()
    if not all_csv_options:
        st.info("dataフォルダに *_FP.csv がありません。")
    else:
        target_csv = st.selectbox(
            "対象データ (csv)",
            options=all_csv_options,
            index=0,
            help="この計測を誰のものか決めます",
            key="target_csv_select",
        )

        # このcsvに現在割り当たってる値を取得
        row_now = df_all[df_all["csv_path"] == target_csv].head(1)
        current_user_val = str(row_now["user"].iloc[0]) if not row_now.empty and pd.notna(row_now["user"].iloc[0]) else ""
        current_handed_val = str(row_now["利き手"].iloc[0]) if not row_now.empty and pd.notna(row_now["利き手"].iloc[0]) else ""
        current_height_val = str(row_now["身長"].iloc[0]) if not row_now.empty and pd.notna(row_now["身長"].iloc[0]) else ""
        current_weight_val = str(row_now["体重"].iloc[0]) if not row_now.empty and pd.notna(row_now["体重"].iloc[0]) else ""

        # ==== 2. セッション初期化 ====
        for key in [
            "edit_user", "edit_handed", "edit_height", "edit_weight",
            "bound_csv",
            "pending_confirm",          # ← 確認待ちフラグ
            "pending_target_csv",       # ← 確認対象のcsv
            "pending_payload",          # ← 保存予定の内容
        ]:
            if key not in st.session_state:
                st.session_state[key] = "" if key != "pending_confirm" else False

        # CSV切り替え時はフォームを最新状態でリセットし、確認フラグも解除
        if st.session_state["bound_csv"] != target_csv:
            st.session_state["edit_user"] = current_user_val
            st.session_state["edit_handed"] = current_handed_val
            st.session_state["edit_height"] = current_height_val
            st.session_state["edit_weight"] = current_weight_val
            st.session_state["bound_csv"] = target_csv
            st.session_state["pending_confirm"] = False
            st.session_state["pending_target_csv"] = ""
            st.session_state["pending_payload"] = {}

        # ==== 3. 動画プレビュー ====
        st.markdown("##### 動画プレビュー")
        mp4_candidate = (DATA_DIR / target_csv).with_suffix(".mp4")
        if mp4_candidate.exists():
            st.video(str(mp4_candidate))
        else:
            st.info("対応する動画(.mp4)が見つかりませんでした。")

        # ==== 4. ユーザー情報（左:既存, 真ん中:矢印, 右:フォーム） ====
        st.markdown("##### ユーザー情報")
        st.caption("既存ユーザーリストからデータを読み込み採用  \nもしくは新規にユーザー情報を記入しデータベースを更新してください")

        pl_df = load_userlist(USERLIST_PATH)
        existing_users = (
            pl_df["user"]
            .astype(str)
            .fillna("")
            .str.strip()
            .replace("nan", "")
            .tolist()
        )
        existing_users = sorted({p for p in existing_users if p})

        left_col_inner, mid_col, right_col_inner = st.columns([0.3, 0.2, 0.5])

        # 左：既存プレイヤー選択
        with left_col_inner:
            chosen_existing_user = st.selectbox(
                "ユーザーリスト",
                options=["(選ばない)"] + existing_users,
                index=0,
                key="existing_user_select",
                help="選んで➡を押すと右フォームに反映されます",
            )

        # 中央：➡ボタン
        with mid_col:
            st.markdown("<div style='height:1.9em'></div>", unsafe_allow_html=True)

            def load_from_existing():
                """選んだ既存プレイヤーの情報をフォームにコピー"""
                if chosen_existing_user == "(選ばない)":
                    st.session_state["edit_user"] = ""
                    st.session_state["edit_handed"] = ""
                    st.session_state["edit_height"] = ""
                    st.session_state["edit_weight"] = ""
                    return
                row_pl = pl_df[pl_df["user"] == chosen_existing_user].head(1)
                if len(row_pl) > 0:
                    st.session_state["edit_user"] = chosen_existing_user
                    st.session_state["edit_handed"] = (
                        str(row_pl["利き手"].iloc[0]) if pd.notna(row_pl["利き手"].iloc[0]) else ""
                    )
                    st.session_state["edit_height"] = (
                        str(row_pl["身長"].iloc[0]) if pd.notna(row_pl["身長"].iloc[0]) else ""
                    )
                    st.session_state["edit_weight"] = (
                        str(row_pl["体重"].iloc[0]) if pd.notna(row_pl["体重"].iloc[0]) else ""
                    )
                # 既存プレイヤーを読み込んだあとも、まだ「pending_confirm」は触らない

            st.button("採用　➡", on_click=load_from_existing, key="btn_load_user")

        # 右：フォーム（タイル配置）
        with right_col_inner:
            tile_cols = st.columns([0.28, 0.18, 0.18, 0.18])
            with tile_cols[0]:
                st.text_input("user名", key="edit_user")
            with tile_cols[1]:
                st.text_input("利き手", key="edit_handed")
            with tile_cols[2]:
                st.text_input("身長", key="edit_height")
            with tile_cols[3]:
                st.text_input("体重", key="edit_weight")

        # ==== 5. 保存ボタン or 上書き確認 ====

        # フォームの内容
        form_user = st.session_state["edit_user"].strip()
        form_handed = (st.session_state["edit_handed"] or "").strip()
        form_height = (st.session_state["edit_height"] or "").strip()
        form_weight = (st.session_state["edit_weight"] or "").strip()

        # userlist 上の既存プロファイルを取得
        def _norm(x): 
            return "" if pd.isna(x) else str(x).strip()

        row_exist = pl_df[pl_df["user"].astype(str).str.strip() == form_user]
        is_existing_user = bool(form_user) and not row_exist.empty

        if is_existing_user:
            # 既存プロファイル（現在の登録値）
            exist_handed = _norm(row_exist["利き手"].iloc[0])
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
                    }
                else:
                    # 新規 or 既存だが値は同一 → そのまま保存
                    assign_user_and_save_all(
                        target_csv=target_csv,
                        user=form_user,
                        handed=form_handed,
                        height=form_height,
                        weight=form_weight,
                    )
                    st.success("保存しました。")
                    st.rerun()

            st.button(
                "💾　データベースを更新",
                key="save_button",
                on_click=on_press_save,
            )

        else:
            # 確認ダイアログ（上書き時のみ）
            pld = st.session_state["pending_payload"]
            old = pl_df[pl_df["user"].astype(str).str.strip() == pld["user"]].head(1)
            old_h = _norm(old["利き手"].iloc[0]); old_ht = _norm(old["身長"].iloc[0]); old_w = _norm(old["体重"].iloc[0])

            st.error(
                f"⚠️ 既存ユーザー『{pld['user']}』の登録値を上書きします。\n\n"
                f"利き手: {old_h} → {pld['handed']}\n"
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


