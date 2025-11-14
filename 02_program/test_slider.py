import streamlit as st
import cv2
from pathlib import Path

st.set_page_config(page_title="スライダーテスト", layout="wide")
st.title("🎬 スライダーテスト")

# 動画ファイルのパス
video_path = Path("aa.mp4")

# 動画情報を取得
@st.cache_resource
def get_video_info(path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None, 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return cap, total_frames

# フレームを取得
def get_frame(path, frame_number):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    cap.release()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None

# 動画情報を読み込み
if video_path.exists():
    _, total_frames = get_video_info(video_path)
    
    st.write(f"総フレーム数: {total_frames}")
    
    # スライダー
    frame_num = st.slider("フレーム番号", 0, total_frames - 1, 0)
    
    st.write(f"選択されたフレーム: {frame_num}")
    
    # フレームを表示
    frame = get_frame(video_path, frame_num)
    if frame is not None:
        st.image(frame, channels="RGB", use_container_width=True)
else:
    st.error(f"動画ファイルが見つかりません: {video_path}")

