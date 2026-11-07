# -*- coding: utf-8 -*-
"""影像處理實驗室 (Image Lab) - Streamlit
執行: streamlit run app.py
"""
import io
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image, ImageOps

st.set_page_config(page_title="影像處理實驗室", layout="wide")

# ---------------------------------------------------------------- 工具函式
def load_image(file) -> tuple[np.ndarray, dict]:
    """讀入 PNG/JPG/BMP，回傳 RGB uint8 陣列與中繼資訊。"""
    raw = file.getvalue()
    pil = Image.open(io.BytesIO(raw))
    fmt, mode = pil.format, pil.mode
    pil = ImageOps.exif_transpose(pil).convert("RGB")  # 去除 alpha、統一 3 通道
    arr = np.array(pil)
    h, w = arr.shape[:2]
    info = {
        "檔名": file.name,
        "格式": fmt,
        "解析度 (寬×高)": f"{w} × {h} px",
        "原始色彩模式": mode,
        "色彩維度 (處理用)": f"{arr.shape[2]} 通道 (RGB), {arr.dtype}",
        "檔案大小": f"{len(raw)/1024:.1f} KB",
        "像素總數": f"{w*h:,}",
    }
    return arr, info


def channel_views(rgb: np.ndarray) -> dict:
    """回傳各顯示模式 -> (影像, 直方圖資料)。"""
    yuv = cv2.cvtColor(rgb, cv2.COLOR_RGB2YUV)
    v = {"Original": rgb}
    for i, n in enumerate("RGB"):
        v[n] = rgb[:, :, i]
    for i, n in enumerate("YUV"):
        v[n] = yuv[:, :, i]
    return v


def plot_hist(mode: str, img: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 3))
    bins = np.arange(257)
    if mode == "Original":
        for i, (c, n) in enumerate(zip("rgb", "RGB")):
            h = np.bincount(img[:, :, i].ravel(), minlength=256)
            ax.plot(range(256), h, color=c, label=n, lw=1)
        ax.legend()
        ax.set_title("Histogram - Original (R/G/B)")
    else:
        h = np.bincount(img.ravel(), minlength=256)
        col = {"R": "r", "G": "g", "B": "b"}.get(mode, "gray")
        ax.bar(range(256), h, width=1.0, color=col)
        ax.set_title(f"Histogram - {mode} channel")
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity (0-255)")
    ax.set_ylabel("Pixel count")
    fig.tight_layout()
    return fig


PRESETS = {
    "恆等 (Identity)": "0 0 0\n0 1 0\n0 0 0",
    "銳化 (Sharpen)": "0 -1 0\n-1 5 -1\n0 -1 0",
    "邊緣偵測 (Laplacian)": "-1 -1 -1\n-1 8 -1\n-1 -1 -1",
    "高斯平滑 3×3": "1 2 1\n2 4 2\n1 2 1",
    "高斯平滑 5×5": "1 4 6 4 1\n4 16 24 16 4\n6 24 36 24 6\n4 16 24 16 4\n1 4 6 4 1",
    "浮雕 (Emboss)": "-2 -1 0\n-1 1 1\n0 1 2",
    "均值模糊 3×3": "1 1 1\n1 1 1\n1 1 1",
    "自訂": "0 0 0\n0 1 0\n0 0 0",
}


def parse_kernel(text: str) -> np.ndarray:
    rows = [r for r in text.replace(",", " ").strip().splitlines() if r.strip()]
    k = np.array([[float(x) for x in r.split()] for r in rows], dtype=np.float64)
    if k.ndim != 2 or k.shape[0] != k.shape[1] or k.shape[0] % 2 == 0:
        raise ValueError(f"卷積核必須為奇數階方陣 (3×3、5×5…)，目前為 {k.shape}")
    return k


def normalize_kernel(k: np.ndarray, how: str) -> np.ndarray:
    if how == "除以總和 (sum=1)":
        s = k.sum()
        return k / s if abs(s) > 1e-12 else k  # 總和為 0 (邊緣/浮雕類) 不縮放
    if how == "除以絕對值總和":
        s = np.abs(k).sum()
        return k / s if s > 0 else k
    return k


def convolve(rgb: np.ndarray, k: np.ndarray, offset: float) -> np.ndarray:
    """真正的卷積 = 旋轉 180° 的相關運算；邊界採鏡射 (reflect101)。"""
    kf = cv2.flip(k, -1)
    out = cv2.filter2D(rgb.astype(np.float32), cv2.CV_32F, kf.astype(np.float32),
                       delta=offset, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(out, 0, 255).astype(np.uint8)


def psnr(ref: np.ndarray, test: np.ndarray) -> tuple[float, float]:
    mse = np.mean((ref.astype(np.float64) - test.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf"), 0.0
    return 10 * np.log10(255.0 ** 2 / mse), mse


@st.cache_resource
def get_cascade():
    if not hasattr(cv2, "CascadeClassifier"):
        raise RuntimeError(f"目前 OpenCV {cv2.__version__} 不含 CascadeClassifier，請安裝 opencv-python>=4.8,<5")
    path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(path)
    if clf.empty():
        raise RuntimeError(f"無法載入 Haar 模型：{path}")
    return clf


def detect_faces(rgb, scale, neighbors, min_size):
    gray = cv2.equalizeHist(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    boxes, _, weights = get_cascade().detectMultiScale3(
        gray, scaleFactor=scale, minNeighbors=neighbors,
        minSize=(min_size, min_size), outputRejectLevels=True)
    out = rgb.copy()
    res = []
    for (x, y, w, h), s in zip(boxes, np.ravel(weights)):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), max(2, rgb.shape[1] // 400))
        cv2.putText(out, f"{s:.2f}", (x, max(15, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    max(0.5, rgb.shape[1] / 1200), (255, 255, 0), 2)
        res.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h), "信心分數": round(float(s), 3)})
    return out, res


# ---------------------------------------------------------------- 介面
st.title("🖼️ 影像處理實驗室")
up = st.sidebar.file_uploader("載入影像 (PNG / JPG / BMP)", type=["png", "jpg", "jpeg", "bmp"])
if up is None:
    st.info("請從左側上傳影像開始。")
    st.stop()

try:
    rgb, info = load_image(up)
except Exception as e:
    st.error(f"影像讀取失敗：{e}")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["① 影像資訊", "② 色彩空間與直方圖", "③ 卷積濾波 & PSNR", "④ 人臉偵測", "📘 說明"])

with tab1:
    c1, c2 = st.columns([2, 1])
    c1.image(rgb, caption="原始影像", use_container_width=True)
    c2.table(info)

with tab2:
    views = channel_views(rgb)
    mode = st.radio("顯示模式", ["Original", "R", "G", "B", "Y", "U", "V"], horizontal=True,
                    format_func=lambda m: {"Original": "Original", "R": "R 通道", "G": "G 通道", "B": "B 通道",
                                           "Y": "Y 亮度", "U": "U 色度", "V": "V 色度"}[m])
    img = views[mode]
    a, b = st.columns(2)
    a.image(img, caption=mode, use_container_width=True, clamp=True)
    fig = plot_hist(mode, img)
    b.pyplot(fig)
    plt.close(fig)

with tab3:
    left, right = st.columns([1, 2])
    with left:
        name = st.selectbox("預設核", list(PRESETS))
        text = st.text_area("卷積核 (每列一行，數值以空白或逗號分隔)", PRESETS[name], height=170,
                            key=f"k_{name}")
        norm = st.selectbox("正規化", ["不正規化", "除以總和 (sum=1)", "除以絕對值總和"])
        offset = st.slider("輸出偏移 (浮雕建議 128)", 0, 255, 128 if "浮雕" in name else 0)
    try:
        k0 = parse_kernel(text)
        k = normalize_kernel(k0, norm)
        filt = convolve(rgb, k, offset)
        p, mse = psnr(rgb, filt)
        with left:
            st.caption(f"核大小 {k.shape[0]}×{k.shape[1]}，總和 = {k.sum():.4f}")
            st.dataframe(np.round(k, 4), hide_index=True)
        with right:
            view = st.radio("比對方式", ["並排", "疊加 (混合滑桿)"], horizontal=True)
            if view == "並排":
                x, y = st.columns(2)
                x.image(rgb, caption="濾波前", use_container_width=True)
                y.image(filt, caption="濾波後", use_container_width=True)
            else:
                alpha = st.slider("濾波後比例", 0.0, 1.0, 0.5)
                st.image(cv2.addWeighted(rgb, 1 - alpha, filt, alpha, 0), use_container_width=True)
            m1, m2 = st.columns(2)
            m1.metric("PSNR (dB)", "∞ (完全相同)" if np.isinf(p) else f"{p:.2f}")
            m2.metric("MSE", f"{mse:.2f}")
        st.session_state["filtered"] = filt
    except Exception as e:
        right.error(f"卷積核輸入錯誤：{e}")

with tab4:
    src = st.radio("偵測來源", ["原始影像", "濾波後影像"], horizontal=True)
    c1, c2, c3 = st.columns(3)
    sf = c1.slider("scaleFactor", 1.05, 1.5, 1.1, 0.01)
    nb = c2.slider("minNeighbors", 1, 10, 5)
    ms = c3.slider("最小人臉尺寸 (px)", 20, 200, 40)
    base = rgb if src == "原始影像" else st.session_state.get("filtered", rgb)
    out, faces = detect_faces(base, sf, nb, ms)
    st.metric("偵測到的人臉數量", len(faces))
    st.image(out, use_container_width=True)
    if faces:
        st.dataframe(faces, hide_index=True)
    st.caption("信心分數為 Haar-Cascade 最後一級分類器的 levelWeight，非 0–1 機率；數值越大越可信。")

with tab5:
    st.markdown(open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md")
                else "請參考 README.md")
