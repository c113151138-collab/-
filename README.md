# 影像處理實驗室 — 說明文件

## 安裝與執行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 功能對照
| 需求 | 實作 |
|---|---|
| 影像載入 | PIL 讀取 PNG/JPG/BMP，自動 EXIF 轉正、轉為 RGB，顯示解析度、色彩維度、檔案大小 |
| 色彩空間 | Original / R / G / B（RGB 分量）與 Y / U / V（`cv2.COLOR_RGB2YUV`） |
| 直方圖 | 橫軸像素強度 0–255、縱軸像素數量；切換模式即時重繪（Original 為 RGB 三色疊圖） |
| 卷積 | 自訂奇數階方陣核（3×3、5×5…）、預設核、三種正規化、輸出偏移、並排/疊加比對 |
| PSNR | 原圖 vs 濾波後，即時顯示 dB 與 MSE |
| 人臉偵測 | OpenCV Haar-Cascade，畫出邊界框、顯示人臉總數與信心分數 |

## 演算法與公式

### 2D 卷積
對核 \(K\)（大小 \((2r+1)\times(2r+1)\)）：

$$g(x,y)=\sum_{i=-r}^{r}\sum_{j=-r}^{r}K(i,j)\,f(x-i,\,y-j)\;+\;\delta$$

實作上以核旋轉 180° 後做相關運算（`cv2.flip` + `cv2.filter2D`），邊界採鏡射延拓，結果裁切至 [0,255]。\(\delta\) 為輸出偏移。

### 正規化
- 除以總和：\(K' = K/\sum K\)（總和為 0 時不縮放，如邊緣/浮雕核）
- 除以絕對值總和：\(K' = K/\sum |K|\)

### MSE 與 PSNR
影像大小 \(M\times N\)、通道數 \(C\)，原圖 \(I\)、濾波後 \(K\)：

$$\mathrm{MSE}=\frac{1}{MNC}\sum_{c}\sum_{x,y}\bigl[I(x,y,c)-K(x,y,c)\bigr]^2$$

$$\mathrm{PSNR}=10\log_{10}\!\left(\frac{\mathrm{MAX}_I^{2}}{\mathrm{MSE}}\right)=20\log_{10}\!\left(\frac{255}{\sqrt{\mathrm{MSE}}}\right)\ \text{dB}$$

8 位元影像 \(\mathrm{MAX}_I=255\)；MSE=0 時 PSNR 為 ∞。PSNR 越低，表示濾波造成的差異越大（銳化、浮雕通常很低，輕微平滑較高）。

### 人臉偵測
使用 Viola–Jones 的 Haar-Cascade（`haarcascade_frontalface_default.xml`，隨 OpenCV 內建，無需另外下載）。先轉灰階並直方圖等化，再以 `detectMultiScale3(outputRejectLevels=True)` 取得邊界框與信心分數（levelWeights）。

## 限制
- Haar-Cascade 對側臉、遮擋、強烈光影較敏感；可調整 scaleFactor / minNeighbors。
- 帶透明通道的圖片會被轉為 RGB（透明處以黑色呈現）。
