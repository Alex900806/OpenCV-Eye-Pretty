# OpenCV Eye Pretty

這是一個使用 OpenCV 和 Face Recognition 來辨識人臉與眼睛，並測量你的眼睛是否達到黃金比例的系統，測量完成後會給予一個分數供參考，可使用「即時影像」或是「圖片影像」來辨識。

## 功能特色

- 透過人臉與眼睛辨識，算出你眼睛比例的分數，並即時顯示出來。

## 使用技術

- Python, OpenCV, Face Recognition

## 操作概念

- 即時影像

  - 開啟攝影機
  - 開始辨識，將每一幀的大小調整為原來的四分之一，以加快處理速度
  - 辨認人臉的 model 使用速度與效能介於 hog 跟 cnn 的 small
  - 並在人臉的圖像上辨識左眼與右眼，以減少計算量
  - 辨識完成後進行分數計算，且在畫面上顯示分數

- 圖片影像

  - 讀取圖片，並將圖片轉為灰度，增加辨識度
  - 開始人臉辨識，並在上面進行左右眼辨識
  - 辨識完成後進行分數計算，且在畫面上顯示分數

- 眼睛評估方式

  - 眼睛寬度：臉部寬度 = 1：1
  - 眼睛寬度： 左右眼距離 = 1：1

## 操作方法

- 安裝 OpenCV Face Recognition Setuptools（環境無法運行時安裝）

```bash
  pip install opencv-python face_recognition setuptools
```

- 執行程式

```bash
  python eye_pretty_webcam.py
```

```bash
  python eye_pretty_img.py
```

## 學習成果

- 學習使用 OpenCV 的預訓練的臉部和眼睛偵測模型
- 學習使用 OpenCV 進行圖像處理，包括調整圖像大小、轉換圖像顏色空間
- 學習使用 face_recognition 庫進行人臉識別，並標記人臉位置

## 效果展示

![image](https://github.com/Alex900806/OpenCV-Eye-Pretty/blob/main/images/imgTest.png)
