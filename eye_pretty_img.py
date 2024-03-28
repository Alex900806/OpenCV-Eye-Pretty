import cv2
import face_recognition

# 讀取圖像
frame = cv2.imread("images/profile.jpg")

# 初始化左眼和右眼的座標及大小變數
LEX, LEY, LEW, LEH = None, None, None, None
REX, REY, REW, REH = None, None, None, None

# 載入OpenCV提供的預訓練的臉部與眼睛偵測模型
load_path = cv2.data.haarcascades
face_cascade = cv2.CascadeClassifier(load_path + "haarcascade_frontalface_default.xml")
left_eye_cascade = cv2.CascadeClassifier(load_path + "haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(
    load_path + "haarcascade_righteye_2splits.xml"
)

# 將圖像轉換為灰度
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# 使用人臉偵測器進行人臉偵測
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
face_locations = face_recognition.face_locations(frame)

# 遍歷每個偵測到的人臉
for top, right, bottom, left in face_locations:
    # 繪製人臉矩形
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)

    # 擷取左眼區域並檢測左眼
    left_eye_region = gray[top:bottom, left : int((left + right) / 2)]
    left_eyes = left_eye_cascade.detectMultiScale(
        left_eye_region, scaleFactor=1.1, minNeighbors=7
    )

    # 遍歷左眼偵測結果
    for lex, ley, lew, leh in left_eyes:
        if ley + leh < (bottom - top) / 2:
            # 繪製左眼框
            cv2.rectangle(
                frame,
                (left + lex, top + ley),
                (left + lex + lew, top + ley + leh),
                (0, 255, 0),
                3,
            )
            # 更新左眼座標和大小變數
            LEX, LEY, LEW, LEH = lex, ley, lew, leh

    # 擷取右眼區域並檢測右眼
    right_eye_region = gray[top:bottom, int((left + right) / 2) : right]
    right_eyes = right_eye_cascade.detectMultiScale(
        right_eye_region, scaleFactor=1.1, minNeighbors=7
    )

    # 遍歷右眼偵測結果
    for rex, rey, rew, reh in right_eyes:
        if rey + reh < (bottom - top) / 2:
            # 繪製右眼框
            cv2.rectangle(
                frame,
                (int((left + right) / 2) + rex, top + rey),
                (int((left + right) / 2) + rex + rew, top + rey + reh),
                (0, 255, 255),
                3,
            )
            # 更新右眼座標和大小變數
            REX, REY, REW, REH = rex, rey, rew, reh

        # 計算分數
        if LEX is not None and REX is not None:
            # 計算 score1
            value1 = ((REW + LEW) / 2) * 5 / (right - left)
            score1 = (1 - abs(value1 - 1)) * 100
            score1 = max(0, min(100, score1))

            # 計算 score2
            eye_distance = (int((left + right) / 2) + REX) - (left + LEX + LEW)
            value2 = eye_distance / ((REW + LEW) / 2)
            score2 = (1 - abs(value2 - 1)) * 100
            score2 = max(0, min(100, score2))

            # 計算 score
            score = (score1 + score2) / 2

            # 在人臉下方繪製一個標籤
            cv2.rectangle(
                frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED
            )
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(
                frame,
                "Score: " + str(score1)[:4],
                (left + 6, bottom - 6),
                font,
                0.9,
                (255, 255, 255),
                1,
            )

# 顯示處理後的圖像
cv2.imshow("Is Your Eye Pretty ?", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
