import cv2
import face_recognition

# 初始化攝影機
video_capture = cv2.VideoCapture(0)

# 初始化左眼和右眼的座標及大小變數
LEX, LEY, LEW, LEH = None, None, None, None
REX, REY, REW, REH = None, None, None, None

# 初始化左眼和右眼的偵測器
load_path = cv2.data.haarcascades
left_eye_cascade = cv2.CascadeClassifier(load_path + "haarcascade_lefteye_2splits.xml")
right_eye_cascade = cv2.CascadeClassifier(
    load_path + "haarcascade_righteye_2splits.xml"
)

# 運行攝影機
while True:
    # 抓取每一幀
    ret, frame = video_capture.read()

    # 將每一幀的大小調整為 1/4 大小以加快人臉識別處理速度
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # 將圖像從 BGR 色彩（OpenCV 使用）轉換為 RGB 色彩（face_recognition 使用）
    rgb_small_frame = small_frame[:, :, ::-1]

    # 在當前幀中找到所有人臉和人臉編碼（速度跟效能介於 hog 跟 cnn 之間）
    face_locations = face_recognition.face_locations(
        rgb_small_frame, number_of_times_to_upsample=2, model="small"
    )

    # 遍歷每個人臉位置
    for top, right, bottom, left in face_locations:
        # 因為我們在 1/4 大小的幀中檢測到的人臉位置，所以將其放大回原始尺寸
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        # 在人臉周圍畫一個框
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 4)

        # 擷取左眼區域並檢測左眼
        left_eye_region = frame[top:bottom, left : int((left + right) / 2)]
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
        right_eye_region = frame[top:bottom, int((left + right) / 2) : right]
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
        if len(left_eyes) > 0 and len(right_eyes) > 0:
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

    # 顯示畫面
    cv2.imshow("Is Your Eye Pretty ?", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
