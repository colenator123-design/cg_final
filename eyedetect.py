import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# 初始化Face Mesh模型
face_mesh = mp_face_mesh.FaceMesh()

# 開啟視訊鏡頭
cap = cv2.VideoCapture(0)

while cap.isOpened():
    # 讀取一幀影像
    success, image = cap.read()
    if not success:
        print("無法讀取視訊鏡頭中的影像，請確認是否連接正常。")
        break
    
    # 將影像傳入Face Mesh模型進行偵測
    results = face_mesh.process(image)

    # 檢查是否偵測到臉部關鍵點
    if results.multi_face_landmarks:
        # 取得第一個偵測到的臉部關鍵點
        face_landmarks = results.multi_face_landmarks[0]
        
        # 取得眼睛的關鍵點索引
        left_eye_landmark_index = 362
        left = 359
        right_eye_landmark_index = 133
        right = 130

        # 取得左眼和右眼的座標
        left_eye_coords = face_landmarks.landmark[left_eye_landmark_index]
        left_coords = face_landmarks.landmark[left]
        right_eye_coords = face_landmarks.landmark[right_eye_landmark_index]
        right_coords = face_landmarks.landmark[right]

        # 將座標轉換為畫面上的位置
        image_rows, image_cols, _ = image.shape
        left_eye_x, left_eye_y = int(left_eye_coords.x * image_cols), int(left_eye_coords.y * image_rows)
        right_eye_x, right_eye_y = int(right_eye_coords.x * image_cols), int(right_eye_coords.y * image_rows)
        left_x, left_y = int(left_coords.x * image_cols), int(left_coords.y * image_rows)
        right_x, right_y = int(right_coords.x * image_cols), int(right_coords.y * image_rows)

        # 在畫面上繪製眼睛的位置
        cv2.circle(image, (left_eye_x, left_eye_y), 5, (0, 255, 0), -1)
        cv2.circle(image, (right_eye_x, right_eye_y), 5, (0, 255, 0), -1)
        cv2.circle(image, (left_x, left_y), 5, (0, 255, 0), -1)
        cv2.circle(image, (right_x, right_y), 5, (0, 255, 0), -1)
    
    # 顯示畫面
    cv2.imshow('Face Mesh Eye Detection', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# 釋放資源
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
