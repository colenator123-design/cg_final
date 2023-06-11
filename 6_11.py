import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import mediapipe as mp
import math

center = (0, 0)
dZ = 3.0
mp_face_mesh = mp.solutions.face_mesh

# 初始化Face Mesh模型
face_mesh = mp_face_mesh.FaceMesh()

def draw_cube():
    glBegin(GL_QUADS)
    glColor3f(1, 0, 0) # 设置颜色为红色
    glVertex3f(-0.5, -0.5, 0.5) # 设置顶点
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    
    glColor3f(0, 1, 0) # 设置颜色为绿色
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    
    glColor3f(0, 0, 1) # 设置颜色为蓝色
    glVertex3f(-0.5, 0.5, -0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, 0.5, -0.5)
    
    glColor3f(1, 1, 0) # 设置颜色为黄色
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, -0.5, 0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    
    glColor3f(0, 1, 1) # 设置颜色为青色
    glVertex3f(0.5, -0.5, -0.5)
    glVertex3f(0.5, 0.5, -0.5)
    glVertex3f(0.5, 0.5, 0.5)
    glVertex3f(0.5, -0.5, 0.5)
    
    glColor3f(1, 0, 1) # 设置颜色为品红色
    glVertex3f(-0.5, -0.5, -0.5)
    glVertex3f(-0.5, -0.5, 0.5)
    glVertex3f(-0.5, 0.5, 0.5)
    glVertex3f(-0.5, 0.5, -0.5)
    glEnd()

def draw_grid():
    # 畫一個立方體
    glBegin(GL_LINES)
    for i in range (-10, 20):
        color = (i + 10.0) / 30.0
        glColor3f(color, color, color)
        # 前面
        glVertex3f(-1.75, -1, i)
        glVertex3f(-1.75, 1, i)
        glVertex3f(-1.75, 1, i)
        glVertex3f(1.75, 1, i)
        glVertex3f(1.75, 1, i)
        glVertex3f(1.75, -1, i)
        glVertex3f(1.75, -1, i)
        glVertex3f(-1.75, -1, i)
        # 邊線
        for j in range (-7, 8):
            j /= 4.0
            glVertex3f(j, -1, i)
            glVertex3f(j, -1, i-1)
            glVertex3f(j, 1, i)
            glVertex3f(j, 1, i-1)
            glVertex3f(j, 1, i)
            glVertex3f(j, 1, i-1)
            glVertex3f(j, -1, i)
            glVertex3f(j, -1, i-1)
            
        for j in range (-10, 11):
            j /= 10.0
            glVertex3f(-1.75, j, i)
            glVertex3f(-1.75, j, i-1)
            glVertex3f(-1.75, j, i)
            glVertex3f(-1.75, j, i-1)
            glVertex3f(1.75, j, i)
            glVertex3f(1.75, j, i-1)
            glVertex3f(1.75, j, i)
            glVertex3f(1.75, j, i-1)
    glEnd()
    
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    width, height = glfw.get_framebuffer_size(window)
    aspect_ratio = width / height
    gluPerspective(45, aspect_ratio, 0.1, 100)
    
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # 調整視角位置
    
    eyeX = -1.75 + 3.5 * (center[0] / 640.0)
    eyeY = 1 - 2 * (center[1] / 480.0)
    gluLookAt(eyeX, eyeY, dZ/10.0, 0, 0, 0, 0, 1, 0)
    
    draw_grid()
    draw_cube()
    glfw.swap_buffers(window)

if __name__ == '__main__':
    cap=cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    detector=FaceDetector(minDetectionCon=0.8)
    
    if not glfw.init():
        raise Exception("glfw init failed")
    
    window = glfw.create_window(1280, 720, "OpenGL Window", None, None)
    if not window:
        glfw.terminate()
        raise Exception("glfw window creation failed")
    
    glfw.make_context_current(window)
    glEnable(GL_DEPTH_TEST)
    
    while not glfw.window_should_close(window):
        glfw.poll_events()
        display()
        if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
            glfw.set_window_should_close(window, True)    
        #capture
        success,img=cap.read()
        results = face_mesh.process(img)
        img = cv2.flip(img, flipCode=1) # 左右翻轉圖像
        img, bboxs = detector.findFaces(img)
        #設定臉部中心座標
        if bboxs:
            center = bboxs[0]["center"]
            
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
            image_rows, image_cols, _ = img.shape
            left_eye_x, left_eye_y = int(left_eye_coords.x * image_cols), int(left_eye_coords.y * image_rows)
            right_eye_x, right_eye_y = int(right_eye_coords.x * image_cols), int(right_eye_coords.y * image_rows)
            left_x, left_y = int(left_coords.x * image_cols), int(left_coords.y * image_rows)
            right_x, right_y = int(right_coords.x * image_cols), int(right_coords.y * image_rows)
            #計算距離
            width = image_cols
            height = image_rows 
            dx = left_x - left_eye_x
            dX = 3.5
            normalizedFocaleX = 1.40625
            fx = min(width, height) * normalizedFocaleX
            dZ = (fx * (dX / dx))            

        cv2.imshow("image",img)
        if cv2.waitKey(1) == 27: # 按下 ESC 鍵退出程序並釋放攝像頭
            break
    
    cap.release()  # 釋放攝像頭
    cv2.destroyAllWindows()  # 關閉所有視窗
    glfw.terminate()
    
  
