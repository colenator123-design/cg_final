import glfw
from OpenGL.GL import *
from OpenGL.GLU import *
from cvzone.FaceDetectionModule import FaceDetector
import cv2
import mediapipe as mp

center = (0, 0)

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
    for i in range (-10, 5):
        color = (i + 10.0) / 15.0
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
    
    print(center)
    eyeX = -1.75 + 3.5 * (center[0] / 640.0)
    eyeY = 1 - 2 * (center[1] / 480.0)
    gluLookAt(eyeX, eyeY, 3.4, 0, 0, 0, 0, 1, 0)
    
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
        img = cv2.flip(img, flipCode=1) # 左右翻轉圖像
        img, bboxs = detector.findFaces(img)
        if bboxs:
            # bboxInfo - "id","bbox","score","center"
            center = bboxs[0]["center"]
        cv2.imshow("image",img)
        if cv2.waitKey(1) == 27: # 按下 ESC 鍵退出程序並釋放攝像頭
            break
        
    
    cap.release()  # 釋放攝像頭
    cv2.destroyAllWindows()  # 關閉所有視窗
    glfw.terminate()
    
  
