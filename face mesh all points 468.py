import cv2
import time
import mediapipe as mp
cam=cv2.VideoCapture(0)
mpface=mp.solutions.face_mesh
myface=mpface.FaceMesh(max_num_faces=2)
mp_draw=mp.solutions.drawing_utils
draw_spec=mp_draw.DrawingSpec(thickness=1,circle_radius=1,color=(0,255,0))
while True:
    ret,frame=cam.read()
    image_rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results=myface.process(image_rgb)
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            mp_draw.draw_landmarks(frame,landmarks,mpface.FACEMESH_TESSELATION,draw_spec,draw_spec)
            for id,val in enumerate(landmarks.landmark):
                h,w,c=frame.shape
                cx=int(val.x*w)
                cy=int(val.y*h)
                print(id,cx,cy)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
