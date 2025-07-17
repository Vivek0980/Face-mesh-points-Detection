# Face Landmark Detection using MediaPipe and OpenCV

This project uses **MediaPipe's FaceMesh** and **OpenCV** to detect facial landmarks in real-time using your webcam.

---

## 📌 Features

- Detects up to 2 faces in a webcam feed.
- Draws 468 facial landmarks with tessellation.
- Prints the landmark ID and its `(x, y)` pixel coordinates.

---

## 🛠 Requirements

Make sure you have Python installed (preferably 3.7+), then install the required libraries:

```bash
pip install opencv-python mediapipe
```

---

## ▶️ How to Run

1. Save the following Python code in a file named `face_mesh_tracker.py`
2. Run it using:

```bash
python face_mesh_tracker.py
```

3. Press **`q`** to quit the webcam window.

---

## 💻 Code

```python
import cv2
import time
import mediapipe as mp

# Initialize webcam
cam = cv2.VideoCapture(0)

# Initialize MediaPipe FaceMesh
mpface = mp.solutions.face_mesh
myface = mpface.FaceMesh(max_num_faces=2)

# Drawing utilities
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1, color=(0, 255, 0))

# Start capturing frames
while True:
    ret, frame = cam.read()
    if not ret:
        break

    # Convert BGR to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image to find face landmarks
    results = myface.process(image_rgb)

    # If landmarks are found
    if results.multi_face_landmarks:
        for landmarks in results.multi_face_landmarks:
            # Draw facial landmarks on the frame
            mp_draw.draw_landmarks(frame, landmarks, mpface.FACEMESH_TESSELATION, draw_spec, draw_spec)

            # Loop through individual landmarks
            for id, val in enumerate(landmarks.landmark):
                h, w, c = frame.shape
                cx = int(val.x * w)
                cy = int(val.y * h)
                print(id, cx, cy)  # Print landmark ID and pixel coordinates

    # Show the frame
    cv2.imshow('frame', frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cam.release()
cv2.destroyAllWindows()
```

---

## 🧠 Explanation

- **MediaPipe FaceMesh**: A lightweight face landmark detector that tracks 468 points on the face.
- **DrawingSpec**: Used to customize landmark drawing (thickness, radius, color).
- **`enumerate(landmarks.landmark)`**: Gives index and each landmark's normalized coordinates.
- **Pixel Conversion**: `val.x * w`, `val.y * h` converts normalized coordinates to actual pixel positions.

---

## 📷 Output

- A real-time webcam window with green dots and lines forming the face mesh.
- Printed landmark positions (like `0 320 240`) in the terminal.

---

## ❓Troubleshooting

- If the webcam doesn't open, make sure no other app is using it.
- If `cv2` window closes immediately, check if `ret` is False — camera may not be accessible.
- Add a small delay like `time.sleep(0.05)` if CPU usage is too high.

---

## 🧑‍💻 Author

**Vivek Nani**

---

Happy Coding! 😊

