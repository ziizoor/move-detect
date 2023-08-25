from flask import Flask, Response
import cv2
import numpy as np

app = Flask(__name__)

THRESHOLD_AREA = 500
WAIT_KEY_DELAY = 70

def generate_frames():
    cap = cv2.VideoCapture(0)  # Usar la cámara por defecto (cambiar el índice si tienes múltiples cámaras)
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    
    area_pts = np.array([[240, 320], [480, 320], [620, cap.get(4)], [50, cap.get(4)]])

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        imAux = np.zeros(shape=gray.shape, dtype=np.uint8)
        imAux = cv2.drawContours(imAux, [area_pts], -1, 255, -1)
        image_area = cv2.bitwise_and(gray, gray, mask=imAux)

        fgmask = fgbg.apply(image_area)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
        fgmask = cv2.dilate(fgmask, None, iterations=2)

        cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

        for cnt in cnts:
            if cv2.contourArea(cnt) > THRESHOLD_AREA:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return (
        "<h1>Previsualización de Video</h1>"
        "<img src='/video_feed' width='640' height='480'>"
    )

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
