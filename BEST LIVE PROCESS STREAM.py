from flask import Flask, Response
import cv2
import threading
import time

app = Flask(__name__)

# Load the video file (update the path)
video_path = r"C:\Users\Gautam Pothurajula\Downloads\Project\edit.MP4"
video = cv2.VideoCapture(video_path)

# Get FPS to maintain proper timing
fps = video.get(cv2.CAP_PROP_FPS) or 30  # Default to 30 FPS if FPS is unknown
frame_delay = 1 / fps  # Delay per frame

# Global variable to store the latest frame
latest_frame = None
lock = threading.Lock()

def capture_frames():
    global latest_frame
    while True:
        success, frame = video.read()
        if not success:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart video
            continue

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        with lock:
            latest_frame = buffer.tobytes()

        time.sleep(frame_delay)  # Maintain original FPS timing

# Start frame capture in a separate thread
thread = threading.Thread(target=capture_frames, daemon=True)
thread.start()

def generate_frames():
    global latest_frame
    while True:
        with lock:
            if latest_frame is None:
                continue  # Skip if no frame is available

            frame = latest_frame

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return "<h2>Live Video Stream</h2><img src='/video' width='640'>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
