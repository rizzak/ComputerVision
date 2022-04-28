from aiohttp import request, web
import json
import cv2
import mediapipe as mp
import numpy as np
import base64
from utils.helpers import current_milli_time

async def detect(request):
    data = await request.post()
    try:
        content = data['my_img'].file.read()
    except KeyError as e:
        response_obj = {'error': str(e)}
        return web.Response(text=json.dumps(response_obj), status=400)

    start = current_milli_time()
    mp_drawing = mp.solutions.drawing_utils
    try:
        im_arr = np.frombuffer(content, dtype=np.uint8)  # im_arr is one-dim Numpy array
        image = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        with mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.4) as face_detection:
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            # Draw the face detection annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image, detection)
                text = 'yes'
            else:
                text = 'no'
    except Exception as e:
        response_obj = {'error': str(e)}
        return web.Response(text=json.dumps(response_obj), status=400)

    end = current_milli_time()
    totalTime = end - start

    print(f"Time: {totalTime} Face: {text}")

    cv2.putText(image, f'Detect time: {int(totalTime)}', (20,70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    base64_str = cv2.imencode('.jpg',image)[1].tobytes()
    base64_str = base64.b64encode(base64_str).decode('utf8')
    response_obj = {'face': text, 'image': base64_str}
    return web.json_response(response_obj)

async def handle(request):
    response_obj = {'status': 'success'}
    return web.Response(text=json.dumps(response_obj), status=200)

app = web.Application()
app.router.add_get('/', handle)
app.router.add_post('/detect', detect)
web.run_app(app)
