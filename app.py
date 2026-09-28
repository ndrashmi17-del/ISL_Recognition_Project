import os
import time
import threading

import cv2
import numpy as np
import mediapipe as mp
import pyttsx3

from flask import Flask, Response, jsonify, render_template
from tensorflow.keras.models import load_model


# ============================================================
# FLASK
# ============================================================

app = Flask(__name__)


# ============================================================
# PATHS
# ============================================================

# app.py is expected to be inside:
# D:\ISL_Recognition_Project

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "sign_model.keras"
)

NORMALIZATION_PATH = os.path.join(
    BASE_DIR,
    "models",
    "normalization.npz"
)


# ============================================================
# SETTINGS
# ============================================================

FRAMES = 30
FEATURES = 255

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Recognition sensitivity
HAND_START_FRAMES = 3
HAND_RELEASE_FRAMES = 5

STABLE_PREDICTIONS = 2

MIN_PREDICTION_GAP = 1.0

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# SIGN ORDER
# IMPORTANT:
# This MUST be exactly the same order used during training.
# ============================================================

ACTIONS = [
    "Hello",
    "Thank You",
    "Help",
    "Yes",
    "No",
    "Please",
    "Sorry",
    "Good Morning",
    "Good Night",
    "How Are You",
    "Welcome",
    "Come",
    "Go",
    "Stop",
    "Eat",
    "Drink",
    "Water",
    "Home",
    "Friend",
    "Love"
]

NUM_CLASSES = len(ACTIONS)


# ============================================================
# FACE LANDMARK INDICES
# 10 landmarks × 3 values = 30 features
# ============================================================

FACE_INDICES = [
    1,
    10,
    33,
    61,
    152,
    199,
    263,
    291,
    454,
    467
]


# ============================================================
# TRANSLATIONS
# ============================================================

TRANSLATIONS = {

    "English": {
        "Hello": "Hello",
        "Thank You": "Thank You",
        "Help": "Help",
        "Yes": "Yes",
        "No": "No",
        "Please": "Please",
        "Sorry": "Sorry",
        "Good Morning": "Good Morning",
        "Good Night": "Good Night",
        "How Are You": "How Are You",
        "Welcome": "Welcome",
        "Come": "Come",
        "Go": "Go",
        "Stop": "Stop",
        "Eat": "Eat",
        "Drink": "Drink",
        "Water": "Water",
        "Home": "Home",
        "Friend": "Friend",
        "Love": "Love"
    },

    "Kannada": {
        "Hello": "ನಮಸ್ಕಾರ",
        "Thank You": "ಧನ್ಯವಾದಗಳು",
        "Help": "ಸಹಾಯ",
        "Yes": "ಹೌದು",
        "No": "ಇಲ್ಲ",
        "Please": "ದಯವಿಟ್ಟು",
        "Sorry": "ಕ್ಷಮಿಸಿ",
        "Good Morning": "ಶುಭೋದಯ",
        "Good Night": "ಶುಭ ರಾತ್ರಿ",
        "How Are You": "ಹೇಗಿದ್ದೀರಿ",
        "Welcome": "ಸ್ವಾಗತ",
        "Come": "ಬನ್ನಿ",
        "Go": "ಹೋಗಿ",
        "Stop": "ನಿಲ್ಲಿಸಿ",
        "Eat": "ತಿನ್ನಿ",
        "Drink": "ಕುಡಿಯಿರಿ",
        "Water": "ನೀರು",
        "Home": "ಮನೆ",
        "Friend": "ಸ್ನೇಹಿತ",
        "Love": "ಪ್ರೀತಿ"
    },

    "Hindi": {
        "Hello": "नमस्ते",
        "Thank You": "धन्यवाद",
        "Help": "मदद",
        "Yes": "हाँ",
        "No": "नहीं",
        "Please": "कृपया",
        "Sorry": "माफ़ कीजिए",
        "Good Morning": "सुप्रभात",
        "Good Night": "शुभ रात्रि",
        "How Are You": "आप कैसे हैं",
        "Welcome": "स्वागत है",
        "Come": "आइए",
        "Go": "जाइए",
        "Stop": "रुकिए",
        "Eat": "खाइए",
        "Drink": "पीजिए",
        "Water": "पानी",
        "Home": "घर",
        "Friend": "दोस्त",
        "Love": "प्यार"
    }
}


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not os.path.exists(NORMALIZATION_PATH):
    raise FileNotFoundError(
        f"Normalization file not found:\n"
        f"{NORMALIZATION_PATH}"
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("SIGN-SPEAK")
print("========================================")

print("\nLoading model...")
print(MODEL_PATH)

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# LOAD NORMALIZATION
# ============================================================

print("\nLoading normalization values...")

normalization = np.load(
    NORMALIZATION_PATH
)

NORMALIZATION_MEAN = normalization["mean"].astype(
    np.float32
)

NORMALIZATION_STD = normalization["std"].astype(
    np.float32
)

print("Normalization loaded successfully.")


# ============================================================
# VERIFY NORMALIZATION SHAPE
# ============================================================

print(
    "Normalization mean shape:",
    NORMALIZATION_MEAN.shape
)

print(
    "Normalization std shape:",
    NORMALIZATION_STD.shape
)


# ============================================================
# MEDIAPIPE
# ============================================================

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


# ============================================================
# GLOBAL VARIABLES
# ============================================================

camera = None
camera_thread = None

camera_lock = threading.Lock()

latest_frame = None

camera_running = False

current_sign = ""

sentence = ""

selected_language = "English"

sequence = []

recognition_state = "camera_off"

hand_start_counter = 0

hand_release_counter = 0

prediction_history = []

last_prediction_time = 0.0

speech_lock = threading.Lock()


# ============================================================
# EXTRACT 255 FEATURES
# ============================================================

def extract_keypoints(results):

    # --------------------------------------------------------
    # LEFT HAND
    # 21 landmarks × 3 = 63
    # --------------------------------------------------------

    if results.left_hand_landmarks:

        left_hand = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        left_hand = np.zeros(
            63,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # RIGHT HAND
    # 21 landmarks × 3 = 63
    # --------------------------------------------------------

    if results.right_hand_landmarks:

        right_hand = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        right_hand = np.zeros(
            63,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # POSE
    # 33 landmarks × 3 = 99
    # --------------------------------------------------------

    if results.pose_landmarks:

        pose = np.array(
            [
                [lm.x, lm.y, lm.z]
                for lm in results.pose_landmarks.landmark
            ],
            dtype=np.float32
        ).flatten()

    else:

        pose = np.zeros(
            99,
            dtype=np.float32
        )


    # --------------------------------------------------------
    # FACE
    # 10 landmarks × 3 = 30
    # --------------------------------------------------------

    face_values = []

    if results.face_landmarks:

        for idx in FACE_INDICES:

            lm = results.face_landmarks.landmark[idx]

            face_values.extend(
                [lm.x, lm.y, lm.z]
            )

    else:

        face_values = [0.0] * 30


    face = np.array(
        face_values,
        dtype=np.float32
    )


    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    keypoints = np.concatenate(
        [
            left_hand,
            right_hand,
            pose,
            face
        ]
    ).astype(np.float32)


    # --------------------------------------------------------
    # VERIFY 255 FEATURES
    # --------------------------------------------------------

    if keypoints.shape[0] != FEATURES:

        raise ValueError(
            f"Expected {FEATURES} features, "
            f"but got {keypoints.shape[0]}"
        )


    return keypoints


# ============================================================
# HAND DETECTION
# ============================================================

def hand_detected(results):

    return (
        results.left_hand_landmarks is not None
        or
        results.right_hand_landmarks is not None
    )


# ============================================================
# TEXT TO SPEECH
# ============================================================

def speak_text(text):

    if not text:
        return


    def worker():

        try:

            with speech_lock:

                engine = pyttsx3.init()

                engine.setProperty(
                    "rate",
                    150
                )

                engine.setProperty(
                    "volume",
                    1.0
                )

                engine.say(text)

                engine.runAndWait()

                engine.stop()

        except Exception as e:

            print(
                "Speech error:",
                e
            )


    threading.Thread(
        target=worker,
        daemon=True
    ).start()


# ============================================================
# TRANSLATE SIGN
# ============================================================

def translate_sign(sign):

    language_data = TRANSLATIONS.get(
        selected_language,
        TRANSLATIONS["English"]
    )

    return language_data.get(
        sign,
        sign
    )


# ============================================================
# PREDICT SIGN
# ============================================================

def predict_sign(sequence_data):

    print("\n========================================")
    print("PREDICTION STARTED")
    print("========================================")

    # Convert sequence to NumPy
    sequence_array = np.asarray(
        sequence_data,
        dtype=np.float32
    )

    print("Sequence shape:", sequence_array.shape)

    # Check exact input shape
    if sequence_array.shape != (FRAMES, FEATURES):
        raise ValueError(
            f"Wrong sequence shape: "
            f"{sequence_array.shape}. "
            f"Expected {(FRAMES, FEATURES)}"
        )

    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------
    print("Normalizing sequence...")

    normalized = (
        sequence_array[np.newaxis, :, :]
        - NORMALIZATION_MEAN
    ) / NORMALIZATION_STD

    normalized = normalized.astype(np.float32)

    print(
        "Model input shape:",
        normalized.shape
    )

    print(
        "Model input finite:",
        np.isfinite(normalized).all()
    )

    # --------------------------------------------------------
    # DIRECT MODEL CALL
    # --------------------------------------------------------
    print("Calling model...")

    start_time = time.time()

    predictions = model(
        normalized,
        training=False
    ).numpy()[0]

    elapsed = time.time() - start_time

    print(
        f"Model prediction completed in "
        f"{elapsed:.2f} seconds"
    )

    # --------------------------------------------------------
    # GET RESULT
    # --------------------------------------------------------
    predicted_index = int(
        np.argmax(predictions)
    )

    confidence = float(
        predictions[predicted_index]
    )

    predicted_sign = ACTIONS[
        predicted_index
    ]

    print(
        "Predicted index:",
        predicted_index
    )

    print(
        "Predicted sign:",
        predicted_sign
    )

    print(
        "Confidence:",
        f"{confidence:.4f}"
    )

    print("========================================\n")

    return predicted_sign, confidence

# ============================================================
# ACCEPT PREDICTION
# ============================================================

def accept_prediction(
    sign,
    confidence
):

    global current_sign
    global sentence
    global prediction_history
    global last_prediction_time


    # --------------------------------------------------------
    # CONFIDENCE CHECK
    # --------------------------------------------------------

    if confidence < CONFIDENCE_THRESHOLD:

        print(
            f"Rejected: {sign} "
            f"({confidence:.2f})"
        )

        return


    # --------------------------------------------------------
    # TIME GAP
    # --------------------------------------------------------

    now = time.time()

    if (
        now - last_prediction_time
        <
        MIN_PREDICTION_GAP
    ):

        return


    # --------------------------------------------------------
    # STORE PREDICTION
    # --------------------------------------------------------

    prediction_history.append(
        sign
    )


    if len(prediction_history) > STABLE_PREDICTIONS:

        prediction_history = (
            prediction_history[
                -STABLE_PREDICTIONS:
            ]
        )


    # --------------------------------------------------------
    # WAIT FOR STABLE PREDICTION
    # --------------------------------------------------------

    if (
        len(prediction_history)
        <
        STABLE_PREDICTIONS
    ):

        return


    if len(
        set(prediction_history)
    ) != 1:

        return


    # --------------------------------------------------------
    # ACCEPT
    # --------------------------------------------------------

    accepted_sign = (
        prediction_history[-1]
    )

    current_sign = accepted_sign

    translated = translate_sign(
        accepted_sign
    )


    # --------------------------------------------------------
    # ADD TO SENTENCE
    # --------------------------------------------------------

    if sentence:

        sentence += (
            " " + translated
        )

    else:

        sentence = translated


    last_prediction_time = now


    print(
        f"✓ Recognized: "
        f"{accepted_sign} "
        f"({confidence:.2f}) "
        f"→ {translated}"
    )


    # --------------------------------------------------------
    # SPEAK
    # --------------------------------------------------------

    speak_text(
        translated
    )


    prediction_history.clear()


# ============================================================
# CAMERA LOOP
# ============================================================

def camera_loop():

    global camera
    global camera_running
    global latest_frame

    global sequence
    global recognition_state

    global hand_start_counter
    global hand_release_counter

    global prediction_history


    print(
        "Camera thread started."
    )


    holistic = None

    cap = None


    try:

        # ----------------------------------------------------
        # OPEN CAMERA
        # ----------------------------------------------------

        cap = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )


        if not cap.isOpened():

            cap = cv2.VideoCapture(
                0
            )


        if not cap.isOpened():

            recognition_state = (
                "camera_error"
            )

            print(
                "ERROR: Could not open webcam"
            )

            camera_running = False

            return


        # ----------------------------------------------------
        # CAMERA SETTINGS
        # ----------------------------------------------------

        cap.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            CAMERA_WIDTH
        )

        cap.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            CAMERA_HEIGHT
        )


        camera = cap

        camera_running = True

        recognition_state = "waiting"


        print(
            "Camera ready."
        )

        print(
            "Show your sign..."
        )


        # ----------------------------------------------------
        # MEDIAPIPE HOLISTIC
        # ----------------------------------------------------

        holistic = mp_holistic.Holistic(

            static_image_mode=False,

            model_complexity=1,

            smooth_landmarks=True,

            enable_segmentation=False,

            refine_face_landmarks=False,

            min_detection_confidence=0.5,

            min_tracking_confidence=0.5
        )


        # ====================================================
        # MAIN CAMERA LOOP
        # ====================================================

        while camera_running:


            success, frame = cap.read()


            if not success:

                time.sleep(
                    0.05
                )

                continue


            # ------------------------------------------------
            # MIRROR CAMERA
            # ------------------------------------------------

            frame = cv2.flip(
                frame,
                1
            )


            # ------------------------------------------------
            # MEDIAPIPE
            # ------------------------------------------------

            image = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image.flags.writeable = False

            results = holistic.process(
                image
            )

            image.flags.writeable = True


            # ------------------------------------------------
            # DRAW LANDMARKS
            # ------------------------------------------------

            if results.pose_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS
                )


            if results.left_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )


            if results.right_hand_landmarks:

                mp_drawing.draw_landmarks(
                    frame,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS
                )


            # ------------------------------------------------
            # HAND DETECTION
            # ------------------------------------------------

            detected = hand_detected(
                results
            )


            # =================================================
            # WAITING
            # =================================================

            if recognition_state == "waiting":

                sequence.clear()

                prediction_history.clear()

                hand_release_counter = 0


                if detected:

                    hand_start_counter += 1

                else:

                    hand_start_counter = 0


                if (
                    hand_start_counter
                    >=
                    HAND_START_FRAMES
                ):

                    sequence = []

                    recognition_state = (
                        "collecting"
                    )

                    hand_start_counter = 0

                    print(
                        "→ Collecting..."
                    )


            # =================================================
            # COLLECTING
            # =================================================

            elif recognition_state == "collecting":

                try:

                    keypoints = (
                        extract_keypoints(
                            results
                        )
                    )

                    sequence.append(
                        keypoints
                    )


                except Exception as e:

                    print(
                        "Feature error:",
                        e
                    )

                    sequence.clear()

                    recognition_state = (
                        "waiting"
                    )


                if len(sequence) >= FRAMES:

                    recognition_state = (
                        "predicting"
                    )


            # =================================================
            # PREDICTING
            # =================================================

            elif recognition_state == "predicting":

                try:

                    predicted_sign, confidence = (
                        predict_sign(
                            sequence[:FRAMES]
                        )
                    )


                    print(
                        f"Prediction: "
                        f"{predicted_sign} | "
                        f"Confidence: "
                        f"{confidence:.2f}"
                    )


                    accept_prediction(
                        predicted_sign,
                        confidence
                    )


                except Exception as e:

                    print(
                        "Prediction error:",
                        e
                    )


                sequence.clear()

                hand_start_counter = 0

                hand_release_counter = 0

                recognition_state = (
                    "waiting_for_release"
                )


            # =================================================
            # WAITING FOR HAND RELEASE
            # =================================================

            elif (
                recognition_state
                ==
                "waiting_for_release"
            ):

                if detected:

                    hand_release_counter = 0

                else:

                    hand_release_counter += 1


                if (
                    hand_release_counter
                    >=
                    HAND_RELEASE_FRAMES
                ):

                    sequence.clear()

                    prediction_history.clear()

                    recognition_state = (
                        "waiting"
                    )

                    print(
                        "→ Ready again\n"
                    )


            # ------------------------------------------------
            # STATUS
            # ------------------------------------------------

            status_text = {

                "waiting":
                    "Show a sign",

                "collecting":
                    f"Collecting "
                    f"{len(sequence)}/{FRAMES}",

                "predicting":
                    "Recognizing...",

                "waiting_for_release":
                    "Release hand",

                "camera_error":
                    "Camera error"

            }.get(
                recognition_state,
                recognition_state
            )


            # ------------------------------------------------
            # STATUS BOX
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (10, 10),
                (630, 55),
                (0, 0, 0),
                -1
            )


            cv2.putText(
                frame,
                status_text,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # CURRENT SIGN
            # ------------------------------------------------

            if current_sign:

                cv2.putText(
                    frame,
                    f"Sign: {current_sign}",
                    (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            # ------------------------------------------------
            # ENCODE FRAME
            # ------------------------------------------------

            success_encode, buffer = (
                cv2.imencode(
                    ".jpg",
                    frame,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        85
                    ]
                )
            )


            if success_encode:

                with camera_lock:

                    latest_frame = (
                        buffer.tobytes()
                    )


            time.sleep(
                0.01
            )


    except Exception as e:

        recognition_state = (
            "camera_error"
        )

        print(
            "Camera error:",
            e
        )


    finally:

        camera_running = False


        if holistic:

            try:

                holistic.close()

            except Exception:

                pass


        if cap:

            try:

                cap.release()

            except Exception:

                pass


        camera = None

        print(
            "Camera stopped."
        )


# ============================================================
# START CAMERA
# ============================================================

def start_camera():

    global camera_thread
    global camera_running


    if camera_running:

        return True


    camera_thread = threading.Thread(
        target=camera_loop,
        daemon=True
    )


    camera_thread.start()


    time.sleep(
        0.4
    )


    return True


# ============================================================
# STOP CAMERA
# ============================================================

def stop_camera():

    global camera_running
    global camera
    global latest_frame


    camera_running = False


    if camera:

        try:

            camera.release()

        except Exception:

            pass


    camera = None


    with camera_lock:

        latest_frame = None


    return True


# ============================================================
# VIDEO STREAM
# ============================================================

def generate_video():

    while True:

        if not camera_running:

            time.sleep(
                0.1
            )

            continue


        with camera_lock:

            frame = latest_frame


        if frame is None:

            time.sleep(
                0.03
            )

            continue


        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + frame
            + b"\r\n"
        )


        time.sleep(
            0.01
        )


# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def index():

    return render_template(
        "index.html"
    )


# ============================================================
# START CAMERA ROUTE
# ============================================================

@app.route(
    "/start_camera",
    methods=["GET", "POST"]
)
def start_camera_route():

    try:

        start_camera()

        return jsonify({
            "success": True
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ============================================================
# STOP CAMERA ROUTE
# ============================================================

@app.route(
    "/stop_camera",
    methods=["GET", "POST"]
)
def stop_camera_route():

    try:

        stop_camera()

        return jsonify({
            "success": True
        })


    except Exception as e:

        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


# ============================================================
# VIDEO ROUTE
# ============================================================

@app.route("/video")
def video():

    return Response(
        generate_video(),
        mimetype=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )


# ============================================================
# STATUS ROUTE
# ============================================================

@app.route("/status")
def status():

    return jsonify({

        "camera_running":
            camera_running,

        "current_sign":
            current_sign,

        "sentence":
            sentence,

        "language":
            selected_language,

        "state":
            recognition_state,

        "sequence_length":
            len(sequence),

        "sequence_total":
            FRAMES
    })


# ============================================================
# LANGUAGE ROUTE
# ============================================================

@app.route(
    "/set_language/<language>",
    methods=["GET", "POST"]
)
def set_language(language):

    global selected_language


    mapping = {

        "english": "English",

        "kannada": "Kannada",

        "hindi": "Hindi"
    }


    selected_language = mapping.get(
        language.lower(),
        "English"
    )


    return jsonify({

        "success": True,

        "language":
            selected_language
    })


# ============================================================
# CLEAR SENTENCE
# ============================================================

@app.route(
    "/clear",
    methods=["GET", "POST"]
)
def clear():

    global current_sign
    global sentence
    global sequence
    global prediction_history


    current_sign = ""

    sentence = ""

    sequence.clear()

    prediction_history.clear()


    return jsonify({
        "success": True
    })


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print("\n========================================")
    print("SIGN-SPEAK")
    print("========================================")

    print(
        "Model:",
        MODEL_PATH
    )

    print(
        "Normalization:",
        NORMALIZATION_PATH
    )

    print(
        "Signs:",
        NUM_CLASSES
    )

    print(
        "Features:",
        FEATURES
    )

    print(
        "Frames:",
        FRAMES
    )

    print(
        "\nOpen → http://127.0.0.1:5000"
    )

    print(
        "========================================\n"
    )


    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False
    )