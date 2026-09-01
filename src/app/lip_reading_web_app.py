"""
Lip reading web demonstration.

This reproduces the way the original project was presented: an OpenCV window
shows the camera feed while a locally served web page reports what the system
is doing. The two were separate in the original because video could not be
transmitted to the page reliably.

The pipeline itself is imported from src/pipeline.py, so this and the desktop
application run identical logic.
"""

import os
import cv2
import sys
import time
import signal
import threading
import mediapipe as mp

from flask import Flask, jsonify, render_template

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FRAMES
from pipeline import (clear_runtime, normalise_extracted_frames,
                      crop_extracted_frames, build_grid,
                      load_lip_model, predict_word,
                      load_t5_model, generate_sentence)

HOST = "127.0.0.1"
PORT = 5000

mp_hands = mp.solutions.hands

def is_open_hand(hand_landmarks):
    """ Check if all fingers are extended (open hand) """
    for finger_tip, finger_pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        if hand_landmarks.landmark[finger_tip].y > hand_landmarks.landmark[finger_pip].y:
            return False
    return True


def is_closed_fist(hand_landmarks):
    """ Check if all fingers are folded (closed fist) """
    for finger_tip, finger_pip in [
        (mp_hands.HandLandmark.INDEX_FINGER_TIP, mp_hands.HandLandmark.INDEX_FINGER_PIP),
        (mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP),
        (mp_hands.HandLandmark.RING_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_PIP),
        (mp_hands.HandLandmark.PINKY_TIP, mp_hands.HandLandmark.PINKY_PIP)
    ]:
        if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_pip].y:
            return False
    return True

app = Flask(__name__)

status_messages = []
status_lock = threading.Lock()

new_session_requested = threading.Event()
ready_for_next = threading.Event()


def post_status(message, result=False):
    """Add a line for the web page to display.

    Result lines are shown differently on the page, so they are tagged here
    rather than being guessed at by the page itself."""
    with status_lock:
        status_messages.append({"text": message, "result": result})
    print(message)


def reset_status():
    with status_lock:
        status_messages.clear()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/get-status")
def get_status():
    with status_lock:
        return jsonify(status=list(status_messages))


@app.route("/record-again", methods=["POST"])
def record_again():
    reset_status()
    ready_for_next.clear()
    new_session_requested.set()
    return jsonify(ok=True)


@app.route("/session-state")
def session_state():
    return jsonify(ready=ready_for_next.is_set())


@app.route("/quit", methods=["POST"])
def quit_app():
    os.kill(os.getpid(), signal.SIGINT)
    return jsonify(ok=True)


def load_models():
    post_status("Loading the lip reading model.")
    load_lip_model(verbose=False)
    post_status("Loading the sentence generation model.")
    load_t5_model(verbose=False)


def run_camera():
    """Capture on gesture in an OpenCV window, reporting progress to the page.

    The camera stays open between sessions so that a second recording does not
    require waiting for it to initialise again."""
    load_models()
    post_status("Setting up the camera. This takes about 20 seconds.")

    cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not cap.isOpened():
        post_status("The camera could not be opened.")
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    ok, _ = cap.read()
    if not ok:
        cap.release()
        post_status("The camera opened but no frame could be read.")
        return

    clear_runtime(verbose=False)
    post_status("Camera ready. Show an open hand to start recording.")

    recording = False
    awaiting_button = False
    captured_frames = []
    t_start = None
    elapsed = 0.0

    with mp_hands.Hands(model_complexity=0,
                        min_detection_confidence=0.8,
                        min_tracking_confidence=0.5) as hands:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if awaiting_button and new_session_requested.is_set():
                new_session_requested.clear()
                clear_runtime(verbose=False)
                captured_frames = []
                elapsed = 0.0
                t_start = None
                awaiting_button = False
                post_status("Camera ready. Show an open hand to start recording.")

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.flip(image, 1)
            image.flags.writeable = False
            results = hands.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = cv2.resize(image, (640, 360))

            if results.multi_hand_landmarks and not awaiting_button:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(121, 22, 76), thickness=2, circle_radius=4),
                        mp.solutions.drawing_utils.DrawingSpec(
                            color=(121, 44, 250), thickness=2, circle_radius=2))

                    if is_open_hand(hand_landmarks):
                        if not recording:
                            recording = True
                            captured_frames = []
                            t_start = time.time()
                            post_status("Recording. Speak now.")

                    elif is_closed_fist(hand_landmarks):
                        if recording:
                            recording = False
                            awaiting_button = True
                            elapsed = time.time() - t_start

            if recording:
                captured_frames.append(frame.copy())
                elapsed = time.time() - t_start
                if len(captured_frames) >= 300:
                    recording = False
                    awaiting_button = True

            if recording:
                cv2.putText(image, f"REC  {elapsed:5.2f}s   {len(captured_frames)} frames",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.circle(image, (615, 25), 8, (0, 0, 255), -1)

            cv2.imshow("Lip Reading - camera", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # if awaiting_button and captured_frames:
            #     session_active.clear()
            #     frames_to_process = captured_frames
            #     captured_frames = []
            #     process_recording(frames_to_process, elapsed)

            if awaiting_button and captured_frames:
                frames_to_process = captured_frames
                captured_frames = []
                threading.Thread(target=process_recording,
                                 args=(frames_to_process, elapsed),
                                 daemon=True).start()

    cap.release()
    cv2.destroyAllWindows()


def process_recording(frames, elapsed):
    """Save the captured frames and run the pipeline over them."""
    fps = len(frames) / elapsed if elapsed else 0
    post_status(f"Recorded {len(frames)} frames in {elapsed:.2f} seconds "
                f"({fps:.1f} frames per second).")

    for i, f in enumerate(frames, start=1):
        cv2.imwrite(os.path.join(FRAMES, f"{i:02d}.png"), f)
    post_status(f"Saved {len(frames)} frames.")

    run_pipeline()
    post_status("To record another word, press the button below. "
                "This will clear the messages above.")

    ready_for_next.set()


def run_pipeline():
    """Process the saved frames and report each step to the page."""
    try:
        post_status("Normalising the recording to 60 frames.")
        stats = normalise_extracted_frames(verbose=False)
        if stats["padded"]:
            post_status(f"Padded {stats['padded']} frames to reach 60.")
        elif stats["trimmed_start"] or stats["trimmed_end"]:
            post_status(f"Trimmed {stats['trimmed_start']} frames from the start "
                        f"and {stats['trimmed_end']} from the end.")

        post_status("Locating and cropping the lip region.")
        crop_stats = crop_extracted_frames(verbose=False)
        if crop_stats["failed"]:
            post_status(f"The lips could not be found in {len(crop_stats['failed'])} "
                        f"frames. These were filled from neighbouring frames.")

        post_status("Assembling the frames into a grid image.")
        build_grid(verbose=False)

        post_status("Recognising the word.")
        word, probs = predict_word(verbose=False)
        confidence = float(max(probs))
        post_status(f"The word is \"{word}\", recognised with "
                    f"{confidence * 100:.1f} percent confidence.", True)

        post_status("Generating a sentence.")
        sentence = generate_sentence(word, verbose=False)
        post_status(sentence, True)

    except Exception as exc:
        post_status(f"The recording could not be processed: {exc}")


def main():
    post_status("Server started.")
    print(f"open http://{HOST}:{PORT} in a browser")

    threading.Thread(target=run_camera, daemon=True).start()

    app.run(host=HOST, port=PORT, debug=False, use_reloader=False)

if __name__ == "__main__":
    main()