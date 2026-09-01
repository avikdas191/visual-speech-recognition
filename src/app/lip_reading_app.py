"""
Lip reading desktop application.

Gesture-controlled capture, lip region cropping, word recognition and
sentence generation, presented in a single window.
"""

import os
import cv2
import sys
import time
import threading
import tkinter as tk
import mediapipe as mp
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FRAMES
from pipeline import (clear_runtime, normalise_extracted_frames,
                      crop_extracted_frames, build_grid,
                      load_lip_model, predict_word,
                      load_t5_model, generate_sentence)

# colours taken from the original chat interface stylesheet
BG_DARK = "#26333d"
CARD_BG = "#1a1d24"
BUBBLE = "#52acff"
TEXT_LIGHT = "#ffffff"
TEXT_MUTED = "#9aa0a8"
PANE_BG = "#12151a"

PREVIEW_WIDTH = 640
PREVIEW_HEIGHT = 360

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

class LipReadingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Lip Reading and Sentence Generation")
        self.root.configure(bg=BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.hands = None
        self.running = False
        self.current_image = None

        self.recording = False
        self.finished = False
        self.captured_frames = []
        self.t_start = None
        self.elapsed = 0.0
        self.models_ready = False

        self._build_layout()
        self.root.after(100, self.start_camera)

    def _build_layout(self):
        header = tk.Frame(self.root, bg=CARD_BG, padx=16, pady=12)
        header.pack(fill=tk.X)

        tk.Label(header, text="Lip Reading", bg=CARD_BG, fg=TEXT_LIGHT,
                 font=("Segoe UI", 15, "bold")).pack(anchor="w")
        tk.Label(header, text="Visual word recognition and sentence generation",
                 bg=CARD_BG, fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(anchor="w")

        holder = tk.Frame(self.root, bg="black",
                          width=PREVIEW_WIDTH, height=PREVIEW_HEIGHT)
        holder.pack(padx=16, pady=16)
        holder.pack_propagate(False)

        self.preview = tk.Label(holder, bg="black")
        self.preview.pack(fill=tk.BOTH, expand=True)

        self.result = tk.Frame(self.root, bg=PANE_BG, padx=16, pady=14)

        self.word_label = tk.Label(self.result, text="", bg=PANE_BG,
                                   fg=TEXT_LIGHT, font=("Segoe UI", 14, "bold"),
                                   anchor="w", justify=tk.LEFT)
        self.word_label.pack(fill=tk.X)

        self.sentence_label = tk.Label(self.result, text="", bg=PANE_BG,
                                       fg=BUBBLE, font=("Segoe UI", 11),
                                       wraplength=600, anchor="w", justify=tk.LEFT)
        self.sentence_label.pack(fill=tk.X, pady=(6, 0))
        
        self.footer = tk.Frame(self.root, bg=BG_DARK)
        self.footer.pack(fill=tk.X, padx=16, pady=(0, 16))

        text_column = tk.Frame(self.footer, bg=BG_DARK)
        text_column.pack(side=tk.LEFT, fill=tk.X, expand=True, anchor="w")

        self.status = tk.Label(text_column, text="starting", bg=BG_DARK,
                               fg=TEXT_MUTED, font=("Segoe UI", 10), anchor="w")
        self.status.pack(fill=tk.X)

        self.waiting = tk.Label(text_column, text="", bg=BG_DARK,
                                fg=BUBBLE, font=("Segoe UI", 10, "italic"),
                                anchor="w")
        self.waiting.pack(fill=tk.X, pady=(4, 0))

        self.quit_button = tk.Button(self.footer, text="Quit",
                                     command=self.on_close,
                                     bg=CARD_BG, fg=TEXT_LIGHT,
                                     font=("Segoe UI", 10), relief=tk.FLAT,
                                     padx=12, pady=6)
        self.quit_button.pack(side=tk.RIGHT)

        self.again_button = tk.Button(self.footer, text="Record another word",
                                      command=self.reset_session,
                                      bg=BUBBLE, fg=TEXT_LIGHT,
                                      font=("Segoe UI", 10), relief=tk.FLAT,
                                      padx=12, pady=6, state=tk.DISABLED)
        self.again_button.pack(side=tk.RIGHT, padx=(0, 8))


    def set_status(self, text, waiting=False):
        self.status.config(text=text)
        self.waiting.config(text="please wait..." if waiting else "")

    def show_result(self, word, confidence, sentence):
        self.word_label.config(
            text=f"{word}  -  {confidence * 100:.1f}% confidence")
        self.sentence_label.config(text=sentence)
        self.result.pack(fill=tk.X, padx=16, pady=(0, 8), before=self.footer)

    def clear_result(self):
        self.word_label.config(text="")
        self.sentence_label.config(text="")
        self.result.pack_forget()

    def start_camera(self):
        """Load models and open the camera, both on a background thread."""
        self.set_status("starting up", True)
        threading.Thread(target=self._startup, daemon=True).start()

    def _startup(self):
        self.root.after(0, self.set_status, "loading the lip reading model", True)
        load_lip_model(verbose=False)

        self.root.after(0, self.set_status, "loading the sentence generation model", True)
        load_t5_model(verbose=False)

        self.models_ready = True
        self.root.after(0, self.set_status, "setting up the camera, this takes about 20 seconds", True)
        self._open_camera()

    def _open_camera(self):
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
        if not cap.isOpened():
            self.root.after(0, self.set_status, "camera could not be opened")
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        ok, _ = cap.read()
        if not ok:
            cap.release()
            self.root.after(0, self.set_status, "camera opened but no frame could be read")
            return

        self.cap = cap
        self.hands = mp.solutions.hands.Hands(model_complexity=0,
                                              min_detection_confidence=0.8,
                                              min_tracking_confidence=0.5)
        self.running = True
        clear_runtime(verbose=False)
        self.root.after(0, self.set_status, "camera ready - show an open hand to start")
        self.root.after(0, self.update_frame)

    def update_frame(self):
        """Read one frame, run gesture detection, draw overlays, then reschedule."""
        if not self.running or self.cap is None:
            return
        
        ok, frame = self.cap.read()
        if not ok:
            self.root.after(30, self.update_frame)
            return

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.flip(image, 1)
        image.flags.writeable = False
        results = self.hands.process(image)
        image.flags.writeable = True

        if results.multi_hand_landmarks and not self.finished:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(76, 22, 121), thickness=2, circle_radius=4),
                    mp.solutions.drawing_utils.DrawingSpec(
                        color=(250, 44, 121), thickness=2, circle_radius=2))

                if is_open_hand(hand_landmarks):
                    if not self.recording and self.models_ready:
                        self.recording = True
                        self.captured_frames = []
                        self.t_start = time.time()
                        self.set_status("recording - speak now")

                elif is_closed_fist(hand_landmarks):
                    if self.recording:
                        self.recording = False
                        self.finished = True
                        self.elapsed = time.time() - self.t_start
                        self.stop_recording()

        if self.recording:
            self.captured_frames.append(frame.copy())
            self.elapsed = time.time() - self.t_start
            if len(self.captured_frames) >= 300:
                self.recording = False
                self.finished = True
                self.stop_recording()

        display = cv2.resize(image, (PREVIEW_WIDTH, PREVIEW_HEIGHT))

        if self.recording:
            cv2.putText(display,
                        f"REC  {self.elapsed:5.2f}s   {len(self.captured_frames)} frames",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.circle(display, (PREVIEW_WIDTH - 25, 25), 8, (255, 0, 0), -1)

        pil_image = Image.fromarray(display)
        if self.current_image is None:
            self.current_image = ImageTk.PhotoImage(pil_image)
            self.preview.config(image=self.current_image)
        else:
            self.current_image.paste(pil_image)

        self.root.after_idle(self.update_frame)

    def stop_recording(self):
        """Write captured frames to disk on a background thread."""
        count = len(self.captured_frames)
        fps = count / self.elapsed if self.elapsed else 0
        self.set_status(f"recorded {count} frames in {self.elapsed:.2f}s ({fps:.1f} fps) - saving", True)
        frames = self.captured_frames
        self.captured_frames = []
        threading.Thread(target=self._write_frames, args=(frames,), daemon=True).start()

    def _write_frames(self, frames):
        for i, f in enumerate(frames, start=1):
            cv2.imwrite(os.path.join(FRAMES, f"{i:02d}.png"), f)
        self.root.after(0, self._writing_done, len(frames))

    def _writing_done(self, count):
        self.set_status(f"saved {count} frames", True)
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        try:
            self.root.after(0, self.set_status, "normalising the recording to 60 frames", True)
            stats = normalise_extracted_frames(verbose=False)
            if stats["padded"]:
                self.root.after(0, self.set_status,
                                f"padded {stats['padded']} frames to reach 60", True)
            elif stats["trimmed_start"] or stats["trimmed_end"]:
                self.root.after(0, self.set_status,
                                f"trimmed {stats['trimmed_start']} frames from the start "
                                f"and {stats['trimmed_end']} from the end", True)

            self.root.after(0, self.set_status, "locating and cropping the lip region", True)
            crop_stats = crop_extracted_frames(verbose=False)
            if crop_stats["failed"]:
                self.root.after(0, self.set_status,
                                f"the lips could not be found in {len(crop_stats['failed'])} "
                                f"frames, filled from neighbouring frames", True)

            self.root.after(0, self.set_status, "assembling the frames into a grid image", True)
            build_grid(verbose=False)

            self.root.after(0, self.set_status, "recognising the word", True)
            word, probs = predict_word(verbose=False)
            confidence = float(max(probs))

            self.root.after(0, self.set_status, "generating a sentence", True)
            sentence = generate_sentence(word, verbose=False)

            self.root.after(0, self.show_result, word, confidence, sentence)

        except Exception as exc:
            self.root.after(0, self.set_status, f"the recording could not be processed: {exc}")

        self.root.after(0, self._pipeline_done)

    def _pipeline_done(self):
        self.set_status("press record another word to continue - this clears the current result")
        self.again_button.config(state=tk.NORMAL)

    def reset_session(self):
        """Clear the previous recording and allow another word to be captured."""
        self.again_button.config(state=tk.DISABLED)
        self.set_status("clearing previous recording", True)
        threading.Thread(target=self._reset_worker, daemon=True).start()

    def _reset_worker(self):
        clear_runtime(verbose=False)
        self.root.after(0, self.clear_result)
        self.captured_frames = []
        self.elapsed = 0.0
        self.t_start = None
        self.recording = False
        self.finished = False
        self.root.after(0, self.set_status, "ready - show an open hand to start")

    def on_close(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        if self.hands is not None:
            self.hands.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    LipReadingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()