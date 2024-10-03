# import queue
# import threading
# import time

# import cv2
# from vidgear.gears import CamGear, StreamGear

# class VideoCapture:
#     def __init__(self, name: str):
#         self.cap = CamGear(name).start()
#         self.q = queue.Queue()
#         t = threading.Thread(target=self._reader)
#         t.daemon = True
#         t.start()

#     def _reader(self):
#         while True:
#             max_retries = 99  # Maximum number of retries
#             retries = 0
#             while retries < max_retries:
#                 try:
#                     frame = self.cap.read()
#                     if frame is None:
#                         raise ValueError("Frame is None")
#                     if not self.q.empty():
#                         try:
#                             self.q.get_nowait()  # Discard previous (unprocessed) frame
#                         except queue.Empty:
#                             continue
#                     self.q.put(frame)
#                 except ValueError as e:
#                     with open('./logs/catch.error.log', '+a') as archivo:
#                         archivo.write(f'Error in decode frame {retries}: {e}' + '\n')
#                     print(f"Error while reading frame: {e}")
#                     retries += 1
#                     time.sleep(0.5)  # Wait a bit before retrying
#             if retries == max_retries:
#                 print("Failed to read frame after maximum retries.")

#     def read(self) -> cv2.Mat:
#         frame = self.q.get()
#         return frame

import cv2
import queue
import threading

class VideoCapture:
   def __init__(self, name: str):
       self.cap = cv2.VideoCapture(name)
       self.q = queue.Queue()
       t = threading.Thread(target=self._reader)
       t.daemon = True
       t.start()

   def _reader(self):
       while True:
           ret, frame = self.cap.read()
           if not ret:
               break
           if not self.q.empty():
               try:
                  self.q.get_nowait()  # descartar fotograma anterior (no procesado)
               except queue.Empty:
                  pass
           self.q.put(frame)

   def read(self) -> cv2.Mat:
       return self.q.get()
