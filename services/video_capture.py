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