import os
# Mostrar solo errores de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Desabilitar GPU ( correr en CPU )
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import StreamingResponse
from starlette.requests import Request
import asyncio

from vidgear.gears import VideoGear

from typing import Callable
import numpy as np 
if not os.path.exists('plates'):
       os.makedirs('plates')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
import time
from alpr.alpr import ALPR
from argparse import ArgumentParser
import yaml
import logging
from timeit import default_timer as timer
import cv2
import pytesseract
import tempfile
from datetime import datetime
fechaActual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import ffmpeg
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
arrayReconocidos = [];
class ImageEventHandler(FileSystemEventHandler):

    image_path = os.path.join(os.path.dirname(__file__), 'plates')
    
    def on_created(self, event):
        if not event.is_directory:
            # Apply OCR to the new image
            image_path = event.src_path
            image = cv2.imread(image_path)
            text = pytesseract.image_to_string(image, lang='spa')
            #print(text, f"OCR Result: {text}")
            
def on_created(src_path: str):
        # Apply OCR to the new image
        print(src_path)
        image = cv2.imread(src_path)
        # text = pytesseract.image_to_string(image, lang='spa')
        #print(text, f"OCR Result: {text}")

def save_temp_file(file: bytes, callback: Callable[[str], any]):
    with tempfile.NamedTemporaryFile(delete=True) as temp:
        data = file
        temp.write(data)
        temp_path = temp.name
        print(temp_path)
        return callback(temp_path)


logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def read_image(path: str):
    #img = Image.open(path).convert("1")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 
    cv2.dilate(img, (5, 5), img)
    text = pytesseract.image_to_string(img)
    if( text == ""):
        return False
    
    exist = text in arrayReconocidos
    if( not exist ):
        arrayReconocidos.append(text)
    return not exist
    return text;
#    text = pytesseract.image_to_string(img, lang='eng')
    
app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

async def gen_frames(cfg, demo=True, benchmark=True, save_vid=False):
    alpr = ALPR(cfg['modelo'], cfg['db'])
    video_path = cfg['video']['fuente']
    cap = cv2.VideoCapture(video_path)
    is_img = cv2.haveImageReader(video_path)  # que hace? is_img
    cv2_wait = 0 if is_img else 1
    logger.info(f'Se va analizar la fuente: {video_path}')
    observer = Observer()
    observer.schedule(ImageEventHandler(), path='plates', recursive=False)
    observer.daemon = True
    observer.start()
    # if save_vid:
    #     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    #     size = (width, height)
        # fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # out = cv2.VideoWriter('alpr-result.avi', fourcc, 20.0, size)
    # Cada cuantos frames hacer inferencia
    intervalo_reconocimiento = cfg['video']['frecuencia_inferencia']
    if not is_img:
        logger.info(f'El intervalo del reconocimiento para el video es de: {intervalo_reconocimiento}')

    count = 1

    try:
        stream = VideoGear(source='rtsp://admin:abc123**@192.168.7.136').start()
        
        while True:
            ret, frame = cap.read()
            frame = cv2 .cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = stream.read()
            if ret is False:
                break  
                
                        # Reduce video quality using ffmpeg-python
            (
                ffmpeg
                .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(*frame.shape[1::-1]))
                .output('pipe:', format='rawvideo', pix_fmt='rgb24', vcodec='libx264', crf=24)
                .overwrite_output()
                .run_async(pipe_stdin=True, pipe_stdout=True)
            )
            
            if demo:
            
                plate_foto, total_time = alpr.mostrar_predicts(frame)
             
            out_boxes, __, _, num_boxes = alpr.bboxes
            image_h, image_w, _ = plate_foto.shape
            for i in range(num_boxes[0]):
                coor = out_boxes[0][i]
                x1 = int(coor[1] * image_w)
                y1 = int(coor[0] * image_h)
                x2 = int(coor[3] * image_w)
                y2 = int(coor[2] * image_h) 
            
              # Obtén las dimensiones originales del frame
            height, width, _ = frame.shape
            
             # Define las nuevas dimensiones aquí. Por ejemplo, para reducir a la mitad:
            new_width = width // 2
            new_height = height // 2
            
            resized_frame = cv2.resize(frame, (new_width, new_height), interpolation = cv2.INTER_AREA)
            
            _, buffer = cv2.imencode('.jpg', resized_frame)
            resized_frame = buffer.tobytes()
            
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + resized_frame + b'\r\n')
    finally:
        stream.stop()

        # frame_id += 1


if __name__ == '__main__':

    try:
        parser = ArgumentParser()
        parser.add_argument("--cfg", dest="cfg_file", help="Path del archivo de config, \
                            default: ./config.yaml", default='config.yaml')
        parser.add_argument("--demo", dest="demo",
                            action='store_true', help="En vez de guardar las patentes, mostrar las predicciones")
        # parser.add_argument("--guardar_video", dest="save_video",
        #                     action='store_true', help="Guardar video en ./alpr-result.avi")
        # parser.add_argument("--benchmark", dest="bench",
        #                     action='store_true', help="Medir la inferencia (incluye todo el pre/post processing")
        args = parser.parse_args()
        with open(args.cfg_file, 'r') as stream:
            try:
                cfg = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                logger.exception(exc)
        gen_frames(cfg, args.demo, args.bench)
    except Exception as e:
        logger.exception(e)




@app.get("/video_feed/")
async def video_feed():
    # Load the configuration here (this is just an example, you'll need to adjust it)
    with open('config.yaml', 'r') as stream:
        cfg = yaml.safe_load(stream)
    return StreamingResponse(gen_frames(cfg), media_type="multipart/x-mixed-replace;boundary=frame")
