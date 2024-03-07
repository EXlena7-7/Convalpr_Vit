import os
import hashlib
# Mostrar solo errores de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Desabilitar GPU ( correr en CPU )
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
from fastapi import FastAPI, Response, UploadFile, File
from fastapi.responses import StreamingResponse
from starlette.requests import Request
import asyncio
from vidgear.gears import CamGear
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
import tempfile
from datetime import datetime
fechaActual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from sqlalchemy import desc
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base

# import ffmpeg
from PIL import Image
import easyocr

SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost/prueba" 

engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)

Base.metadata.create_all(bind=engine)

arrayReconocidos = []
class ImageEventHandler(FileSystemEventHandler):

    image_path = os.path.join(os.path.dirname(__file__), 'plates')
    hash_dict = {}
    def on_created(self, event):
        if not event.is_directory:
        
            # Apply OCR to the new image
            image_path = event.src_path
            
            # Aqui es la lectura de la imagen 
            image = cv2.imread(image_path)



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
    reader = easyocr.Reader(['en'])  # Carga el modelo de EasyOCR para inglés

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    cv2.dilate(img, (5, 5), img)

    # Utiliza EasyOCR para reconocer texto en la imagen
    result = reader.readtext(img)

    # Verifica si se detectó algún texto
    if( result == ""):
        return False


async def read_ocr(path: str):
    reader = easyocr.Reader(["es"] , gpu=False)
    directory = './plates'

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            filepath = os.path.join(directory, filename)

        image = cv2.imread(filepath)
        text = reader.readtext(image, paragraph=False)

        resultados = [...]
        # Extrae solo el texto de cada predicción
        for res in text:

            text = res[1]  

            print("Patente: ", text) 
    
            exist = text in arrayReconocidos
            if( not exist ):
                arrayReconocidos.append(text)
            return not exist

    
app = FastAPI()


origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="./plates"), name="static")

@app.get("/placas")
def get_images(page: int = Query(1, gt=0), page_size: int = Query(10, gt=0, le=100)):
    # Calcular el índice de inicio y fin para la paginación
    start_index = (page - 1) * page_size
    end_index = start_index + page_size

    # Crear una sesión
    db = SessionLocal()

    # Consultar todas las imágenes almacenadas en la base de datos, ordenadas por ID en orden descendente
    images = db.query(Image).order_by(desc(Image.id)).offset(start_index).limit(page_size).all()

    # Cerrar la sesión
    db.close()

    # Si no se encontraron imágenes en la página especificada, lanzar una excepción 404
    if not images:
        raise HTTPException(status_code=404, detail="No se encontraron imágenes en la página especificada")

    # Devolver las imágenes como respuesta
    return images


@app.get("/ultimo")
def return_images():
    list = os.listdir('./plates')

    ultimo_elemento = list[-1]
    return ultimo_elemento;

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

    intervalo_reconocimiento = cfg['video']['frecuencia_inferencia']
    if not is_img:
        logger.info(f'El intervalo del reconocimiento para el video es de: {intervalo_reconocimiento}')

    count = 1

    try:
        stream = CamGear(source='rtsp://admin:Vt3lc4123@38.51.120.236:8061').start()
        
        while True:
            ret, frame = cap.read()
          
            if ret is False:
                break  
            frame = cv2 .cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = stream.read()
            if frame is None or np.sum(frame) ==  0:
                continue 
                        # Reduce video quality using ffmpeg-python
            # (
            #     ffmpeg
            #     .input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(*frame.shape[1::-1]))
            #     .output('pipe:', format='rawvideo', pix_fmt='rgb24', vcodec='libx264', crf=24)
            #     .overwrite_output()
            #     .run_async(pipe_stdin=True, pipe_stdout=True)
            # )
            
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
                new_frame = plate_foto.copy()[y1:y2, x1:x2]
                
                image = cv2.imencode('.jpg', new_frame)[1].tobytes()
                
                frame_name = 'plates\plate_{}.jpg'.format(count)
                print(frame_name);
                image = cv2.imwrite(frame_name,new_frame)
                valor = read_image(frame_name)
                if ( valor == False ) :
                    try:
                        os.remove(frame_name)
                        print('hora')
                    except FileNotFoundError:    
                        print('El archivo no se encontro')


 
                db = SessionLocal()

                # Crear un objeto de la clase Image utilizando el nombre del archivo
                new_image = Image(file_name=frame_name)

                # Agregar el objeto a la sesión y confirmar los cambios
                db.add(new_image)
                db.commit()
            
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
    # root_folder = './plates'
    # duplicates = find_duplicate_files(root_folder)
    # remove_duplicate_files(duplicates)
    try:
        parser = ArgumentParser()
        parser.add_argument("--cfg", dest="cfg_file", help="Path del archivo de config, \
                            default: ./config.yaml", default='config.yaml')
        parser.add_argument("--demo", dest="demo",
                            action='store_true', help="En vez de guardar las patentes, mostrar las predicciones")

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
    try:
        # Load the configuration here (this is just an example, you'll need to adjust it)
        with open('config.yaml', 'r') as stream:
            cfg = yaml.safe_load(stream)
        return StreamingResponse(gen_frames(cfg), media_type="multipart/x-mixed-replace;boundary=frame")

    except Exception as e:
        error_message = {"Cámara no encontrada": str(e)}
    return error_message

