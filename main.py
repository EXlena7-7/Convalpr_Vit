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
from services.video_capture import VideoCapture
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
import tempfile
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



def save_temp_file(file_content: bytes, callback: callable):
    # Crear un archivo temporal en la carpeta 'plates' y eliminarlo automáticamente después de su uso
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True, dir="plates") as temp_file:
        # Escribir el contenido del archivo en el archivo temporal
        temp_file.write(file_content)
        temp_file.flush()

        # Obtener la ruta del archivo temporal
        temp_path = temp_file.name

        # Llamar a la función de callback con la ruta del archivo temporal
        return callback(temp_path)

# Ejemplo de uso
def callback_function(file_path):
    print("La imagen se guardó temporalmente en:", file_path)
    # Aquí puedes hacer cualquier cosa con la imagen
    # Por ejemplo, cargarla, procesarla, etc.

# Llamada a la función save_temp_file con algún contenido de archivo y la función de callback
save_temp_file(b'Contenido de la imagen', callback_function)

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

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

async def save_plate(plate_foto: np.ndarray, alpr: ALPR, count: int):
    out_boxes, __, _, num_boxes = alpr.bboxes
    image_h, image_w, _ = plate_foto.shape
    for i in range(num_boxes[0]):
        coor = out_boxes[0][i]
        x1 = int(coor[1] * image_w)
        y1 = int(coor[0] * image_h)
        x2 = int(coor[3] * image_w)
        y2 = int(coor[2] * image_h)
        new_frame = plate_foto.copy()[y1:y2, x1:x2]

        # image = cv2.imencode('.jpg', new_frame)[1].tobytes()

        # frame_name = 'plates\plate_{}.jpg'.format(count)
        # print(frame_name)
        # # cv2.imshow("plate", image)
        # image = cv2.imwrite(frame_name,new_frame)
        # valor = alpr.mostrar_predicts(frame_name)
        # if ( valor == False ) :
        #     try:
        #         os.remove(frame_name)
        #         print('hora')
        #     except FileNotFoundError:    
        #         print('El archivo no se encontro')


async def resize_frame_to_bytes(frame: cv2.Mat):
    # Obtén las dimensiones originales del frame
    height, width, _ = frame.shape

    # Define las nuevas dimensiones aquí. Por ejemplo, para reducir a la mitad:
    new_width = width // 4
    new_height = height // 4

    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation = cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


async def gen_frames(cfg):
    alpr = ALPR(cfg['modelo'], cfg['db'])
    video_path = cfg['video']['fuente']
    cap = VideoCapture(video_path)

    count = 1

    while True:
        try:
            # stream = VideoCapture(video_path)
            await asyncio.sleep(0.30)
            frame = cap.read()

            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = stream.read()

            plate_foto, total_time = alpr.mostrar_predicts(frame)

            # print(f"Total time: {total_time:.2f} seconds")

            asyncio.ensure_future(save_plate(plate_foto, alpr, count))

            # count = count + 1

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + (await resize_frame_to_bytes(frame)) + b'\r\n')

        except asyncio.CancelledError:
            print('Coneccion cerrada')
            raise e
        except Exception as e:
            print(e)
            raise e



if __name__ == '__main__':

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

