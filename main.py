import os
import pandas as pd

import time
import hashlib
import cvzone

import itertools
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
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, HTTPException, Query
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Boolean, Integer, String, Float, DateTime, func
from sqlalchemy import desc, asc
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import tempfile
# import ffmpeg
from PIL import Image
# import easyocr
from typing import Dict

from ultralytics import YOLO
model = YOLO('yolov8s.pt')
from tracker import Tracker
from sort import *
import math
from poligono import *
classnames  = []
with open('coco.txt','r') as f:
    classnames = f.read().splitlines()

# funtions:
tracker = Sort(max_age=20)
line = [50, 550, 3900, 550]
counter = []

tracker = Tracker()
line = [100, 550, 2900, 550]

my_file = open(r"./coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Clases de interés
classes_of_interest = ["bus", "truck", "car"]

cy1 = 322
cy2 = 368
offset = 6

vh_down = {}
vh_up = {}


# Define el modelo base
Base = declarative_base()


# Define la clase del modelo para la tabla de placas y cámaras
class PlateCamera(Base):
    __tablename__ = 'registros'
    id = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False, unique=True)
    placa = Column(String)
    # ip_camera = Column(String)
    camara = Column(String)    
    interseccion = Column(String)
    momento = Column(DateTime, default=datetime.now)
    
    
# Configura la conexión a la base de datos PostgreSQL
engine = create_engine('postgresql://postgres:password@localhost/api')
# engine = create_engine('postgresql://postgres:123456@192.168.7.246/detecion_semaforos')

# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost/prueba" 
# engine = create_engine(SQLALCHEMY_DATABASE_URL)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


# Crea todas las tablas definidas en los modelos en la base de datos
Base.metadata.create_all(engine)

# # Crea una sesión de SQLAlchemy
Session = sessionmaker(bind=engine)
session = Session()


def get_plate_cameras(pag: int = 0, limit: int = 10):
    # Crear una sesión
    db = SessionLocal()
    try:
        offset = pag * limit
        # Realizar la consulta y devolver los resultados
        return db.query(PlateCamera).offset(offset).limit(limit).all()
    finally:
        # Cerrar la sesión
        db.close()
        
def get_total_plates():
    db = SessionLocal()
    try:
        return db.query(PlateCamera).count();
    finally:
        db.close()
        
def get_last_plate_numbers():
    db = SessionLocal()
    try:
        # plate_numbers = db.query(PlateCamera.plate_number).order_by(desc(PlateCamera.created_at)).limit(5).all()
        plate_numbers = db.query(PlateCamera.placa).order_by(desc(PlateCamera.id)).limit(5).all()
        
        print (plate_numbers)
        
        return [plate_number[0] for plate_number in plate_numbers]
    
        # plate_numbers_flat = [plate_number[0] for plate_number in plate_numbers]

# # Verificar si todos los números de placa son iguales
# if len(set(plate_numbers_flat)) == 1:
#     # Si todos son iguales, devolver una lista con un solo elemento
#     return [plate_numbers_flat[0]]
# else:
#     # Si hay variación en los números de placa, devolver la lista completa
#     return plate_numbers_flat
 
    finally:
        db.close()
        

class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)

# Base.metadata.create_all(bind=engine)

arrayReconocidos = []
# class ImageEventHandler(FileSystemEventHandler):

#     image_path = os.path.join(os.path.dirname(__file__), 'plates')
#     hash_dict = {}
#     def on_created(self, event):
#         if not event.is_directory:
        
#             # Apply OCR to the new image
#             image_path = event.src_path
            
#             # Aqui es la lectura de la imagen 
#             image = cv2.imread(image_path)



# def save_temp_file(file_content: bytes, callback: callable):
#     # Crear un archivo temporal en la carpeta 'plates' y eliminarlo automáticamente después de su uso
#     with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True, dir="plates") as temp_file:
#         # Escribir el contenido del archivo en el archivo temporal
#         temp_file.write(file_content)
#         temp_file.flush()

#         # Obtener la ruta del archivo temporal
#         temp_path = temp_file.name

#         # Llamar a la función de callback con la ruta del archivo temporal
#         return callback(temp_path)

# # Ejemplo de uso
# def callback_function(file_path):
#     print("La imagen se guardó temporalmente en:", file_path)
#     # Aquí puedes hacer cualquier cosa con la imagen
#     # Por ejemplo, cargarla, procesarla, etc.

# # Llamada a la función save_temp_file con algún contenido de archivo y la función de callback
# save_temp_file(b'Contenido de la imagen', callback_function)

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

ruta_configuracion = "config.yaml"



# def obtener_ip_y_puerto_camara_desde_configuracion(ruta_configuracion):
#     with open(ruta_configuracion, 'r') as f:
#         config = yaml.safe_load(f)
#         fuente = config.get('video', {}).get('fuente', None)
#         if fuente:
#             # Verificar si la fuente es una URL RTSP
#             if fuente.startswith('rtsp://'):
#                 # Extraer la parte de la URL que contiene la IP y el puerto
#                 inicio_ip = fuente.find('@') + 1
#                 final_ip = fuente.find(':', inicio_ip)
#                 ip_camara = fuente[inicio_ip:final_ip]
#                 # Extraer el puerto de la URL RTSP
#                 inicio_puerto = final_ip + 1
#                 final_puerto = fuente.find('/', inicio_puerto)
#                 puerto_camara = fuente[inicio_puerto:final_puerto]
#                 # print('Puerto:',puerto_camara)
#                 return ip_camara, puerto_camara
#             else:
#                 raise ValueError("La fuente no es una URL RTSP válida.")
#         else:
#             raise ValueError("La fuente no está especificada en el archivo de configuración.")


@app.get("/plate_cameras/")
def read_plate_cameras(pag: int = 0, limit: int = 10):
    cameras = get_plate_cameras(pag,limit)
    total = get_total_plates()
    pages = total % limit
    return {
        'total': total,
        'pages': pages,
        'data': cameras,
    }

@app.get("/last_plate/")
def read_last_plate_numbers():
    plate_numbers = get_last_plate_numbers()
    return plate_numbers


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


# class GeneradorID:
#     def __init__(self):
#         self.contador = 0
    
#     def generar_id(self):
#         self.contador += 1
#         return self.contador
    
# generador = GeneradorID()
# id_1 = generador.generar_id()

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
        
        # Guardar los datos en la base de datos
        plate_number = alpr.plate
        
        if len(plate_number) >= 6:  # Validación de longitud mínima de placa
            # ip_camera = obtener_ip_y_puerto_camara_desde_configuracion(ruta_configuracion)[0]
            ip_camera = 192.121
            nueva_entrada = PlateCamera(placa=plate_number, camara=ip_camera, interseccion="AVENIDA RAFAEL GONZALEZ CON JACINTO LARA") 
            session.add(nueva_entrada)
            session.commit()

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
    CamGear  = VideoCapture(video_path)
    placas = []  # Diccionario para almacenar placas y sus IDs
    count = 0
    vh_up={}
    counter1=[]

    tracker=Tracker()

    while True:
        try:
            # stream = VideoCapture(video_path)
            await asyncio.sleep(0.30)
            frame = CamGear.read()
            count += 1
            if count % 3 != 0:
                continue
            cy1=500
            cy2=668
            offset=6

            vh_down={}
            counter=[]

            vh_up={}
            counter1=[]
            offset=6
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # frame = stream.read()
            # Definir ROI
            x = 320
            y = 200
            w = 800
            h = 490
            # Extraer ROI
            roi = frame[y:y+h, x:x+w].copy()
            personas_cont=[]
            results=model.predict(roi)

            # frame = cv2.cvtColor(frasme, cv2.COLOR_BGR2RGB)
            # frame = stream.read()

            plate_foto, total_time = alpr.mostrar_predicts(roi)

             # Verificar si la placa ya está en el diccionario
            if alpr.plate in placas:
                # Si la placa ya está en el diccionario, no necesitamos hacer nada más
                pass
            else:
                # Si la placa es nueva, la agregamos al diccionario con su ID correspondiente
                
                #placas[alpr.plate] = count
                if len(placas) == 50:
                    placas.pop(0)
                    
                placas.append(alpr.plate)
                
            results = model.predict(frame)
            a = results[0].boxes.data
            px = pd.DataFrame(a).astype("float")
            list = []
            
           # Inicializa el contador de vehículos
            carros_cont=[]
            for numero in results[0].boxes.cls:
            # Incrementar el conteo del número actua
                if numero==2:
                #    print('222')
                #    carros_cont += 1
                   carros_cont.append(numero)
                # elif numero==0:
                #     personas_cont.append(numero)
            carros_cont=len(carros_cont)
            # personas_cont=len(personas_cont)
            results = model.predict(roi)
            
            a=results[0].boxes.data
            px=pd.DataFrame(a).astype("float")
            # cv2.rectangle(roi, (x, y), (x+w, y+h), (255, 0, 0), 2)
        #    print(px)
            list=[]
            for index,row in px.iterrows():
                x1=int(row[0])
                y1=int(row[1])
                x2=int(row[2])
                y2=int(row[3])
                d=int(row[5])
                c=class_list[d]
                if "bus" in c or "truck" in c or "car" in c:
                    list.append([x1,y1,x2,y2])
            bbox_id=tracker.update(list)
            for bbox in bbox_id:
                x3,y3,x4,y4,id=bbox
                cx=int(x3+x4)//2
                cy=int(y3+y4)//2
        
                cv2.rectangle(roi,(x3,y3),(x4,y4),(0,255,255),2)
            
            
                if cy1<(cy+offset) and cy1 > (cy-offset):
                    vh_down[id]=time.time()
                if id in vh_down:
                    if cy2<(cy+offset) and cy2 > (cy-offset):
                        counter.append(id)

            cv2.line(roi,(170,cy1),(1900,cy1),(255,255,255),3)

            # cv2.putText(roi,('L1'),(260,180),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),4)

            # cv2.line(roi,(177,cy2),(1927,cy2),(255,255,255),4)
            
            # cv2.putText(roi,('L2'),(282,367),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),4)
            d=(len(counter))
            u=(len(counter1))
            # cv2.putText(frame,('goingdown:-')+str(d),(60,90),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),4)

            # cv2.putText(frame,('goingup:-')+str(u),(60,130),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
            # area de reportes:
            
            # cv2.putText(frame,('vehiculos: -> ')+str(carros_cont),(80,109),cv2.FONT_HERSHEY_COMPLEX,1.4,(255,0,0),3)
            cvzone.putTextRect(frame,f'Total Vehicles ={str(carros_cont)}',[290,34],thickness=4,scale=2.3,border=2)
            # cv2.putText(frame,('personas en area : -> ')+str(personas_cont),(60,790),cv2.FONT_HERSHEY_COMPLEX,1.5,(0,0,255),3)
            # # end
            # ingresa el area procesada al frame principal
            frame[y:y+h, x:x+w]=roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

            # plate_foto, total_time = alpr.mostrar_predicts(frame)
            plate_foto, total_time = alpr.mostrar_predicts(roi)
            


             # Verificar si la placa ya está en el diccionario
            if alpr.plate in placas:
                # Si la placa ya está en el diccionario, no necesitamos hacer nada más
                pass
            else:
                # Si la placa es nueva, la agregamos al diccionario con su ID correspondiente
                
                #placas[alpr.plate] = count
                if len(placas) == 100:
                    placas.pop(0)
                    
                placas.append(alpr.plate)        
                # count = len(placas)
                # placas.append("Nueva placa")
                #count += 1
                print('Placas Detectadas: ', len(placas))
                
                # Mostrar los IDs de las detecciones en la parte inferior del frame
                #cv2.putText(frame, f'Vehiculos: {id}', (50, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                # print("Contador:", count)
                
                
                
                # Función para guardar las placas en la base de datos
                asyncio.ensure_future(save_plate(plate_foto, alpr, count))
                
                # ruta_configuracion = "config.yaml"  # Ruta de tu archivo config.yaml
                # ip_camara = obtener_ip_y_puerto_camara_desde_configuracion(ruta_configuracion)
                # print("CAMARA IP, puerto :", ip_camara)

            # count = count + 1

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + (await resize_frame_to_bytes(frame)) + b'\r\n')

        except asyncio.CancelledError:
            print('Coneccion cerrada')
            raise e
            break
        except Exception as e:
            print(e)
            raise e
            break



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
        with open('config.yaml', 'r') as stream:
            cfg = yaml.safe_load(stream)
        return StreamingResponse(gen_frames(cfg), media_type="multipart/x-mixed-replace;boundary=frame")

    except Exception as e:
        error_message = {"Cámara no encontrada": str(e)}
    return error_message

