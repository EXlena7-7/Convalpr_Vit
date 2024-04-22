import cv2
import os
from camaras import *
from camaras.init import *
from db.coneccion import *
from db.models import *
import pandas as pd
from sort import *
import math
from data import *
import time
import hashlib
import cvzone
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

from services.video_capture import VideoCapture
import logging
from timeit import default_timer as timer
from datetime import datetime
fechaActual = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
# from watchdog.observers import Observer
# from watchdog.events import FileSystemEventHandler
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, Query
import tempfile
from PIL import Image
from ultralytics import YOLO
model = YOLO('yolov8s.pt')


classnames  = []
with open('coco.txt','r') as f:
    classnames = f.read().splitlines()

# funtions:
tracker = Sort(max_age=20)
line = [50, 550, 3900, 550]
counter = []



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
    

    finally:
        db.close()
        

arrayReconocidos = []


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
def poligonDeInteres(frame):
    img =frame.copy()

    px1,py1,px2,py2,px3,py3,px4,py4 = conf_camara[0]['camara3']['poligono'].values()
    print(px1)

     # Definir los puntos iniciales del polígono (aquí puedes ajustar según tu necesidad)
    poligonos = np.array([[px1, py1], [px2, py2], [px3, py3 ],[px4, py4]], np.int32)
   #   # Dibujar el polígono en la imagen
    
   #  # Crear una máscara negra del mismo tamaño que la imagen original
    mask = np.zeros_like(img)
    
   #  # Dibujar el polígono en la máscara
    cv2.fillPoly(mask, [poligonos], (255, 255, 255))
    
   #  # Usar la máscara para dejar en negro el exterior del polígono en la imagen original
    img[mask == 0] = 0
    
   #  # Dibujar el polígono en la imagen
    cv2.polylines(frame, [poligonos], isClosed=True, color=(0, 255, 0), thickness=2)
    # cv2.imshow('mascara', img)
    if cv2.waitKey(1) & 0xFF == 27:
        pass
    # cv2.release()
    # Mostrar la imagen con el polígono dibujado en pantalla
    
    # print('llego')
    return img

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
        # print('Que Diablos eres:',plate_foto)
        # cv2.imshow('placa',new_frame)
        # result2=poligonDeInteres(frame)
        # cv2.rectangle(result2, (x1, y1), (x2, y2), (255, 0, 0), 2)
        # new_frame, total_time = alpr.mostrar_predicts(result2)
        # Guardar los datos en la base de datos
        plate_number = alpr.plate
        
        if len(plate_number) >= 6:  # Validación de longitud mínima de placa
            ip_camera = obtener_ip_y_puerto_camara_desde_configuracion(ruta_configuracion)[0]
            nueva_entrada = PlateCamera(placa=plate_number, camara=ip_camera, interseccion="AVENIDA RAFAEL GONZALEZ CON JACINTO LARA") 
            session.add(nueva_entrada)
            session.commit()

async def resize_frame_to_bytes(frame: cv2.Mat):
    # Obtén las dimensiones originales del frame
    height, width, _ = frame.shape

    # Define las nuevas dimensiones aquí. Por ejemplo, para reducir a la mitad:
    new_width = width // 4
    new_height = height // 4

    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation = cv2.INTER_AREA)

    _, buffer = cv2.imencode('.jpg', frame)
    return buffer.tobytes()


# def VehiclesInArea(array):
    # cvzone.putTextRect(frame,f'Vehiculos en Area ={len(array)}',[290,74],thickness=4,scale=2.3,border=2)
    # print(array)

async def gen_frames(cfg):
    alpr = ALPR(cfg['modelo'], cfg['db'])
    video_path = cfg['video']['fuente']
    CamGear  = VideoCapture(video_path)
    placas = []  # Diccionario para almacenar placas y sus IDs
    counter = []
    count=0

    while True:
        try:
            # stream = VideoCapture(video_path)
            await asyncio.sleep(0.30)
            frame = CamGear.read()
            count += 1
            if count % 3 != 0:
                continue
            detecciones=[]
            detections = np.empty((0,5))
            # plate_foto, total_time = alpr.mostrar_predicts(frame)
            result2=poligonDeInteres(frame)
            
            result = model(result2,stream=1)
            for info in result:
                boxes = info.boxes
                for box in boxes:
                    x1,y1,x2,y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls)
                    detecciones.append(cls)
                    # VehiclesInArea(detecciones)
                    # print(cls,'angel seguridad')
                    classindex = box.cls[0]
                    conf = math.ceil(conf * 100)
                    classindex = int(classindex)
                    objectdetect = classnames[classindex]
                    if objectdetect == 'car' or objectdetect == 'bus' or objectdetect =='truck' or objectdetect =='motorcycle' and conf >60:
                        x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                        new_detections = np.array([x1,y1,x2,y2,conf])
                        detections = np.vstack((detections,new_detections))
                        #vamos hacer la prueba
                        # print(f'frame {type(frame)} y frame poligono {type(result2)}')
                      
                    plate_foto, total_time = alpr.mostrar_predicts(frame)
                    # print('Cafe:',list(rectangulo_placas))
                    cv2.imshow('plate', plate_foto)
                    # if cv2.waitKey(1) & 0xFF == 27:
                    #         pass
                    track_result = tracker.update(detections)
                    cv2.line(frame,(line[0],line[1]),(line[2],line[3]),(0,255,255),7)
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
    
                for results in track_result:
                    x1,y1,x2,y2,id = results
                    x1, y1, x2, y2, id = int(x1), int(y1), int(x2), int(y2),int(id)

                    w,h = x2-x1,y2-y1
                    cx,cy = x1+w//2 , y1+h//2

                    cv2.circle(frame,(cx,cy),6,(0,0,255),-1)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),3)
                    cvzone.putTextRect(frame,f'{id}',
                                    [x1+8,y1-12],thickness=2,scale=1.5)

                    if line[0] < cx <line[2] and line[1] -20 <cy <line[1]+20:
                        cv2.line(frame, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 15)
                        if counter.count(id) == 0:
                            counter.append(id)
            cvzone.putTextRect(frame,f'Total Vehicles ={len(counter)}',[290,34],thickness=4,scale=2.3,border=2)
            
            # Función para guardar las placas en la base de datos
            asyncio.ensure_future(save_plate(plate_foto, alpr, count))
                
            # ruta_configuracion = "config.yaml"  # Ruta de tu archivo config.yaml
            ip_camara = obtener_ip_y_puerto_camara_desde_configuracion(ruta_configuracion)
            print("CAMARA IP, puerto :", ip_camara)

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
        with open(r'config.yaml', 'r') as stream:
            cfg = yaml.safe_load(stream)
        return StreamingResponse(gen_frames(cfg), media_type="multipart/x-mixed-replace;boundary=frame")

    except Exception as e:
        error_message = {"Cámara no encontrada": str(e)}
    return error_message