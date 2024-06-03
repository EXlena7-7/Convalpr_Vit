import os
import cv2
import uvicorn
# import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
        "*",   
    ]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],)

# Definir la URL de la cámara IP y la ruta de salida directamente en el código
camera_ip = 'rtsp://admin:Covv01%2A.@200.109.234.154:554/' 
output_directory = './capturas'  # Ruta de salida donde se guardarán las capturas  # Ruta de salida donde se guardarán las capturas
capture_filename = 'latest_capture.jpg'  # Nombre del archivo de la captura más reciente

# Crear la carpeta de salida si no existe
if not os.path.exists(output_directory):
    os.makedirs(output_directory)

@app.get("/capture")
def capture_image():
    # Verificar si ya existe una captura
    latest_capture_path = os.path.join(output_directory, capture_filename)
    if os.path.exists(latest_capture_path):
        return FileResponse(latest_capture_path, media_type='image/jpeg')
    
    # Abrir la cámara IP usando la URL proporcionada
    cap = cv2.VideoCapture(camera_ip)

    if not cap.isOpened():
        raise HTTPException(status_code=500, detail="No se pudo abrir la cámara IP")

    # Leer un frame de la cámara
    ret, frame = cap.read()

    if not ret:
        raise HTTPException(status_code=500, detail="No se pudo capturar una imagen")

    # Guardar la imagen en disco en formato JPEG con calidad reducida
    quality = 50  # Ajusta la calidad entre 0 y 100 (menor valor, menor calidad y tamaño)
    success = cv2.imwrite(latest_capture_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        raise HTTPException(status_code=500, detail="Error al guardar la imagen")

    # Liberar la cámara
    cap.release()

    return FileResponse(latest_capture_path, media_type='image/jpeg')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
