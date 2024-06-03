import cv2
import numpy as np
from conf_camara import conf_camara 


def poligonDeInteres(frame):
    img =frame.copy()

    px1,py1,px2,py2,px3,py3,px4,py4 = conf_camara[0]['camara4']['poligono'].values()
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

