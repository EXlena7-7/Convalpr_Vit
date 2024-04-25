import cv2
import numpy as np


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

# obj conf:
conf_camara=[{
   'camara1':{
     'roi':{
        'x' : 400,
        'y' : 400,
        'w' : 400,
        'h' : 300 ,

     },
      'poligono':{
          'px1':1050,
          'py1':1000,
          'px2':1000,
          'py2':658,
          'px3':1510,
          'py3':650,
          'px4':1920,
          'py4':900,
          

     },

     'lineas':[
      {
         'linea':{
            'cy1':180,
            'cx1':0,
            'cx2':700,
            'lcolor1':255,
            'lcolor2':255,
            'lcolor3':255,
            'boder':1,
            'tx1':0,
            'tx2':180,
            'sizetx':0.8,
            'grosor':2,
            'etiqueta':'L11',
            'txcolor1':0,
            'txcolor2':255,
            'txcolor3':255,
         }
      },      {
         'linea':{
            'cy1':368,
            'cx1':177,
            'cx2':927,
            'lcolor1':255,
            'lcolor2':255,
            'lcolor3':255,
            'boder':1,
            'tx1':182,
            'tx2':367,
            'sizetx':0.8,
            'grosor':2,
            'etiqueta':'L22',
            'txcolor1':0,
            'txcolor2':255,
            'txcolor3':255,


         }
      }

   ],
   'conte_vehiculos':0,
   'conte_personas':0,
   },
    'camara2':{
     'roi':{
        'x' : 1000,
        'y' : 600,
        'w' : 700,
        'h' : 300 ,

     },
      'roi2':{
        'x' : 0,
        'y' : 0,
        'w' : 0,
        'h' : 0 ,

     },
     'poligono':{
          'px1':1050,
          'py1':1000,
          'px2':1000,
          'py2':658,
          'px3':1510,
          'py3':650,
          'px4':1920,
          'py4':900,
          

     },

     'lineas':[
      {
         'linea':{
            'cy1':880,
            'cx1':1100,
            'cx2':1900,
            'lcolor1':255,
            'lcolor2':255,
            'lcolor3':255,
            'boder':10,
            'tx1':1050,
            'tx2':860,
            'sizetx':0.8,
            'grosor':2,
            'etiqueta':'conteo caracas',
            'txcolor1':0,
            'txcolor2':255,
            'txcolor3':255,
           
         }
      },      {
         'linea':{
            'cy1':369,
            'cx1':177,
            'cx2':927,
            'lcolor1':255,
            'lcolor2':255,
            'lcolor3':255,
            'boder':1,
            'tx1':182,
            'tx2':367,
            'sizetx':0.8,
            'grosor':2,
            'etiqueta':'L22',
            'txcolor1':0,
            'txcolor2':255,
            'txcolor3':255,


         }
      }

   ],
   'conte_vehiculos':0,
   'conte_personas':0,
   },
   # conf camara ccaracas
       'camara3':{
     'roi':{
        'x' : 1000,
        'y' : 600,
        'w' : 700,
        'h' : 300 ,

     },
      'roi2':{
        'x' : 0,
        'y' : 0,
        'w' : 0,
        'h' : 0 ,

     },
     'poligono':{
          'px1':350,
          'py1':700,
          'px2':300,
          'py2':200,
          'px3':450,
          'py3':250,
          'px4':1000,
          'py4':700,
          

     },

     'lineas':[
      {
         'linea':{
            'cy1':570,
            'cx1':340,
            'cx2':840,
            'lcolor1':255,
            'lcolor2':255,
            'lcolor3':255,
            'boder':10,
            'tx1':1050,
            'tx2':860,
            'sizetx':0.8,
            'grosor':2,
            'etiqueta':'conteo',
            'txcolor1':0,
            'txcolor2':255,
            'txcolor3':255,
           
         }
      },      {
         'linea':{
            'cy1':369,
            'cx1':177,
            'cx2':927,
            'lcolor1':255,
            'lcolor2':255,
            'lcolor3':255,
            'boder':1,
            'tx1':182,
            'tx2':367,
            'sizetx':0.8,
            'grosor':2,
            'etiqueta':'L22',
            'txcolor1':0,
            'txcolor2':255,
            'txcolor3':255,


         }
      }

   ],
   'conte_vehiculos':0,
   'conte_personas':0,
   },
},
]