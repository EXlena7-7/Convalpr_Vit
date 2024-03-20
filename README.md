# Convalpr_Vit

# Aplicacion de Api con FastAPI para detectar placas

# Video de prueba en assets/sample.mp4
#Ruta del Video de Prueba
https://drive.google.com/file/d/1K_29oQHqrTK23-KP8h5mbsKztkDBIt0m/view?usp=sharing


# Documentacion 
<!-- En el Siguiente Fragmento de Codigo se Obtuvieron los recortes de las placas detetactas
de la funcion generadora dentro de modulo ocr.py
"""
        # Hacer prediction
        pred = self.cnn_ocr_model(patente_recortada)
        return pred[next(iter(pred))].numpy()
"""
 -->

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

<!-- En el Modulo global ocr quedo aparte debido a que se hisieron unas pruebas para obtener el ocr del
directorio de plates donde se guardaban las placas recortadas y a su vez se le aplico easyocr
 -->

 """
import os
import cv2
import easyocr

reader = easyocr.Reader(["es"] , gpu=False)

# Directory containing the images
directory = './plates'

# Iterate over all files in the directory
for filename in os.listdir(directory):
    if filename.endswith(".jpg"):  # Add more conditions if you have different file types
        # Construct full file path
        filepath = os.path.join(directory, filename)
        
        # Read the image
        image = cv2.imread(filepath)
        

result = reader.readtext(image, paragraph=False)
# Supongamos que 'resultados' es una lista de predicciones
resultados = [...]  # Lista de predicciones
# Extrae solo el texto de cada predicción

for res in result:

    texto_reconocido = res[1]  

print("patente: ",texto_reconocido)
 """


<!-- Ademas para aplicar el ocr se esta utilizando alpr para obtener el caracteres de las patente -->

"""
 # print('Patente Camara 1: ',plate, 'Confianza: ',avg * 100, '%')
            plate = (''.join(plate)).replace('_', '')
            # print('Patente Camara 1: ',plate)
            self.plate = plate
            if avg > self.ocr.confianza_avg and self.ocr.none_low(probs, thresh=self.ocr.none_low_thresh):
"""

<!-- de la variable plate dentro de la clase class ALPR(SqlSaver): fue resignada asimisma con: -->
"""
self.plate = None
"""

<!-- Y mostrada en el modulo principal main.py como: -->
"""
alpr.plate
"""