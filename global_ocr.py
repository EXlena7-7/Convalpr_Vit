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

# import cv2
# import easyocr

# reader = easyocr.Reader(["es"] , gpu=False)
# image = cv2.imread("plate_33.jpg")
# result = reader.readtext(image, paragraph=False)

# for res in result:
#     print("res:", res)
#     print("placa:", result)
    
# cv2.imshow("Image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()