import cv2
import easyocr

reader = easyocr.Reader(["es"] , gpu=False)
image = cv2.imread("plate_1.jpg")
result = reader.readtext(image, paragraph=False)

for res in result:
    print("res:", res)
    print("placa:", result)
    
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()