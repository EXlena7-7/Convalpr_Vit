from timeit import default_timer as timer
import cv2
import numpy as np
from .detector import PlateDetector
from .ocr import PlateOCR
from .saver import SqlSaver
import yaml

ruta_configuracion = "config.yaml"

def obtener_ip_camara_desde_configuracion(ruta_configuracion):
    with open(ruta_configuracion, 'r') as f:
        config = yaml.safe_load(f)
        fuente = config.get('video', {}).get('fuente', None)
        if fuente:
            # Verificar si la fuente es una URL RTSP
            if fuente.startswith('rtsp://'):
                # Extraer la parte de la URL que contiene la IP
                inicio_ip = fuente.find('@') + 1
                final_ip = fuente.find(':', inicio_ip)
                ip_camara = fuente[inicio_ip:final_ip]
                ip_camara = obtener_ip_camara_desde_configuracion(ruta_configuracion)
                print("IP de la cámara:", ip_camara)
                return ip_camara
                
            else:
                raise ValueError("La fuente no es una URL RTSP válida.")
        else:
            raise ValueError("La fuente no está especificada en el archivo de configuración.")
           
class ALPR(SqlSaver):
    detector: PlateDetector = None
    iter_coords: tuple = None
    def __init__(self, cfg: dict, cfg_db: dict):
        super().__init__(
            frequency_insert=cfg_db['insert_frequency'],
            db_path=cfg_db['path']
        )
        input_size = cfg['resolucion_detector']
        if input_size not in (384, 512, 608):
            raise ValueError(
                'Modelo detector no existe! Opciones { 384, 512, 608 }'
            )
        detector_path = f'alpr/models/detection/tf-yolo_tiny_v4-{input_size}x{input_size}-custom-anchors/'
        self.detector = PlateDetector(
            detector_path, input_size, score=cfg['confianza_detector']
        )
        self.ocr = PlateOCR(
            cfg['numero_modelo_ocr'], cfg['confianza_avg_ocr'], cfg['confianza_low_ocr']
        )
        self.guardar_bd = cfg_db['guardar']
        self.plate = None

    def predict(self, frame: np.ndarray) -> list:
        """
        Devuelve todas las patentes reconocidas
        a partir de un frame. Si self.guardar_bd = True
        entonces cada n patentes se guardan en la base de datos

        Parametros:
            frame: np.ndarray sin procesar (Colores en orden: RGB)
        Returns:
            Una lista con todas las patentes reconocidas
        """
        # Preprocess
        input_img = self.detector.preprocess(frame)
        # Inference
        yolo_out = self.detector.predict(input_img)
        # Bounding Boxes despues de NMS
        self.bboxes = self.detector.procesar_salida_yolo(yolo_out)
        # Hacer OCR a cada patente localizada
        self.iter_coords = self.detector.yield_coords(frame, self.bboxes)
        patentes = self.ocr.predict(self.iter_coords, frame)

        if self.guardar_bd:
            self.update_in_memory(patentes)
        return patentes

    def mostrar_predicts(self, frame: np.ndarray):
        """
        Mostrar localizador + reconocedor

        Parametros:
            frame: np.ndarray sin procesar (Colores en orden: RGB)
        Returns:
            frame con el bounding box de la patente y
            la prediccion del texto de la patente

            total_time: tiempo de inferencia sin contar el dibujo
            de los rectangulos
        """
        total_time = 0
        start = timer()
        # Preprocess
        input_img = self.detector.preprocess(frame)
        # Inference
        yolo_out = self.detector.predict(input_img)
        # Bounding Boxes despues de NMS
        self.bboxes = self.detector.procesar_salida_yolo(yolo_out)
        # Hacer y mostrar OCR
        self.iter_coords = self.detector.yield_coords(frame, self.bboxes)
        end = timer()
        total_time += end - start
        fontScale = 1.25
        for yolo_prediction in self.iter_coords:
            x1, y1, x2, y2, _ = yolo_prediction
            #
            cv2.rectangle(frame, (x1, y1), (x2, y2), (36, 255, 12), 2)
            #
            start = timer()
            plate, probs = self.ocr.predict_ocr(x1, y1, x2, y2, frame)
            total_time += timer() - start
            avg = np.mean(probs)
            # print('Patente Camara 1: ',plate, 'Confianza: ',avg * 100, '%')
            plate = (''.join(plate)).replace('_', '')
            
            
            # print('Patente Camara 1: ',plate)
            self.plate = plate
            if avg > self.ocr.confianza_avg and self.ocr.none_low(probs, thresh=self.ocr.none_low_thresh):
                
                mostrar_txt = f'{plate} {avg * 100:.2f}%'
                # print(mostrar_txt) //Ocr de las placas por defecto!
                cv2.putText(img=frame, text=mostrar_txt, org=(x1 - 20, y1 - 15),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale,
                            color=[0, 0, 0], lineType=cv2.LINE_AA, thickness=6)
                cv2.putText(img=frame, text=mostrar_txt, org=(x1 - 20, y1 - 15),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=fontScale,
                            color=[255, 255, 255], lineType=cv2.LINE_AA, thickness=2)
        return frame, total_time