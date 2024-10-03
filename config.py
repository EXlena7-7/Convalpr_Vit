config = {
    'video': {
        # Fuente de la cámara o video
        'fuente': 'rtsp://admin:Covv01%2A.@200.109.234.154:554',

        # Cada cuantos frames hacer inferencia
        'frecuencia_inferencia': 30,
    },

    'modelo': {
        # Resolución del detector (opciones pueden ser {608, 512, 384})
        'resolucion_detector': 608,

        # Confianza del detector (Yolo objectness/score)
        'confianza_detector': 0.25,

        # Número del modelo OCR (opciones {1, 2, 3, 4})
        'numero_modelo_ocr': 4,

        # Confianza promedio para los caracteres de las placas (OCR)
        'confianza_avg_ocr': 0.3,

        # Confianza mínima para caracteres individuales
        'confianza_low_ocr': 0.10,
    },

    'db': {
        # Guardar en base de datos SQLite
        'guardar': True,

        # Frecuencia de inserción de placas en la base de datos
        'insert_frequency': 1,

        # Ruta a la base de datos
        'path': 'db/plates.db',
    }
}
