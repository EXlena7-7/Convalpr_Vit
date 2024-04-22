import yaml

ruta_configuracion ="C://Users//VIT//Desktop//EMPRESA_VIT_API_PROCESAMIENTO_IMAGEN//Convalpr_Vit//config.yaml"

def obtener_ip_y_puerto_camara_desde_configuracion(ruta_configuracion):
    with open(ruta_configuracion, 'r') as f:
        config = yaml.safe_load(f)
        fuente = config.get('video', {}).get('fuente', None)
        if fuente:
            # Verificar si la fuente es una URL RTSP
            if fuente.startswith('rtsp://'):
                # Extraer la parte de la URL que contiene la IP y el puerto
                inicio_ip = fuente.find('@') + 1
                final_ip = fuente.find(':', inicio_ip)
                ip_camara = fuente[inicio_ip:final_ip]
                # Extraer el puerto de la URL RTSP
                inicio_puerto = final_ip + 1
                final_puerto = fuente.find('/', inicio_puerto)
                puerto_camara = fuente[inicio_puerto:final_puerto]
                # print('Puerto:',puerto_camara)
                return ip_camara, puerto_camara
            else:
                raise ValueError("La fuente no es una URL RTSP válida.")
        else:
            raise ValueError("La fuente no está especificada en el archivo de configuración.")