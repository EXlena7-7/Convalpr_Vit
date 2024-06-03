from sqlalchemy import String, Boolean, Integer, Column, DateTime
from datetime import datetime
from db.coneccion import Base

# Definir los registros de las placas
class PlateCamera(Base):
    __tablename__ = 'registros'
    id = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False, unique=True)
    placa = Column(String)
    # ip_camera = Column(String)
    camara = Column(String)    
    interseccion = Column(String)
    momento = Column(DateTime, default=datetime.now)
    
# Definir la tabla de coordenadas
class Coordenada(Base):
    __tablename__ = 'coordenadas'
    id = Column(Integer, primary_key=True)
    x = Column(String)
    y = Column(String)
    z = Column(String)
    a = Column(String)
    linea1 = Column(String)
    linea2 = Column(String)
    create_at = Column(DateTime)
    
class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)



