from sqlalchemy import create_engine, String, Integer, Column, Numeric, DateTime, ForeignKey
from datetime import datetime
from db.coneccion import Base
from sqlalchemy.orm import relationship

# # Definir los registros de las placas
class PlateCamera(Base):
    __tablename__ = 'registros'
    placa  = Column(String, primary_key=True, index=True, nullable=False, unique=True)
    camara = Column(String)
    interseccion = Column(String)
    momento = Column(DateTime, default=datetime.now)
    id_cantidad = Column(String, ForeignKey('vehiculos.cantidad'))

    vehiculo = relationship("Vehiculo")

class Vehiculo(Base):
    __tablename__ = 'vehiculos'
    cantidad = Column(String, primary_key=True, index=True)
    momento = Column(DateTime, default=datetime.now)

# # Definir la tabla de coordenadas
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



