
from sqlite3 import Date
from xmlrpc.client import DateTime
from sqlalchemy import String, Boolean, Integer, Column,JSON,func # type: ignore
from datetime import datetime

from db.coneccion import Base

class PlateCamera(Base):
    __tablename__ = 'registros'
    id = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False, unique=True)
    placa = Column(String)
    # ip_camera = Column(String)
    camara = Column(String)    
    interseccion = Column(String)
    # momento = Column(DateTime, default=datetime.now)
    
class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)

# aqui vamos a consultar la otra db
# BASE PARA OTRA DB
class Conf_camara(Base):
    __tablename__ = "conf_obj"
    id = Column(Integer, primary_key=True)
    # conf_camara = Column( NestedMutableJson, nullable=False)
    conf_camara = Column( JSON, nullable=False)
    # create_at = Column(Date, nullable=False, default=func.now())


