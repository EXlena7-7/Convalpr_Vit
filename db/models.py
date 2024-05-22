
from sqlalchemy import String, Boolean, Integer, Column, DateTime
from datetime import datetime

from db.coneccion import Base

class PlateCamera(Base):
    __tablename__ = 'registros'
    id = Column(Integer, primary_key=True, autoincrement=True, index=True, nullable=False, unique=True)
    placa = Column(String)
    # ip_camera = Column(String)
    camara = Column(String)    
    interseccion = Column(String)
    momento = Column(DateTime, default=datetime.now)
    
    
class Image(Base):
    __tablename__ = "images"
    id = Column(Integer, primary_key=True, index=True)
    file_name = Column(String, index=True)



