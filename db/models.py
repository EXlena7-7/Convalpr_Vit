
from sqlalchemy import String, Boolean, Integer, Column, Date
from datetime import datetime

from db.coneccion import Base

class Plate(Base):
    __tablename__='plates'
    id=Column(Integer, primary_key=True, nullable=True)
    number_plate=Column(String(20),nullable=False,unique=False)
    infraction=Column(Boolean, default=False)
    created_at=Column(Date, default=datetime.now())
    camera_ip=Column(String,nullable=False)
    interseccion=Column(String,nullable=False)
    


    def __repr__(self):
        return f"<Plate number_plate={self.number_plate}>"

