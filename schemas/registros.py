from pydantic import BaseModel
from datetime import datetime
from typing import List

class RegistroBody(BaseModel):
    camara: str
    placa: str
    imagen: str
    interseccion: str

class RegistroDates(BaseModel):
    init: datetime
    end: datetime

class RegistroBaseResponse(BaseModel):
    id: int
    # camara: int
    camara: str
    placa: str
    interseccion: str
    # imagen: str
    momento: datetime
    # fecha: datetime


class RegistroResponse(BaseModel):
    result: RegistroBaseResponse | None


class RegistrosPagResponse(BaseModel):
    result: List[RegistroBaseResponse] | List
