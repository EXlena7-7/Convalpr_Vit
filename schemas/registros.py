from pydantic import BaseModel
from datetime import datetime
from typing import List

class RegistroBody(BaseModel):
    camara: int
    placa: str
    imagen: str

class RegistroDates(BaseModel):
    init: datetime
    end: datetime

class RegistroBaseResponse(BaseModel):
    id: int
    camara: int
    placa: str
    imagen: str
    momento: datetime
    fecha: datetime


class RegistroResponse(BaseModel):
    result: RegistroBaseResponse | None


class RegistrosPagResponse(BaseModel):
    result: List[RegistroBaseResponse] | List
