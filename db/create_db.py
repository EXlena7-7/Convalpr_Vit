from db.coneccion import Base, engine
from db.models import Plate

print("creating database....")

Base.metadata.create_all(engine)
