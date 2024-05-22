from db.coneccion import Base, engine
from db.models import PlateCamera, Image
print("creating database....")

Base.metadata.create_all(engine)
