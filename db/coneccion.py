from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Boolean, Integer, String, Float, DateTime, func
from sqlalchemy import desc, asc
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base

# Configura la conexión a la base de datos PostgreSQL
engine = create_engine('postgresql://postgres:password@localhost/api')
# engine = create_engine('postgresql://postgres:123456@192.168.7.246/detecion_semaforos')

# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:password@localhost/prueba" 
# engine = create_engine(SQLALCHEMY_DATABASE_URL)


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Define el modelo base
Base = declarative_base()

# Crea todas las tablas definidas en los modelos en la base de datos
Base.metadata.create_all(engine)

# # Crea una sesión de SQLAlchemy
Session = sessionmaker(bind=engine)
session = Session()
