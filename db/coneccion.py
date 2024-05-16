from sqlalchemy.orm import declarative_base
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine, Column, Boolean, Integer, String, Float, DateTime, func
from sqlalchemy import desc, asc
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base

engine = create_engine('postgresql://postgres:123456@localhost/placas_registradas')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)
session = Session()

# Connection details for the second database
SECOND_DB_ENGINE_TYPE = 'postgresql'
SECOND_DB_USERNAME = 'postgres'
SECOND_DB_PASSWORD = '123456'
SECOND_DB_HOST = 'localhost'
SECOND_DB_NAME = 'placas_registradas'

# Create the second engine
# second_engine = create_engine(
#     f"{SECOND_DB_ENGINE_TYPE}://{SECOND_DB_USERNAME}:{SECOND_DB_PASSWORD}@{SECOND_DB_HOST}/{SECOND_DB_NAME}"
# )
second_engine = create_engine('postgresql://postgres:123456@localhost/apimvc')
print(second_engine)
print(engine,'  aaaaaca')
a=input('hasta aqui')
# Create a session maker for the second database
SecondSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=second_engine)

# Example: Interact with the second database
second_session = SecondSessionLocal()
# Use second_session for queries, inserts, updates, etc.
second_session.close()
