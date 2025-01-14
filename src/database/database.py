from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import logging

logger = logging.getLogger("llama_index_api")

class Database:
    def __init__(self, db_url: str = "sqlite:///./messages.db"):
        self.engine = create_engine(db_url)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.Base = declarative_base()
        
    def init_db(self):
        from .models import Base
        Base.metadata.create_all(bind=self.engine)
        logger.info("Database initialized")

    def get_session(self):
        session = self.SessionLocal()
        try:
            yield session
        finally:
            session.close() 