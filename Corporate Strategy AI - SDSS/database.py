import os
import datetime

from dotenv import load_dotenv

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime
)

from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker

from cryptography.fernet import Fernet


# Load .env
load_dotenv()

ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")

cipher_suite = Fernet(ENCRYPTION_KEY.encode())

Base = declarative_base()

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


class DecisionHistory(Base):

    __tablename__ = "decision_history"

    id = Column(Integer, primary_key=True, index=True)

    timestamp = Column(
        DateTime,
        default=datetime.datetime.utcnow
    )

    strategy_name = Column(String)

    final_score = Column(Float)

    consistency_ratio = Column(Float)

    encrypted_metadata = Column(String)


def encrypt_data(data):

    return cipher_suite.encrypt(
        data.encode()
    ).decode()


def save_decision(
    strategy,
    score,
    cr,
    metadata
):

    db = SessionLocal()

    encrypted_metadata = encrypt_data(metadata)

    record = DecisionHistory(
        strategy_name=strategy,
        final_score=score,
        consistency_ratio=cr,
        encrypted_metadata=encrypted_metadata
    )

    db.add(record)

    db.commit()

    db.close()


Base.metadata.create_all(bind=engine)