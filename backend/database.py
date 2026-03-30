from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import os

# SQLite database file
# SQLite/PostgreSQL database file - uses environment variable for production flexibility
DB_URL = os.environ.get("DATABASE_URL", "sqlite:///./zurastock.db")

# Fix for SQLAlchemy 1.4+: PostgreSQL URLs must start with 'postgresql://'
if DB_URL.startswith("postgres://"):
    DB_URL = DB_URL.replace("postgres://", "postgresql://", 1)

# SQLite requires 'check_same_thread=False', PostgreSQL DOES NOT.
if "sqlite" in DB_URL:
    engine = create_engine(DB_URL, connect_args={"check_same_thread": False})
else:
    engine = create_engine(DB_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password_hash = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    watchlist = relationship("Watchlist", back_populates="owner")
    portfolio = relationship("Portfolio", back_populates="owner")

class Watchlist(Base):
    __tablename__ = "watchlists"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, index=True)
    added_at = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="watchlist")

class Portfolio(Base):
    __tablename__ = "portfolios"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String, index=True)
    quantity = Column(Integer)
    avg_price = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    
    owner = relationship("User", back_populates="portfolio")

def init_db():
    Base.metadata.create_all(bind=engine)
    # Create a default user for the demo if not exists
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == "demo_user").first()
        if not user:
            new_user = User(username="demo_user", password_hash="demo_pass") # In real app use hash
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            user = new_user

        # Seed Silver ETFs into watchlist for the user
        silver_mfts = ["HDFCSILVER", "SBISILVER", "SILVERBEES", "SILVERIETF"]
        for sym in silver_mfts:
            exists = db.query(Watchlist).filter(Watchlist.user_id == user.id, Watchlist.symbol == sym).first()
            if not exists:
                db.add(Watchlist(user_id=user.id, symbol=sym))
        db.commit()
    finally:
        db.close()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
