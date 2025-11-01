from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from src.database import get_db
from src.auth.schemas import CreateUser, UserResponse
from src.auth.service import create_new_user

router = APIRouter()

@router.post("/signup", response_model=UserResponse)
def create_user(user: CreateUser, db: Session = Depends(get_db)):
    return create_new_user(db, user)
