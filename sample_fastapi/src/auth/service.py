from fastapi import HTTPException
from sqlalchemy.orm import Session
import uuid
from src.auth.models import User
from src.auth.schemas import CreateUser, UserResponse

def get_user_by_email(db: Session, email: str):
    return db.query(User).filter(User.email == email).first()

def create_new_user(db: Session, user: CreateUser) -> UserResponse:
    if get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already exists")
    new_user = User(
        id=uuid.uuid4(),
        email=user.email,
        password=user.password  # NOTE: hash passwords in production
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return UserResponse(id=str(new_user.id), email=user.email)
