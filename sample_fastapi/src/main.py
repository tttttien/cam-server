from fastapi import FastAPI
from src.database import engine
from src.auth.models import Base
from src.auth import routers

app = FastAPI()

# Include auth router
app.include_router(routers.router)

# Auto-create tables
@app.on_event("startup")
def startup_event():
    Base.metadata.create_all(bind=engine)

@app.get("/", tags=["root"])
def read_root():
    return {"message": "Welcome to my app"}
