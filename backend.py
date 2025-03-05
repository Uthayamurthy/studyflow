import random
import string
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import List
from sqlmodel import Field, Session, SQLModel, create_engine, select

class Session_Info(SQLModel, table=True):
    id: str = Field(primary_key=True)
    description: str | None = Field(default="New Session")

sqlite_file_name = "app_data/sessions.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db()
    yield

def gen_session_key():
    chars = string.ascii_letters + string.digits
    session_id = ''.join(random.choices(chars, k=8))
    return session_id

def create_db():
    try:
        SQLModel.metadata.create_all(engine)
    except:
        import os
        os.mkdir("app_data")
        SQLModel.metadata.create_all(engine)
app = FastAPI(lifespan=lifespan)

@app.get("/session_ids/", response_model=List[Session_Info])
def get_session_ids():
    with Session(engine) as sql_session:
        app_sessions = sql_session.exec(select(Session_Info)).all()
        return app_sessions

@app.get("/new_session_id/", response_model=Session_Info)
def create_new_session():
    new_id = gen_session_key()
    new_session = Session_Info(id=new_id)
    with Session(engine) as sql_session:
        sql_session.add(new_session)
        sql_session.commit()
    return {"id": new_id}

@app.delete("/session_id/{session_id}")
def delete_session(session_id: str):
    try:
        with Session(engine) as sql_session:
            statement = select(Session_Info).where(Session_Info.id == session_id)
            results = sql_session.exec(statement)
            session = results.one()
            sql_session.delete(session)
            sql_session.commit()
        return {"message": "Session deleted"}
    except:
        return {"message": "Session not found"}
    