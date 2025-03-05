import uuid
import json
import os
from fastapi import FastAPI
from contextlib import asynccontextmanager
from typing import List
from litellm import completion
from prompts import title_gen_prompt
from sqlmodel import Field, Session, SQLModel, create_engine, select
try:
    from api_key import gemini_key
except:
    print("API Key not found. Please create a \"api_key.py\" file and say your key as \"gemini_key='<your_key>'\" ")
    exit(0)

class Session_Info(SQLModel, table=True):
    id: str = Field(primary_key=True)
    description: str | None = Field(default="New Session")
    chat_history : str | None = Field(default="[]")
    
sqlite_file_name = "app_data/sessions.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_db()
    os.environ["GEMINI_API_KEY"] = gemini_key
    yield

def gen_session_key():
    session_id = str(uuid.uuid4())
    return session_id

def create_db():
    try:
        SQLModel.metadata.create_all(engine)
    except:
        os.mkdir("app_data")
        SQLModel.metadata.create_all(engine)
app = FastAPI(lifespan=lifespan)

def gen_description(user_query: str, session_id: str):
    message = title_gen_prompt.ingest_args(user_prompt=user_query)
    response = completion(
        model="gemini/gemini-2.0-flash-lite", 
        messages=[{"role": "user", "content": message}]
    )
    title = response['choices'][0]['message']['content']
    with Session(engine) as sql_session:
        statement = select(Session_Info).where(Session_Info.id == session_id)
        result = sql_session.exec(statement).one()
        result.description = title
        sql_session.add(result)
        sql_session.commit()
        sql_session.refresh(result)

@app.get("/sessions_info/", response_model=List[Session_Info])
def get_session_ids():
    with Session(engine) as sql_session:
        app_sessions = sql_session.exec(select(Session_Info)).all()
        return app_sessions

@app.get("/new_session/", response_model=Session_Info)
def create_new_session():
    new_id = gen_session_key()
    new_session = Session_Info(id=new_id)
    with Session(engine) as sql_session:
        sql_session.add(new_session)
        sql_session.commit()
    return {"id": new_id}

@app.delete("/session/{session_id}")
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

@app.get("/chat_data/{session_id}")
def get_chat_data(session_id: str):
    try:
        with Session(engine) as sql_session:
            statement = select(Session_Info).where(Session_Info.id == session_id)
            result = sql_session.exec(statement).one()
            chat_history = json.loads(result.chat_history)
        return {'chat_history': chat_history}
    except json.JSONDecodeError:
        return {"chat_history": None}
    except:
        return {"chat_history": "Session not found"}

@app.post("/chat_completion/{session_id}")
def process_chat(session_id: str, user_query: str):
    with Session(engine) as sql_session:
        statement = select(Session_Info).where(Session_Info.id == session_id)
        result = sql_session.exec(statement).one()
        chat_history = json.loads(result.chat_history)
        if len(chat_history) == 0:
            gen_description(user_query, session_id)
        user_chat = {"role": "user", "content": user_query}
        chat_history.append(user_chat)
        response = completion(
               model="gemini/gemini-2.0-flash", 
                messages=chat_history
            )
        assistant_chat = {"role": "assistant", "content": response['choices'][0]['message']['content']}
        chat_history.append(assistant_chat)
        print(f'Chat History: {chat_history}')
        result.chat_history = json.dumps(chat_history)
        sql_session.add(result)
        sql_session.commit()
        sql_session.refresh(result)

    return {'assistant_message': assistant_chat["content"]}