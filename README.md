# Study Flow
## An AI Powered Study Assistant

### Installation Instruction for backend: 
- Create an virtual environment
```bash
python3 -m venv env
source env/bin/activate
```
- Install the requirements
```bash
pip install -r requirements.txt
```
> Note: Please create a "api_key.py" file and save your key as a string " gemini_key='<your_key>' ". If you don't have an api key, please get it from https://aistudio.google.com

### Usage Instructions for backend:
Just run:
```bash
fastapi dev backend.py
```
Now you can access the server and api endpoints from:
```
127.0.0.1:8000
```
To get the docs for the api go to docs url:
```
http://127.0.0.1:8000/docs
```

### Usage Instructions for frontend:
Go here: https://github.com/vicfic18/studyflow