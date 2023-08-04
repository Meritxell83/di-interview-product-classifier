The scripts are implemented using Python 3.10

The API is implemented using FastAPI and uvicorn (added to requirements.txt)

To executed run the command: uvicorn main:app --reload --port 5000

In a separate window execute: 
curl -XPOST 'http://127.0.0.1:5000/classify' 
-H 'Content-Type: application/json' 
-d '{ "title": "my product title containing aquarelle" }'

