from fastapi import FastAPI

app = FastAPI()

@app.get("/{name}")
async def hello_fastAPI(name):
    return {"intro": f"Hello, {name[0].upper()}{name[1:]}!"}