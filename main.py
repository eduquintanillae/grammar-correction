from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from modules.gec import GrammarErrorCorrector
import uvicorn
from fastapi import FastAPI, HTTPException, Request

load_dotenv()

app = FastAPI()


@app.get("/health")
async def health_check():
    try:
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/correct_grammar")
async def correct_grammar(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        model = data.get("model", "gpt-4.1-mini")
        gec = GrammarErrorCorrector(model)
        response = gec.correct(text)
        return {
            "status": "success",
            "data": response["model_response"],
            "total_time": response["total_time"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
