from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
from modules.gec import GrammarErrorCorrector

load_dotenv()

app = FastAPI()


@app.get("/health")
async def health_check():
    try:
        return {"status": "healthy"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/correct_grammar")
async def correct_grammar(text: str, model: str = "gpt-5-mini"):
    try:
        gec = GrammarErrorCorrector(model)
        response = gec.correct(text)
        return {
            "status": "success",
            "data": response["model_response"],
            "total_time": response["total_time"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
