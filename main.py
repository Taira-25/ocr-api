from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from google.cloud import vision
import os
import tempfile

# サービスアカウントキーのパス
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/etc/secrets/gcp-key.json'

app = FastAPI()
client = vision.ImageAnnotatorClient()

@app.post("/ocr_file")
async def ocr_from_file(file: UploadFile = File(...)):
    try:
        # 一時ファイルとして画像を保存
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(contents)
            tmp_file_path = tmp_file.name

        # 画像をGoogle Vision APIで読み取る
        with open(tmp_file_path, "rb") as image_file:
            image = vision.Image(content=image_file.read())

        result = client.document_text_detection(image=image)

        return {"filename": file.filename, "extracted_text": result.full_text_annotation.text.strip()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
