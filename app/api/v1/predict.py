from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from app.core.config import settings
from app.core.logger import logger
from app.services.feature import extract_features, save_temp_file

import joblib, os, uuid, numpy as np

router = APIRouter()

_model_bundle = joblib.load(settings.MODEL_PATH)
model = _model_bundle.get("model", _model_bundle)

@router.get("/health")
def health_check():
    return {"status": "healthy", "message": "수박 당도 예측 서버 실행 중"}

@router.post("/predict")
async def predict(file: UploadFile = File(...)):
    temp_path = None
    try:
        # 파일 확장자 검사
        ext = os.path.splitext(file.filename)[-1].lower()
        if ext not in [".wav", ".m4a", ".mp3"]:
            return JSONResponse(
                status_code=400, 
                content={
                    "success": False,
                    "error": "지원하지 않는 파일 형식입니다. .wav, .m4a, .mp3 파일만 업로드 가능합니다."
                }
            )

        # 임시 파일 저장
        temp_path = f"temp_{uuid.uuid4()}{ext}"
        with open(temp_path, "wb") as f:
            f.write(await file.read())
        
        logger.info(f"파일 업로드 완료: {file.filename} -> {temp_path}")

        # 특성 추출 및 예측
        features = extract_features(temp_path)
        
        # 예측 (스케일러 없음)
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0] if hasattr(model, 'predict_proba') else None
        
        logger.info(f"🎯 예측 결과: {prediction} ({'높음' if prediction == 1 else '낮음'})")
        if probability is not None:
            logger.info(f"   - 확률: 낮음={probability[0]:.3f}, 높음={probability[1]:.3f}")
        
        # 결과 반환
        result = {
            "success": True,
            "filename": file.filename,
            "prediction": int(prediction),
            "result": "높음" if prediction == 1 else "낮음",
            "confidence": float(max(probability)) if probability is not None else None
        }
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"예측 중 오류 발생: {e}")
        return JSONResponse(
            status_code=500, 
            content={
                "success": False,
                "error": f"예측 중 오류가 발생했습니다: {str(e)}"
            }
        )
    finally:
        # 임시 파일 삭제
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"임시 파일 삭제: {temp_path}")
            except Exception as e:
                logger.warning(f"임시 파일 삭제 실패: {e}")