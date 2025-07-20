# tongtongtong-api

FastAPI로 구현된 오디오 기반 수박 숙성도 예측 서버입니다. 사용자의 오디오 파일을 업로드하면 특징을 추출하고, 학습된 모델을 사용해 숙성도(낮은 소리/높은 소리)를 예측합니다.

📂 폴더 구조
```
├── app/
│   ├── core/          # 설정 및 로거
│   │   ├── config.py  # 환경변수(BaseSettings) 설정
│   │   └── logger.py  # 로깅 설정
│   ├── services/      # 비즈니스 로직 (특성 추출, 임시 파일 저장)
│   │   └── feature.py
│   ├── api/           # HTTP 라우터
│   │   └── v1/
│   │       └── predict.py
│   └── main.py        # FastAPI 앱 초기화 및 라우터 조립
├── models/            # ML 모델(.pkl) 파일
│   └── best_model.pkl
├── src/               # ML 모델 코드
│   └── data/
│       ├── feature_extractor.py
│       └── preprocessor.py
├── .env.example       # 환경변수 예시
├── requirements.txt   # Python 패키지 목록
└── README.md          # 프로젝트 설명서
```

🚀 빠른 시작

1. 가상환경
```
python3 -m venv .venv
source .venv/bin/activate
```

2. 의존성 설치
```
pip install --upgrade pip
pip install -r requirements.txt
```

3. 환경변수 설정
```
cp .env.example .env
```

4. 서버 실행
```
uvicorn app.main:app --host $FASTAPI_HOST --port $FASTAPI_PORT --reload
```
