import os, uuid, json
from datetime import datetime
import numpy as np, librosa, joblib
from src.data.feature_extractor import AudioFeatureExtractor
# from src.data.preprocessor import AudioPreprocessor 
from app.core.logger import logger

def log_audio_info(y, sr, file_path):
    """오디오 파일의 기본 정보를 로깅"""
    duration = len(y) / sr
    max_amplitude = np.max(np.abs(y))
    rms_energy = np.sqrt(np.mean(y**2))
    
    logger.info(f"📁 파일 정보: {os.path.basename(file_path)}")
    logger.info(f"   - 샘플링 레이트: {sr} Hz")
    logger.info(f"   - 길이: {duration:.2f}초 ({len(y)} 샘플)")
    logger.info(f"   - 최대 진폭: {max_amplitude:.6f}")
    logger.info(f"   - RMS 에너지: {rms_energy:.6f}")
    logger.info(f"   - 다이나믹 레인지: {20 * np.log10(max_amplitude / (rms_energy + 1e-8)):.2f} dB")

def log_feature_details(feature_name, feature_data, feature_mean):
    """개별 특성의 상세 정보를 로깅"""
    logger.info(f"🔍 {feature_name} 분석:")
    logger.info(f"   - 원본 shape: {feature_data.shape}")
    logger.info(f"   - 평균값 shape: {feature_mean.shape}")
    logger.info(f"   - 값 범위: [{np.min(feature_data):.6f}, {np.max(feature_data):.6f}]")
    logger.info(f"   - 평균값 범위: [{np.min(feature_mean):.6f}, {np.max(feature_mean):.6f}]")
    logger.info(f"   - 표준편차: {np.std(feature_data):.6f}")
    
    # 각 계수별 상세 정보 (MFCC, Chroma만)
    if feature_name in ["MFCC", "Chroma"]:
        logger.info(f"   - 계수별 평균값:")
        for i, val in enumerate(feature_mean):
            logger.info(f"     [{i+1:2d}] {val:10.6f}")

def log_feature_statistics(features, feature_names):
    """전체 특성의 통계 정보를 로깅"""
    logger.info(f"📊 전체 특성 통계:")
    logger.info(f"   - 총 특성 수: {len(features)}")
    logger.info(f"   - 값 범위: [{np.min(features):.6f}, {np.max(features):.6f}]")
    logger.info(f"   - 평균: {np.mean(features):.6f}")
    logger.info(f"   - 표준편차: {np.std(features):.6f}")
    logger.info(f"   - 중간값: {np.median(features):.6f}")
    
    # 특성 그룹별 통계
    mfcc_features = features[:13]
    chroma_features = features[13:25]
    other_features = features[25:]
    
    logger.info(f"   - MFCC 그룹 (1-13): 평균={np.mean(mfcc_features):.6f}, 표준편차={np.std(mfcc_features):.6f}")
    logger.info(f"   - Chroma 그룹 (14-25): 평균={np.mean(chroma_features):.6f}, 표준편차={np.std(chroma_features):.6f}")
    logger.info(f"   - 기타 그룹 (26-{len(features)}): 평균={np.mean(other_features):.6f}, 표준편차={np.std(other_features):.6f}")

def save_features_to_json(features, feature_names, file_path):
    """특성을 JSON 파일로 저장"""
    try:
        feature_data = {
            "timestamp": datetime.now().isoformat(),
            "source_file": os.path.basename(file_path),
            "feature_count": len(features),
            "features": {
                name: float(value) for name, value in zip(feature_names, features)
            },
            "statistics": {
                "min": float(np.min(features)),
                "max": float(np.max(features)),
                "mean": float(np.mean(features)),
                "std": float(np.std(features)),
                "median": float(np.median(features))
            }
        }
        
        json_filename = f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(feature_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 특성 데이터 저장됨: {json_filename}")
        
    except Exception as e:
        logger.warning(f"특성 JSON 저장 실패: {e}")

def extract_features(file_path, sr=22050):
    """
    51개 특성 추출 (AudioFeatureExtractor 기반)
    """
    try:
        logger.info(f"🎵 특성 추출 시작: {os.path.basename(file_path)}")
        # 오디오 파일 로드
        y, sr = librosa.load(file_path, sr=sr)
        log_audio_info(y, sr, file_path)

        # 51개 특성 추출
        extractor = AudioFeatureExtractor()
        features = extractor.extract_all_features(y, sr)
        feature_names = extractor.get_feature_names()

        # 전체 특성 통계
        log_feature_statistics(features, feature_names)

        # 특성 상세 로깅
        logger.info(f"📋 === 추출된 {len(features)}개 특성 상세 ===")
        for i, (name, value) in enumerate(zip(feature_names, features)):
            status = "⚠️ " if np.isnan(value) or np.isinf(value) else "✅ "
            logger.info(f"  {status}[{i+1:2d}] {name:20s}: {value:12.6f}")

        # NaN/Inf 확인
        nan_count = np.sum(np.isnan(features))
        inf_count = np.sum(np.isinf(features))
        if nan_count > 0 or inf_count > 0:
            logger.error(f"❌ 특성 품질 이슈 발견!")
            logger.error(f"   - NaN 개수: {nan_count}개")
            logger.error(f"   - Inf 개수: {inf_count}개")
            for i, (name, value) in enumerate(zip(feature_names, features)):
                if np.isnan(value) or np.isinf(value):
                    logger.error(f"   - 문제 특성: [{i+1}] {name} = {value}")
        else:
            logger.info("✅ 모든 특성이 정상입니다")

        # 특성 데이터를 JSON으로 저장
        save_features_to_json(features, feature_names, file_path)

        logger.info(f"🎯 특성 추출 완료: {os.path.basename(file_path)}")
        logger.info("=" * 80)

        return features.reshape(1, -1)

    except Exception as e:
        logger.error(f"❌ 특성 추출 실패: {file_path}")
        logger.error(f"   - 오류: {e}")
        logger.error("=" * 80)
        raise

async def save_temp_file(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.filename)[1]
    temp_path = f"tmp_{uuid.uuid4()}{ext}"
    with open(temp_path, "wb") as f:
        f.write(await uploaded_file.read())
    return temp_path
