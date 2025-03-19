from sentence_transformers import SentenceTransformer
import time
import os

def load_model(model_name, use_cache=True):
    """
    Args:
        model_name (str): 로드할 모델 이름 (Hugging Face 모델 ID)
        use_cache (bool): 캐시 사용 여부
    Returns:
        SentenceTransformer: 로드된 모델 객체
    """
    print(f"모델 '{model_name}' 로딩 중...")
    start_time = time.time()
    
    # 캐시 사용 설정
    if not use_cache:
        os.environ["HF_HUB_OFFLINE"] = "1"
    
    # 모델 로드
    model = SentenceTransformer(model_name)
    
    print(f"모델 로딩 완료 ({time.time() - start_time:.2f}초)")
    return model

def get_available_models():
    """
    Returns:
        dict: 모델 이름과 설명을 포함하는 딕셔너리
    """
    return {
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1": "XLM-R 기반 multilingual paraphrase 모델",
        "sentence-transformers/stsb-xlm-r-multilingual": "XLM-R 기반 STS 학습 multilingual 모델",
        "sentence-transformers/xlm-r-large-en-ko-nli-ststb": "한국어-영어 특화 XLM-R 모델",
        "sentence-transformers/distiluse-base-multilingual-cased-v1": "경량화된 multilingual 모델"
    }
