import numpy as np
import time

def generate_embeddings(model, texts, normalize=True):
    """
    Args:
        model: SentenceTransformer 모델
        texts (list): 임베딩을 생성할 텍스트 리스트
        normalize (bool): 정규화 여부
        
    Returns:
        numpy.ndarray: 임베딩 배열
    """
    print("임베딩 생성 중...")
    start_time = time.time()
    
    embeddings = model.encode(texts, normalize_embeddings=normalize)
    
    print(f"임베딩 생성 완료: {embeddings.shape} ({time.time() - start_time:.2f}초)")
    return embeddings

def calculate_similarities(embeddings):
    """
    Args:
        embeddings (numpy.ndarray): 임베딩 배열
        
    Returns:
        numpy.ndarray: 유사도 행렬
    """
    print("유사도 계산 중...")
    start_time = time.time()
    
    n = embeddings.shape[0]
    similarities = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            similarities[i][j] = np.dot(embeddings[i], embeddings[j])
    
    print(f"유사도 계산 완료 ({time.time() - start_time:.2f}초)")
    return similarities
