import os
import numpy as np
import json
import time

def setup_output_directory(output_dir="output"):
    """
    Args:
        output_dir (str): 출력 디렉토리 경로
    
    Returns:
        str: 생성된 출력 디렉토리 경로
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"출력 디렉토리 준비 완료: {output_dir}")
    return output_dir

def save_embeddings(embeddings, similarities, texts, text_types, item_indices, 
                   model_name, output_dir="output"):
    """
    Args:
        embeddings (numpy.ndarray): 임베딩 배열
        similarities (numpy.ndarray): 유사도 행렬
        texts (list): 텍스트 리스트
        text_types (list): 텍스트 유형 리스트
        item_indices (list): 항목 인덱스 리스트
        model_name (str): 모델 이름
        output_dir (str): 출력 디렉토리
    """
    print("임베딩 및 메타데이터 저장 중...")
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(os.path.join(output_dir, "multilingual_embeddings.npy"), embeddings)
    np.save(os.path.join(output_dir, "similarities.npy"), similarities)
    
    metadata = {
        "model": model_name,
        "dimension": embeddings.shape[1],
        "sample_count": len(texts),
        "text_types": text_types,
        "item_indices": item_indices
    }
    
    with open(os.path.join(output_dir, "embeddings_info.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    
    print(f"임베딩 및 메타데이터 저장 완료 ({time.time() - start_time:.2f}초)")

def print_summary(model_name, embeddings, start_time, output_dir="output"):
    """
    Args:
        model_name (str): 모델 이름
        embeddings (numpy.ndarray): 임베딩 배열
        start_time (float): 시작 시간
        output_dir (str): 출력 디렉토리
    """
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("실행 결과 요약")
    print("="*50)
    print(f"모델: {model_name}")
    print(f"임베딩 크기: {embeddings.shape}")
    print(f"총 소요 시간: {total_time:.2f}초")
    print("\n생성된 파일:")
    print(f"- {output_dir}/embeddings_pca.png")
    print(f"- {output_dir}/embeddings_interactive.html")
    print(f"- {output_dir}/item_connections.html")
    print(f"- {output_dir}/embeddings_connected.html")
    print(f"- {output_dir}/multilingual_embeddings.npy")
    print(f"- {output_dir}/similarities.npy")
    print(f"- {output_dir}/embeddings_info.json")
    print("="*50)
