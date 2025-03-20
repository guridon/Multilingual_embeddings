from code_switching_analyzer import CodeSwitchingAnalyzer
import numpy as np
import json
import os
import time

def main():
    start_time = time.time()
    # 출력 디렉토리 설정
    output_dir = "output_analysis_category"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 저장된 임베딩 로드
    print("저장된 임베딩 로드 중...")
    embeddings = np.load('output/multilingual_embeddings.npy')
    print(f"임베딩 로드 완료: {embeddings.shape}")
    
    # 2. 데이터셋 로드 (원본 텍스트 데이터)
    print("데이터셋 로드 중...")
    with open('code-switch.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 텍스트 추출
    texts = []
    for item in data:
        texts.extend([item["EtoK"], item["KtoE"], item["English"], item["Korean"]])
    
    # 텍스트 유형과 항목 인덱스 생성
    text_types = ["EtoK", "KtoE", "English", "Korean"] * len(data)
    item_indices = []
    for i in range(len(data)):
        item_indices.extend([i] * 4)  # 각 항목당 4개 문장
    
    print(f"데이터셋 로드 완료: {len(texts)}개 문장, {len(data)}개 항목")
    
    # 3. 임베딩 메타데이터 로드 (선택사항)
    model_name = "저장된 임베딩"
    try:
        with open('output/embeddings_info.json', 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        model_name = metadata.get('model', '저장된 임베딩')
        print(f"메타데이터 로드 완료: 모델 = {model_name}")
    except (FileNotFoundError, json.JSONDecodeError):
        print("메타데이터 파일을 찾을 수 없거나 로드할 수 없습니다. 기본값 사용.")
    
    # 4. analyzer 생성 및 실행
    print("코드 스위칭 분석기 초기화 및 실행...")
    with open('./category/category_map.json', 'r', encoding='utf-8') as f:
        category_map = json.load(f)


    analyzer = CodeSwitchingAnalyzer(
        embeddings=embeddings,
        texts=texts,
        text_types=text_types,
        item_indices=item_indices,
        model_name=model_name,
        item_categories=category_map,
        output_dir=output_dir
    )
    
    # 모든 분석 실행
    analyzer.run_all_analyses()
    
    # 실행 완료 메시지
    total_time = time.time() - start_time
    print("\n" + "="*50)
    print("코드 스위칭 분석 완료!")
    print("="*50)
    print(f"총 소요 시간: {total_time:.2f}초")
    print(f"분석 결과는 {output_dir} 디렉토리에 저장되었습니다.")
    print("\n생성된 파일:")
    print(f"- {output_dir}/cs_vs_baseline_pairs.html (코드 스위칭-베이스라인 쌍 시각화)")
    print(f"- {output_dir}/cs_vs_baseline_comparison.html (유형별 평균 유사도)")
    print(f"- {output_dir}/cs_vs_baseline_distribution.html (항목별 유사도 분포)")
    print(f"- {output_dir}/cs_extreme_examples.html (극단 사례 분석)")
    print(f"- {output_dir}/cs_analysis_report.md (분석 보고서)")
    print("="*50)

if __name__ == "__main__":
    main()
