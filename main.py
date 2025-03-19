import time
import argparse
from model_loader import load_model
from dataset_loader import load_code_switch_dataset
from embedding_generator import generate_embeddings, calculate_similarities
from visualizer import (reduce_dimensions, plot_embeddings_matplotlib, 
                        plot_embeddings_plotly, plot_connections_by_item,
                        plot_connections_by_similarity)
from utils import setup_output_directory, save_embeddings, print_summary

def parse_args():
    parser = argparse.ArgumentParser(description='multilingual embeddings analysis')
    parser.add_argument('--model', type=str, 
                        default='sentence-transformers/paraphrase-xlm-r-multilingual-v1',
                        help='사용할 모델 이름')
    parser.add_argument('--dataset', type=str, 
                        default='code-switch.json',
                        help='데이터셋 파일 경로')
    parser.add_argument('--output', type=str, 
                        default='output',
                        help='출력 디렉토리')
    parser.add_argument('--threshold', type=float, 
                        default=0.7,
                        help='유사도 임계값')
    parser.add_argument('--normalize', action='store_true',
                        help='임베딩 정규화 적용')
    return parser.parse_args()

def main():
    start_time = time.time()
    args = parse_args()
    output_dir = setup_output_directory(args.output)
    
    # 모델 로드
    model = load_model(args.model)
    # 데이터셋 로드
    texts, text_types, item_indices, data = load_code_switch_dataset(args.dataset)
    # 임베딩 생성
    embeddings = generate_embeddings(model, texts, normalize=args.normalize)
    # 유사도 계산
    similarities = calculate_similarities(embeddings)
    # 차원 축소
    embeddings_2d = reduce_dimensions(embeddings)
    
    # Matplotlib 시각화
    plot_embeddings_matplotlib(embeddings_2d, texts, text_types, output_dir)
    # Plotly 인터랙티브 시각화
    plot_embeddings_plotly(embeddings_2d, texts, text_types, item_indices, args.model, output_dir)
    # 항목별 연결 시각화
    plot_connections_by_item(embeddings_2d, texts, text_types, item_indices, similarities, args.model, output_dir)
    # 유사도 기반 연결 시각화
    plot_connections_by_similarity(embeddings_2d, texts, text_types, item_indices, 
                                   similarities, args.threshold, args.model, output_dir)
    # 임베딩 및 메타데이터 저장
    save_embeddings(embeddings, similarities, texts, text_types, item_indices, args.model, output_dir)
    print_summary(args.model, embeddings, start_time, output_dir)

if __name__ == "__main__":
    main()
