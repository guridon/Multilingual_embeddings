import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import os
import time

def reduce_dimensions(embeddings, n_components=2):
    """ 
    Args:
        embeddings (numpy.ndarray): 임베딩 배열
        n_components (int): 축소할 차원 수
        
    Returns:
        numpy.ndarray: 축소된 임베딩 배열
    """
    print(f"PCA 차원 축소 중 (차원: {embeddings.shape[1]} -> {n_components})...")
    start_time = time.time()
    
    pca = PCA(n_components=n_components)
    reduced_embeddings = pca.fit_transform(embeddings)
    
    print(f"PCA 차원 축소 완료 ({time.time() - start_time:.2f}초)")
    return reduced_embeddings

def plot_embeddings_matplotlib(reduced_embeddings, texts, text_types, output_dir="output"):
    """
    Args:
        reduced_embeddings (numpy.ndarray): 축소된 임베딩 배열
        texts (list): 텍스트 리스트
        text_types (list): 텍스트 유형 리스트
        output_dir (str): 출력 디렉토리
    """
    print("Matplotlib 시각화 생성 중...")
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14, 10))
    type_colors = {
        "EtoK": "red",
        "KtoE": "blue",
        "English": "green",
        "Korean": "purple"
    }
    
    # 문장 유형별 시각화
    for text_type in set(text_types):
        indices = [i for i, t in enumerate(text_types) if t == text_type]
        plt.scatter(
            reduced_embeddings[indices, 0], 
            reduced_embeddings[indices, 1], 
            c=type_colors.get(text_type, "gray"),
            label=text_type,
            alpha=0.7,
            s=100
        )
    
    # idx 추가
    for i, txt in enumerate(texts):
        short_text = txt[:20] + "..." if len(txt) > 20 else txt
        plt.annotate(
            short_text, 
            (reduced_embeddings[i, 0], reduced_embeddings[i, 1]),
            fontsize=8,
            alpha=0.8
        )
    
    plt.title('다국어 임베딩 PCA 시각화')
    plt.legend()
    plt.tight_layout()
    
    # save
    output_file = os.path.join(output_dir, "embeddings_pca.png")
    plt.savefig(output_file, dpi=300)
    
    print(f"Matplotlib 시각화 저장 완료: {output_file} ({time.time() - start_time:.2f}초)")

def plot_embeddings_plotly(reduced_embeddings, texts, text_types, item_indices, model_name, output_dir="output"):
    """
    Args:
        reduced_embeddings (numpy.ndarray): 축소된 임베딩 배열
        texts (list): 텍스트 리스트
        text_types (list): 텍스트 유형 리스트
        item_indices (list): 항목 인덱스 리스트
        model_name (str): 모델 이름
        output_dir (str): 출력 디렉토리
    """
    print("Plotly 인터랙티브 시각화 생성 중...")
    start_time = time.time()
    os.makedirs(output_dir, exist_ok=True)
    
    type_colors = {
        "EtoK": "red",
        "KtoE": "blue",
        "English": "green",
        "Korean": "purple"
    }
    
    fig = go.Figure()
    
    for text_type in set(text_types):
        indices = [i for i, t in enumerate(text_types) if t == text_type]
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[indices, 0],
            y=reduced_embeddings[indices, 1],
            mode='markers+text',
            name=text_type,
            text=[texts[i][:30] + "..." if len(texts[i]) > 30 else texts[i] for i in indices],
            textposition='top center',
            marker=dict(
                size=10, 
                color=type_colors.get(text_type, "gray"),
                opacity=0.7
            ),
            hoverinfo='text',
            hovertext=[f"Type: {text_type}<br>Item: {item_indices[i]}<br>Text: {texts[i]}" for i in indices]
        ))
    
    fig.update_layout(
        title=f'다국어 임베딩 시각화 ({model_name})',
        width=1000,
        height=800,
        legend=dict(
            title="문장 유형",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    output_file = os.path.join(output_dir, "embeddings_interactive.html")
    fig.write_html(output_file)
    
    print(f"Plotly 인터랙티브 시각화 저장 완료: {output_file} ({time.time() - start_time:.2f}초)")
    
    return fig

def plot_connections_by_item(reduced_embeddings, texts, text_types, item_indices, 
                            similarities, model_name, output_dir="output"):
    """
    Args:
        reduced_embeddings (numpy.ndarray): 축소된 임베딩 배열
        texts (list): 텍스트 리스트
        text_types (list): 텍스트 유형 리스트
        item_indices (list): 항목 인덱스 리스트
        similarities (numpy.ndarray): 유사도 행렬
        model_name (str): 모델 이름
        output_dir (str): 출력 디렉토리
    """
    print("항목 간 연결 시각화 생성 중...")
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    type_colors = {
        "EtoK": "red",
        "KtoE": "blue",
        "English": "green",
        "Korean": "purple"
    }
    
    fig = go.Figure()
    for text_type in set(text_types):
        indices = [i for i, t in enumerate(text_types) if t == text_type]
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[indices, 0],
            y=reduced_embeddings[indices, 1],
            mode='markers',
            name=text_type,
            marker=dict(
                size=10, 
                color=type_colors.get(text_type, "gray"),
                opacity=0.7
            ),
            hoverinfo='text',
            hovertext=[f"Type: {text_type}<br>Item: {item_indices[i]}<br>Text: {texts[i]}" for i in indices]
        ))
    
    # 고유 항목 인덱스 추출
    unique_item_indices = set(item_indices)
    
    # 동일 항목 내 문장 간 연결선 추가
    for item_idx in unique_item_indices:
        # 해당 항목의 모든 문장 인덱스 찾기
        item_text_indices = [j for j, idx in enumerate(item_indices) if idx == item_idx]
        
        # 해당 항목 내 모든 문장 쌍에 대해 연결선 추가
        for j in range(len(item_text_indices)):
            for k in range(j+1, len(item_text_indices)):
                idx1 = item_text_indices[j]
                idx2 = item_text_indices[k]
                sim = similarities[idx1][idx2]
                
                # 유사도에 따라..
                line_width = max(1, sim * 5)
                opacity = min(1.0, max(0.2, sim))
                
                fig.add_trace(go.Scatter(
                    x=[reduced_embeddings[idx1, 0], reduced_embeddings[idx2, 0]],
                    y=[reduced_embeddings[idx1, 1], reduced_embeddings[idx2, 1]],
                    mode='lines',
                    line=dict(
                        width=line_width, 
                        color=f'rgba(0,0,0,{opacity})'
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"Similarity: {sim:.4f}<br>From: {text_types[idx1]}<br>To: {text_types[idx2]}"
                ))
    fig.update_layout(
        title=f'동일 항목 내 문장 간 연결 시각화 ({model_name})',
        width=1000,
        height=800,
        legend=dict(
            title="문장 유형",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    output_file = os.path.join(output_dir, "item_connections.html")
    fig.write_html(output_file)
    
    print(f"항목 연결 시각화 저장 완료: {output_file} ({time.time() - start_time:.2f}초)")
    
    return fig

def plot_connections_by_similarity(reduced_embeddings, texts, text_types, item_indices,
                                  similarities, threshold=0.7, model_name="", output_dir="output"):
    """
    Args:
        reduced_embeddings (numpy.ndarray): 축소된 임베딩 배열
        texts (list): 텍스트 리스트
        text_types (list): 텍스트 유형 리스트
        item_indices (list): 항목 인덱스 리스트
        similarities (numpy.ndarray): 유사도 행렬
        threshold (float): 유사도 임계값
        model_name (str): 모델 이름
        output_dir (str): 출력 디렉토리
    """
    print(f"유사도 기반 연결 시각화 생성 중 (임계값: {threshold})...")
    start_time = time.time()
    
    os.makedirs(output_dir, exist_ok=True)
    type_colors = {
        "EtoK": "red",
        "KtoE": "blue",
        "English": "green",
        "Korean": "purple"
    }

    fig = go.Figure()
    
    # 모든 문장 점 추가
    for text_type in set(text_types):
        indices = [i for i, t in enumerate(text_types) if t == text_type]
        fig.add_trace(go.Scatter(
            x=reduced_embeddings[indices, 0],
            y=reduced_embeddings[indices, 1],
            mode='markers+text',
            name=text_type,
            text=[texts[i][:20] + "..." if len(texts[i]) > 20 else texts[i] for i in indices],
            textposition='top center',
            marker=dict(
                size=10, 
                color=type_colors.get(text_type, "gray"),
                opacity=0.7
            ),
            hoverinfo='text',
            hovertext=[f"Type: {text_type}<br>Item: {item_indices[i]}<br>Text: {texts[i]}" for i in indices]
        ))
    
    # 임계값 이상의 유사도를 가진 문장 쌍 연결
    connection_count = 0
    for i in range(len(texts)):
        for j in range(i+1, len(texts)):
            if similarities[i][j] > threshold:
                connection_count += 1
                
                # 동일 항목인 경우와 다른 항목인 경우 색상 구분
                color = 'rgba(0,0,255,0.3)' if item_indices[i] == item_indices[j] else 'rgba(255,0,0,0.3)'
                
                fig.add_trace(go.Scatter(
                    x=[reduced_embeddings[i, 0], reduced_embeddings[j, 0]],
                    y=[reduced_embeddings[i, 1], reduced_embeddings[j, 1]],
                    mode='lines',
                    line=dict(
                        width=similarities[i][j] * 3, 
                        color=color
                    ),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"Similarity: {similarities[i][j]:.4f}<br>{text_types[i]} - {text_types[j]}<br>Same item: {item_indices[i] == item_indices[j]}"
                ))
    
    fig.update_layout(
        title=f'유사 텍스트 연결 시각화 (임계값: {threshold}, 연결: {connection_count}개)',
        width=1000,
        height=800,
        legend=dict(
            title="문장 유형",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )
    
    output_file = os.path.join(output_dir, "embeddings_connected.html")
    fig.write_html(output_file)
    
    print(f"유사도 기반 연결 시각화 저장 완료: {output_file} ({time.time() - start_time:.2f}초)")
    
    return fig
