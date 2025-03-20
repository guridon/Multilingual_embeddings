import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
import itertools
import json


class CodeSwitchingAnalyzer:
    def __init__(self, embeddings, texts, text_types, item_indices, model_name, 
                 item_categories=None, output_dir="output"):
        """
        Args:
            embeddings (numpy.ndarray): 임베딩 배열
            texts (list): 텍스트 리스트
            text_types (list): 텍스트 유형 리스트 (EtoK, KtoE, English, Korean)
            item_indices (list): 항목 인덱스 리스트
            model_name (str): 모델 이름
            item_categories (dict): 항목별 카테고리 정보 (예: {0: "tech", 1: "tech", ...})
            output_dir (str): 출력 디렉토리
        """
        self.embeddings = embeddings
        self.texts = texts
        self.text_types = text_types
        self.item_indices = item_indices
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        # 카테고리 정보 로드
        self.item_categories = item_categories or {}
        self.category_colors = {
            "tech": "#2ecc71",       
            "academic": "#9b59b6",   
            "geopolitics": "#e74c3c", 
            "culture": "#3498db",    
            "regional": "#f39c12",  
            "language": "#1abc9c",   
            "food": "#e67e22"        
        }
        self.category_descriptions = {
            "tech": "기술/과학",
            "academic": "학술/철학",
            "geopolitics": "역사/지정학",
            "culture": "글로벌 문화",
            "regional": "지역 특수 문화",
            "language": "언어/번역",
            "food": "음식/요리"
        }
        
        self.reduced_embeddings = None
        # 분석할 모든 유형 쌍
        self.all_type_pairs = [
            # 코드 스위칭 vs 베이스라인
            ("EtoK", "English"),
            ("KtoE", "Korean"),
            # 코드 스위칭 간 비교
            ("EtoK", "KtoE"),
            # 베이스라인 간 비교
            ("English", "Korean"),
            # 크로스 비교
            ("EtoK", "Korean"),
            ("KtoE", "English")
        ]
        self.type_colors = {
            "EtoK": "red",
            "KtoE": "blue",
            "English": "green",
            "Korean": "purple"
        }
        self.type_descriptions = {
            "EtoK": "영→한 코드스위칭",
            "KtoE": "한→영 코드스위칭",
            "English": "영어",
            "Korean": "한국어"
        }
        
        print(f"Code switch analyzer 초기화 완료: {model_name}")
        print(f"분석할 표현 유형: {', '.join(set(text_types))}")
        if item_categories:
            print(f"카테고리 정보 로드 완료: {len(item_categories)}개 항목")
        
    def reduce_dimensions(self, n_components=2):
        print("임베딩 차원 축소 중...")
        pca = PCA(n_components=n_components)
        self.reduced_embeddings = pca.fit_transform(self.embeddings)
        print(f"차원 축소 완료: {self.embeddings.shape[1]} → {n_components}")
        return self.reduced_embeddings
    
    def calculate_all_similarity_metrics(self):
        print("모든 표현 유형 쌍 간의 유사도 계산 중...")
        
        # 결과 저장을 위한 데이터 구조
        results = []
        # 고유 항목 인덱스 목록
        unique_items = sorted(set(self.item_indices))
        for item_idx in unique_items:
            # 현재 항목의 인덱스 찾기
            item_positions = [i for i, idx in enumerate(self.item_indices) if idx == item_idx]
            
            # 현재 항목의 텍스트와 타입
            item_texts = [self.texts[i] for i in item_positions]
            item_types = [self.text_types[i] for i in item_positions]
            
            # 현재 항목의 임베딩
            item_embeddings = [self.embeddings[i] for i in item_positions]
            
            # 텍스트 유형별 임베딩 인덱스 찾기
            type_to_idx = {t: i for i, t in enumerate(item_types)}
            
            # 항목의 카테고리 정보 추가
            item_category = self.item_categories.get(str(item_idx), self.item_categories.get(item_idx, "other"))
            
            # 모든 가능한 유형 쌍에 대해 유사도 계산
            for type1, type2 in self.all_type_pairs:
                if type1 in type_to_idx and type2 in type_to_idx:
                    idx1 = type_to_idx[type1]
                    idx2 = type_to_idx[type2]
                    
                    emb1 = item_embeddings[idx1]
                    emb2 = item_embeddings[idx2]
                    
                    # 코사인 유사도 계산 (1 - 코사인 거리)
                    similarity = 1 - cosine(emb1, emb2)
                    
                    # 결과 저장
                    results.append({
                        "item_idx": item_idx,
                        "type1": type1,
                        "type2": type2,
                        "text1": item_texts[idx1],
                        "text2": item_texts[idx2],
                        "similarity": similarity,
                        "category": item_category  # 카테고리 정보 추가
                    })
        
        self.similarity_results = results
        
        # 유형 쌍별 평균 유사도 계산
        pair_averages = {}
        for type1, type2 in self.all_type_pairs:
            pair_sims = [r["similarity"] for r in results if (r["type1"] == type1 and r["type2"] == type2)]
            if pair_sims:
                pair_name = f"{type1}-{type2}"
                pair_averages[pair_name] = {
                    "mean": np.mean(pair_sims),
                    "std": np.std(pair_sims),
                    "count": len(pair_sims),
                    "min": np.min(pair_sims),
                    "max": np.max(pair_sims),
                    "type1": type1,
                    "type2": type2
                }
                print(f"{pair_name}: 평균 유사도 = {pair_averages[pair_name]['mean']:.4f} (± {pair_averages[pair_name]['std']:.4f})")
        
        self.pair_averages = pair_averages
        
        return results
    
    def plot_comprehensive_similarity_comparison(self):
        """모든 표현 유형 쌍 간의 유사도 비교 시각화"""
        if not hasattr(self, 'pair_averages'):
            self.calculate_all_similarity_metrics()
        
        print("포괄적 유사도 비교 시각화 생성 중...")
        
        # 데이터프레임 변환
        data = []
        for pair_name, stats in self.pair_averages.items():
            type1, type2 = stats['type1'], stats['type2']
            color1 = self.type_colors.get(type1, "#808080") 
            color2 = self.type_colors.get(type2, "#808080") 
            
            # 두 색상 중간값 계산
            import re
            def hex_to_rgb(hex_color):
                # HEX 색상 코드를 RGB로 변환
                hex_color = hex_color.lstrip('#')
                return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            color_map = {
                "red": "#FF0000",
                "blue": "#0000FF",
                "green": "#00FF00",
                "purple": "#800080",
                "gray": "#808080"
            }
            
            color1 = color_map.get(color1, color1)
            color2 = color_map.get(color2, color2)
            
            # RGB 평균 계산
            rgb1 = hex_to_rgb(color1)
            rgb2 = hex_to_rgb(color2)
            avg_rgb = [(r1 + r2) // 2 for r1, r2 in zip(rgb1, rgb2)]
            
            # 다시 HEX로 변환
            avg_color = "#{:02x}{:02x}{:02x}".format(*avg_rgb)
            
            data.append({
                "pair": pair_name,
                "description": f"{self.type_descriptions[stats['type1']]} → {self.type_descriptions[stats['type2']]}",
                "mean": stats["mean"],
                "std": stats["std"],
                "count": stats["count"],
                "type1": stats["type1"],
                "type2": stats["type2"],
                "color": avg_color 
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values(by="mean", ascending=False)
        
        # Plotly 막대 그래프
        fig = px.bar(
            df, 
            x="pair", 
            y="mean", 
            error_y="std",
            title=f'모든 표현 유형 쌍 간의 유사도 비교 ({self.model_name})',
            labels={"pair": "표현 유형 쌍", "mean": "평균 유사도"},
            text_auto='.4f',
            hover_data=["description", "count"],
            color="pair",  # 각 쌍을 색상으로 구분
            color_discrete_map=dict(zip(df["pair"], df["color"]))  # 커스텀 색상 매핑
        )
        
        fig.update_layout(
            width=900,
            height=600,
            yaxis_range=[0, 1],
            xaxis_title="표현 유형 쌍",
            yaxis_title="평균 유사도"
        )
        # update x축 라벨 
        fig.update_xaxes(
            ticktext=df["description"].tolist(),
            tickvals=df["pair"].tolist(),
            tickangle=45
        )

        fig.update_layout(showlegend=False)
        fig.write_html(os.path.join(self.output_dir, "comprehensive_similarity_comparison.html"))
        print(f"포괄적 유사도 비교 시각화 저장 완료: {self.output_dir}/comprehensive_similarity_comparison.html")
        return fig

    
    def plot_item_clusters(self):
        """같은 항목 내 모든 표현의 클러스터 시각화"""
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
            
        print("항목별, 표현 유형별 클러스터 시각화 생성 중...")
        fig = go.Figure()
        
        # 모든 임베딩 포인트 추가 (표현 유형별 색상)
        for text_type in set(self.text_types):
            indices = [i for i, t in enumerate(self.text_types) if t == text_type]
            fig.add_trace(go.Scatter(
                x=self.reduced_embeddings[indices, 0],
                y=self.reduced_embeddings[indices, 1],
                mode='markers',
                name=self.type_descriptions.get(text_type, text_type),
                marker=dict(
                    size=10, 
                    color=self.type_colors.get(text_type, "gray"),
                    opacity=0.7
                ),
                hoverinfo='text',
                hovertext=[f"Type: {text_type}<br>Item: {self.item_indices[i]}<br>Text: {self.texts[i]}" for i in indices]
            ))
        
        # 같은 항목 내 모든 표현을 연결하는 선 추가
        unique_item_indices = sorted(set(self.item_indices))
        
        for item_idx in unique_item_indices:
            # 현재 항목의 인덱스 찾기
            item_positions = [i for i, idx in enumerate(self.item_indices) if idx == item_idx]
            
            # 현재 항목의 타입
            item_types = [self.text_types[i] for i in item_positions]
            
            # 타입별 인덱스 매핑
            type_to_pos = {item_types[i]: item_positions[i] for i in range(len(item_types))}
            
            # 같은 항목 내 모든 표현 쌍을 연결
            for type1, type2 in itertools.combinations(type_to_pos.keys(), 2):
                pos1 = type_to_pos[type1]
                pos2 = type_to_pos[type2]
                
                # 코사인 유사도 계산
                similarity = 1 - cosine(self.embeddings[pos1], self.embeddings[pos2])
                
                # 선 두께와 투명도 조정
                line_width = max(0.5, similarity * 3)
                opacity = min(0.8, max(0.1, similarity))
                
                # 선 추가 (반투명 회색)
                fig.add_trace(go.Scatter(
                    x=[self.reduced_embeddings[pos1, 0], self.reduced_embeddings[pos2, 0]],
                    y=[self.reduced_embeddings[pos1, 1], self.reduced_embeddings[pos2, 1]],
                    mode='lines',
                    line=dict(width=line_width, color=f'rgba(100,100,100,{opacity})'),
                    showlegend=False,
                    hoverinfo='text',
                    hovertext=f"Item: {item_idx}<br>{type1} → {type2}<br>Similarity: {similarity:.4f}"
                ))
        fig.update_layout(
            title=f'항목별, 표현 유형별 임베딩 클러스터 ({self.model_name})',
            width=1000,
            height=800,
            legend=dict(
                title="표현 유형",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99
            )
        )
        
        # 저장
        fig.write_html(os.path.join(self.output_dir, "item_clusters.html"))
        print(f"항목 클러스터 시각화 저장 완료: {self.output_dir}/item_clusters.html")
        
        return fig
    
    def plot_enhanced_clusters(self):
        """
        향상된 클러스터 시각화: 
        - 같은 예제는 같은 색상
        - 같은 쿼리 유형은 같은 마커 모양
        """
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
                
        print("향상된 클러스터 시각화 생성 중...")
        
        # 고유 항목 인덱스 및 유형 목록
        unique_items = sorted(set(self.item_indices))
        unique_types = sorted(set(self.text_types))
        
        # 항목별 색상 매핑 (충분히 다양한 색상을 사용)
        import matplotlib.cm as cm
        import random
        
        # 색상 생성 - 예제 수가 많으면 랜덤 컬러를 생성
        if len(unique_items) > 20:
            item_colors = {item_idx: f"#{random.randint(0, 0xFFFFFF):06x}" for item_idx in unique_items}
        else:
            # 항목 수가 적으면 colormap 사용 (더 일관된 색상)
            colormap = cm.get_cmap('hsv', len(unique_items))
            item_colors = {
                item_idx: f"rgba({int(r*255)},{int(g*255)},{int(b*255)},0.8)" 
                for item_idx, (r, g, b, _) in zip(unique_items, [colormap(i) for i in range(len(unique_items))])
            }
        
        # 유형별 마커 형태 매핑
        type_symbols = {
            "EtoK": "circle",        # 원형 (코드 스위칭)
            "KtoE": "diamond",       # 다이아몬드 (코드 스위칭)
            "English": "square",     # 사각형 (비 코드 스위칭)
            "Korean": "triangle-up"  # 삼각형 (비 코드 스위칭)
        }
        
        # 새 figure 생성
        fig = go.Figure()
        
        # 각 항목-유형 조합에 대한 포인트 그룹 추가
        for item_idx in unique_items:
            for text_type in unique_types:
                # 현재 항목 및 유형에 해당하는 인덱스 찾기
                indices = [i for i, (idx, t) in enumerate(zip(self.item_indices, self.text_types)) 
                          if idx == item_idx and t == text_type]
                
                # 해당하는 데이터가 있는 경우만 처리
                if indices:
                    # 항목 색상 및 유형 심볼 가져오기
                    color = item_colors[item_idx]
                    symbol = type_symbols.get(text_type, "circle")
                    
                    # 항목-유형 조합 이름 (범례 표시용)
                    if len(indices) == 1:  # 일반적으로 각 항목-유형 조합은 하나의 문장
                        text = self.texts[indices[0]]
                        short_text = text[:30] + "..." if len(text) > 30 else text
                        name = f"Item {item_idx} ({self.type_descriptions.get(text_type, text_type)}): {short_text}"
                    else:
                        name = f"Item {item_idx} ({self.type_descriptions.get(text_type, text_type)})"
                    
                    # 시각화에 추가
                    fig.add_trace(go.Scatter(
                        x=self.reduced_embeddings[indices, 0],
                        y=self.reduced_embeddings[indices, 1],
                        mode='markers',
                        marker=dict(
                            color=color,
                            symbol=symbol,
                            size=12,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        name=name,
                        legendgroup=f"item_{item_idx}",
                        showlegend=True,
                        hovertext=[f"Item: {item_idx}<br>Type: {text_type}<br>Text: {self.texts[i]}" for i in indices],
                        hoverinfo='text'
                    ))
        
        # 같은 항목 내 다른 유형 간 연결선 추가 (유사도 시각화)
        for item_idx in unique_items:
            # 현재 항목에 해당하는 인덱스 찾기
            item_positions = [i for i, idx in enumerate(self.item_indices) if idx == item_idx]
            
            # 현재 항목의 타입
            item_types = [self.text_types[i] for i in item_positions]
            
            # 타입별 인덱스 매핑
            type_to_pos = {item_types[i]: item_positions[i] for i in range(len(item_positions))}
            
            # 코드 스위칭과 베이스라인 쿼리 쌍 분석을 위한 특별 연결
            special_pairs = [
                ("EtoK", "English"),  # 영→한 코드스위칭 vs 영어
                ("KtoE", "Korean")    # 한→영 코드스위칭 vs 한국어
            ]
            
            # 특별 연결 먼저 처리 (코드 스위칭과 베이스라인 비교)
            for type1, type2 in special_pairs:
                if type1 in type_to_pos and type2 in type_to_pos:
                    pos1 = type_to_pos[type1]
                    pos2 = type_to_pos[type2]
                    
                    # 두 포인트 간 유사도 계산
                    similarity = 1 - cosine(self.embeddings[pos1], self.embeddings[pos2])
                    
                    # 유사도에 따라 선 스타일 설정
                    line_width = max(1.5, similarity * 4)
                    
                    # 유사도에 따른 색상 (높을수록 진한 초록, 낮을수록 빨강)
                    g_value = max(0, min(255, int(255*similarity)))
                    b_value = max(0, min(255, int(50*(1-similarity))))
                    line_color = f'rgba(0,{g_value},{b_value},0.7)'
                    
                    # 코드 스위칭-베이스라인 쌍 연결선 추가
                    fig.add_trace(go.Scatter(
                        x=[self.reduced_embeddings[pos1, 0], self.reduced_embeddings[pos2, 0]],
                        y=[self.reduced_embeddings[pos1, 1], self.reduced_embeddings[pos2, 1]],
                        mode='lines',
                        line=dict(width=line_width, color=line_color, dash='dash'),
                        name=f"Item {item_idx}: {type1}→{type2} (Sim: {similarity:.3f})",
                        legendgroup=f"item_{item_idx}_comparison",
                        showlegend=False,
                        hoverinfo='text',
                        hovertext=f"Item: {item_idx}<br>{type1}→{type2}<br>Similarity: {similarity:.4f}"
                    ))
        fig.update_layout(
            title=f'코드 스위칭 분석: 항목별 색상, 쿼리 유형별 도형 ({self.model_name})',
            width=1200,
            height=800,
            legend=dict(
                title="항목 및 쿼리 유형",
                groupclick="toggleitem"
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # 범례 그룹화 - 같은 항목은 같은 색상으로 표시
        fig.update_layout(
            legend_tracegroupgap=5  # 각 그룹 간격
        )
        
        # 주석 추가 - 쿼리 유형별 마커 설명
        annotations = []
        for i, (type_name, symbol) in enumerate(type_symbols.items()):
            annotations.append(dict(
                x=0.01,
                y=0.99 - i*0.05,
                xref="paper",
                yref="paper",
                text=f"{self.type_descriptions.get(type_name, type_name)}: {symbol} 형태",
                showarrow=False,
                font=dict(size=12)
            ))
        
        fig.update_layout(annotations=annotations)
        os.makedirs(self.output_dir, exist_ok=True)
        fig.write_html(os.path.join(self.output_dir, "enhanced_clusters.html"))
        print(f"향상된 클러스터 시각화 저장 완료: {self.output_dir}/enhanced_clusters.html")
        
        return fig

    def plot_item_similarity_heatmap(self):
        """항목 내 표현 유형 간 유사도 히트맵"""
        if not hasattr(self, 'similarity_results'):
            self.calculate_all_similarity_metrics()
            
        print("항목 내 표현 유형 간 유사도 히트맵 생성 중...")
        
        # 데이터프레임 변환
        df = pd.DataFrame(self.similarity_results)
        
        # 유형 이름 변환
        df['type1_desc'] = df['type1'].map(self.type_descriptions)
        df['type2_desc'] = df['type2'].map(self.type_descriptions)
        
        # 각 항목에 대한 유사도 행렬 계산
        unique_items = sorted(set(df['item_idx']))
        unique_types = sorted(set(self.text_types))
        
        # 평균 유사도 행렬 계산
        avg_sim_matrix = np.zeros((len(unique_types), len(unique_types)))
        for i, type1 in enumerate(unique_types):
            for j, type2 in enumerate(unique_types):
                if i == j:
                    avg_sim_matrix[i, j] = 1.0  # 자기 자신과의 유사도는 1
                else:
                    sims = df[(df['type1'] == type1) & (df['type2'] == type2)]['similarity'].values
                    if len(sims) > 0:
                        avg_sim_matrix[i, j] = np.mean(sims)
        
        # 히트맵 데이터 준비
        heatmap_data = []
        for i, type1 in enumerate(unique_types):
            for j, type2 in enumerate(unique_types):
                heatmap_data.append({
                    'type1': self.type_descriptions[type1],
                    'type2': self.type_descriptions[type2],
                    'similarity': avg_sim_matrix[i, j]
                })
        
        heatmap_df = pd.DataFrame(heatmap_data)
        
        # 히트맵 생성
        fig = px.density_heatmap(
            heatmap_df,
            x='type1',
            y='type2',
            z='similarity',
            title=f'표현 유형 간 평균 유사도 히트맵 ({self.model_name})',
            labels={'type1': '표현 유형 1', 'type2': '표현 유형 2', 'similarity': '평균 유사도'},
            color_continuous_scale='RdYlGn',
            range_color=[0, 1],
            text_auto='.3f'
        )
        fig.update_layout(
            width=700,
            height=700
        )
        fig.write_html(os.path.join(self.output_dir, "type_similarity_heatmap.html"))
        
        print(f"표현 유형 간 유사도 히트맵 저장 완료: {self.output_dir}/type_similarity_heatmap.html")
        
        return fig
    
    def analyze_item_similarity_distribution(self):
        """항목 내 표현 유형 간 유사도 분포 분석"""
        if not hasattr(self, 'similarity_results'):
            self.calculate_all_similarity_metrics()
            
        print("항목 내 표현 유형 간 유사도 분포 분석 중...")
        df = pd.DataFrame(self.similarity_results)
        
        # 각 쌍에 쌍 이름 추가 (가독성 향상)
        df['pair_name'] = df.apply(
            lambda row: f"{self.type_descriptions.get(row['type1'], row['type1'])} → {self.type_descriptions.get(row['type2'], row['type2'])}",
            axis=1
        )
        
        # 1. 유형 쌍별 유사도 분포 시각화 (간소화된 방식)
        try:
            # 각 유형 쌍에 대한 별도의 박스플롯
            fig_box = px.box(
                df,
                x='pair_name',
                y='similarity',
                color='pair_name',
                title=f'표현 유형 쌍별 유사도 분포 ({self.model_name})',
                labels={'pair_name': '표현 유형 쌍', 'similarity': '유사도'},
                points='all',
                hover_data=['item_idx', 'text1', 'text2']
            )
            
            # 레이아웃 설정
            fig_box.update_layout(
                width=1000,
                height=600,
                xaxis_title="표현 유형 쌍",
                yaxis_title="유사도",
                xaxis={'categoryorder': 'mean descending'}  # 평균 기준 내림차순 정렬
            )
            
            # X축 라벨 회전
            fig_box.update_xaxes(tickangle=45)
            fig_box.write_html(os.path.join(self.output_dir, "type_similarity_distribution.html"))
            print(f"표현 유형 쌍별 유사도 분포 시각화 저장 완료: {self.output_dir}/type_similarity_distribution.html")
        
        except Exception as e:
            print(f"유형 쌍별 유사도 분포 시각화 생성 중 오류 발생: {e}")
            print("이 시각화는 건너뜁니다.")
        
        # 2. 표현 유형 쌍별 상자그림 (각 쌍별로 별도 그래프)
        try:
            # 유형 쌍별 그룹화
            pair_groups = df.groupby(['type1', 'type2'])
            
            # 각 유형 쌍에 대한 별도 그래프 생성
            fig_multi = go.Figure()
            
            for (type1, type2), group in pair_groups:
                pair_name = f"{self.type_descriptions.get(type1, type1)} → {self.type_descriptions.get(type2, type2)}"
                
                # 색상 결정 (type1과 type2의 조합)
                color1 = self.type_colors.get(type1, "#808080")
                color2 = self.type_colors.get(type2, "#808080")
                
                # 상자그림 추가
                fig_multi.add_trace(go.Box(
                    y=group['similarity'],
                    name=pair_name,
                    boxpoints='all',  # 모든 점 표시
                    jitter=0.3,  # 점들이 겹치지 않도록
                    pointpos=-1.8,  # 점을 상자 왼쪽에 배치
                    marker=dict(
                        color=color1,
                        line=dict(width=1, color=color2)
                    ),
                    hovertemplate=(
                        "유사도: %{y:.4f}<br>" +
                        "항목: %{customdata[0]}"
                    ),
                    customdata=group[['item_idx']].values
                ))
            
            # 레이아웃 설정
            fig_multi.update_layout(
                title=f'표현 유형 쌍별 유사도 분포 상세 ({self.model_name})',
                yaxis_title='유사도',
                boxmode='group',
                width=1000,
                height=600,
                yaxis=dict(range=[0, 1])
            )
            fig_multi.write_html(os.path.join(self.output_dir, "type_similarity_boxplots.html"))
            print(f"표현 유형 쌍별 상세 상자그림 저장 완료: {self.output_dir}/type_similarity_boxplots.html")
            
        except Exception as e:
            print(f"표현 유형 쌍별 상자그림 생성 중 오류 발생: {e}")
            print("이 시각화는 건너뜁니다.")
        
        # 3. 항목별 클러스터 밀집도 분석
        # (같은 항목 내 모든 표현 쌍의 평균 유사도)
        item_cluster_scores = {}
        for item_idx in sorted(set(df['item_idx'])):
            item_sims = df[df['item_idx'] == item_idx]['similarity'].values
            item_cluster_scores[item_idx] = np.mean(item_sims)
        
        # 4. 항목별 클러스터 밀집도 시각화
        try:
            item_df = pd.DataFrame([
                {'item_idx': item_idx, 'cluster_score': score}
                for item_idx, score in item_cluster_scores.items()
            ])
            
            # 카테고리 정보 추가
            if self.item_categories:
                item_df['category'] = item_df['item_idx'].apply(
                    lambda idx: self.item_categories.get(str(idx), self.item_categories.get(idx, "other"))
                )
                item_df['category_desc'] = item_df['category'].apply(
                    lambda cat: self.category_descriptions.get(cat, cat)
                )
                
                # 카테고리별 색상 매핑
                color_map = {cat: self.category_colors.get(cat, "#808080") for cat in item_df['category'].unique()}
    
                # 카테고리별로 색상 지정
                fig_items = px.bar(
                    item_df.sort_values('cluster_score', ascending=False),
                    x='item_idx',
                    y='cluster_score',
                    color='category_desc',
                    title=f'항목별 클러스터 밀집도 ({self.model_name})',
                    labels={'item_idx': '항목 인덱스', 'cluster_score': '클러스터 밀집도 (평균 유사도)', 'category_desc': '카테고리'},
                    color_discrete_map={desc: color_map.get(cat, "#808080") for cat, desc in self.category_descriptions.items()}
                )
            else:
                fig_items = px.bar(
                    item_df.sort_values('cluster_score', ascending=False),
                    x='item_idx',
                    y='cluster_score',
                    title=f'항목별 클러스터 밀집도 ({self.model_name})',
                    labels={'item_idx': '항목 인덱스', 'cluster_score': '클러스터 밀집도 (평균 유사도)'},
                    color='cluster_score',
                    color_continuous_scale='RdYlGn',
                    range_color=[0, 1]
                )
            
            # 레이아웃 설
            fig_items.update_layout(
                width=1000,
                height=600,
                yaxis_range=[0, 1],
                xaxis_title="항목 인덱스 (클러스터 밀집도 내림차순)",
                yaxis_title="클러스터 밀집도 (평균 유사도)"
            )
            fig_items.write_html(os.path.join(self.output_dir, "item_cluster_scores.html"))
            print(f"항목별 클러스터 밀집도 시각화 저장 완료: {self.output_dir}/item_cluster_scores.html")
            
        except Exception as e:
            print(f"항목별 클러스터 밀집도 시각화 생성 중 오류 발생: {e}")
            print("이 시각화는 건너뜁니다.")
        
        # 5. 상위/하위 10개 항목에 대한 추가 분석
        try:
            # 상위/하위 10개 항목 선택
            top_items = item_df.nlargest(10, 'cluster_score')['item_idx'].tolist()
            bottom_items = item_df.nsmallest(10, 'cluster_score')['item_idx'].tolist()
            
            # 상위/하위 항목 시각화
            top_bottom_df = df[df['item_idx'].isin(top_items + bottom_items)].copy()
            top_bottom_df['category'] = top_bottom_df['item_idx'].apply(
                lambda x: '상위 10개 항목' if x in top_items else '하위 10개 항목'
            )
            
            fig_top_bottom = px.box(
                top_bottom_df,
                x='pair_name',
                y='similarity',
                color='category',
                facet_row='category',
                title=f'상위/하위 10개 항목의 유형 쌍별 유사도 ({self.model_name})',
                labels={'pair_name': '표현 유형 쌍', 'similarity': '유사도', 'category': '항목 그룹'},
                color_discrete_map={'상위 10개 항목': 'green', '하위 10개 항목': 'red'},
                points='all'
            )
            
            # 레이아웃 설정
            fig_top_bottom.update_layout(
                width=1000,
                height=800,
                xaxis_title="표현 유형 쌍",
                yaxis_title="유사도"
            )
            
            # X축 라벨 회전
            fig_top_bottom.update_xaxes(tickangle=45)
            fig_top_bottom.write_html(os.path.join(self.output_dir, "top_bottom_items_comparison.html"))
            print(f"상위/하위 항목 비교 시각화 저장 완료: {self.output_dir}/top_bottom_items_comparison.html")
            
        except Exception as e:
            print(f"상위/하위 항목 비교 시각화 생성 중 오류 발생: {e}")
            print("이 시각화는 건너뜁니다.")
        
        # 6. 항목별 클러스터 밀집도 상세 테이블 생성
        try:
            # 항목별 텍스트 정보 수집
            item_details = {}
            for item_idx in sorted(set(self.item_indices)):
                item_positions = [i for i, idx in enumerate(self.item_indices) if idx == item_idx]
                item_texts = {}
                for pos in item_positions:
                    text_type = self.text_types[pos]
                    item_texts[text_type] = self.texts[pos]
                item_details[item_idx] = item_texts
            
            # 항목별 밀집도와 텍스트 정보를 합친 데이터프레임 생성
            detailed_rows = []
            for item_idx in sorted(set(self.item_indices)):
                row = {
                    'item_idx': item_idx,
                    'cluster_score': item_cluster_scores.get(item_idx, 0),
                    'EtoK': item_details[item_idx].get('EtoK', ''),
                    'KtoE': item_details[item_idx].get('KtoE', ''),
                    'English': item_details[item_idx].get('English', ''),
                    'Korean': item_details[item_idx].get('Korean', '')
                }
                # 카테고리 정보 추가
                if self.item_categories:
                    category = self.item_categories.get(str(item_idx), self.item_categories.get(item_idx, "other"))
                    row['category'] = category
                    row['category_desc'] = self.category_descriptions.get(category, category)
                
                detailed_rows.append(row)
            
            # 데이터프레임 생성 및 밀집도 기준 정렬
            detailed_df = pd.DataFrame(detailed_rows)
            detailed_df = detailed_df.sort_values('cluster_score', ascending=False)
            
            # HTML 테이블 생성
            html_table = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>항목별 클러스터 밀집도 상세 정보</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; }}
                    th {{ background-color: #f2f2f2; text-align: left; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .high-score {{ background-color: rgba(0, 255, 0, 0.2); }}
                    .medium-score {{ background-color: rgba(255, 255, 0, 0.2); }}
                    .low-score {{ background-color: rgba(255, 0, 0, 0.2); }}
                    .score-cell {{ font-weight: bold; text-align: center; }}
                    .category-badge {{
                        display: inline-block;
                        padding: 3px 8px;
                        border-radius: 12px;
                        font-size: 12px;
                        color: white;
                    }}
                    .tech {{ background-color: #2ecc71; }}
                    .academic {{ background-color: #9b59b6; }}
                    .geopolitics {{ background-color: #e74c3c; }}
                    .culture {{ background-color: #3498db; }}
                    .regional {{ background-color: #f39c12; }}
                    .language {{ background-color: #1abc9c; }}
                    .food {{ background-color: #e67e22; }}
                    .other {{ background-color: #95a5a6; }}
                    
                    .filters {{
                        margin-bottom: 20px;
                        padding: 10px;
                        background-color: #f8f9fa;
                        border-radius: 5px;
                    }}
                    select, button {{
                        padding: 5px 10px;
                        margin-right: 10px;
                    }}
                    button {{
                        background-color: #4CAF50;
                        color: white;
                        border: none;
                        cursor: pointer;
                        border-radius: 3px;
                    }}
                    button:hover {{
                        background-color: #45a049;
                    }}
                </style>
            </head>
            <body>
                <h1>항목별 클러스터 밀집도 상세 정보 ({self.model_name})</h1>
                <p>총 {len(detailed_df)}개 항목의 클러스터 밀집도와 4가지 표현 내용</p>
            """
            
            # 카테고리 정보가 있는 경우 필터 추가
            if self.item_categories:
                html_table += """
                <div class="filters">
                    <label for="categoryFilter">카테고리 필터:</label>
                    <select id="categoryFilter">
                        <option value="all">모든 카테고리</option>
                """
                
                # 유일한 카테고리 목록
                unique_categories = sorted(set(detailed_df['category']))
                for cat in unique_categories:
                    cat_desc = self.category_descriptions.get(cat, cat)
                    html_table += f'<option value="{cat}">{cat_desc}</option>\n'
                
                html_table += """
                    </select>
                    
                    <label for="scoreFilter">밀집도 필터:</label>
                    <select id="scoreFilter">
                        <option value="all">모든 밀집도</option>
                        <option value="high">높음 (0.7 이상)</option>
                        <option value="medium">중간 (0.4 - 0.7)</option>
                        <option value="low">낮음 (0.4 미만)</option>
                    </select>
                    
                    <button onclick="applyFilters()">필터 적용</button>
                </div>
                """
            # 테이블 시작
            html_table += """
                <table id="detailTable">
                    <tr>
                        <th>항목 인덱스</th>
            """
            # 카테고리 컬럼 추가
            if self.item_categories:
                html_table += "<th>카테고리</th>\n"
                
            html_table += """
                        <th>클러스터 밀집도</th>
                        <th>EtoK (영→한 코드스위칭)</th>
                        <th>KtoE (한→영 코드스위칭)</th>
                        <th>English (영어)</th>
                        <th>Korean (한국어)</th>
                    </tr>
            """
            
            # 각 항목에 대한 행 추가
            for _, row in detailed_df.iterrows():
                score = row['cluster_score']
                if score >= 0.7:
                    score_class = "high-score"
                elif score >= 0.4:
                    score_class = "medium-score"
                else:
                    score_class = "low-score"
                
                # 행 시작 - 카테고리 데이터 속성 추가
                cat_attr = ""
                if self.item_categories:
                    cat_attr = f" data-category=\"{row['category']}\" data-score=\"{score}\""
                
                html_table += f"\n<tr{cat_attr}>\n<td>{row['item_idx']}</td>\n"
                
                # 카테고리 열 추가
                if self.item_categories:
                    cat = row['category']
                    cat_desc = row.get('category_desc', self.category_descriptions.get(cat, cat))
                    html_table += f'<td><span class="category-badge {cat}">{cat_desc}</span></td>\n'
                
                # 나머지 열 추가
                html_table += f"""
                    <td class="score-cell {score_class}">{score:.4f}</td>
                    <td>{row['EtoK'][:100]}{'...' if len(row['EtoK']) > 100 else ''}</td>
                    <td>{row['KtoE'][:100]}{'...' if len(row['KtoE']) > 100 else ''}</td>
                    <td>{row['English'][:100]}{'...' if len(row['English']) > 100 else ''}</td>
                    <td>{row['Korean'][:100]}{'...' if len(row['Korean']) > 100 else ''}</td>
                </tr>
                """
            
            # 필터링 스크립트 추가
            if self.item_categories:
                html_table += """
                <script>
                function applyFilters() {
                    var categoryFilter = document.getElementById('categoryFilter').value;
                    var scoreFilter = document.getElementById('scoreFilter').value;
                    
                    var rows = document.querySelectorAll('#detailTable tr:not(:first-child)');
                    
                    rows.forEach(function(row) {
                        var category = row.getAttribute('data-category');
                        var score = parseFloat(row.getAttribute('data-score'));
                        
                        var categoryMatch = (categoryFilter === 'all' || category === categoryFilter);
                        var scoreMatch = true;
                        
                        if (scoreFilter === 'high') {
                            scoreMatch = score >= 0.7;
                        } else if (scoreFilter === 'medium') {
                            scoreMatch = score >= 0.4 && score < 0.7;
                        } else if (scoreFilter === 'low') {
                            scoreMatch = score < 0.4;
                        }
                        
                        if (categoryMatch && scoreMatch) {
                            row.style.display = '';
                        } else {
                            row.style.display = 'none';
                        }
                    });
                }
                </script>
                """
            
            html_table += """
                </table>
            </body>
            </html>
            """
            with open(os.path.join(self.output_dir, "item_cluster_details.html"), 'w', encoding='utf-8') as f:
                f.write(html_table)
            
            print(f"항목별 클러스터 밀집도 상세 테이블 저장 완료: {self.output_dir}/item_cluster_details.html")
            
        except Exception as e:
            print(f"항목별 클러스터 밀집도 상세 테이블 생성 중 오류 발생: {e}")
            print("이 시각화는 건너뜁니다.")
        
        return {
            'type_similarity': fig_box if 'fig_box' in locals() else None,
            'item_cluster_scores': fig_items if 'fig_items' in locals() else None,
            'cluster_scores': item_cluster_scores,
            'top_bottom_comparison': fig_top_bottom if 'fig_top_bottom' in locals() else None
        }

    # 코드 스위칭 효과 분석
    def analyze_code_switching_effect(self):
        if not hasattr(self, 'similarity_results'):
            self.calculate_all_similarity_metrics()
            
        print("코드 스위칭 효과 분석 중...")
        df = pd.DataFrame(self.similarity_results)
        
        # 코드 스위칭 쌍과 그렇지 않은 쌍 구분
        code_switching_pairs = [("EtoK", "English"), ("KtoE", "Korean")]
        other_pairs = [("EtoK", "KtoE"), ("English", "Korean"), ("EtoK", "Korean"), ("KtoE", "English")]
        
        # 전체 데이터에서 코드 스위칭 효과 분석
        cs_sims = []
        for type1, type2 in code_switching_pairs:
            cs_sims.extend(df[(df['type1'] == type1) & (df['type2'] == type2)]['similarity'].tolist())
        
        other_sims = []
        for type1, type2 in other_pairs:
            other_sims.extend(df[(df['type1'] == type1) & (df['type2'] == type2)]['similarity'].tolist())
        
        cs_mean = np.mean(cs_sims) if cs_sims else 0
        other_mean = np.mean(other_sims) if other_sims else 0
        
        # 카테고리별 코드 스위칭 효과 분석
        cs_data = []
        
        if self.item_categories:
            # 카테고리별 코드 스위칭 효과
            for category in set(self.item_categories.values()):
                # 해당 카테고리 항목 인덱스
                category_items = [item_idx for item_idx, item_cat in self.item_categories.items() 
                                 if item_cat == category]
                
                category_items_str = [str(item) for item in category_items]
                category_items_int = [int(item) if str(item).isdigit() else item for item in category_items]
                
                # 카테고리 항목 필터링
                category_df = df[df['item_idx'].isin(category_items_str) | df['item_idx'].isin(category_items_int)]
                
                # 코드 스위칭 쌍 유사도
                cs_similarities = []
                for type1, type2 in code_switching_pairs:
                    pair_df = category_df[(category_df['type1'] == type1) & (category_df['type2'] == type2)]
                    if len(pair_df) > 0:
                        cs_similarities.extend(pair_df['similarity'].tolist())
                
                # 기타 쌍 유사도
                other_similarities = []
                for type1, type2 in other_pairs:
                    pair_df = category_df[(category_df['type1'] == type1) & (category_df['type2'] == type2)]
                    if len(pair_df) > 0:
                        other_similarities.extend(pair_df['similarity'].tolist())
                
                # 통계 계산
                if cs_similarities and other_similarities:
                    cs_data.append({
                        'category': self.category_descriptions.get(category, category),
                        'category_code': category,
                        'cs_mean': np.mean(cs_similarities),
                        'cs_std': np.std(cs_similarities),
                        'other_mean': np.mean(other_similarities),
                        'other_std': np.std(other_similarities),
                        'difference': np.mean(cs_similarities) - np.mean(other_similarities),
                        'cs_count': len(cs_similarities),
                        'other_count': len(other_similarities)
                    })
        
        # 코드 스위칭 효과 시각화
        try:
            # 1. 전체 효과 비교
            data = pd.DataFrame([
                {'group': '코드 스위칭 쌍', 'mean': cs_mean, 'std': np.std(cs_sims) if cs_sims else 0, 'count': len(cs_sims)},
                {'group': '다른 쌍', 'mean': other_mean, 'std': np.std(other_sims) if other_sims else 0, 'count': len(other_sims)}
            ])
            
            fig_overall = px.bar(
                data,
                x='group',
                y='mean',
                error_y='std',
                title=f'코드 스위칭 효과 비교 ({self.model_name})',
                labels={'group': '쌍 유형', 'mean': '평균 유사도'},
                text_auto='.4f',
                color='group',
                color_discrete_map={'코드 스위칭 쌍': '#1abc9c', '다른 쌍': '#3498db'}
            )
            
            fig_overall.update_layout(
                width=800,
                height=500,
                yaxis_range=[0, 1]
            )
            
            fig_overall.write_html(os.path.join(self.output_dir, "code_switching_effect.html"))
            print(f"코드 스위칭 효과 비교 시각화 저장 완료: {self.output_dir}/code_switching_effect.html")
            
            # 2. 카테고리별 효과 비교 (카테고리 정보가 있는 경우)
            if cs_data:
                cs_df = pd.DataFrame(cs_data)
                cs_df = cs_df.sort_values('difference', ascending=False)
                
                # 카테고리별 효과 비교 바 차트
                fig_category = px.bar(
                    cs_df,
                    x='category',
                    y='difference',
                    title=f'카테고리별 코드 스위칭 효과 차이 ({self.model_name})',
                    labels={'category': '카테고리', 'difference': '코드 스위칭 효과 (평균 유사도 차이)'},
                    text_auto='.4f',
                    color='category_code',
                    color_discrete_map={cat: color for cat, color in self.category_colors.items()}
                )
                
                fig_category.update_layout(
                    width=1000,
                    height=600,
                    showlegend=False
                )
                
                fig_category.update_xaxes(tickangle=45)
                
                fig_category.write_html(os.path.join(self.output_dir, "category_cs_effect.html"))
                print(f"카테고리별 코드 스위칭 효과 시각화 저장 완료: {self.output_dir}/category_cs_effect.html")
                
                # 3. 상세 유사도 비교
                fig_detail = go.Figure()
                
                for i, row in cs_df.iterrows():
                    category = row['category']
                    fig_detail.add_trace(go.Bar(
                        x=[f"{category} (코드 스위칭)"],
                        y=[row['cs_mean']],
                        error_y=dict(type='data', array=[row['cs_std']]),
                        name=f"{category} (코드 스위칭)",
                        marker_color=self.category_colors.get(row['category_code'], "#808080")
                    ))
                    
                    fig_detail.add_trace(go.Bar(
                        x=[f"{category} (기타)"],
                        y=[row['other_mean']],
                        error_y=dict(type='data', array=[row['other_std']]),
                        name=f"{category} (기타)",
                        marker_color=self.category_colors.get(row['category_code'], "#808080"),
                        marker_pattern_shape='x'
                    ))
                
                fig_detail.update_layout(
                    title=f'카테고리별 코드 스위칭 vs 기타 쌍 유사도 ({self.model_name})',
                    width=1200,
                    height=700,
                    yaxis_title='평균 유사도',
                    yaxis_range=[0, 1],
                    barmode='group'
                )
                
                fig_detail.write_html(os.path.join(self.output_dir, "category_detail_comparison.html"))
                print(f"카테고리별 상세 비교 시각화 저장 완료: {self.output_dir}/category_detail_comparison.html")
                
                # HTML 요약 테이블 생성
                html_summary = f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>코드 스위칭 효과 분석 요약</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 20px; }}
                        h1, h2 {{ color: #333; }}
                        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                        th {{ background-color: #f2f2f2; }}
                        tr:nth-child(even) {{ background-color: #f9f9f9; }}
                        .positive {{ color: green; font-weight: bold; }}
                        .negative {{ color: red; font-weight: bold; }}
                        .overview {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                        .category-badge {{
                            display: inline-block;
                            padding: 3px 8px;
                            border-radius: 12px;
                            font-size: 12px;
                            color: white;
                            margin-right: 5px;
                        }}
                    </style>
                </head>
                <body>
                    <h1>코드 스위칭 효과 분석 요약 ({self.model_name})</h1>
                    
                    <div class="overview">
                        <h2>전체 효과 분석</h2>
                        <p>
                            <strong>코드 스위칭 쌍 평균 유사도:</strong> {cs_mean:.4f} (샘플 수: {len(cs_sims)})<br>
                            <strong>기타 쌍 평균 유사도:</strong> {other_mean:.4f} (샘플 수: {len(other_sims)})<br>
                            <strong>효과 차이:</strong> <span class="{'positive' if cs_mean > other_mean else 'negative'}">{cs_mean - other_mean:.4f}</span>
                        </p>
                        <p>
                            <strong>코드 스위칭 쌍:</strong> EtoK↔English, KtoE↔Korean<br>
                            <strong>기타 쌍:</strong> EtoK↔KtoE, English↔Korean, EtoK↔Korean, KtoE↔English
                        </p>
                    </div>
                    
                    <h2>카테고리별 효과 분석</h2>
                    <table>
                        <tr>
                            <th>카테고리</th>
                            <th>코드 스위칭 쌍 유사도</th>
                            <th>기타 쌍 유사도</th>
                            <th>효과 차이</th>
                            <th>샘플 수 (코드 스위칭/기타)</th>
                        </tr>
                """
                # 효과 차이 순으로 정렬
                for _, row in cs_df.iterrows():
                    category = row['category']
                    cat_code = row['category_code']
                    diff_class = "positive" if row['difference'] > 0 else "negative"
                    
                    html_summary += f"""
                        <tr>
                            <td><span class="category-badge" style="background-color: {self.category_colors.get(cat_code, '#808080')}">{category}</span></td>
                            <td>{row['cs_mean']:.4f} (±{row['cs_std']:.4f})</td>
                            <td>{row['other_mean']:.4f} (±{row['other_std']:.4f})</td>
                            <td class="{diff_class}">{row['difference']:.4f}</td>
                            <td>{row['cs_count']} / {row['other_count']}</td>
                        </tr>
                    """
                
                html_summary += """
                    </table>
                    
                    <h2>결론</h2>
                    <div class="overview">
                """

                if cs_mean > other_mean:
                    html_summary += f"""
                        <p>코드 스위칭 쌍의 평균 유사도({cs_mean:.4f})가 기타 쌍의 평균 유사도({other_mean:.4f})보다 {cs_mean - other_mean:.4f} 높게 나타났습니다.
                        이는 코드 스위칭이 있는 표현과 해당 베이스라인 언어 간의 의미적 유사성이 잘 유지됨을 의미합니다.</p>
                    """
                else:
                    html_summary += f"""
                        <p>코드 스위칭 쌍의 평균 유사도({cs_mean:.4f})가 기타 쌍의 평균 유사도({other_mean:.4f})보다 {other_mean - cs_mean:.4f} 낮게 나타났습니다.
                        이는 코드 스위칭으로 인해 의미적 차이가 발생할 수 있음을 시사합니다.</p>
                    """

                highest_category = cs_df.iloc[0]['category'] if len(cs_df) > 0 else None
                lowest_category = cs_df.iloc[-1]['category'] if len(cs_df) > 0 else None
                
                if highest_category and lowest_category:
                    html_summary += f"""
                        <p>카테고리별로 살펴보면, <strong>{highest_category}</strong> 카테고리에서 코드 스위칭 효과가 가장 크게 나타났으며,
                        <strong>{lowest_category}</strong> 카테고리에서는 효과가 가장 작거나 부정적으로 나타났습니다.</p>
                    """
                
                html_summary += """
                    </div>
                </body>
                </html>
                """
                with open(os.path.join(self.output_dir, "code_switching_effect_summary.html"), 'w', encoding='utf-8') as f:
                    f.write(html_summary)
                
                print(f"코드 스위칭 효과 요약 저장 완료: {self.output_dir}/code_switching_effect_summary.html")
        
        except Exception as e:
            print(f"코드 스위칭 효과 시각화 생성 중 오류 발생: {e}")
            print("이 시각화는 건너뜁니다.")
        
        return {
            'cs_mean': cs_mean,
            'other_mean': other_mean,
            'difference': cs_mean - other_mean,
            'cs_sims': cs_sims,
            'other_sims': other_sims,
            'category_data': cs_data if 'cs_data' in locals() else []
        }

    # 카테고리별 클러스터 시각화
    def plot_category_clusters(self):
        if self.reduced_embeddings is None:
            self.reduce_dimensions()
        
        if not self.item_categories:
            print("카테고리 정보가 없어 카테고리별 클러스터 시각화를 건너뜁니다.")
            return None
                
        print("카테고리별 클러스터 시각화 생성 중...")
        type_symbols = {
            "EtoK": "circle",       
            "KtoE": "diamond",       
            "English": "square",     
            "Korean": "triangle-up"  
        }
        
        # 새 figure 생성
        fig = go.Figure()
        
        # 각 카테고리 및 표현 유형별 포인트 추가
        for category, color in self.category_colors.items():
            # 해당 카테고리 항목 인덱스 찾기
            category_items = [item_idx for item_idx, item_cat in self.item_categories.items() 
                              if item_cat == category]
            
            # 문자열과 숫자 인덱스 모두 고려
            category_items_str = [str(item) for item in category_items]
            category_items_int = [int(item) if str(item).isdigit() else item for item in category_items]
            
            # 각 표현 유형별 포인트 추가
            for text_type in set(self.text_types):
                # 현재 카테고리 및 표현 유형에 해당하는 인덱스 찾기
                indices = [i for i, (idx, t) in enumerate(zip(self.item_indices, self.text_types)) 
                          if (str(idx) in category_items_str or idx in category_items_int) and t == text_type]
                
                # 해당하는 데이터가 있는 경우만 처리
                if indices:
                    # 마커 심볼 및 이름 가져오기
                    symbol = type_symbols.get(text_type, "circle")
                    category_desc = self.category_descriptions.get(category, category)
                    name = f"{category_desc} - {self.type_descriptions.get(text_type, text_type)}"
                    
                    # 시각화에 추가
                    fig.add_trace(go.Scatter(
                        x=self.reduced_embeddings[indices, 0],
                        y=self.reduced_embeddings[indices, 1],
                        mode='markers',
                        marker=dict(
                            color=color,
                            symbol=symbol,
                            size=10,
                            line=dict(width=1, color='DarkSlateGrey')
                        ),
                        name=name,
                        legendgroup=category,
                        hovertext=[f"Category: {category_desc}<br>Type: {text_type}<br>Item: {self.item_indices[i]}<br>Text: {self.texts[i]}" for i in indices],
                        hoverinfo='text'
                    ))
        
        fig.update_layout(
            title=f'카테고리별 임베딩 클러스터 ({self.model_name})',
            width=1200,
            height=800,
            legend=dict(
                title="카테고리 및 표현 유형",
                groupclick="toggleitem"
            ),
            margin=dict(l=20, r=20, t=60, b=20)
        )
        
        # 같은 카테고리는 같은 색상으로 표시
        fig.update_layout(
            legend_tracegroupgap=5
        )

        annotations = []
        for i, (type_name, symbol) in enumerate(type_symbols.items()):
            annotations.append(dict(
                x=0.01,
                y=0.99 - i*0.05,
                xref="paper",
                yref="paper",
                text=f"{self.type_descriptions.get(type_name, type_name)}: {symbol} 형태",
                showarrow=False,
                font=dict(size=12)
            ))
        
        fig.update_layout(annotations=annotations)
        fig.write_html(os.path.join(self.output_dir, "category_clusters.html"))
        print(f"카테고리별 클러스터 시각화 저장 완료: {self.output_dir}/category_clusters.html")
        
        return fig

    # 카테고리별 항목 클러스터 밀집도 분석
    def analyze_category_distribution(self):
        """카테고리별 항목 클러스터 밀집도 분석"""
        if not hasattr(self, 'similarity_results'):
            self.calculate_all_similarity_metrics()
        
        if not self.item_categories:
            print("카테고리 정보가 없어 카테고리별 밀집도 분석을 건너뜁니다.")
            return None
            
        print("카테고리별 항목 클러스터 밀집도 분석 중...")
        df = pd.DataFrame(self.similarity_results)
        
        # 항목별 클러스터 밀집도 계산
        item_cluster_scores = {}
        for item_idx in sorted(set(df['item_idx'])):
            item_sims = df[df['item_idx'] == item_idx]['similarity'].values
            item_cluster_scores[item_idx] = np.mean(item_sims)
        
        # 항목-카테고리 매핑
        item_categories = {}
        for item_idx in sorted(set(df['item_idx'])):
            str_idx = str(item_idx)
            int_idx = int(item_idx) if str_idx.isdigit() else item_idx
            
            category = self.item_categories.get(str_idx, self.item_categories.get(int_idx, "other"))
            item_categories[item_idx] = category
        
        # 카테고리별 밀집도 통계
        category_scores = {}
        for item_idx, score in item_cluster_scores.items():
            category = item_categories.get(item_idx, "other")
            if category not in category_scores:
                category_scores[category] = []
            category_scores[category].append(score)
        
        # 카테고리별 평균 및 통계 계산
        category_stats = []
        for category, scores in category_scores.items():
            category_stats.append({
                'category': category,
                'category_desc': self.category_descriptions.get(category, category),
                'mean': np.mean(scores),
                'std': np.std(scores),
                'min': np.min(scores),
                'max': np.max(scores),
                'count': len(scores)
            })
        
        category_df = pd.DataFrame(category_stats)
        category_df = category_df.sort_values('mean', ascending=False)
        
        # 시각화
        try:
            # 1. 카테고리별 평균 밀집도 바 차트
            fig_category = px.bar(
                category_df,
                x='category_desc',
                y='mean',
                error_y='std',
                title=f'카테고리별 클러스터 밀집도 평균 ({self.model_name})',
                labels={'category_desc': '카테고리', 'mean': '평균 밀집도'},
                text_auto='.4f',
                color='category',
                color_discrete_map={cat: self.category_colors.get(cat, "#808080") for cat in category_df['category']}
            )
            
            fig_category.update_layout(
                width=1000,
                height=600,
                yaxis_range=[0, 1],
                xaxis_title="카테고리",
                yaxis_title="평균 클러스터 밀집도"
            )
            
            fig_category.update_xaxes(tickangle=45)
            
            fig_category.write_html(os.path.join(self.output_dir, "category_density.html"))
            print(f"카테고리별 밀집도 평균 시각화 저장 완료: {self.output_dir}/category_density.html")
            
            # 2. 카테고리별 밀집도 분포 상자그림
            # 데이터 준비
            boxplot_data = []
            for category, scores in category_scores.items():
                for score in scores:
                    boxplot_data.append({
                        'category': self.category_descriptions.get(category, category),
                        'category_code': category,
                        'cluster_score': score
                    })
            
            boxplot_df = pd.DataFrame(boxplot_data)
            
            fig_box = px.box(
                boxplot_df,
                x='category',
                y='cluster_score',
                color='category_code',
                title=f'카테고리별 클러스터 밀집도 분포 ({self.model_name})',
                labels={'category': '카테고리', 'cluster_score': '클러스터 밀집도'},
                color_discrete_map={cat: self.category_colors.get(cat, "#808080") for cat in boxplot_df['category_code'].unique()},
                points="all"
            )
            
            fig_box.update_layout(
                width=1000,
                height=600,
                yaxis_range=[0, 1],
                showlegend=False
            )
            
            fig_box.update_xaxes(tickangle=45)
            
            fig_box.write_html(os.path.join(self.output_dir, "category_density_distribution.html"))
            print(f"카테고리별 밀집도 분포 시각화 저장 완료: {self.output_dir}/category_density_distribution.html")
            
            # 3. HTML 요약 테이블 생성
            html_summary = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>카테고리별 클러스터 밀집도 분석</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2 {{ color: #333; }}
                    table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                    th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                    th {{ background-color: #f2f2f2; }}
                    tr:nth-child(even) {{ background-color: #f9f9f9; }}
                    .overview {{ background-color: #f8f9fa; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
                    .category-badge {{
                        display: inline-block;
                        padding: 3px 8px;
                        border-radius: 12px;
                        font-size: 12px;
                        color: white;
                        margin-right: 5px;
                    }}
                </style>
            </head>
            <body>
                <h1>카테고리별 클러스터 밀집도 분석 ({self.model_name})</h1>
                
                <div class="overview">
                    <h2>전체 통계</h2>
                    <p>
                        <strong>전체 항목 수:</strong> {len(item_cluster_scores)}<br>
                        <strong>전체 평균 밀집도:</strong> {np.mean(list(item_cluster_scores.values())):.4f}<br>
                        <strong>최대 밀집도:</strong> {np.max(list(item_cluster_scores.values())):.4f}<br>
                        <strong>최소 밀집도:</strong> {np.min(list(item_cluster_scores.values())):.4f}
                    </p>
                </div>
                
                <h2>카테고리별 통계</h2>
                <table>
                    <tr>
                        <th>카테고리</th>
                        <th>항목 수</th>
                        <th>평균 밀집도</th>
                        <th>표준편차</th>
                        <th>최소값</th>
                        <th>최대값</th>
                    </tr>
            """
            
            # 카테고리별 통계 추가
            for _, row in category_df.iterrows():
                category = row['category']
                cat_desc = row['category_desc']
                color = self.category_colors.get(category, "#808080")
                
                html_summary += f"""
                    <tr>
                        <td><span class="category-badge" style="background-color: {color}">{cat_desc}</span></td>
                        <td>{row['count']}</td>
                        <td>{row['mean']:.4f}</td>
                        <td>{row['std']:.4f}</td>
                        <td>{row['min']:.4f}</td>
                        <td>{row['max']:.4f}</td>
                    </tr>
                """
            
            html_summary += """
                </table>
                
                <h2>결론</h2>
                <div class="overview">
            """
            
            # 결론 부분 자동 생성
            highest_category = category_df.iloc[0]['category_desc'] if len(category_df) > 0 else None
            lowest_category = category_df.iloc[-1]['category_desc'] if len(category_df) > 0 else None
            
            if highest_category and lowest_category:
                html_summary += f"""
                    <p>카테고리별 클러스터 밀집도를 분석한 결과, <strong>{highest_category}</strong> 카테고리가 평균 {category_df.iloc[0]['mean']:.4f}로 
                    가장 높은 밀집도를 보였습니다. 이는 이 카테고리의 항목들이 다양한 표현 방식(코드 스위칭, 비 코드 스위칭) 간에 
                    의미적 일관성을 잘 유지하고 있음을 시사합니다.</p>
                    
                    <p>반면, <strong>{lowest_category}</strong> 카테고리는 평균 {category_df.iloc[-1]['mean']:.4f}로 가장 낮은 밀집도를 보였습니다.
                    이는 이 카테고리의 항목들이 서로 다른 표현 방식 간에 의미적 차이가 크게 나타날 수 있음을 의미합니다.</p>
                """
            
            html_summary += """
                </div>
            </body>
            </html>
            """
            with open(os.path.join(self.output_dir, "category_density_summary.html"), 'w', encoding='utf-8') as f:
                f.write(html_summary)
            
            print(f"카테고리별 밀집도 분석 요약 저장 완료: {self.output_dir}/category_density_summary.html")
        
        except Exception as e:
            print(f"카테고리별 밀집도 시각화 생성 중 오류 발생: {e}")
            print("이 시각화는 건너뜁니다.")
        
        return {
            'category_stats': category_stats,
            'item_scores': item_cluster_scores
        }

    def run_all_analyses(self):
        self.reduce_dimensions()
        self.calculate_all_similarity_metrics()
        self.plot_comprehensive_similarity_comparison()
        self.plot_item_clusters()
        self.plot_enhanced_clusters()
        
        # 카테고리 관련 추가 분석
        if hasattr(self, 'item_categories') and self.item_categories:
            self.plot_category_clusters()
            self.analyze_category_distribution()
            self.analyze_code_switching_effect()
        
        self.plot_item_similarity_heatmap()
        self.analyze_item_similarity_distribution()
        
        # 요약 보고서 생성
        self._generate_comprehensive_report()
        
        return {
            "similarity_results": self.similarity_results,
            "pair_averages": self.pair_averages,
            "reduced_embeddings": self.reduced_embeddings
        }
    
    def _generate_comprehensive_report(self):
        """포괄적 분석 결과 요약 보고서 생성"""
        if not hasattr(self, 'pair_averages'):
            self.calculate_all_similarity_metrics()
        df = pd.DataFrame(self.similarity_results)
        
        # 보고서 텍스트 생성
        report = f"""# 코드 스위칭 및 다국어 임베딩 포괄적 분석 보고서

    ## 모델 정보
    - 모델: {self.model_name}
    - 분석 항목 수: {len(set(self.item_indices))}

    ## 표현 유형 간 유사도 분석
    ### 모든 표현 유형 쌍 간의 평균 유사도
    """
        
        # 평균 유사도 표 생성
        report += "| 표현 유형 쌍 | 평균 유사도 | 표준편차 | 최소값 | 최대값 | 데이터 수 |\n"
        report += "|--------------|------------|----------|--------|--------|----------|\n"
        
        # 유사도 내림차순 정렬
        sorted_pairs = sorted(self.pair_averages.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        for pair_name, stats in sorted_pairs:
            type1, type2 = stats['type1'], stats['type2']
            type1_desc = self.type_descriptions.get(type1, type1)
            type2_desc = self.type_descriptions.get(type2, type2)
            report += f"| {type1_desc} → {type2_desc} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | {stats['count']} |\n"
        
        # 항목 클러스터 분석
        item_cluster_scores = {}
        for item_idx in sorted(set(df['item_idx'])):
            item_sims = df[df['item_idx'] == item_idx]['similarity'].values
            item_cluster_scores[item_idx] = np.mean(item_sims)
        
        # 상위/하위 항목 분석
        sorted_items = sorted(item_cluster_scores.items(), key=lambda x: x[1], reverse=True)
        top5_items = sorted_items[:5]
        bottom5_items = sorted_items[-5:]
        
        report += f"""
    ## 항목 클러스터 분석

    ### 항목 클러스터 밀집도 통계
    - 평균 밀집도: {np.mean(list(item_cluster_scores.values())):.4f}
    - 표준편차: {np.std(list(item_cluster_scores.values())):.4f}
    - 최소값: {np.min(list(item_cluster_scores.values())):.4f}
    - 최대값: {np.max(list(item_cluster_scores.values())):.4f}

    ### 밀집도 상위 5개 항목
    """
        
        report += "| 항목 인덱스 | 클러스터 밀집도 |\n"
        report += "|------------|----------------|\n"
        for item_idx, score in top5_items:
            report += f"| {item_idx} | {score:.4f} |\n"
        
        report += f"""
    ### 밀집도 하위 5개 항목
    """
        
        report += "| 항목 인덱스 | 클러스터 밀집도 |\n"
        report += "|------------|----------------|\n"
        for item_idx, score in bottom5_items:
            report += f"| {item_idx} | {score:.4f} |\n"
        
        report += f"""
    ## 코드 스위칭 효과 분석

    ### 코드 스위칭과 베이스라인 비교
    - EtoK → English 평균 유사도: {self.pair_averages.get('EtoK-English', {}).get('mean', 'N/A')}
    - KtoE → Korean 평균 유사도: {self.pair_averages.get('KtoE-Korean', {}).get('mean', 'N/A')}

    ### 코드 스위칭 간 비교
    - EtoK → KtoE 평균 유사도: {self.pair_averages.get('EtoK-KtoE', {}).get('mean', 'N/A')}

    ### 베이스라인 간 비교
    - English → Korean 평균 유사도: {self.pair_averages.get('English-Korean', {}).get('mean', 'N/A')}

    ## 결론

    1. **코드 스위칭 처리 능력**: 
    - 코드 스위칭과 베이스라인 간의 유사도는 평균적으로 {np.mean([self.pair_averages.get('EtoK-English', {}).get('mean', 0), self.pair_averages.get('KtoE-Korean', {}).get('mean', 0)]):.4f}입니다.
    - {'EtoK→English가 KtoE→Korean보다 높은 유사도를 보임' if self.pair_averages.get('EtoK-English', {}).get('mean', 0) > self.pair_averages.get('KtoE-Korean', {}).get('mean', 0) else 'KtoE→Korean이 EtoK→English보다 높은 유사도를 보임'}

    2. **다국어 임베딩 특성**:
    - 같은 의미의 서로 다른 표현 유형들은 임베딩 공간에서 {"밀집된" if np.mean(list(item_cluster_scores.values())) > 0.7 else "비교적 분산된"} 클러스터를 형성합니다.
    - 평균 클러스터 밀집도: {np.mean(list(item_cluster_scores.values())):.4f}

    3. **시각화 결과**:
    다음 파일에서 시각화 결과를 확인할 수 있습니다:
    - 표현 유형 간 유사도 비교: {self.output_dir}/comprehensive_similarity_comparison.html
    - 항목별 클러스터 시각화: {self.output_dir}/item_clusters.html
    - 표현 유형 간 유사도 히트맵: {self.output_dir}/type_similarity_heatmap.html
    - 표현 유형 쌍별 유사도 분포: {self.output_dir}/type_similarity_distribution.html
    - 항목별 클러스터 밀집도: {self.output_dir}/item_cluster_scores.html
    """
        
        # 보고서 저장 (HTML && Markdown 동시 생성)
        report_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>코드 스위칭 분석 보고서</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; }}
                th {{ background-color: #f2f2f2; text-align: left; }}
                .high-score {{ background-color: rgba(0, 255, 0, 0.2); }}
                .medium-score {{ background-color: rgba(255, 255, 0, 0.2); }}
                .low-score {{ background-color: rgba(255, 0, 0, 0.2); }}
                .score-cell {{ font-weight: bold; text-align: center; }}
            </style>
        </head>
        <body>
            <h1>코드 스위칭 및 다국어 임베딩 포괄적 분석 보고서</h1>
            <div style="display: none;">{report}</div>
        </body>
        </html>
        """
        
        # 저장
        with open(os.path.join(self.output_dir, "comprehensive_analysis_report.md"), 'w', encoding='utf-8') as f:
            f.write(report)
        
        with open(os.path.join(self.output_dir, "comprehensive_analysis_report.html"), 'w', encoding='utf-8') as f:
            f.write(report_html)
        
        print(f"포괄적 분석 보고서 저장 완료: {self.output_dir}/comprehensive_analysis_report.md")


