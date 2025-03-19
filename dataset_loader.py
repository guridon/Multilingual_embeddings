import json
import os

def load_code_switch_dataset(file_path="code-switch.json"):
    """
    Args:
        file_path (str): 데이터셋 파일 경로
        
    Returns:
        tuple: (all_texts, text_types, item_indices, data)
    """
    print(f"데이터셋 '{file_path}' 로딩 중...")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], dict):
            raise ValueError("데이터셋 형식이 올바르지 않습니다. 리스트 형태의 객체 배열이어야 합니다.")
        
        expected_keys = ["EtoK", "KtoE", "English", "Korean"]
        if not all(key in data[0] for key in expected_keys):
            raise ValueError(f"예상된 키가 누락되었습니다. 필요한 키: {', '.join(expected_keys)}")
        
        etok_texts = [item["EtoK"] for item in data]
        ktoe_texts = [item["KtoE"] for item in data]
        english_texts = [item["English"] for item in data]
        korean_texts = [item["Korean"] for item in data]
        
        all_texts = []
        all_texts.extend(etok_texts)
        all_texts.extend(ktoe_texts)
        all_texts.extend(english_texts)
        all_texts.extend(korean_texts)
        
        text_types = ["EtoK"] * len(etok_texts) + ["KtoE"] * len(ktoe_texts) + \
                     ["English"] * len(english_texts) + ["Korean"] * len(korean_texts)
        
        item_indices = []
        for i in range(len(data)):
            item_indices.extend([i] * 4)  # 각 항목당 4개 문장
        
        print(f"총 {len(all_texts)}개 문장 로드됨 ({len(data)}개 항목 x 4가지 유형)")
        
        return all_texts, text_types, item_indices, data
        
    except FileNotFoundError:
        print(f"경고: {file_path} 파일을 찾을 수 없습니다. 샘플 데이터셋을 생성합니다.")
        return create_sample_dataset(file_path)
    
    except json.JSONDecodeError:
        raise ValueError("JSON 파일 형식이 올바르지 않습니다.")

def create_sample_dataset(file_path="code-switch.json"):
    """  
    Args:
        file_path (str): 저장할 파일 경로
        
    Returns:
        tuple: (all_texts, text_types, item_indices, data)
    """
    sample_data = [
        {
            "EtoK": "Could you explain how 양자 컴퓨팅 works?",
            "KtoE": "quantum computing 기술은 매우 흥미롭습니다.",
            "English": "Could you explain how quantum computing works?",
            "Korean": "양자 컴퓨팅이 어떻게 작동하는지 설명해주세요."
        },
        {
            "EtoK": "AI models like 생성적 적대 신경망 are revolutionary.",
            "KtoE": "생성적 적대 신경망 is a type of GANs.",
            "English": "AI models like Generative Adversarial Networks are revolutionary.",
            "Korean": "생성적 적대 신경망과 같은 AI 모델은 혁명적입니다."
        }
    ]

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else '.', exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    etok_texts = [item["EtoK"] for item in sample_data]
    ktoe_texts = [item["KtoE"] for item in sample_data]
    english_texts = [item["English"] for item in sample_data]
    korean_texts = [item["Korean"] for item in sample_data]
    
    all_texts = []
    all_texts.extend(etok_texts)
    all_texts.extend(ktoe_texts)
    all_texts.extend(english_texts)
    all_texts.extend(korean_texts)
    
    text_types = ["EtoK"] * len(etok_texts) + ["KtoE"] * len(ktoe_texts) + \
                 ["English"] * len(english_texts) + ["Korean"] * len(korean_texts)
    
    item_indices = []
    for i in range(len(sample_data)):
        item_indices.extend([i] * 4)
    
    print(f"샘플 데이터셋 생성 완료: {len(all_texts)}개 문장 ({len(sample_data)}개 항목 x 4가지 유형)")
    return all_texts, text_types, item_indices, sample_data
