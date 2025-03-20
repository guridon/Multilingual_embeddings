import json

# category.json에서 데이터 로드
with open('category.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
    print(f"category.json 로드 완료: {len(data)}개 항목")

# 인덱스 기반 매핑 생성 (문자열 키)
category_map = {}
for item in data:
    category_map[str(item['idx'])] = item['category']

# 매핑 데이터 저장
with open('category_map.json', 'w', encoding='utf-8') as f:
    json.dump(category_map, f, ensure_ascii=False, indent=2)
