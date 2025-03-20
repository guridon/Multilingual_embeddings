import json
import re

# JS 파일 읽기
with open('category.js', 'r', encoding='utf-8') as f:
    js_content = f.read()

# JavaScript 변수 선언 부분 제거 ('const categorizedData = ' 시작 부분)
json_content = js_content.replace('const categorizedData = ', '').strip()

# 끝부분의 세미콜론 제거
if json_content.endswith('];'):
    json_content = json_content[:-2] + ']'  # 세미콜론만 제거하고 닫는 괄호는 유지

# 주석 제거 (// 로 시작하는 줄 제거)
json_lines = []
for line in json_content.split('\n'):
    if '//' not in line:
        json_lines.append(line)
json_content = '\n'.join(json_lines)

# 속성 이름에 따옴표 추가
# idx: 0 -> "idx": 0
json_content = re.sub(r'(\w+):', r'"\1":', json_content)

try:
    # JSON으로 파싱 테스트
    data = json.loads(json_content)
    
    # JSON 파일로 저장
    with open('category.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print("JSON 변환 완료: category.json 파일 생성됨")
    
    # 카테고리 맵 생성 (인덱스 → 카테고리)
    category_map = {}
    for item in data:
        category_map[str(item['idx'])] = item['category']
    
    # 카테고리 맵 저장
    with open('category_map.json', 'w', encoding='utf-8') as f:
        json.dump(category_map, f, ensure_ascii=False, indent=2)
    
    print("카테고리 맵 생성 완료: category_map.json 파일 생성됨")
    
except json.JSONDecodeError as e:
    print(f"JSON 변환 중 오류 발생: {e}")
    # 디버깅을 위해 변환된 내용의 일부 출력
    print("변환된 내용 일부:")
    print(json_content[:200] + "...")
