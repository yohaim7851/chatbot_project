import os
import json

def save_state(current_path, state):
    if not os.path.exists(f"{current_path}/data"):
        os.makedirs(f"{current_path}/data")
    
    state_dict = {}

    messages = [(m.__class__.__name__, m.content) for m in state["messages"]]
    state_dict["messages"] = messages
    state_dict["task_history"] = [task.to_dict() for task in state.get("task_history", [])]

    # references
    references = state.get("references", {"queries": [], "docs": []})
    state_dict["references"] = {
        "queries": references["queries"], 
        "docs": [doc.metadata for doc in references["docs"]]
    }
    
    with open(f"{current_path}/data/state.json", "w", encoding='utf-8') as f:
        json.dump(state_dict, f, indent=4, ensure_ascii=False)

def get_target(current_path):
    target = '아직 설정된 목표가 없습니다.'

    if os.path.exists(f'{current_path}/data/target.md'):
        with open(f'{current_path}/data/target.md', 'r', encoding='utf-8') as f:
            target = f.read()

    return target

def save_target(current_path, target):
    if not os.path.exists(f'{current_path}/data'):
        os.makedirs(f'{current_path}/data')

    with open(f'{current_path}/data/target.md', 'w', encoding='utf-8') as f:
        f.write(target)

    return target

def parse_user_info(text: str) -> dict:
    """
    사용자 입력 텍스트에서 정보 추출

    예시:
    성별: 남성
    키: 180
    몸무게: 84
    목표: 다이어트

    Returns:
        {
            'gender': '남성' or '여성',
            'height': int (cm),
            'weight': int (kg),
            'goal': '다이어트' or '근육증가' or '체중유지' or '건강관리'
        }
    """
    import re

    user_info = {}

    # 성별 추출
    gender_match = re.search(r'성별\s*[:：]\s*(남성|여성|남|여)', text)
    if gender_match:
        gender = gender_match.group(1)
        user_info['gender'] = '남성' if gender in ['남성', '남'] else '여성'

    # 키 추출
    height_match = re.search(r'키\s*[:：]\s*(\d+)', text)
    if height_match:
        user_info['height'] = int(height_match.group(1))

    # 몸무게 추출
    weight_match = re.search(r'몸무게\s*[:：]\s*(\d+)', text)
    if weight_match:
        user_info['weight'] = int(weight_match.group(1))

    # 목표 추출
    goal_match = re.search(r'목표\s*[:：]\s*(다이어트|근육증가|체중유지|건강관리)', text)
    if goal_match:
        user_info['goal'] = goal_match.group(1)

    return user_info
