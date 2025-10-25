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