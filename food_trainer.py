from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from tools import retrieve, web_search, add_web_pages_json_to_chroma, calculate_nutrition_needs

from utils import save_state, get_target, save_target, parse_user_info
from models import Task, FoodAnalysis
from datetime import datetime
import os
import shutil
from food_volume_analyzer import get_analyzer

filename = os.path.basename(__file__)
absolute_path = os.path.abspath(__file__)
current_path = os.path.dirname(absolute_path)

llm = ChatOpenAI(model='gpt-4o')

class State(TypedDict):
    messages: List[AnyMessage|str]
    task_history: List[Task]
    references: dict
    diet_plan: dict  # 생성된 식단표
    food_analysis: List[FoodAnalysis]  # 음식 분석 결과
    uploaded_images: List[str]  # 업로드된 이미지 경로
    user_info: dict  # 사용자 정보 (성별, 키, 몸무게, 목표)
    user_request: str  # 사용자 요구사항
    diet_iterations: int  # 식단 재생성 횟수
    last_analyzed_image: str  # 마지막으로 분석한 이미지 경로
    awaiting_satisfaction_response: bool  # 만족도 응답 대기 플래그

def quick_router_node(state: State):
    """Quick router를 노드로 실행 (state 반환)"""
    return state

def quick_router(state: State) -> str:
    """빠른 패턴 매칭으로 단순 요청 분류 (supervisor 부하 감소)"""
    print("\n\n====================QUICK ROUTER====================")

    messages = state.get("messages", [])
    if not messages:
        return "supervisor"

    last_message = messages[-1].content.lower() if hasattr(messages[-1], 'content') else str(messages[-1]).lower()

    # 1. 단순 인사/감사 → communicator 직행
    simple_greetings = ['안녕', '하이', 'hi', 'hello', '고마워', '감사', '좋아', '완벽', '최고', '괜찮아', 'ㄱㅅ', 'ㄳ','안녕하세요']
    if any(word in last_message for word in simple_greetings) and len(last_message) < 20:
        print("[QUICK ROUTER] 단순 대화 감지 → communicator")
        return "communicator"

    # 2. 이미지 분석 요청 → food_analyzer (새 이미지만)
    image_keywords = ['이미지', '사진', '분석']
    uploaded_images = state.get("uploaded_images", [])
    last_analyzed_image = state.get("last_analyzed_image", "")

    if uploaded_images:
        latest_image = uploaded_images[-1]

        # 명시적 분석 요청이고, 새 이미지인 경우만 food_analyzer로
        if any(word in last_message for word in image_keywords):
            if latest_image != last_analyzed_image:
                print(f"[QUICK ROUTER] 새 이미지 분석 요청 → food_analyzer")
                return "food_analyzer"
            else:
                print(f"[QUICK ROUTER] 이미 분석된 이미지 - 질문으로 판단 → communicator")
                return "communicator"

    # 3. 식단 생성 요청 → supervisor (복잡한 판단 필요)
    diet_keywords = ['식단', '메뉴', '추천', '계획', '다이어트', '먹을', '레시피']
    if any(word in last_message for word in diet_keywords):
        print("[QUICK ROUTER] 식단 요청 감지 → supervisor")
        return "supervisor"

    # 4. 검색 필요 → supervisor
    search_keywords = ['알려줘', '뭐야', '방법', '어떻게', '정보', '찾아']
    if any(word in last_message for word in search_keywords):
        print("[QUICK ROUTER] 검색 필요 → supervisor")
        return "supervisor"

    # 5. 기본: communicator (단순 대화)
    print("[QUICK ROUTER] 기본 대화 → communicator")
    return "communicator"

def supervisor(state: State):
    print("\n\n====================SUPERVISOR====================")

    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 운동 및 식단 정보를 제공해야한다는 최종 목표를 염두해두고,
        사용자의 요구를 달성하기 위해 현재 해야할 일이 무엇인지 결정한다.

        ** 사용 가능한 Agent **

        [정보 수집 Agent - 독립적 사용 가능]
        - web_search_agent: 최신 정보, 트렌드, 최근 연구 결과 등을 웹에서 검색
          사용 시기: 최신 다이어트 방법, 최근 운동 트렌드, 새로운 영양 정보

        - vector_search_agent: 저장된 전문 운동/영양 자료에서 검색
          사용 시기: 기본 영양 정보, 운동 방법, 식단 구성 원칙

        * 두 검색 agent는 독립적으로 사용 가능
        * 필요시 web_search 후 supervisor를 통해 vector_search 추가 가능

        [실행 Agent]
        - diet_planner: 구체적인 식단표 생성
          * 사용 시기: 식단 생성 요청이 있을 때 
          * 사용자 정보를 기반으로 식단표 생성 사용자 정보가 없을 시 supervisor에게 사용자 정보 요청 가능

        - food_analyzer_agent: 음식 이미지 분석 (볼륨, 칼로리, 영양소)
          사용 시기: 사용자가 이미지를 업로드했을 때

        [사용자 대화 Agent]
        - communicator: 단순 질문 응답, 진행상황 보고, 피드백 수집
          사용 시기: 검색 없이 답변 가능한 대화, 작업 결과 전달

        ** Agent 선택 가이드 **
        1. 단순 대화/인사 → communicator
        2. 최신 정보 필요 → web_search_agent
        3. 전문 지식 필요 → vector_search_agent
        4. 둘 다 필요 → 하나 실행 후 supervisor가 다음 결정
        5. 식단표 요청 → diet_planner
        6. 이미지 분석 → food_analyzer_agent

        ---------------------------------
        previous_target: {target}
        ---------------------------------
        messages: {messages}
        """
    )

    supervisor_chain = supervisor_system_prompt | llm.with_structured_output(Task)

    messages = state.get("messages", [])

    inputs = {
        "messages": messages,
        'target': get_target(current_path)
    }

    task = supervisor_chain.invoke(inputs)
    task_history = state.get("task_history", [])
    task_history.append(task)

    supervisor_message = AIMessage(f'[supervisor] {task}')
    messages.append(supervisor_message)
    print(supervisor_message.content)

    return {
        "messages": messages,
        "task_history": task_history
    }

def supervisor_router(state:State):
    tasks = state.get('task_history', [])
    if not tasks:
        return "communicator"  # task가 없으면 communicator로
    return tasks[-1].agent

def vector_search_agent(state: State):
    print("\n\n============ VECTOR SEARCH AGENT ============")

    tasks = state.get("task_history", [])

    # Task가 없거나 다른 agent의 task인 경우 새로 생성
    if not tasks or tasks[-1].agent != "vector_search_agent":
        new_task = Task(
            agent="vector_search_agent",
            done=False,
            description="벡터 DB에서 전문 운동/영양 정보를 검색한다.",
            done_at=""
        )
        tasks.append(new_task)
        print("[VECTOR SEARCH] Task 자동 생성")

    task = tasks[-1]

    vector_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로, 
        목표달성에 필요한 정보를 벡터 검색을 통해 찾아내는 Agent이다.

        현재 목표 달성을 위해 필요한 정보를 확보하기 위해, 
        다음 내용을 활용해 적절한 벡터 검색을 수행하라. 

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목표: {target}
        """
    )

    # inputs 설정
    mission = task.description
    references = state.get("references", {"queries": [], "docs": []})
    messages = state["messages"]
    target = get_target(current_path)

    inputs = {
        "mission": mission,
        "references": references,
        "messages": messages,
        "target": target
    }

    # LLM과 벡터 검색 모델 연결
    llm_with_retriever = llm.bind_tools([retrieve]) 
    vector_search_chain = vector_search_system_prompt | llm_with_retriever

    # LLM과 벡터 검색 모델 연결
    search_plans = vector_search_chain.invoke(inputs)
    # 검색할 내용 출력
    for tool_call in search_plans.tool_calls:
        print('-----------------------------------', tool_call)
        args = tool_call["args"]

        query = args["query"]
        retrieved_docs = retrieve.invoke(args)
        references["queries"].append(query) 
        references["docs"] += retrieved_docs
    
    unique_docs = []
    unique_page_contents = set()

    for doc in references["docs"]:
        if doc.page_content not in unique_page_contents:
            unique_docs.append(doc)
            unique_page_contents.add(doc.page_content)
    references["docs"] = unique_docs

    # 검색 결과 출력 – 쿼리 출력
    print('Queries:--------------------------')
    queries = references["queries"]
    for query in queries:
        print(query)
    
    # 검색 결과 출력 – 문서 청크 출력
    print('References:--------------------------')
    for doc in references["docs"]:
        print(doc.page_content[:100])
        print('--------------------------')

    # task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # 새로운 task 추가
    new_task = Task(
        agent="communicator",
        done=False,
        description="AI팀의 진행상황을 사용자에게 보고하고, 사용자의 의견을 파악하기 위한 대화를 나눈다",
        done_at=""
    )
    tasks.append(new_task)

    # vector search agent의 작업후기를 메시지로 생성
    msg_str = f"[VECTOR SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    message = AIMessage(msg_str)
    print(msg_str)

    messages.append(message)
    # state 업데이트
    return {
        "messages": messages,
        "task_history": tasks,
        "references": references
    }

def web_search_agent(state: State):
    print("\n\n============ WEB SEARCH AGENT ============")

    tasks = state.get("task_history", [])

    # Task가 없거나 다른 agent의 task인 경우 새로 생성
    if not tasks or tasks[-1].agent != "web_search_agent":
        new_task = Task(
            agent="web_search_agent",
            done=False,
            description="웹에서 최신 운동/영양 정보를 검색한다.",
            done_at=""
        )
        tasks.append(new_task)
        print("[WEB SEARCH] Task 자동 생성")

    task = tasks[-1]
    
    #③ 시스템 프롬프트 정의
    web_search_system_prompt = PromptTemplate.from_template(
        """
        너는 다른 AI Agent 들이 수행한 작업을 바탕으로, 
        목표 달성에 필요한 정보를 웹 검색을 통해 찾아내는 Web Search Agent이다.

        현재 부족한 정보를 검색하고, 복합적인 질문은 나눠서 검색하라.

        - 검색 목적: {mission}
        --------------------------------
        - 과거 검색 내용: {references}
        --------------------------------
        - 이전 대화 내용: {messages}
        --------------------------------
        - 목표(target): {target}
        --------------------------------
        - 현재 시각 : {current_time}
        """
    )
    
    #④ 기존 대화 내용 가져오기
    messages = state.get("messages", [])

    #⑤ 인풋 자료 준비하기
    inputs = {
        "mission": task.description,
        "references": state.get("references", {"queries": [], "docs": []}),
        "messages": messages,
        "target": get_target(current_path),
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    #⑥ LLM과 웹 검색 모델 연결
    llm_with_web_search = llm.bind_tools([web_search])

    #⑦ 시스템 프롬프트와 모델을 연결
    web_search_chain = web_search_system_prompt | llm_with_web_search

    #⑧ 웹 검색 tool_calls 가져오기
    search_plans = web_search_chain.invoke(inputs)

    #⑨ 어떤 내용을 검색했는지 담아두기
    queries = []

    #⑩ 검색 계획(tool_calls)에 따라 검색하기
    for tool_call in search_plans.tool_calls:
        print('-------- web search --------', tool_call)
        args = tool_call["args"]
        
        queries.append(args["query"])

        # (10)  검색 결과를 chroma에 추가
        _, json_path = web_search.invoke(args)
        print('json_path:', json_path)

        # (10)  JSON 파일을 chroma에 추가
        add_web_pages_json_to_chroma(json_path)

    # Task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Communicator로 전환 (강제 vector_search 제거)
    new_task = Task(
        agent="communicator",
        done=False,
        description="웹 검색 결과를 사용자에게 보고한다.",
        done_at=""
    )
    tasks.append(new_task)

    # 작업 후기 메시지
    msg_str = f"[WEB SEARCH AGENT] 웹 검색 완료: {queries}"
    messages.append(AIMessage(msg_str))
    print(msg_str)

    # State 업데이트
    return {
        "messages": messages,
        "task_history": tasks
    }


def diet_planner(state: State):
    print("\n\n====================DIET PLANNER====================")

    tasks = state.get("task_history", [])

    # Task가 없거나 다른 agent의 task인 경우 새로 생성
    if not tasks or tasks[-1].agent != "diet_planner":
        new_task = Task(
            agent="diet_planner",
            done=False,
            description="사용자 정보와 목표에 맞는 맞춤 식단표를 생성한다.",
            done_at=""
        )
        tasks.append(new_task)
        print("[DIET PLANNER] Task 자동 생성")

    task = tasks[-1]

    # 사용자 정보 기반 영양 계산
    user_info = state.get("user_info", {})
    nutrition_needs = {}

    if user_info and all(k in user_info for k in ['gender', 'height', 'weight', 'goal']):
        nutrition_needs = calculate_nutrition_needs(user_info)
        print(f"[DIET PLANNER] 영양 계산 완료: {nutrition_needs}")
    else:
        print("[DIET PLANNER] 사용자 정보 부족으로 기본 영양 계산 사용")
        nutrition_needs = {
            'calories': 2000,
            'protein': 100,
            'carbs': 250,
            'fat': 55,
            'bmr': 1600,
            'tdee': 2000
        }

    # 사용자 목표에 따른 맞춤형 식단 지침
    goal = user_info.get('goal', '체중유지')

    if goal == '다이어트':
        goal_specific_guide = """
        ** 다이어트 모드 특별 지침 **
        - 칼로리를 TDEE의 80%로 제한 (칼로리 부족 유도)
        - 단백질 비율을 높여 근손실 방지 (30%)
        - 저칼로리 고단백 음식 위주 (닭가슴살, 두부, 생선, 채소)
        - 포만감 높은 식이섬유 음식 포함 (현미, 채소, 버섯)
        - 간식은 견과류, 과일, 그릭요거트 등 건강한 옵션
        - 당 함량이 높은 음식 제외
        """
    elif goal == '근육증가':
        goal_specific_guide = """
        ** 근육증가 모드 특별 지침 **
        - 칼로리를 TDEE의 115%로 증가 (칼로리 잉여 유도)
        - 단백질 비율 30%, 탄수화물 50% (근성장 최적화)
        - 고단백 음식 필수 (닭가슴살, 소고기, 계란, 유제품)
        - 운동 후 탄수화물 섭취 강조 (고구마, 현미, 바나나)
        - 간식으로 프로틴 쉐이크, 견과류, 요거트 포함
        - 하루 5-6끼 소량 다회 식사 권장
        """
    else:  # 체중유지 또는 건강관리
        goal_specific_guide = """
        ** 체중유지/건강관리 모드 특별 지침 **
        - 칼로리를 TDEE와 동일하게 유지
        - 균형 잡힌 영양소 비율 (탄수화물 50%, 단백질 25%, 지방 25%)
        - 다양한 음식군을 골고루 섭취
        - 한식 위주의 건강한 식단 구성
        - 신선한 채소와 과일 충분히 포함
        - 규칙적인 식사 시간 유지
        """

    # 시스템 프롬프트 정의
    diet_planner_system_prompt = PromptTemplate.from_template(
        """
        너는 운동 식단 트레이너 AI팀의 식단 설계 전문가(Diet Planner)로서,
        사용자의 목표와 사용자 정보를 종합하여 다이어트 식단에 대한 정보를 제공한다.

        ## 작성 지침
        1. 하루 식단을 아침 점심 저녁 순으로 작성
        2. 각 끼니마다 구체적인 음식명과 분량 명시
        3. 칼로리와 주요 영양소(탄수화물, 단백질, 지방) 표시
        4. 실현 가능하고 한국 음식 위주로 구성
        5. 사용자의 개인 정보(목표, 건강 상태, 선호도)를 최대한 반영
        6. 사용자가 요청한 사항(예: 단백질 식단 위주, 업로드한 이미지에 부족한 영양소 위주)을 최대한 반영
        7. **반드시 아래 일일 영양 목표를 준수하여 식단을 작성하라**
        8. 이모티콘(😊, 👍 등)은 절대 사용하지 말 것

        {goal_guide}

        ## 사용자 요구사항
        {user_request}

        ## 사용자 정보
        {user_info}

        ## 일일 영양 목표 (과학적으로 계산된 값)
        - 기초대사량(BMR): {bmr} kcal
        - 활동대사량(TDEE): {tdee} kcal
        - 목표 칼로리: {calories} kcal
        - 목표 단백질: {protein} g
        - 목표 탄수화물: {carbs} g
        - 목표 지방: {fat} g

        ## 검색된 식단 정보
        {references}

        ## 이전 대화 내용
        {messages}

        ## 출력 형식
        마크다운 형식으로 표를 사용하여 보기 좋게 작성하라.
        하루 식단을 아침, 점심, 저녁, 간식을 구분하여 작성하고,
        일일 총 칼로리와 영양소 합계를 명시하라.
        목표 대비 실제 섭취량을 비교하여 표시하라.
        """
    )

    # 입력 데이터 준비
    messages = state.get("messages", [])
    target = get_target(current_path)
    user_request = state.get("user_request", "")
    references = state.get("references", {"queries": [], "docs": []})

    # references의 docs를 문자열로 변환
    reference_text = ""
    if references.get("docs"):
        reference_text = "\n\n".join([
            f"[출처: {doc.metadata.get('source', 'Unknown')}]\n{doc.page_content[:500]}..."
            for doc in references["docs"][:10]  # 상위 10개로 증가
        ])
    else:
        reference_text = "검색된 참고 자료가 없습니다. 일반적인 건강 식단 원칙을 적용합니다."

    inputs = {
        "user_request": user_request,
        "target": target,
        "goal_guide": goal_specific_guide,
        "user_info": f"성별: {user_info.get('gender', '미설정')}, 키: {user_info.get('height', '미설정')}cm, 몸무게: {user_info.get('weight', '미설정')}kg, 목표: {user_info.get('goal', '미설정')}",
        "bmr": nutrition_needs.get('bmr', 1600),
        "tdee": nutrition_needs.get('tdee', 2000),
        "calories": nutrition_needs.get('calories', 2000),
        "protein": nutrition_needs.get('protein', 100),
        "carbs": nutrition_needs.get('carbs', 250),
        "fat": nutrition_needs.get('fat', 55),
        "references": reference_text,
        "messages": messages[-10:] if len(messages) > 10 else messages  # 최근 10개 메시지만 사용
    }

    # LLM 체인 생성
    diet_planner_chain = diet_planner_system_prompt | llm | StrOutputParser()

    # 식단 생성
    print("\n[DIET PLANNER] 식단표를 생성하고 있습니다...\n")

    diet_plan_text = ""
    for chunk in diet_planner_chain.stream(inputs):
        print(chunk, end='')
        diet_plan_text += chunk

    print("\n")

    # 생성된 식단을 state에 저장
    diet_plan = {
        "plan": diet_plan_text,
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "user_request": user_request,
        "nutrition_needs": nutrition_needs  # 영양 목표 저장
    }

    # 식단을 파일로도 저장 - data 디렉토리 생성 확인
    data_dir = os.path.join(current_path, "data")
    os.makedirs(data_dir, exist_ok=True)

    diet_plan_path = os.path.join(data_dir, f"diet_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    try:
        with open(diet_plan_path, 'w', encoding='utf-8') as f:
            f.write(f"# 식단표\n\n")
            f.write(f"**생성일시**: {diet_plan['created_at']}\n\n")
            if user_request:
                f.write(f"**사용자 요구사항**: {user_request}\n\n")
            f.write(f"---\n\n{diet_plan_text}")
        print(f"[DIET PLANNER] 식단표가 {diet_plan_path}에 저장되었습니다.")
    except Exception as e:
        print(f"[DIET PLANNER WARNING] 파일 저장 실패: {e}")

    # task 완료 처리
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # communicator로 전달할 새 task 추가
    new_task = Task(
        agent="communicator",
        done=False,
        description="생성된 식단표를 사용자에게 보고하고, 피드백을 받는다.",
        done_at=""
    )
    tasks.append(new_task)

    # 메시지 추가
    msg_str = f"[DIET PLANNER] 일주일 식단표 생성 완료"
    messages.append(AIMessage(msg_str))
    print(msg_str)

    return {
        "messages": messages,
        "task_history": tasks,
        "diet_plan": diet_plan
    }


def food_analyzer_agent(state: State):
    print("\n\n====================FOOD ANALYZER AGENT====================")

    tasks = state.get("task_history", [])

    # quick_router에서 직접 호출될 경우 task가 없을 수 있음 - task 생성
    if not tasks or tasks[-1].agent != "food_analyzer_agent":
        new_task = Task(
            agent="food_analyzer_agent",
            done=False,
            description="업로드된 음식 이미지를 분석하여 영양 정보를 제공한다.",
            done_at=""
        )
        tasks.append(new_task)
        print("[FOOD ANALYZER] Task 자동 생성")

    messages = state.get("messages", [])
    uploaded_images = state.get("uploaded_images", [])

    # 업로드된 이미지 확인
    if not uploaded_images:
        msg = "[FOOD ANALYZER] 분석할 이미지가 없습니다. 먼저 이미지를 업로드해주세요."
        messages.append(AIMessage(msg))
        print(msg)

        tasks[-1].done = True
        tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        return {
            "messages": messages,
            "task_history": tasks
        }

    # 가장 최근 이미지 분석
    image_path = uploaded_images[-1]
    print(f"[*] 분석할 이미지: {image_path}")

    try:
        print(f"[FOOD ANALYZER] 이미지 분석을 시작합니다...")

        # 음식 볼륨 분석기 초기화 및 분석
        analyzer = get_analyzer()
        print(f"[FOOD ANALYZER] 분석기 초기화 완료. 볼륨 추정 중...")

        result = analyzer.estimate_volume(
            image_path=image_path,
            plate_diameter_cm=26.0
        )

        print(f"[FOOD ANALYZER] 볼륨 추정 완료. 영양 정보 분석 중...")

        # FoodAnalysis 객체 생성
        analysis = FoodAnalysis(
            food_name=result.get('food_type', '알 수 없는 음식'),
            volume_ml=result.get('volume_ml', 0.0),
            weight_grams=result.get('weight_grams', 0.0),
            calories=result.get('calories', 0),
            protein=result.get('protein', 0.0),
            carbs=result.get('carbs', 0.0),
            fat=result.get('fat', 0.0),
            confidence=result.get('confidence', 0.0),
            image_path=image_path,
            analyzed_at=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        # State 업데이트
        food_analysis_list = state.get("food_analysis", [])
        food_analysis_list.append(analysis)

        # 영양 목표와 비교
        diet_plan = state.get("diet_plan", {})
        nutrition_needs = diet_plan.get("nutrition_needs", {})

        # 결과 메시지 생성
        result_msg = f"""[FOOD ANALYZER] 분석 완료!

    ** {analysis.food_name} **
    측정 정보:
    - 볼륨: {analysis.volume_ml:.1f} mL
    - 무게: {analysis.weight_grams:.1f} g

    영양 정보:
    - 칼로리: {analysis.calories} kcal
    - 단백질: {analysis.protein:.1f} g
    - 탄수화물: {analysis.carbs:.1f} g
    - 지방: {analysis.fat:.1f} g

    신뢰도: {analysis.confidence:.0%}
    """

        # 영양 목표가 있으면 비교 분석 추가
        if nutrition_needs and nutrition_needs.get('calories'):
            cal_target = nutrition_needs.get('calories', 0)
            protein_target = nutrition_needs.get('protein', 0)
            carbs_target = nutrition_needs.get('carbs', 0)
            fat_target = nutrition_needs.get('fat', 0)

            # 일일 목표 대비 퍼센트 계산
            cal_percent = (analysis.calories / cal_target * 100) if cal_target > 0 else 0
            protein_percent = (analysis.protein / protein_target * 100) if protein_target > 0 else 0
            carbs_percent = (analysis.carbs / carbs_target * 100) if carbs_target > 0 else 0
            fat_percent = (analysis.fat / fat_target * 100) if fat_target > 0 else 0

            result_msg += f"""
    일일 목표 대비 섭취량:
    - 칼로리: {analysis.calories}/{cal_target} kcal ({cal_percent:.1f}%)
    - 단백질: {analysis.protein:.1f}/{protein_target} g ({protein_percent:.1f}%)
    - 탄수화물: {analysis.carbs:.1f}/{carbs_target} g ({carbs_percent:.1f}%)
    - 지방: {analysis.fat:.1f}/{fat_target} g ({fat_percent:.1f}%)

    권장사항:
    """
            # 부족한 영양소 피드백
            if protein_percent < 80:
                result_msg += f"- 단백질이 부족합니다. 오늘 {protein_target - analysis.protein:.1f}g 더 섭취가 필요합니다.\n    "
            if carbs_percent < 80:
                result_msg += f"- 탄수화물이 부족합니다. 오늘 {carbs_target - analysis.carbs:.1f}g 더 섭취가 필요합니다.\n    "
            if cal_percent < 80:
                result_msg += f"- 칼로리가 부족합니다. 오늘 {cal_target - analysis.calories}kcal 더 섭취가 필요합니다.\n    "

            # 초과한 영양소 피드백
            if protein_percent > 120:
                result_msg += f"- 단백질이 과다합니다. 목표보다 {analysis.protein - protein_target:.1f}g 초과했습니다.\n    "
            if carbs_percent > 120:
                result_msg += f"- 탄수화물이 과다합니다. 목표보다 {analysis.carbs - carbs_target:.1f}g 초과했습니다.\n    "
            if fat_percent > 120:
                result_msg += f"- 지방이 과다합니다. 목표보다 {analysis.fat - fat_target:.1f}g 초과했습니다.\n    "
            if cal_percent > 120:
                result_msg += f"- 칼로리가 과다합니다. 목표보다 {analysis.calories - cal_target}kcal 초과했습니다.\n    "

        messages.append(AIMessage(result_msg))
        print(result_msg)

        # Task 완료 처리
        tasks[-1].done = True
        tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Communicator로 전환
        new_task = Task(
            agent="communicator",
            done=False,
            description="음식 분석 결과를 사용자에게 보고하고 추가 요청을 확인한다.",
            done_at=""
        )
        tasks.append(new_task)

        # 분석한 이미지 경로 저장 (재분석 방지)
        return {
            "messages": messages,
            "task_history": tasks,
            "food_analysis": food_analysis_list,
            "last_analyzed_image": image_path
        }

    except FileNotFoundError:
        error_msg = f"[FOOD ANALYZER ERROR] 이미지 파일을 찾을 수 없습니다: {image_path}"
        messages.append(AIMessage(error_msg))
        print(error_msg)

        tasks[-1].done = True
        tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Communicator로 전환하여 사용자에게 재업로드 요청
        new_task = Task(
            agent="communicator",
            done=False,
            description="이미지 파일 오류를 사용자에게 알리고 재업로드를 요청한다.",
            done_at=""
        )
        tasks.append(new_task)

        return {
            "messages": messages,
            "task_history": tasks
        }

    except ValueError as e:
        error_msg = f"[FOOD ANALYZER ERROR] 이미지 형식 오류: {str(e)}"
        messages.append(AIMessage(error_msg))
        print(error_msg)

        tasks[-1].done = True
        tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        new_task = Task(
            agent="communicator",
            done=False,
            description="이미지 형식 오류를 사용자에게 알리고 올바른 형식으로 재업로드를 요청한다.",
            done_at=""
        )
        tasks.append(new_task)

        return {
            "messages": messages,
            "task_history": tasks
        }

    except Exception as e:
        error_msg = f"[FOOD ANALYZER ERROR] 분석 중 예상치 못한 오류 발생: {str(e)}\n다시 시도해주세요."
        messages.append(AIMessage(error_msg))
        print(error_msg)
        import traceback
        print(traceback.format_exc())  # 디버깅용 상세 에러 출력

        tasks[-1].done = True
        tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Communicator로 전환
        new_task = Task(
            agent="communicator",
            done=False,
            description="분석 오류를 사용자에게 알리고 다른 이미지로 재시도를 권장한다.",
            done_at=""
        )
        tasks.append(new_task)

        return {
            "messages": messages,
            "task_history": tasks
        }


def nutrition_check_node(state: State):
    """음식 분석 후 영양 상태 평가 및 user_request 업데이트"""
    print("\n\n====================NUTRITION CHECK NODE====================")

    food_analysis = state.get("food_analysis", [])
    diet_plan = state.get("diet_plan", {})
    nutrition_needs = diet_plan.get("nutrition_needs", {})

    # 영양 목표가 없으면 그냥 통과
    if not nutrition_needs or not nutrition_needs.get('calories'):
        print("[NUTRITION CHECK] 영양 목표 없음")
        return state

    # 최근 음식 분석 결과
    if not food_analysis:
        print("[NUTRITION CHECK] 분석 결과 없음")
        return state

    latest = food_analysis[-1]

    # 영양소 비율 계산
    cal_target = nutrition_needs.get('calories', 2000)
    protein_target = nutrition_needs.get('protein', 100)

    cal_percent = (latest.calories / cal_target * 100) if cal_target > 0 else 0
    protein_percent = (latest.protein / protein_target * 100) if protein_target > 0 else 0

    # 심각한 부족 (50% 미만) → 보충 식단 제안을 위한 user_request 업데이트
    if cal_percent < 50 or protein_percent < 50:
        print(f"[NUTRITION CHECK] 심각한 영양 부족 감지 (칼로리: {cal_percent:.1f}%, 단백질: {protein_percent:.1f}%)")

        # 부족한 영양소 정보를 user_request에 추가
        shortage_info = f"현재 섭취: 칼로리 {latest.calories}kcal, 단백질 {latest.protein:.1f}g. "
        shortage_info += f"부족분: 칼로리 {cal_target - latest.calories}kcal, 단백질 {protein_target - latest.protein:.1f}g 보충 필요."

        state["user_request"] = f"음식 분석 결과 영양 부족으로 보충 식단 필요. {shortage_info}"
    else:
        print(f"[NUTRITION CHECK] 영양 상태 양호 (칼로리: {cal_percent:.1f}%, 단백질: {protein_percent:.1f}%)")

    return state

def nutrition_router(state: State) -> str:
    """영양 상태에 따라 다음 노드 결정"""
    food_analysis = state.get("food_analysis", [])
    diet_plan = state.get("diet_plan", {})
    nutrition_needs = diet_plan.get("nutrition_needs", {})

    # 영양 목표가 없으면 communicator로
    if not nutrition_needs or not nutrition_needs.get('calories'):
        print("[NUTRITION ROUTER] 영양 목표 없음 → communicator")
        return "communicator"

    # 최근 음식 분석 결과
    if not food_analysis:
        print("[NUTRITION ROUTER] 분석 결과 없음 → communicator")
        return "communicator"

    latest = food_analysis[-1]

    # 영양소 비율 계산
    cal_target = nutrition_needs.get('calories', 2000)
    protein_target = nutrition_needs.get('protein', 100)

    cal_percent = (latest.calories / cal_target * 100) if cal_target > 0 else 0
    protein_percent = (latest.protein / protein_target * 100) if protein_target > 0 else 0

    # 심각한 부족 (50% 미만) → 보충 식단 제안
    if cal_percent < 50 or protein_percent < 50:
        print(f"[NUTRITION ROUTER] → diet_planner")
        return "diet_planner"

    # 정상 범위 → 피드백만 제공
    print(f"[NUTRITION ROUTER] → communicator")
    return "communicator"

def satisfaction_check(state: State) -> str:
    """식단 생성 후 사용자 만족도 체크 및 재생성 판단"""
    print("\n\n====================SATISFACTION CHECK====================")

    messages = state.get("messages", [])
    iterations = state.get("diet_iterations", 0)
    awaiting_response = state.get("awaiting_satisfaction_response", False)

    MAX_ITERATIONS = 3

    # 최대 반복 횟수 초과
    if iterations >= MAX_ITERATIONS:
        print(f"[SATISFACTION CHECK] 최대 재생성 횟수 도달 ({iterations}회) → END")
        messages.append(AIMessage("최대 재생성 횟수에 도달했습니다. 더 구체적인 요구사항을 말씀해주시면 새로운 식단을 생성해드리겠습니다."))
        state["awaiting_satisfaction_response"] = False
        return "end"

    # 식단이 방금 생성되었는지 확인
    diet_plan = state.get("diet_plan", {})
    if not diet_plan.get("plan"):
        print("[SATISFACTION CHECK] 식단 없음 → communicator")
        return "communicator"

    # 마지막 메시지가 diet_planner에서 온 것인지 확인 (만족도 질문 전송)
    if len(messages) > 0 and not awaiting_response:
        last_msg_content = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

        # 식단표가 방금 생성된 경우 만족도 질문 추가
        if "[DIET PLANNER]" in last_msg_content:
            feedback_msg = AIMessage(
                """
식단표가 생성되었습니다. 만족도를 선택해주세요:

1. 만족 - 이대로 사용하겠습니다
2. 보통 - 괜찮지만 조금 수정이 필요합니다
3. 불만족 - 다시 생성해주세요

번호를 입력하거나 '만족', '보통', '불만족' 중 하나를 입력해주세요.
                """
            )
            messages.append(feedback_msg)
            print(feedback_msg.content)
            # 다음 실행에서 응답 파싱하도록 플래그 설정
            state["awaiting_satisfaction_response"] = True
            return "end"

    # 사용자 응답 파싱 (만족도 질문 후)
    if awaiting_response and len(messages) >= 2:
        user_msg = messages[-1].content.lower() if hasattr(messages[-1], 'content') else str(messages[-1]).lower()

        # 1 또는 "만족" → 종료
        if '1' in user_msg or '만족' in user_msg or 'ok' in user_msg or '좋' in user_msg:
            print("[SATISFACTION CHECK] 만족 → END")
            state["awaiting_satisfaction_response"] = False
            return "end"

        # 2 또는 "보통" → communicator (추가 요청 받기)
        elif '2' in user_msg or '보통' in user_msg:
            print("[SATISFACTION CHECK] 보통 → communicator (추가 요청 대기)")
            msg = AIMessage("어떤 부분을 수정하면 좋을까요? 구체적으로 말씀해주세요.")
            messages.append(msg)
            state["awaiting_satisfaction_response"] = False
            return "communicator"

        # 3 또는 "불만족" → 재생성
        elif '3' in user_msg or '불만족' in user_msg or '불만' in user_msg:
            new_iterations = iterations + 1
            state["diet_iterations"] = new_iterations
            print(f"[SATISFACTION CHECK] 불만족 - 재생성 ({new_iterations}회차) → supervisor")

            # Task 생성
            tasks = state.get("task_history", [])
            new_task = Task(
                agent="diet_planner",
                done=False,
                description=f"식단 재생성 요청 ({new_iterations}회차)",
                done_at=""
            )
            tasks.append(new_task)
            state["awaiting_satisfaction_response"] = False

            return "supervisor"

    # 기본: communicator
    print("[SATISFACTION CHECK] 기본 → communicator")
    return "communicator"

def communicator(state: State):
    print("\n\n====================COMMUNICATOR====================")

    communicator_system_prompt = PromptTemplate.from_template(
        """
            너는 운동 식단 트레이너 AI팀의 커뮤니케이터로서,
            사용자와의 모든 일반적인 대화를 담당한다.

            **주요 역할**:
            1. 단순 질문에 대한 직접 응답
            2. AI 팀의 작업 진행상황 보고
            3. 사용자 피드백 수집
            4. 결과물(식단표, 분석 결과 등) 전달
            5. 다음 단계 안내 및 추가 기능 제안
            6. **이미 분석된 음식 정보를 기반으로 추가 질문에 답변**

            **대화 원칙**:
            - 친절하고 전문적인 톤 유지
            - 필요한 정보만 간결하게 전달
            - 사용자의 다음 요청을 자연스럽게 유도
            - 핵심 정보는 **굵게** 표시

            **상황별 대응**:
            - 식단표 생성 완료 시: 만족도 확인, 수정 요청 안내
            - 음식 분석 완료 시: 건강 조언, 추가 분석 제안
            - **음식 분석 후 추가 질문**: 아래 분석 결과를 활용하여 답변
            - 오류 발생 시: 명확한 해결 방법 제시
            - 일반 대화: 간결하면서도 도움이 되는 답변

            **음식 분석 정보** (질문이 이와 관련된 경우 이 정보 활용):
            {food_analysis_detail}

            현재 목표: {target}
            최근 대화 내역: {messages}
            생성된 식단표: {diet_plan}
        """
    )

    system_chain = communicator_system_prompt | llm

    messages = state['messages']
    diet_plan = state.get('diet_plan', {})
    food_analysis = state.get('food_analysis', [])

    # 식단표 요약
    diet_summary = "없음"
    if diet_plan and diet_plan.get('plan'):
        diet_summary = f"생성됨 ({diet_plan.get('created_at', '시간 미상')})"

    # 음식 분석 상세 정보
    food_analysis_detail = "분석된 음식이 없습니다."
    if food_analysis:
        latest = food_analysis[-1]
        food_analysis_detail = f"""
최근 분석된 음식:
- 음식명: {latest.food_name}
- 칼로리: {latest.calories} kcal
- 단백질: {latest.protein:.1f}g
- 탄수화물: {latest.carbs:.1f}g
- 지방: {latest.fat:.1f}g
- 볼륨: {latest.volume_ml:.1f}mL
- 무게: {latest.weight_grams:.1f}g
"""

    inputs = {
        'messages': messages[-10:] if len(messages) > 10 else messages,  # 최근 10개만
        'target': get_target(current_path),
        'diet_plan': diet_summary,
        'food_analysis_detail': food_analysis_detail
    }

    gathered = None

    print('\nAI\t: ', end='')

    for chunk in system_chain.stream(inputs):
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk

    messages.append(gathered)

    task_history = state.get("task_history", [])

    # Task 자동 생성 (quick_router에서 직접 호출 시)
    if not task_history or task_history[-1].agent != "communicator":
        new_task = Task(
            agent="communicator",
            done=False,
            description="사용자와 대화 진행",
            done_at=""
        )
        task_history.append(new_task)

    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        "messages": messages,
        "task_history": task_history
    }



# Nodes
graph_builder = StateGraph(State)
graph_builder.add_node("quick_router", quick_router_node)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node('web_search_agent', web_search_agent)
graph_builder.add_node("diet_planner", diet_planner)
graph_builder.add_node("food_analyzer_agent", food_analyzer_agent)
graph_builder.add_node("nutrition_check_node", nutrition_check_node)

# Edges
# 시작점: START → quick_router
graph_builder.add_edge(START, "quick_router")

# quick_router에서 조건부 라우팅
graph_builder.add_conditional_edges(
    "quick_router",
    quick_router,
    {
        'communicator': 'communicator',
        'supervisor': 'supervisor',
        'food_analyzer': 'food_analyzer_agent'
    }
)

# Supervisor 라우팅
graph_builder.add_conditional_edges(
    "supervisor",
    supervisor_router,
    {
        'communicator': 'communicator',
        'vector_search_agent': 'vector_search_agent',
        'web_search_agent': 'web_search_agent',
        'diet_planner': 'diet_planner',
        'food_analyzer_agent': 'food_analyzer_agent'
    }
)

# 검색 Agent → communicator
graph_builder.add_edge("web_search_agent", "communicator")
graph_builder.add_edge("vector_search_agent", "communicator")

# food_analyzer → nutrition_check_node → nutrition_router (조건부 식단 생성)
graph_builder.add_edge("food_analyzer_agent", "nutrition_check_node")
graph_builder.add_conditional_edges(
    "nutrition_check_node",
    nutrition_router,
    {
        'diet_planner': 'diet_planner',
        'communicator': 'communicator'
    }
)

# diet_planner → satisfaction_check (만족도 체크)
graph_builder.add_conditional_edges(
    "diet_planner",
    satisfaction_check,
    {
        'communicator': 'communicator',
        'supervisor': 'supervisor',
        'end': END
    }
)

# communicator → 항상 END (사용자 입력 대기)
graph_builder.add_edge("communicator", END)

graph = graph_builder.compile()

graph.get_graph().draw_mermaid_png(
    output_file_path=absolute_path.replace('.py', '.png'),
    max_retries=10,
    retry_delay=2.0
)

state = State(
    messages=[
        SystemMessage(
            f"""
            너희 AI들은 사용자의 요구에 맞는 운동과 식단에 대한 정보를 제공해주는 팀이야.
            사용자가 사용하는 언어로 대화해.

            현재 시간은 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}야.

            """
        )
    ],
    task_history=[],
    references={"queries": [], "docs": []},
    diet_plan={},
    food_analysis=[],
    uploaded_images=[],
    user_request="",
    user_info={},
    diet_iterations=0,
    last_analyzed_image="",
    awaiting_satisfaction_response=False
)

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break

    # 사용자 정보 자동 파싱 및 저장
    parsed_info = parse_user_info(user_input)
    if parsed_info:
        current_user_info = state.get("user_info", {})
        current_user_info.update(parsed_info)
        state["user_info"] = current_user_info
        print(f"\n[INFO] 사용자 정보 업데이트됨: {parsed_info}")
        print(f"[INFO] 현재 정보: 성별={current_user_info.get('gender', '미설정')}, "
              f"키={current_user_info.get('height', '미설정')}cm, "
              f"몸무게={current_user_info.get('weight', '미설정')}kg, "
              f"목표={current_user_info.get('goal', '미설정')}\n")

    # 이미지 업로드 핸들링: !image <path> 형식
    if user_input.startswith('!image '):
        image_path = user_input[7:].strip()

        # 따옴표 제거 (경로에 공백이 있을 경우)
        image_path = image_path.strip('"').strip("'")

        # 파일 존재 확인
        if not os.path.exists(image_path):
            print(f"[ERROR] 이미지 파일을 찾을 수 없습니다: {image_path}")
            print(f"[INFO] 경로를 다시 확인해주세요. 현재 작업 디렉토리: {os.getcwd()}")
            continue

        # 이미지 파일 확장자 확인
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            print(f"[ERROR] 지원되지 않는 이미지 형식입니다.")
            print(f"[INFO] 지원 형식: {', '.join(valid_extensions)}")
            continue

        # uploaded_images 디렉토리에 복사
        upload_dir = os.path.join(current_path, 'uploaded_images')
        os.makedirs(upload_dir, exist_ok=True)

        # 파일 복사
        filename = os.path.basename(image_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{timestamp}_{filename}"
        dest_path = os.path.join(upload_dir, new_filename)

        try:
            shutil.copy2(image_path, dest_path)
            state["uploaded_images"].append(dest_path)
            print(f"\n[SUCCESS] 이미지 업로드 성공!")
            print(f"[INFO] 저장 위치: {dest_path}")
            print(f"\n[INFO] 음식 분석 명령어:")
            print(f"   - '이미지 분석해줘'")
            print(f"   - '음식 영양 정보 알려줘'")
            print(f"   - '칼로리 분석해줘'\n")
        except PermissionError:
            print(f"[ERROR] 파일 접근 권한이 없습니다: {image_path}")
        except Exception as e:
            print(f"[ERROR] 이미지 복사 중 오류가 발생했습니다: {e}")

        continue

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n------------------------MESSAGE COUNT\t', len(state["messages"]))

    save_state(current_path, state)

