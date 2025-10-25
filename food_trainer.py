from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from typing_extensions import TypedDict
from typing import List
from tools import retrieve, web_search, add_web_pages_json_to_chroma

from utils import save_state, get_target, save_target
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
    ai_recommendation: str

def supervisor(state: State):
    print("\n\n====================SUPERVISOR====================")

    supervisor_system_prompt = PromptTemplate.from_template(
        """
        너는 AI 팀의 supervisor로서 AI 팀의 작업을 관리하고 지도한다.
        사용자가 원하는 운동 및 식단 정보를 제공해야한다는 최종 목표를 염두해두고,
        사용자의 요구를 달성하기 위해 현재 해야할 일이 무엇인지 결정한다.


        supervisor가 활용할 수 있는 agent는 다음과 같다:

        [정보 수집 Agent]
        -web_search_agent: 웹 검색을 통해 최신 정보나 일반적인 정보를 확보한다.
        -vector_search_agent: 벡터 DB 검색을 통해 전문적인 운동/영양 정보를 확보한다.

        [실행 Agent]
        -diet_planner: 수집된 정보를 바탕으로 구체적인 식단표를 생성한다.
        -food_analyzer_agent: 사용자가 업로드한 음식 이미지를 분석하여 볼륨, 무게, 칼로리, 영양소 정보를 추출한다.

        [사용자 대화 Agent]
        -communicator: 단순 질문 응답, 진행상황 보고, 사용자 피드백 수집 등 일반적인 대화를 처리한다.


        ** Agent 선택 기준 **
        1. 단순 질문/대화 → communicator 직행
        2. 정보 검색 필요 → web_search_agent 또는 vector_search_agent
        3. 식단표 생성 → diet_planner
        4. 음식 이미지 분석 → food_analyzer_agent

        아래 내용을 고려하여, 현재 해야할 일이 무엇인지, 적절한 agent를 선택하라.

        ---------------------------------
        - previous_target: {target}
        ---------------------------------
        - messages: {messages}

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
    task = state['task_history'][-1]
    return task.agent

def vector_search_agent(state: State):
    print("\n\n============ VECTOR SEARCH AGENT ============")
    
    tasks = state.get("task_history", [])
    task = tasks[-1]
    if task.agent != "vector_search_agent":
        raise ValueError(f"Vector Search Agent가 아닌 agent가 Vector Search Agent를 시도하고 있습니다.\n {task}")

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
        "outline": target
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
        retrieved_docs = retrieve(args)
		#① (1) 결과 담아 두기
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


def business_analyst(state: State):
    print("\n\n====================BUSINESS ANALYST====================")
    business_analyst_system_prompt = PromptTemplate.from_template(
        """
        너는 식단과 운동에 대한 정보를 제공하는 AI팀의 비즈니스 애널리스트로서,
        AI 팀의 진행상황과 "사용자 요구 사항"을 토대로,
        현 시점에서 'ai_recommendation'과 최근 사용자의 발언을 바탕으로 요구사항이 무엇인지 판단한다.
        지난 요구 사항이 달성되었는지 판단하고, 현 시점에서 어떤 작업을 해야 하는지 결정한다.

        **분석 지침**:
        1. 사용자의 의도를 명확히 파악
        2. 이전 작업과의 연관성 분석
        3. 우선순위가 높은 작업 식별
        4. 실행 가능한 구체적인 방법 제시

        다음과 같은 템플릿 형태로 반환한다:

        '''
        목표: [사용자가 원하는 최종 결과]
        방법: [목표 달성을 위한 구체적인 실행 방법]
        우선순위: [high/medium/low]
        '''

        ---------------------------------
        *AI 권장사항(ai_recommendation)*: {ai_recommendation}
        ---------------------------------
        *최근 사용자 발언*: {user_last_comment}
        ---------------------------------
        *참고자료*: {references}
        ---------------------------------
        *사용자 업로드 이미지*: {user_upload_img}
        ---------------------------------
        *이전 식단표*: {diet_plan}
        ---------------------------------
        *이전 음식 분석*: {food_analysis}
        """
    )
    ba_chain = business_analyst_system_prompt | llm | StrOutputParser()

    messages = state['messages']

    # 최근 사용자 발언 찾기
    user_last_comment = None
    for m in messages[::-1]:
        if isinstance(m, HumanMessage):
            user_last_comment = m.content
            break

    # 입력 자료 준비
    uploaded_images = state.get("uploaded_images", [])
    diet_plan = state.get("diet_plan", {})
    food_analysis = state.get("food_analysis", [])

    inputs = {
        "ai_recommendation": state.get("ai_recommendation", "없음"),
        "references": state.get("references", {"queries": [], "docs": []}),
        "user_upload_img": f"{len(uploaded_images)}개 업로드됨" if uploaded_images else "없음",
        "messages": messages[-5:] if len(messages) > 5 else messages,  # 최근 5개만
        "user_last_comment": user_last_comment or "없음",
        "diet_plan": "생성됨" if diet_plan.get("plan") else "없음",
        "food_analysis": f"{len(food_analysis)}개 분석됨" if food_analysis else "없음"
    }

    #⑤ (4) 시스템 프롬프트를 통해 사용자 요구사항을 분석
    user_request = ba_chain.invoke(inputs)

    #⑥ (5) businessage analyst의 결과를 메시지에 추가
    business_analyst_message = f"[Business Analyst] {user_request}"
    print(business_analyst_message)
    messages.append(AIMessage(business_analyst_message))

    save_state(current_path, state) #⑦ (6) 현재 state 내용 저장

    return {
        "messages": messages,
        "user_request": user_request,
        "ai_recommendation": ""
    }

def outline_reviewer(state: State): # ①
    print("\n\n============ OUTLINE REVIEWER ============")

    # ② 시스템 프롬프트 정의
    outline_reviewer_system_prompt = PromptTemplate.from_template(
        """
        너는 AI팀의 목표 리뷰어로서, AI팀이 작성한 정보를 검토하고 문제점을 지적한다. 

        - 정보가 사용자의 요구사항을 충족시키는지 여부
        - 정보의 논리적인 흐름이 적절한지 여부
        - 근거에 기반하지 않은 내용이 있는지 여부
        - 주어진 참고자료(references)를 충분히 활용했는지 여부
        - 참고자료가 충분한지, 혹은 잘못된 참고자료가 있는지 여부
        - example.com 같은 더미 URL이 있는지 여부: 
        - 실제 페이지 URL이 아닌 대표 URL로 되어 있는 경우 삭제 해야함: 어떤 URL이 삭제되어야 하는지 명시하라.
        - 기타 리뷰 사항

        그 분석 결과를 설명하고, 다음 어떤 작업을 하면 좋을지 제안하라.
        
        - 분석결과: outline이 사용자의 요구사항을 충족시키는지 여부
        - 제안사항: (vector_search_agent, communicator 중 어떤 agent를 호출할지)

        ------------------------------------------
        user_request: {user_request}
        ------------------------------------------
        references: {references}
        ------------------------------------------
        messages: {messages}
        """
    )
    # ③ inputs에 들어갈 내용 정리    
    user_request = state.get("user_request", None)
    references = state.get("references", {"queries": [], "docs": []})
    messages = state.get("messages", [])

    inputs = {
        "user_request": user_request,
        "references": references,
        "messages": messages
    }

    # 시스템 프롬프트와 모델을 연결
    outline_reviewer_chain = outline_reviewer_system_prompt | llm

    # ④ 목차 리뷰
    review = outline_reviewer_chain.stream(inputs)

    gathered = None

    for chunk in review:
        print(chunk.content, end='')

        if gathered is None:
            gathered = chunk
        else:
            gathered += chunk
    # ⑤ outline_review 에이전트의 작업 후기를 메시지에 추가
    if '[OUTLINE REVIEW AGENT]' not in gathered.content:
        gathered.content = f"[OUTLINE REVIEW AGENT] {gathered.content}"

    print(gathered.content)
    messages.append(gathered)

    # ⑥ ai_recommendation은 목차 리뷰 결과를 사용
    ai_recommendation = gathered.content

    return {"messages": messages, "ai_recommendation": ai_recommendation} # ⑦


def web_search_agent(state: State): #① (0)
    print("\n\n============ WEB SEARCH AGENT ============")

    # 작업 리스트 가져와서 web search agent 가 할 일인지 확인하기
    tasks = state.get("task_history", [])
    task = tasks[-1]

    if task.agent != "web_search_agent":
        raise ValueError(f"Web Search Agent가 아닌 agent가 Web Search Agent를 시도하고 있습니다.\n {task}")
    
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

    #⑪ (11) task 완료
    tasks[-1].done = True
    tasks[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    #⑪ (11) 새로운 task 추가
    task_desc = "AI팀의 세부 목표를 결정하기 위한 정보를 벡터 검색을 통해 찾아낸다."
    task_desc += f" 다음 항목이 새로 추가되었다\n: {queries}"
    
    new_task = Task(
        agent="vector_search_agent",
        done=False,
        description=task_desc,
        done_at=""
    )

    tasks.append(new_task)

    #⑫ (12) 작업 후기 메시지
    msg_str = f"[WEB SEARCH AGENT] 다음 질문에 대한 검색 완료: {queries}"
    messages.append(AIMessage(msg_str))

    #⑬ (13) state 업데이트
    return {
        "messages": messages,
        "task_history": tasks
    }


def diet_planner(state: State):
    print("\n\n====================DIET PLANNER====================")

    # 현재 task 확인
    tasks = state.get("task_history", [])
    task = tasks[-1]

    if task.agent != "diet_planner":
        raise ValueError(f"Diet Planner가 아닌 agent가 Diet Planner를 시도하고 있습니다.\n {task}")

    # 시스템 프롬프트 정의
    diet_planner_system_prompt = PromptTemplate.from_template(
        """
        너는 운동 식단 트레이너 AI팀의 식단 설계 전문가(Diet Planner)로서,
        사용자의 목표와 검색된 정보를 바탕으로 구체적이고 실행 가능한 식단표를 생성한다.

        ## 작성 지침
        1. 일주일 식단표를 요일별로 작성 (월요일~일요일)
        2. 각 끼니마다 구체적인 음식명과 분량 명시
        3. 칼로리와 주요 영양소(탄수화물, 단백질, 지방) 표시
        4. 실현 가능하고 한국 음식 위주로 구성
        5. 사용자가 실천하기 쉽도록 간단한 조리법도 포함
        6. 사용자의 개인 정보(목표, 건강 상태, 선호도)를 최대한 반영

        ## 사용자 요구사항
        {user_request}

        ## 사용자 목표
        {target}

        ## 검색된 식단 정보
        {references}

        ## 이전 대화 내용
        {messages}

        ## 출력 형식
        마크다운 형식으로 표를 사용하여 보기 좋게 작성하라.
        각 요일별로 아침, 점심, 저녁, 간식을 구분하여 작성하고,
        일일 총 칼로리와 영양소 합계를 명시하라.
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
        "user_request": user_request
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
    task = tasks[-1]

    if task.agent != "food_analyzer_agent":
        raise ValueError(f"Food Analyzer Agent가 아닌 agent가 실행되고 있습니다.\n {task}")

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

        # 결과 메시지 생성
        result_msg = f"""[FOOD ANALYZER] 분석 완료!

🍽️ **{analysis.food_name}**
━━━━━━━━━━━━━━━━━━━━━━━━
📊 측정 정보
  • 볼륨: {analysis.volume_ml:.1f} mL
  • 무게: {analysis.weight_grams:.1f} g

🔥 영양 정보
  • 칼로리: {analysis.calories} kcal
  • 단백질: {analysis.protein:.1f} g
  • 탄수화물: {analysis.carbs:.1f} g
  • 지방: {analysis.fat:.1f} g

✓ 신뢰도: {analysis.confidence:.0%}
"""
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

        return {
            "messages": messages,
            "task_history": tasks,
            "food_analysis": food_analysis_list
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

            **대화 원칙**:
            - 친절하고 전문적인 톤 유지
            - 필요한 정보만 간결하게 전달
            - 사용자의 다음 요청을 자연스럽게 유도
            - 이모지를 적절히 활용하여 가독성 향상
            - 핵심 정보는 **굵게** 표시

            **상황별 대응**:
            - 식단표 생성 완료 시: 만족도 확인, 수정 요청 안내
            - 음식 분석 완료 시: 건강 조언, 추가 분석 제안
            - 오류 발생 시: 명확한 해결 방법 제시
            - 일반 대화: 간결하면서도 도움이 되는 답변

            현재 목표: {target}
            최근 대화 내역: {messages}
            생성된 식단표: {diet_plan}
            분석된 음식: {food_analysis}
        """
    )

    system_chain = communicator_system_prompt | llm

    messages = state['messages']
    diet_plan = state.get('diet_plan', {})
    food_analysis = state.get('food_analysis', [])

    # 식단표와 음식 분석 요약
    diet_summary = "없음"
    if diet_plan and diet_plan.get('plan'):
        diet_summary = f"생성됨 ({diet_plan.get('created_at', '시간 미상')})"

    food_summary = "없음"
    if food_analysis:
        latest = food_analysis[-1]
        food_summary = f"{latest.food_name} ({latest.calories} kcal)"

    inputs = {
        'messages': messages[-10:] if len(messages) > 10 else messages,  # 최근 10개만
        'target': get_target(current_path),
        'diet_plan': diet_summary,
        'food_analysis': food_summary
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

    if task_history[-1].agent != "communicator":
        raise ValueError(f"Communicator가 아닌 agent가 대화를 시도하고 있습니다.\n {task_history[-1]}")
    
    task_history[-1].done = True
    task_history[-1].done_at = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    return {
        "messages": messages,
        "task_history": task_history
    }

#Node
graph_builder = StateGraph(State)
graph_builder.add_node("business_analyst", business_analyst)
graph_builder.add_node("supervisor", supervisor)
graph_builder.add_node("communicator", communicator)
graph_builder.add_node("vector_search_agent", vector_search_agent)
graph_builder.add_node('web_search_agent', web_search_agent)
graph_builder.add_node("diet_planner", diet_planner)
graph_builder.add_node("food_analyzer_agent", food_analyzer_agent)

#Edge
graph_builder.add_edge(START, "business_analyst")  # 내부 기획 완료 후 supervisor로 복귀
graph_builder.add_edge("business_analyst", "supervisor")
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

graph_builder.add_edge("web_search_agent", "vector_search_agent")
graph_builder.add_edge("vector_search_agent", "supervisor")  # 정보 수집 후 supervisor로 복귀
graph_builder.add_edge("diet_planner", "communicator")  # 식단표 생성 후 사용자에게 전달
graph_builder.add_edge("food_analyzer_agent", "business_analyst")  # 분석 결과 사용자에게 전달
graph_builder.add_edge("communicator", END)  # 사용자 대화 후 종료

graph = graph_builder.compile()

graph.get_graph().draw_mermaid_png(output_file_path = absolute_path.replace('.py', '.png'))

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
    ai_recommendation=""
)

while True:
    user_input = input("\nUser\t: ").strip()

    if user_input.lower() in ['exit', 'quit', 'q']:
        print("Goodbye!")
        break

    # 이미지 업로드 핸들링: !image <path> 형식
    if user_input.startswith('!image '):
        image_path = user_input[7:].strip()

        # 따옴표 제거 (경로에 공백이 있을 경우)
        image_path = image_path.strip('"').strip("'")

        # 파일 존재 확인
        if not os.path.exists(image_path):
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
            print(f"💡 경로를 다시 확인해주세요. 현재 작업 디렉토리: {os.getcwd()}")
            continue

        # 이미지 파일 확장자 확인
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
        if not any(image_path.lower().endswith(ext) for ext in valid_extensions):
            print(f"❌ 지원되지 않는 이미지 형식입니다.")
            print(f"✅ 지원 형식: {', '.join(valid_extensions)}")
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
            print(f"\n✅ 이미지 업로드 성공!")
            print(f"📁 저장 위치: {dest_path}")
            print(f"\n📸 음식 분석 명령어:")
            print(f"   - '이미지 분석해줘'")
            print(f"   - '음식 영양 정보 알려줘'")
            print(f"   - '칼로리 분석해줘'\n")
        except PermissionError:
            print(f"❌ 파일 접근 권한이 없습니다: {image_path}")
        except Exception as e:
            print(f"❌ 이미지 복사 중 오류가 발생했습니다: {e}")

        continue

    state["messages"].append(HumanMessage(user_input))
    state = graph.invoke(state)

    print('\n------------------------MESSAGE COUNT\t', len(state["messages"]))

    save_state(current_path, state)

