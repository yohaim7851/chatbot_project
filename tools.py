from tavily import TavilyClient
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
import json
import os
absolute_path = os.path.abspath(__file__) # 현재 파일의 절대 경로 반환
current_path = os.path.dirname(absolute_path) # 현재 .py 파일이 있는 폴더 경로

# RAG를 위한 설정
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# 오픈AI Embedding 설정
embedding = OpenAIEmbeddings(model='text-embedding-3-large')

# 크로마 DB 저장 경로 설정
persist_directory = f"{current_path}/data/chroma_store"

# Chroma 객체 생성
vectorstore = Chroma(
    persist_directory=persist_directory,
    embedding_function=embedding
)

@tool
def web_search(query: str):
    """
    주어진 query에 대해 웹검색을 하고, 결과를 반환한다.

    Args:
        query (str): 검색어

    Returns:
        dict: 검색 결과
    """
    client = TavilyClient()

    content = client.search(
        query, 
        search_depth="advanced",
        include_raw_content=True,
    )
    
    results = content["results"] 

    for result in results:
        if result["raw_content"] is None:
            try:
                result["raw_content"] = load_web_page(result["url"])
            except Exception as e:
                print(f"Error loading page: {result['url']}")
                print(e)
                result["raw_content"] = result["content"]

    resources_json_path = f'{current_path}/data/resources_{datetime.now().strftime('%Y_%m%d_%H%M%S')}.json'
    with open(resources_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
   
    return results, resources_json_path  # 검색 결과와 JSON 파일 경로 반환


def web_page_to_document(web_page):
    # raw_content와 content 중 정보가 많은 것을 page_content로 한다.
    if len(web_page['raw_content']) > len(web_page['content']):
        page_content = web_page['raw_content']
    else:
        page_content = web_page['content']
    # 랭체인 Document로 변환
    document = Document(
        page_content=page_content,
        metadata={
            'title': web_page['title'],
            'source': web_page['url']
        }
    )

    return document


def web_page_json_to_documents(json_file):
    with open(json_file, "r", encoding='utf-8') as f:
        resources = json.load(f)

    documents = []

    for web_page in resources:
        document = web_page_to_document(web_page)
        documents.append(document)

    return documents


def split_documents(documents, chunk_size=1000, chunk_overlap=100):
    print('Splitting documents...')
    print(f"{len(documents)}개의 문서를 {chunk_size}자 크기로 중첩 {chunk_overlap}자로 분할합니다.\n")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    splits = text_splitter.split_documents(documents)

    print(f"총 {len(splits)}개의 문서로 분할되었습니다.")
    return splits

# documents를 chroma DB에 저장하는 함수
def documents_to_chroma(documents, chunk_size=1000, chunk_overlap=100):
    print("Documents를 Chroma DB에 저장합니다.")

    # documents의 url 가져오기
    urls = [document.metadata['source'] for document in documents]

    # 이미 vectorstore에 저장된 urls 가져오기
    stored_metadatas = vectorstore._collection.get()['metadatas'] 
    stored_web_urls = [metadata['source'] for metadata in stored_metadatas] 

    # 새로운 urls만 남기기
    new_urls = set(urls) - set(stored_web_urls)

    # 새로운 urls에 대한 documents만 남기기
    new_documents = []

    for document in documents:
        if document.metadata['source'] in new_urls:
            new_documents.append(document)
            print(document.metadata)

    # 새로운 documents를 Chroma DB에 저장
    splits = split_documents(new_documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 크로마 DB에 저장
    if splits:
        vectorstore.add_documents(splits)
    else:
        print("No new urls to process")

# json 파일에서 documents를 만들고, 그 documents들을 Chroma DB에 저장
def add_web_pages_json_to_chroma(json_file, chunk_size=1000, chunk_overlap=100):
    documents = web_page_json_to_documents(json_file)
    documents_to_chroma(
        documents, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )


def load_web_page(url: str):
    loader = WebBaseLoader(url, verify_ssl=False)

    content = loader.load()
    raw_content = content[0].page_content.strip()  

    while '\n\n\n' in raw_content or '\t\t\t' in raw_content:
        raw_content = raw_content.replace('\n\n\n', '\n\n')
        raw_content = raw_content.replace('\t\t\t', '\t\t')
        
    return raw_content

@tool
def retrieve(query: str, top_k: int=5):
    """
    주어진 query에 대해 벡터 검색을 수행하고, 결과를 반환한다.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    retrieved_docs = retriever.invoke(query)

    return retrieved_docs

def calculate_nutrition_needs(user_info: dict) -> dict:
    """
    사용자 정보 기반 일일 영양소 요구량 계산

    Args:
        user_info: {
            'gender': '남성' or '여성',
            'height': int (cm),
            'weight': int (kg),
            'goal': '다이어트' or '근육증가' or '체중유지' or '건강관리'
        }

    Returns:
        {
            'calories': int,  # 일일 칼로리 (kcal)
            'protein': int,   # 일일 단백질 (g)
            'carbs': int,     # 일일 탄수화물 (g)
            'fat': int        # 일일 지방 (g)
        }
    """
    gender = user_info.get('gender', '남성')
    height = user_info.get('height', 170)
    weight = user_info.get('weight', 70)
    goal = user_info.get('goal', '체중유지')

    # 1. 기초대사량 (BMR) 계산 - Harris-Benedict 공식
    if gender == '남성':
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * 30)  # 나이 30세 가정
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * 30)

    # 2. 활동대사량 (TDEE) 계산 - 활동량 보통 (1.55 적용)
    tdee = bmr * 1.55

    # 3. 목표에 따른 칼로리 및 영양소 비율 조정
    if goal == '다이어트':
        calories = int(tdee * 0.8)  # 20% 감소
        protein_ratio = 0.30  # 30%
        carbs_ratio = 0.40    # 40%
        fat_ratio = 0.30      # 30%
    elif goal == '근육증가':
        calories = int(tdee * 1.15)  # 15% 증가
        protein_ratio = 0.30  # 30%
        carbs_ratio = 0.50    # 50%
        fat_ratio = 0.20      # 20%
    elif goal == '체중유지':
        calories = int(tdee)
        protein_ratio = 0.25  # 25%
        carbs_ratio = 0.50    # 50%
        fat_ratio = 0.25      # 25%
    else:  # 건강관리
        calories = int(tdee)
        protein_ratio = 0.20  # 20%
        carbs_ratio = 0.55    # 55%
        fat_ratio = 0.25      # 25%

    # 4. 영양소 계산 (칼로리 기준)
    # 단백질: 1g = 4kcal
    # 탄수화물: 1g = 4kcal
    # 지방: 1g = 9kcal
    protein = int((calories * protein_ratio) / 4)
    carbs = int((calories * carbs_ratio) / 4)
    fat = int((calories * fat_ratio) / 9)

    return {
        'calories': calories,
        'protein': protein,
        'carbs': carbs,
        'fat': fat,
        'bmr': int(bmr),
        'tdee': int(tdee)
    }
