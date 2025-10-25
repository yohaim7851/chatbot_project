from pydantic import BaseModel, Field
from typing import Literal, Optional

class Task(BaseModel):
    agent: Literal[
        "strategist",
        "communicator",
        "vector_search_agent",
        "web_search_agent",
        "diet_planner",
        "food_analyzer_agent"
    ] = Field(
        ...,
        description="""
        작업을 수행하는 agent의 종류.
        -strategist: 사용자의 요구사항이 불명확할 때, AI 팀의 세부 목표를 설정한다.
        -communicator: AI팀에서 해야할 일을 스스로 판단할 수 없을 때 사용한다,사용자에게 진행사항을 보고하고, 다음 지시를 물어본다.
        -vector_search_agent: 벡터 DB 검색을 통해 목표 달성에 필요한 정보를 확보한다.
        -web_search_agent: 웹 검색을 통해 목표 달성에 필요한 정보를 확보한다.
        -diet_planner: 수집된 정보와 사용자 요구사항을 바탕으로 구체적인 식단표를 생성한다.
        -food_analyzer_agent: 사용자가 업로드한 음식 이미지를 분석하여 볼륨, 무게, 영양 정보를 추출한다.
        """
    )
	
    done: bool = Field(..., description="종료 여부")
    description: str = Field(..., description="어떤 작업을 해야 하는지에 대한 설명")
	
    done_at: str = Field(..., description="할 일이 완료된 날짜와 시간")
	
    def to_dict(self):
        return {
            "agent": self.agent,
            "done": self.done,
            "description": self.description,
            "done_at": self.done_at
        }


class FoodAnalysis(BaseModel):
    """음식 분석 결과"""
    food_name: str = Field(..., description="음식 이름")
    volume_ml: float = Field(..., description="볼륨 (mL)")
    weight_grams: float = Field(..., description="무게 (g)")
    calories: int = Field(default=0, description="칼로리 (kcal)")
    protein: float = Field(default=0.0, description="단백질 (g)")
    carbs: float = Field(default=0.0, description="탄수화물 (g)")
    fat: float = Field(default=0.0, description="지방 (g)")
    confidence: float = Field(default=0.0, description="신뢰도 (0~1)")
    image_path: str = Field(..., description="분석한 이미지 경로")
    analyzed_at: str = Field(..., description="분석 시간")

    def to_dict(self):
        return {
            "food_name": self.food_name,
            "volume_ml": self.volume_ml,
            "weight_grams": self.weight_grams,
            "calories": self.calories,
            "protein": self.protein,
            "carbs": self.carbs,
            "fat": self.fat,
            "confidence": self.confidence,
            "image_path": self.image_path,
            "analyzed_at": self.analyzed_at
        }