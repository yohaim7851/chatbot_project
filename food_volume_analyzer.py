"""
Food Volume Analyzer - Chatbot Integration Module
food_trainer.py와 통합하기 위한 음식 볼륨 분석 모듈 (food_db.csv 활용)
"""

import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import pipeline
from ultralytics import YOLO
import cv2
from scipy.spatial import ConvexHull
from typing import Dict, Tuple, Optional
import warnings
import os
warnings.filterwarnings('ignore')


class FoodVolumeAnalyzer:
    """Hugging Face 모델 기반 음식 볼륨 분석기 (food_db.csv 통합)"""

    def __init__(self, food_db_path: str = "food_db.csv"):
        """
        초기화

        Args:
            food_db_path: 한국 식품영양성분 DB 파일 경로
        """
        self.device = 0 if torch.cuda.is_available() else -1

        print("\n[FoodVolumeAnalyzer] 초기화 중...")

        # 1. Depth Estimation 모델
        print("  [1/3] Depth Estimation 모델 로딩...")
        self.depth_estimator = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Large-hf",
            device=self.device
        )

        # 2. YOLO Segmentation 모델
        print("  [2/3] YOLO Segmentation 모델 로딩...")
        self.segmentor = YOLO('yolov8n-seg.pt')
        if self.device >= 0:
            self.segmentor.to('cuda')

        # 3. 한국 식품영양성분 DB 로드
        print("  [3/3] 식품영양성분 DB 로딩...")
        self.food_db = self._load_food_db(food_db_path)

        # 기본 밀도 DB (폴백용)
        self.default_density = {
            'rice': 0.85, 'chicken': 1.05, 'meat': 1.08, 'beef': 1.10,
            'pork': 1.08, 'fish': 1.05, 'soup': 1.00, 'stew': 0.95,
            'noodles': 0.70, 'pasta': 0.72, 'salad': 0.40,
            'vegetables': 0.50, 'bread': 0.30, 'pizza': 0.65,
            'fruit': 0.85, 'default': 0.80
        }

        print(f"✓ 초기화 완료")
        print(f"  - GPU: {'사용' if self.device >= 0 else '미사용'}")
        print(f"  - 식품 DB: {len(self.food_db):,}개 항목")

    def _load_food_db(self, db_path: str) -> pd.DataFrame:
        """한국 식품영양성분 DB 로드"""
        if not os.path.exists(db_path):
            print(f"  ⚠️ {db_path} 파일을 찾을 수 없습니다. 기본 DB 사용")
            return pd.DataFrame()

        try:
            # CSV 로드 (인코딩 자동 감지)
            df = pd.read_csv(db_path, encoding='cp949')  # 한글 인코딩

            # 필요한 컬럼만 추출 (실제 파일의 컬럼명에 맞게 수정)
            # 실제 컬럼명: 식품명, 에너지(kcal), 단백질(g), 탄수화물(g), 지방(g)
            required_cols = ['식품명', '에너지(kcal)', '단백질(g)', '탄수화물(g)', '지방(g)']

            # 컬럼 존재 확인
            available_cols = [col for col in required_cols if col in df.columns]

            if len(available_cols) < 2:
                print(f"  ⚠️ 필수 컬럼이 부족합니다: {available_cols}")
                return pd.DataFrame()

            # 데이터 정제
            df_clean = df[available_cols].copy()
            df_clean = df_clean.dropna(subset=['식품명'])  # 식품명 없는 행 제거

            # 숫자형 컬럼 변환
            for col in available_cols[1:]:
                if col in df_clean.columns:
                    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

            return df_clean

        except Exception as e:
            print(f"  ⚠️ DB 로드 실패: {e}")
            return pd.DataFrame()

    def estimate_volume(
        self,
        image_path: str,
        plate_diameter_cm: float = 26.0
    ) -> Dict:
        """
        음식 볼륨 추정 (food_db.csv 활용)

        Args:
            image_path: 이미지 경로
            plate_diameter_cm: 접시 직경 (cm)

        Returns:
            {
                'volume_ml': float,
                'weight_grams': float,
                'food_type': str,
                'calories': int,
                'protein': float,
                'carbs': float,
                'fat': float,
                'confidence': float
            }
        """
        try:
            print(f"\n[분석 시작] {os.path.basename(image_path)}")

            # 1. 이미지 로드
            image = Image.open(image_path).convert('RGB')
            image_np = np.array(image)

            # 2. 깊이맵 생성
            print("  [1/6] 깊이맵 생성...")
            depth_map = self._estimate_depth(image)

            # 3. 세그멘테이션
            print("  [2/6] 음식 세그멘테이션...")
            mask, food_class = self._segment_food(image_np)
            food_pixel_ratio = mask.sum() / (mask.shape[0] * mask.shape[1])
            print(f"    → 음식 영역: {food_pixel_ratio*100:.1f}%")

            # 4. 깊이 스케일링
            print("  [3/6] 깊이 스케일 조정...")
            scaled_depth = self._scale_depth(depth_map, mask, plate_diameter_cm)

            # 5. 포인트 클라우드 생성
            print("  [4/6] 3D 포인트 클라우드 생성...")
            point_cloud = self._depth_to_pointcloud(
                scaled_depth, mask, 70.0, image.size
            )
            print(f"    → 포인트: {len(point_cloud):,}개")

            # 6. 볼륨 계산
            print("  [5/6] 볼륨 계산...")
            volume_ml = self._calculate_volume(point_cloud)
            print(f"    → 볼륨: {volume_ml:.1f} mL")

            # 7. food_db.csv에서 음식 정보 조회
            print("  [6/6] 영양 정보 조회...")
            food_info = self._query_food_db(food_class, volume_ml)

            result = {
                'volume_ml': round(volume_ml, 1),
                'weight_grams': round(food_info['weight'], 1),
                'food_type': food_info['name'],
                'calories': food_info['calories'],
                'protein': food_info['protein'],
                'carbs': food_info['carbs'],
                'fat': food_info['fat'],
                'confidence': 0.85,
                'yolo_class': food_class
            }

            print(f"\n✓ 분석 완료:")
            print(f"  - 음식: {result['food_type']}")
            print(f"  - 무게: {result['weight_grams']}g")
            print(f"  - 칼로리: {result['calories']}kcal\n")

            return result

        except Exception as e:
            print(f"[ERROR] 볼륨 추정 실패: {e}")
            return {
                'volume_ml': 0.0,
                'weight_grams': 0.0,
                'food_type': 'unknown',
                'calories': 0,
                'protein': 0.0,
                'carbs': 0.0,
                'fat': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }

    def _estimate_depth(self, image: Image.Image) -> np.ndarray:
        """깊이맵 생성"""
        depth_output = self.depth_estimator(image)
        depth_array = np.array(depth_output['depth']).astype(np.float32)

        # 정규화
        d_min, d_max = depth_array.min(), depth_array.max()
        if d_max > d_min:
            depth_array = (depth_array - d_min) / (d_max - d_min)

        return depth_array

    def _segment_food(self, image_np: np.ndarray) -> Tuple[np.ndarray, str]:
        """음식 세그멘테이션"""
        results = self.segmentor(image_np, verbose=False)

        # 마스크가 없으면 중앙 영역 사용
        if len(results) == 0 or results[0].masks is None or len(results[0].masks) == 0:
            h, w = image_np.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            mask[int(h*0.15):int(h*0.85), int(w*0.15):int(w*0.85)] = True
            return mask, "음식"

        # 가장 큰 마스크 선택
        masks = results[0].masks.data.cpu().numpy()
        mask_idx = np.argmax([m.sum() for m in masks])
        largest_mask = masks[mask_idx]

        # 리사이즈
        h, w = image_np.shape[:2]
        mask_resized = cv2.resize(
            largest_mask.astype(np.uint8), (w, h),
            interpolation=cv2.INTER_NEAREST
        )

        # 클래스명 (영어 → 한글 매핑)
        class_name = "음식"
        if results[0].boxes is not None and len(results[0].boxes) > mask_idx:
            class_id = int(results[0].boxes[mask_idx].cls.item())
            class_name_en = results[0].names[class_id]
            class_name = self._translate_food_name(class_name_en)

        return mask_resized > 0.5, class_name

    def _translate_food_name(self, en_name: str) -> str:
        """영어 음식명 → 한글"""
        translation = {
            'banana': '바나나', 'apple': '사과', 'orange': '오렌지',
            'broccoli': '브로콜리', 'carrot': '당근', 'pizza': '피자',
            'hot dog': '핫도그', 'donut': '도넛', 'cake': '케이크',
            'sandwich': '샌드위치', 'bowl': '밥', 'cup': '국',
            'bottle': '음료'
        }
        return translation.get(en_name, en_name)

    def _scale_depth(
        self, depth: np.ndarray, mask: np.ndarray, plate_cm: float
    ) -> np.ndarray:
        """깊이 스케일링 (접시 직경 기준)"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0 or len(contours[0]) < 5:
            return depth * (plate_cm / 100)

        try:
            ellipse = cv2.fitEllipse(max(contours, key=cv2.contourArea))
            measured_diameter_px = max(ellipse[1])
            scale = (plate_cm / 100) / (measured_diameter_px / depth.shape[1])
            return depth * scale
        except:
            return depth * (plate_cm / 100)

    def _depth_to_pointcloud(
        self, depth: np.ndarray, mask: np.ndarray,
        fov: float, image_size: Tuple[int, int]
    ) -> np.ndarray:
        """3D 포인트 클라우드 생성"""
        h, w = depth.shape
        f = w / (2 * np.tan(np.radians(fov) / 2))
        cx, cy = w / 2, h / 2

        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth
        x = (u - cx) * z / f
        y = (v - cy) * z / f

        points = np.stack([x, y, z], axis=-1)
        food_points = points[mask]

        # 이상치 제거
        valid = (food_points[:, 2] > 0.01) & (food_points[:, 2] < 2.0)
        return food_points[valid]

    def _calculate_volume(self, point_cloud: np.ndarray) -> float:
        """볼륨 계산 (Convex Hull)"""
        if len(point_cloud) < 4:
            return 0.0

        try:
            hull = ConvexHull(point_cloud)
            volume_ml = hull.volume * 1e6  # m³ → mL

            # 합리적 범위 체크 (10mL ~ 5000mL)
            if 10 <= volume_ml <= 5000:
                return volume_ml
        except:
            pass

        # 폴백: 바운딩 박스
        bbox = point_cloud.max(axis=0) - point_cloud.min(axis=0)
        volume_ml = np.prod(bbox) * 1e6 * 0.6  # 충진율 60%
        return max(volume_ml, 10.0)

    def _query_food_db(self, food_name: str, volume_ml: float) -> Dict:
        """
        food_db.csv에서 음식 정보 조회

        Returns:
            {
                'name': str,
                'weight': float (g),
                'calories': int,
                'protein': float (g),
                'carbs': float (g),
                'fat': float (g)
            }
        """
        # DB가 비어있으면 기본값
        if self.food_db.empty:
            return self._get_default_nutrition(food_name, volume_ml)

        # 음식명으로 검색 (부분 일치)
        matched = self.food_db[
            self.food_db['식품명'].str.contains(food_name, na=False, case=False)
        ]

        if len(matched) == 0:
            # 매칭 실패 시 유사한 음식으로 대체
            return self._get_default_nutrition(food_name, volume_ml)

        # 첫 번째 매칭 항목 사용
        food_data = matched.iloc[0]

        # 밀도 추정 (100mL 기준 무게 계산)
        # 대부분의 음식은 0.6~1.2 g/mL 범위
        estimated_density = self._estimate_density(food_name)
        weight_grams = volume_ml * estimated_density

        # 영양 정보 계산 (100g 기준 → 실제 무게 기준)
        calories_per_100g = food_data.get('에너지(kcal)', 0)
        protein_per_100g = food_data.get('단백질(g)', 0)
        carbs_per_100g = food_data.get('탄수화물(g)', 0)
        fat_per_100g = food_data.get('지방(g)', 0)

        return {
            'name': food_data['식품명'],
            'weight': weight_grams,
            'calories': int(calories_per_100g * weight_grams / 100),
            'protein': round(protein_per_100g * weight_grams / 100, 1),
            'carbs': round(carbs_per_100g * weight_grams / 100, 1),
            'fat': round(fat_per_100g * weight_grams / 100, 1)
        }

    def _estimate_density(self, food_name: str) -> float:
        """음식명으로 밀도 추정 (g/mL)"""
        # 음식 카테고리별 밀도
        if any(kw in food_name for kw in ['밥', '쌀', 'rice']):
            return 0.85
        elif any(kw in food_name for kw in ['고기', '닭', '소고기', '돼지', 'meat', 'chicken', 'beef', 'pork']):
            return 1.08
        elif any(kw in food_name for kw in ['국', '찌개', '탕', 'soup', 'stew']):
            return 1.00
        elif any(kw in food_name for kw in ['면', '라면', '파스타', 'noodle', 'pasta']):
            return 0.72
        elif any(kw in food_name for kw in ['빵', '케이크', 'bread', 'cake']):
            return 0.35
        elif any(kw in food_name for kw in ['샐러드', '채소', 'salad', 'vegetable']):
            return 0.45
        elif any(kw in food_name for kw in ['과일', 'fruit']):
            return 0.85
        else:
            return self.default_density.get('default', 0.80)

    def _get_default_nutrition(self, food_name: str, volume_ml: float) -> Dict:
        """기본 영양 정보 (DB 매칭 실패 시)"""
        density = self._estimate_density(food_name)
        weight_grams = volume_ml * density

        # 기본 칼로리 추정 (100g당)
        calories_per_100g = 150  # 평균값

        return {
            'name': food_name,
            'weight': weight_grams,
            'calories': int(calories_per_100g * weight_grams / 100),
            'protein': round(weight_grams * 0.08, 1),  # 8% 단백질
            'carbs': round(weight_grams * 0.25, 1),   # 25% 탄수화물
            'fat': round(weight_grams * 0.05, 1)      # 5% 지방
        }


# 전역 인스턴스 (싱글톤 패턴)
_analyzer_instance = None

def get_analyzer():
    """싱글톤 분석기 인스턴스 반환"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = FoodVolumeAnalyzer()
    return _analyzer_instance


if __name__ == "__main__":
    # 테스트
    analyzer = get_analyzer()
    result = analyzer.estimate_volume("food.jpg")

    print("\n" + "="*60)
    print("📊 최종 결과")
    print("="*60)
    print(f"음식:      {result['food_type']}")
    print(f"볼륨:      {result['volume_ml']} mL")
    print(f"무게:      {result['weight_grams']} g")
    print(f"칼로리:    {result['calories']} kcal")
    print(f"단백질:    {result['protein']} g")
    print(f"탄수화물:  {result['carbs']} g")
    print(f"지방:      {result['fat']} g")
    print(f"신뢰도:    {result['confidence']:.1%}")
    print("="*60)
