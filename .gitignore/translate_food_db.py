"""
food_db.csv의 한국어 음식명을 영어로 번역하여 새로운 컬럼 추가
"""

import pandas as pd
from openai import OpenAI
import time
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI()

def translate_food_names_batch(food_names, batch_size=50):
    """배치로 음식명 번역"""
    translations = []

    for i in tqdm(range(0, len(food_names), batch_size)):
        batch = food_names[i:i+batch_size]

        # 프롬프트 생성
        food_list = "\n".join([f"{idx+1}. {name}" for idx, name in enumerate(batch)])

        prompt = f"""Translate the following Korean food names to English.
Keep the translation concise and use common English food terminology.
Return ONLY the numbered list of translations in the same order, one per line.

Korean food names:
{food_list}

English translations:"""

        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional translator specializing in Korean food terminology."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            # 응답 파싱
            result = response.choices[0].message.content.strip()
            batch_translations = []

            for line in result.split('\n'):
                line = line.strip()
                if line and '. ' in line:
                    # "1. Translation" 형식에서 번역만 추출
                    translation = line.split('. ', 1)[1].strip()
                    batch_translations.append(translation)

            # 배치 크기와 번역 결과 크기가 맞지 않으면 원본 사용
            if len(batch_translations) != len(batch):
                print(f"\nWarning: Batch size mismatch. Using original names.")
                batch_translations = batch

            translations.extend(batch_translations)

            # API rate limit 방지
            time.sleep(0.5)

        except Exception as e:
            print(f"\nError translating batch: {e}")
            # 에러 발생 시 원본 그대로 사용
            translations.extend(batch)
            time.sleep(1)

    return translations


def main():
    print("Loading food_db.csv...")
    df = pd.read_csv('food_db.csv', encoding='cp949')

    print(f"Total rows: {len(df)}")
    print(f"Columns: {df.columns.tolist()[:5]}...")

    # 제품명 컬럼 찾기 (인덱스 1이 제품명)
    food_name_col = df.columns[1]
    print(f"\nFood name column: {food_name_col}")

    # 고유한 음식명만 추출 (중복 제거)
    unique_foods = df[food_name_col].unique().tolist()
    print(f"Unique food names: {len(unique_foods)}")

    # 번역 실행
    print("\nTranslating food names...")
    translations_dict = {}

    # 고유 음식명에 대해서만 번역
    unique_translations = translate_food_names_batch(unique_foods, batch_size=50)

    # 딕셔너리 생성
    for korean, english in zip(unique_foods, unique_translations):
        translations_dict[korean] = english

    # 전체 데이터프레임에 영어명 매핑
    df['food_name_en'] = df[food_name_col].map(translations_dict)

    # UTF-8로 저장
    output_file = 'food_db_with_english.csv'
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved to {output_file}")

    # 샘플 출력
    print("\nSample translations:")
    print(df[[food_name_col, 'food_name_en']].head(10).to_string())

    # 번역 통계
    print(f"\n📊 Translation Statistics:")
    print(f"Total rows: {len(df)}")
    print(f"Unique Korean names: {len(unique_foods)}")
    print(f"Successfully translated: {df['food_name_en'].notna().sum()}")
    print(f"Missing translations: {df['food_name_en'].isna().sum()}")


if __name__ == "__main__":
    main()
