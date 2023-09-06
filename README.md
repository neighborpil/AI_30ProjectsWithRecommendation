# AI_30ProjectsWithRecommendation

## Terminology
- Filter bubble: 추천결과와 유저 반응이 상호작용하여 한정된 주제에 갇히는 현상
- 연관규칙 추천: 같이 자주 구매된 상품을 함께 추천하는 기법


## Traditional methods
- Collaborative filtering: recommend similar users's choice
- Contents-based filtering: recommend similar items
- Precautions
    + 변수의 종류에 따른 다른 처리 필요
        - 연속형 변수: 정규화 후 수치 형태로 사용
        - 범주형 변수: one-hot 등의 가공 방식 사용
- 유사도 계산 방법: 각각을 대칭 행렬로 만들어서 계산
    + Cosine Similarity
    + Euclidean Distance
    + Pearson correlation coeffeicient

### Contents-based filtering
- 장점
    + 적은 상호작용 데이터에서도 사용 가능
    + 유저-아이템간 상호작용이 거의 없는 경우에도 사용 가능, cold start 상황에 적합
    + 어떤 feature가 추천에 도움이 되었는지 근거를 찾을 수 있음
- 단점
    + 아이템 feature의 한계: feature정보로만 유사도를 계산하기 때문에 feature의 품질에 크게 영향을 받음
    + 신규유저 cold start: 아무런 아이템도 클릭하지 않은 유저에게는 추천이 어려움
- 컨텐츠 필터링 구현 절차
    1. 피처 추출
         + 아이템간 유사성 계산 위해 아이템 특징 추출
         + 일반적인 아이템은 상품의 특징을 담은 tabular feature 사용
         + 텍스트의 경우 TF-IDF와 같은 기법을 적용
    2. 사용자 프로필 정보 생성
    3. 유사도 계산: 코사인 유사도 자주 사용
    4. 랭킹: 상위 아이템 추천
 
##### Terminology
- BoW(Bag of Words)
- 불용어(stop words) : a, the, 관사 등 크게 의미지 있지 않은 단어들
- TF-IDF: 여기서 자주 나오는 단어는 중요하지만, 어디서나 자주 나오는 단어는 중요하지 않다
    + W_x_y = tf_x_y * log(N / df_x)
    + TF(Term Frequence): 문장 내에서 자주 등장하는 단어는 더 중요하다, 문서 y내에서 x라는 단어의 등장 빈도
    + IDF(Inverse Document Frequency): 여러 문장에 걸쳐 자주 등장하는 단어는 별로 안 중요하다,
        - df_x: x라는 단어를 포함하고 있는 문서 수
        - N: 전체 문서의 수
        - log: 지나치게 큰 수가 나오는 것을 방지하기 위한 normalize 기능 수행
     
        - 
