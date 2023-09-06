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
 
##### * Terminology
- BoW(Bag of Words)
- 불용어(stop words) : a, the, 관사 등 크게 의미지 있지 않은 단어들
- TF-IDF: 여기서 자주 나오는 단어는 중요하지만, 어디서나 자주 나오는 단어는 중요하지 않다
    + W_x_y = tf_x_y * log(N / df_x)
    + TF(Term Frequence): 문장 내에서 자주 등장하는 단어는 더 중요하다, 문서 y내에서 x라는 단어의 등장 빈도
    + IDF(Inverse Document Frequency): 여러 문장에 걸쳐 자주 등장하는 단어는 별로 안 중요하다,
        - df_x: x라는 단어를 포함하고 있는 문서 수
        - N: 전체 문서의 수
        - log: 지나치게 큰 수가 나오는 것을 방지하기 위한 normalize 기능 수행
     

### Collaborative filtering(CF)
- 너가 좋아했던 그거, **그거 찾던 애들**은 이것도 찾던데?
- 행: 유저, 열: 아이템, 각각의 아이템에 대한 유저의 평점을 사용하여 interaction matrix 만듬
- 구현 절차
    1. 유저-아이템 상호작용 행렬 생성: 유저, 아이템, 평가
    2. 유사도 계산
         + 유저기반 협업필터링의 경우 평가한 아이템의 index를 feature로 간주
         + 아이템기반 협업필터링의 경우 각 아이템을 평가한 유저들의 index를 feature로 간주
         + 이러한 관점에서 각 유저별/아이템별 유사도를 계산
    4. 이웃 지정
         + 추천 대상의 스코어를 생성할 이웃 유저/아이템을 지정
         + 이웃의 수는 고정시킬수도, 그때그때 적용할 수도 있음
         + 보통 높은 유사도의 유저 n명을 사용하는 전략을 쉽게 떠울릴 수 있음
    6. 랭킹
         + 이웃들의 상호작용 정보를 토대로 추천 스코어를 계산
- 장점
    + 개인화된 추천
    + 잠재적 특징 사용
        - 아이템/유저 feature로 명시적으로 드러나지 않은 선호 정보도 반응 정보로부터 추출하여 사용
    + 데이터가 쌓일수록 성능이 높아짐
- 단점
    + 아이템 cold start: 한번도 추천되지 않은 아이템은 추천되기 어려움
    + 유저 cold start: 과거 상호작용 데이터에 의존하기 때문에 cold user에게는 추천이 어려움
    + 낮은 확장가능성: 아이템/유저의 수가 늘어남에 따라 연산량이 급격히 증가함
 
### Hybrid filtering(Contents-based filtering & Collaborative filtering)
- 두개 다 사용, 성능면에서 좋음
- 초반엔 컨텐츠 필터링으로, 상호작용이 많은 고객은 협업 필터링으로
- Ensemble 기법
    + 여러 모델의 결과값에 평균 내어 사용
    + 모델의 성능을 기준으로 좋은 성능에 큰 점수 가중치 부여
- Switching
    + 콜드 유저를 대상으로 아이템 기반 추천, 활동 기록이 쌓인 유저는 협업 필터링으로
- Feature combination
    + 앞선 모델(보통 협업 필터링)의 추천 결과를 이후에 등장하는 메인 모델의 feature로 사용
    + 협업 필터링과 추천 결과를 고려해 컨텐츠 필터링을 적용함으로써 두 모델의 장점을 적절히 조합 가능
    + 임베딩 + 부스ㅇ 모델 결합을 이 모델의 예시로 볼 수 있음
- 장점
    + 성능 향상
    + 추천 범위, 다양성이 증가 & cold start문제 개선
- 단점
    + 복잡도 증가

















