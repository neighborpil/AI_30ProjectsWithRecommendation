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

### 추천 방법
- 메모리 기반 추천
    + 추천이 필요할 때마다 연산하여 추천을 제공
    + 구현이 쉽지만 계산이 오래 걸리며, 제한된 성능
    + 협업 필터링, Contents-based filtering
- 모델 기반 추천
    + 다량의 데이터로 학습한 모델을 통하여 리소스 효율적인 추천 제공
    + 학습에 많은 데이터와 리소스가 필요하지만 여러 feature 정보로부터 패턴을 학습함으로써 상대적으로 더 높은 성능을 얻음

### 문제 해결 방법
1. 문제 정의
2. 모델 정의
3. 가르치기
4. 평가하기

##### Terminology
- EDA:  Exploratory Data Analysis,

#### 함수와 모형(Model)
- 머신러닝에서는 x가 주어졌을 때에 y를 뱉어내는 함수를 얻는 것이 목적
- 모델 종류
    + 결정적 모형: 무결하고 완전한 관계를 나타내는 모형. 섭씨/화씨
    + **통계적 모형**: 입력과 출력의 경향성을 나타내는 모형. 오차항을 포함. 몸무게와 키의 상관관계
 
### 손실함수 대표 2개
- 회귀: MSE, RMSE
- 분류: Cross Entropy, 실제값(P)과 예측값(Q)의 확률 분포 차이
    + H(P, Q) = - Sigma P(X_i)log_2Q(X_i)

### 검증 방식
- Hold-Out: Training set / Validation Set
- K-fold cross-validation
- LOOCV: Leave One Cross Validation


## Confusion Matrix
TP - FP <br />
FN - TN

- FP: Type1 error, 아닌데 맞다고 하는 경우
- FN: Type2 error, 맞는데 아니다고 하는 경우

- Precisoin = ΣTP / Σ(TP + FP), 참이라고 한 것 중에 실제로 몇개가 참이었는지
- Recall = ΣTP / Σ(TP + FN), 실제 true인 것 중 몇개를 true로 예측하였는지
- Accuracy = Σ(TP + TN) / Σ(TP + FP + FN + TN)
- F1 score = 2 * (Precision * Recall) / (Precision + Recall)

### AUC(Area Under Curve) - 무작위로 찍은 것보다 얼마나 잘했냐
-  ROC curve의 하단 부의 넓이 비율을 나타내는 값으로 값이 높을수록 좋음
-  ROC(Receiver-Operating Characteristic) curve: 이진(binary) 분류 모델의 threshold에 따라 변화하는 성능 값을 나타낸 곡선
    + X축: FP rate, Y축: TP rate
    + Perfect classifier에 가까울수록 성능이 좋다(즉 AUC가 크다)
    + Random classifier에 가까울수록 성능이 낮다(즉 AUC가 작다)
- 추천시스템에서 사용
    + 추천시스템에서는 class 불균형한 데이터가 대부분이다
    + AUC는 class 불균형에 무관하다
    + 추천에는 절대적인 값보다 어느것보다 좋다는 비교를 하는것이라 AUC는 전반적으로 나타내기 때문에 좋다?


### Ensenble 기법
- 배깅(Bagging) : Bootstrap Aggregating의 약자, 다르게 샘플링한 데이터로 각각 다른 모델을 학습
    + bootstrap: 원본 데이터셋에서 복원 추춸(sampling with replacement)한 데이터셋, 데이터의 다양성 확보, 분산을 줄이는 것이 목표
    + aggregation: 각 데이터셋에서 샘플링 된 결과를 결합
- 부스팅(Boosting): 모델을 순차적으로 학습시키며, 이전 모델의 약점을 해결하며 성능을 높임
    + 여러 weak learner를 사용하여 순차적으로 학습
    + **이전 차례에 잘 못 맞춘 문제에 집중**하여 더 잘 풀도록 유도하는 알고리즘
    + Bagging과는 달리, 이전 모델의 예측 성능에 따라 weighted sampling을 수행함
    + 모델의 **편향**을 줄이는데 목표
    + XGBoostt(Level-wise tree grouth adopted): 모든 feature에 대하여 동등하게 학습
    + LGBM(Leaf-wise tree grouth adopted): 구분을 잘하는 feature에 대하여 집중하여 학습
        - 장점
            + 높은 성능, prediction/classification에서 딥러닝 방법론에 비교할 수 있을 정도로 추천성능이 높므
            + 확장 가능성, feature 히스토그램화 방식으로 자원의 효율적인 학습이 가능함
            + 구현 및 관리의 효율성: 범용 알고리즘의 특성상 API가 잘 활발히 관리/개선되고 있음
            + 타 모듈(하이퍼 파라미터 최적화, 서빙 관련 등)과 호환성이 높음
            + 해석 가능성: 행렬분해 기반 방법론에 배해 모델의 판단을 해석하기 용이함
            + 자체적으로 제공하는 Feature Importance 및 기타 XAI 알고리즘을 손쉽게 적용할 수 있음
    + GOSS(Gradient-based One-side Sampling): 구분을 잘하는 feature에 대하여 가중치를 주어 학습, 마이너한 feature에 대해서도 작은 비율이지만 학습에 사용함으로써 효율성을 높일 수 있다
- 스태킹(Stacking): 서로 다른 모델의 출력에서 최종 출력을 만드는 메타 모델, 여러개의 모델의 결과값을 feature처럼 학습시켜 다시 사용


### Matrix Factorization(행렬 분해)
- 평점 행렬을 두개의 작은 행렬로 쪼개어, 그들의 곱으로 유저의 평점을 예측하는 모델
- 각각 행렬은 유저 잠재 요인(User latent factor), 아이템 잠재 요인(Item latent factor)을 나타냄






















