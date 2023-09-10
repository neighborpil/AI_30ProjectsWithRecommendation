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


## Matrix Factorization(행렬 분해)
- 평점 행렬을 두개의 작은 행렬로 쪼개어, 그들의 곱으로 유저의 평점을 예측하는 모델
- 각각 행렬은 유저 잠재 요인(User latent factor), 아이템 잠재 요인(Item latent factor)을 나타냄
- latent factor를 가중치로 두어 실제 평점을 가장 잘 복원 할 수 있는 latent factor를 찾음
- 최적화 방법
    + SGD(Stochastic Gradient Descent)
        - r_ui: 유저 u의 아이템 i에 대한 평점(실제 값)
        - p^T_u q_i: 행렬 P와 Q에서 가져온, 유저 u와 아이템 i에 대한 latent factor의 내적값
    + ALS(Alternating Least Squares)
        + SGD와 동일한 Loss function을 사용
        + 최적화 시에, 두 행렬 중 하나의 행렬을 freeze하고 다른 한 행렬만을 최적화
- latent factor의 경우에는 사람이 해석 할 수 없는 의미가 숨어있는 수치 값으로 나타나게 됨
- 이러한 latent factor를 어떻게 찾는지가 Matrix Factorization 알고리즘의 핵심적 요소
- Matrix Factorization 알고리즘의 종류에는 SVD, NMF 등이 있음

##### Terminology
- convex: 볼록한
- eigenvector: 고유 벡터

### SVD(Singular Value Decomposition)
- 특이값 분해
- 얼굴 이미지를 행렬로 표현하여 중요한 latent factor K개만 사용하여 원래 행렬을 복원한 그림
- k가 늘어갈수록 선명해지지만, 그 증가하는 선명도는 점차 줄어듬
- eigenface: 얼굴 인식을 위한 eigenvector들의 집합

### 특이값 분해의 기하적 설명
- M = UΣV^T
- 고유값 분해(Eigen Decomposition): 정방행렬(m*m)을 고유벡터 행렬과 고유값 행렬로 분해하는 기법
- 특이값 분해(Singular Value Decomposition): 정방이 아닌 m*n 행렬을 U, V라는 직교행렬과 고유값 행렬 Sigma로 분해하는 기법

### truncated-SVD
- SVD 이후 생성된 M개의 singular value 중, 가장 중요한 k개의 singular value만을 남김으로써 truncate함
- 이와 같은 방식으로 U와 V의 low-rank matrix를 구하고, 각 행렬 벡터는 latent factor를 나타내는 low-rank factor로 간주 할 수 있음
- 이 truncate된 행렬로 복원한 기존 행려르이 근사행렬의 원소에는 유저가 평가하지 않은 아이템에 대한 평점 근사치가 들어가 있음

## Factorization Machine
- tree 기반 모델(Contents-based filtering), Matrix Factorization(Collaborative filtering) 이 두가지의 접근 방식을 결합한 방식

- MF 한계
    + Interaction Matrix 형태로 입력을 받음
    + Interaction Matrix가 매우 Sparse할 경우, 잘 동작하지 않는 문제
    + 신규 유저에 대한 추천이 어려움
- (Tree 기반) Regression 모델의 한계
    + Interaction Matrix를 직접적으로 활용 할 수 없음
    + Feature간의 Interaction을 반영 할 수 없음.
- FM은 interaction도 활용하고 featrue도 활용하는 방식
- 구성
    + 각 행이 회귀 문제의 sample과 같이 X 피처와 y 타겟으로 구성
    + 유저 및 아이템을 가리키는 one-hot vector를 구성
    + 이후 추가로 다양한 피처, 암식적 정보등을 포함

##### Terminology
- CVR: Conversion rate, 마케팅에서 광고같은 것을 했을때에 전환률

### FFM
- Field-aware Factorization Machine)
- FM에서 파생된 모델로, CTR(Click Through Rate) prediction 문제에 주로 사용
- FM에선느 feature마다 1개의 latent vector를 사용
- FFM에서는 feature가 갖는 여러 값(field)마다 latent vector를 할당하여, 더욱 복잡한 feature interaction을 학습 할 수 있도록 구성


## 추천 시스템의 평가 방법
1. 정확도 지표(Precision, Recall등) 적절한지 여부를 측정
2. Business Metric과 관련된 기타 지표

- 기존 모델 메트릭은 그대로 쓸 수 없다.
- precision, recall, accuracy등의 지표는 단순히 정답과 예측 값의 비율을 측정. 즉 **순서정보**에 대한 가중치가 전혀 반영되지 않음
- 추천시스템과 같은 Information Retrieval 문제에서는, 맞췄는지 여부 뿐 아니라 얼마나 잘 맞추었는지가 중요함(상대적 순서가 중요)
- 가령, 나와 꼭 맞는 영화를 10번째에 놓아서 맞춘 것과 1번째 놓아서 맞춘 것은 추천의 성능이 다르다고 볼 수 있음
- 따라서 순서 정보까지 반영 할 수 있는 지표가 필요
- 관련된 것이 언제 처음 나왔는지가 중요
    + MRR
    + MAP
    + NDCG

## 정확도 기반 평가 지표
### MRR(Mean Reciprocal Rank)
- binary relevance based metrics로, 이진적으로 좋은 추천인지 나쁜 추천인지를 가려내는 지표
- "유관 상품이 최초로 등장한 곳은 몇번째인가"를 측정함으로써 계산
- 아이템이 3개일때에 관련 있는 것(Relevant Item)이 3번째로 등장하였을 경우는 1/3, 첫번째로 등장하였을 경우는 1 이와같이 측정 후 모든 아이템의 합을 구함
- 장점
    + 계산이 쉽고, 해석이 간단
    + 가장 처음 등장하는 유관 상품에 초점을 둠, 유저가 "나를 위한 최적상품"을 찾고 있을 때에 가장 적합
    + 고객이 잘 알고 있는 아이템을 찾고 있을 때에 적합
- 단점
    + 추천 리스트의 나머지 제품들에 대해서는 측정을 하지 않음. 리스트의 하나에만 집중함
    + 유관 상품을 1개 품고 있는 리스트나 여러개 품고 있는 리스트나 동일하게 취급
    + 유관 상품 목록을 원하는 경우(여러개 유관품목 중 비교를 원하는 경우) 좋은 척도가 아님

### MAP(Mean Average Precision)
- 맞춘 곳까지 precision을 구하고, 또 구한다
- 역시 binary relevance based metrics로, 이진적으로 좋은 추천인지 나쁜 추천인지를 가려내는 지표
- 리스트 내에 유관상품이 등장할때마다 precision을 구하고, 이를 평균(Average Precision)을 냄
- 결과로 생성된 여러 리스트간의 평균을 내서 (Mean AP) 평가하는 방식
- 장점
    + Precision-Recall 곡선 아래의 복잡한 영역을 단일 평점으로 나타낼 수 있음
    + 랭킹이 높은 아이템에 대해서 더 높은 가중치를 부여함. 이는 대부분의 추천 상황을 생각해보면 타당한 처리 방식
- 단점
    + binary 상황에서만 작동함.
    + 평점의 scale이 이진적이지 않을 경우(1~5점의 평점)는 적합하지 않음

### NDCG(Normalized Discounted Cumulative Gain)
- 특정 고객에게 5개의 상품을 추천하는 경우, 고객은 각각의 아이템에 대하여 각각의 유관도(relevance)를 가짐
    + ['사과': 0.9, '오렌지': 0.8, '코카콜라': 0.7, '카시트': 0.6, '휴지': 0.5]
    + A 알고리즘 [사과, 오렌지, 휴지, 코카콜라, 카시트]의 순으로 추천
    + B 알고리즘 [휴지, 오렌지, 카시트, 코카콜라, 사과] 순으로 추천
 
<img width="701" alt="image" src="https://github.com/neighborpil/AI_30ProjectsWithRecommendation/assets/22423285/8297eecd-f7f0-40ad-b6c0-6619d2bf4190">

- CG(Cumulative Gain): i번째 포지션의 평가된 유관도의 누적합(추천된 총 p번까지의 추천 상품들의)
- DCG(Discounted Cumulative Gain): CG의 각 항을 log(i+1)으로 나누어 줌. 이는 후반부(e.g. 100번째)로 갈수록 분모가 커지는 효과를 낳는데 이로써 가장 중요한 품목을 잘 맞추는 것에 대하여 더 높은 점수를 부여
- IDCG(Ideal Discounted Cumulative Gain): Corpus 내 p번자리까지의 유관한 문서들의 목록에 대한 총 누적 이득(도달 할 수 있는 최고의 값)
- NDCG(Normalized Discounted Cumulative Gain): DCG를 IDCG로 나누어 표준화해준 값
- Average NDCG Across User: 유저간의 NDCG의 총합
- 장점
    + 평가된 유관도 값을 고려함
    + MAP에 비교해서 순위 매겨진 품목들의 위치를 평가하는데 우수함. MAP는 유관/무관만 판단하기 때문
    + Logarithmic discounting factor 텀으로 인해 일관성 있는 척도로 사용될 수 있음
- 단점
    + 불완전 ratings가 있을 경우(평가를 안내린 제품) 문제가 있음
    + IDCG가 0일 경우 직접 다루어 주어야함. 이 경우는 유저가 유관한 제품을 가지고 있지 않을 경우 발생

## 기타 평가 지표
- 추천 시스템은 비지니스 종속성이 크므로 다양한 목표를 얼마나 충족하여 추천하는지에 대한 지표들

### Hit rate
- 유저에게 추천한 것 중, 마음에 드는 것이 있는가?

<img width="612" alt="image" src="https://github.com/neighborpil/AI_30ProjectsWithRecommendation/assets/22423285/cdba9a7c-b16c-413e-a808-20bb3bcf1398">

### Diversity
- 추천된 아이템이 얼마나 다른지
- 비슷한 것만 보여주는 것이 아니므로 추천경험을 개선해 줄수 있음
- 아이템간의 유사도를 계산함으로써 평균을 냄

### Novelty
- 참신성. 추천하는 아이템이 유저에게 얼마나 알려지지 않았는지를 측정
- 모두가 알고 있는 곳보다 숨겨진 곳을 알려줄 경우 높게 나옴
- 추천아이템의 평균적인 인기를 구해서 비교, 추천 아이템의 인기가 낮을 수록 좀더 참신하다고 여겨짐

### Serendipity
- 의도적으로 찾지 않았음에도 뭔가 새로운 좋은 것을 발견하는 일
- 정량화 어려움

## XAI(Explainable AI)
- 설명가능한 AI
- 추천 시스템의 특징
    + 도메인 종속성: 서비스되는 도메인에 따라 아이템이 크게 다름
    + 비지니스 목적의 다양성: 다양한 비지니스 목표가 존재
- 의사결정 프로세스, 입력, 출력에 대하여 명확한 설명을 제공함으로써 AI 모델을 보다 투명하고 해석 가능하게 만드는 것이 목표
- XAI 필요성
    + 머신러닝 모델으 디버깅에 필요. 모델의 feature에 대한 선호도를 분석하므로써 insight를 얻을 수 있음
    + 사용자 신뢰성. 추천의 근거를 제시 할 수 있음
    + 모델 편항 분석. 편항된 모델을 판단하고 이를 보정하는데 사용할 수 있음

### 대리분석(Surrogate Analysis)
- 설명하고자 하는 모델이 지나치게 복잡해서 해석하기 어려울 때에, 해석 가능한 **대리 모델**(surrogate model)을 사용하여 기존의 모델을 분석하는 방법
- SVM 분류기와 같이 성능은 좋지만 해석이 어려운 모델이 있을 때에 Logistic regression 모델처럼 설명 가능성은 높지만 성능은 낮은 모델을 대리 모델로 사용해 해당 모델의 계수를 기반으로 모델 판단 메커니즘을 어림짐작 하는 것
- 모델에 상관없이 적용 가능
- global 대리분석: 전체 모델의 중요 변수를 새로운 모델로 대체해서 파악하는 방법
- local 대리분석: 개별 샘플에 대한 모델의 판단을 분석하는 방법 

### LIME 알고리즘
- XAI 알고리즘의 한 종류
- Local Interpretable Model-agnostic Explanations
    + Interpretable Explanation: 각 예측을 내림에 있어 어떤 feature가 사용되었는지에 대한 설명을 제공한다는 의미
    + Local: observation specific하다는 것을 의미. **한 개인 또는 샘플**에 대하여 내려진 판단이 어떻게 내려진 것인지를 분석
    + Model-agnostic: 어떤 모델을 사용하든지 간에 무관하게 사용될 수 있음을 의미
- 학습 메커니즘
  + 결정 경계를 분석하여 판단 메커니즘을 제공
  1. 데이터 뒤섞기(permute)
  2. 뒤섞은 데이터와 기존 관측치 사이의 거리 측정
  3. blackbox model을 사용해 새로운 데이터를 대상으로 예측 수행
  4. 뒤섞은 데이터로부터 복잡한 모델의 출력을 가장 잘 설명하는 m개의 feature 선택
  5. 여기서 뽑은 m개의 feature로 뒤섞은 데이터를 대상으로 단순한 모델 적합, 유사도 점수를 가중치로 사용
  6. 단순 모델의 가중치는 곧 복잡한 모델의 local한 행동을 설명하는데 사용됨
- LIME의 손실함수

 <img width="1040" alt="image" src="https://github.com/neighborpil/AI_30ProjectsWithRecommendation/assets/22423285/8f0e871d-e0fc-4c45-bb7d-9b09d865dad8">

### SHAP
- SHapley Additive exPlanaitions
- 머신러닝 모델의 예측 결과를 설명하기 위한 알고리즘 중 하나로, Shapley value에 기반함
- Shapley value: 게임이론에서 도입된 개념으로, feature의 전체 결과에 대한 기여도에 따라 게임 내 각 플레이어에게 공을 공평하게 나눌 수 있도록 게산한 값
- feature가 추가되거나 빠질때에 모델의 결과가 얼마나 달라지는지에 대하여 계산

<img width="1077" alt="image" src="https://github.com/neighborpil/AI_30ProjectsWithRecommendation/assets/22423285/b4331560-6e6d-4bc5-b446-4aee87fc25ad">






































 





