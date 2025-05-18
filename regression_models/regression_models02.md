# 회귀분석 02 - 중회귀모형(1)

---

## 목차

**I. 중회귀모형의 기본 개념**
1. [중회귀모형의 정의와 성격](#1-중회귀모형-multiple-linear-regression-model)
2. [중회귀모형의 수학적 표현](#2-중회귀모형의-수학적-표현)
3. [행렬을 이용한 표현 방식](#3-행렬을-이용한-표현-방식)

**II. 다중공선성과 모형 추정**
1. [다중공선성의 개념과 문제점](#4-다중공선성의-개념과-문제점)
2. [다중공선성의 해결 방법](#5-다중공선성의-해결-방법)
3. [최소제곱법을 통한 계수 추정](#6-최소제곱법을-통한-계수-추정)

**III. 모형 평가와 해석**
1. [분산분석과 F-검정](#7-분산분석과-f-검정)
2. [결정계수와 조정된 결정계수](#8-결정계수와-조정된-결정계수)
3. [회귀계수의 해석과 유의성](#9-회귀계수의-해석과-유의성)

---

## I. 중회귀모형의 기본 개념

### 1. 중회귀모형의 정의와 성격

- **중회귀모형의 기본 개념**:
    - 종속변수(Y)의 변화를 설명하기 위하여 **두 개 이상의 독립변수(X)**가 사용되는 선형회귀모형을 의미합니다.
    - **중선형회귀(Multiple Linear Regression Model)** 또는 간단히 **중회귀모형(Multiple Regression Model)**이라고도 부릅니다.
- **독립변수의 수가 k개인 중회귀모형의 수학적 표현**:
    - $Y_i = \beta_0 + \beta_1 X_{i1} + \beta_2 X_{i2} + \dots + \beta_k X_{ik} + \epsilon_i$
        - $Y_i$: i번째 관측치의 종속변수 값
        - $X_{ij}$: i번째 관측치의 j번째 독립변수 값
        - $\beta_0$: **절편 (Intercept)** (모집단의 y절편)
        - $\beta_j$ (j=1, ..., k): j번째 독립변수에 대한 **회귀계수 (Regression Coefficient)**
            - 다른 모든 독립변수들이 일정하다고 가정할 때, 해당 독립변수 $X_j$가 한 단위 증가할 때 종속변수 Y의 평균 변화량 (부분 회귀계수).
        - $\epsilon_i$: **오차항 (Error Term)**
            - 서로 독립이며, 평균이 0이고 분산이 $\sigma^2$인 정규분포를 따르는 확률변수로 가정.
- **행렬을 이용한 중회귀모형 표현**:
    - n개의 관측치에 대한 중회귀모형을 행렬과 벡터를 사용하여 간결하게 표현할 수 있습니다.
    - $\mathbf{Y} = \mathbf{X\beta} + \mathbf{\epsilon}$
        - $\mathbf{Y}$: (n x 1) 종속변수 벡터
        $\mathbf{Y} = \begin{pmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_n \end{pmatrix}$
        - $\mathbf{X}$: (n x (k+1)) 독립변수 행렬 (첫 번째 열은 절편을 위해 1로 채워짐)
        $\mathbf{X} = \begin{pmatrix} 1 & X_{11} & X_{12} & \dots & X_{1k} \\ 1 & X_{21} & X_{22} & \dots & X_{2k} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & X_{n1} & X_{n2} & \dots & X_{nk} \end{pmatrix}$
        - $\mathbf{\beta}$: ((k+1) x 1) 회귀계수 벡터
        $\mathbf{\beta} = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \vdots \\ \beta_k \end{pmatrix}$
        - $\mathbf{\epsilon}$: (n x 1) 오차 벡터
        $\mathbf{\epsilon} = \begin{pmatrix} \epsilon_1 \\ \epsilon_2 \\ \vdots \\ \epsilon_n \end{pmatrix}$
            - $\mathbf{\epsilon}$은 다변량 정규분포 $N(\mathbf{0}, \sigma^2 \mathbf{I})$를 따른다고 가정 (여기서 $\mathbf{I}$는 단위행렬).
- **예시 데이터 (표본상점의 총판매액 자료, `market-2.csv`)**:
    - 종속변수(Y): 총판매액
    - 독립변수(X1): 인테리어비
    - 독립변수(X2): 상점의 크기

### 2. 중회귀모형의 추정

- **회귀계수의 추정**:
    - **최소제곱법 (Method of Least Squares, OLS)**:
        - 오차제곱합 $SSE = \sum_{i=1}^{n} (Y_i - (\hat{\beta}_0 + \hat{\beta}*1 X*{i1} + \dots + \hat{\beta}*k X*{ik}))^2$ 를 최소화하는 회귀계수 추정량 $\hat{\mathbf{\beta}}$를 찾는 방법.
        - 행렬 표현으로는 $SSE = (\mathbf{Y} - \mathbf{X}\hat{\mathbf{\beta}})^T (\mathbf{Y} - \mathbf{X}\hat{\mathbf{\beta}})$ 를 최소화.
        - 이를 $\hat{\mathbf{\beta}}$에 대해 미분하여 0으로 놓고 풀면, 최소제곱추정량 $\hat{\mathbf{\beta}}$는 다음과 같이 구해집니다:
            - $\hat{\mathbf{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}$
            - $(\mathbf{X}^T \mathbf{X})^{-1}$: $\mathbf{X}^T \mathbf{X}$ 행렬의 역행렬.
- **R 활용: 행렬 연산을 이용한 회귀계수 직접 계산 (예제: `market-2.csv`)**:
    - 데이터 로드: `market2 = read.csv("c:/data/reg/market-2.csv")`
    - 독립변수 행렬 $\mathbf{X}$ 생성:
        - `X_vars = market2[,c(2:3)]` (X1, X2 컬럼 선택)
        - `X = cbind(1, X_vars)` (절편항을 위해 첫 번째 열에 1 추가)
        - `X = as.matrix(X)` (행렬로 변환)
    - 종속변수 벡터 $\mathbf{Y}$ 생성:
        - `Y = market2[,4]`
        - `Y = as.matrix(Y)` (행렬로 변환)
    - $\mathbf{X}^T \mathbf{X}$ 계산: `XTX = t(X) %*% X`
    - $(\mathbf{X}^T \mathbf{X})^{-1}$ 계산: `XTXI = solve(XTX)` (`solve` 함수는 역행렬 계산)
    - $\mathbf{X}^T \mathbf{Y}$ 계산: `XTY = t(X) %*% Y`
    - $\hat{\mathbf{\beta}}$ 계산: `beta = XTXI %*% XTY`
    - 결과 (소수점 3자리 반올림):
        - $\hat{\beta}_0$ (절편): `0.850`
        - $\hat{\beta}_1$ (X1 계수): `1.558`
        - $\hat{\beta}_2$ (X2 계수): `0.427`
    - **적합된 회귀식**: $\hat{Y} = 0.850 + 1.558 X_1 + 0.427 X_2$
    - **해석 예시**: 인테리어비(X1)가 10,000만원이고 상점 크기(X2)가 100 $m^2$일 때, 평균 총판매액($\hat{Y}$)은 약 20.7 (10,000만원)으로 추정.
- **잔차(Residual)의 성질**:
    - $\mathbf{e} = \mathbf{Y} - \hat{\mathbf{Y}} = \mathbf{Y} - \mathbf{X}\hat{\mathbf{\beta}}$
    - 여기서 $\hat{\mathbf{Y}} = \mathbf{X}\hat{\mathbf{\beta}} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y} = \mathbf{HY}$
    - $\mathbf{H} = \mathbf{X}(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T$ 를 **햇 행렬(Hat Matrix)**이라고 하며, 이는 대칭이고 멱등행렬(Idempotent Matrix, $\mathbf{H}^2 = \mathbf{H}$)입니다.
    - 잔차의 성질:
        1. 잔차의 합은 0: $\sum e_i = \mathbf{1}^T \mathbf{e} = 0$
        2. 각 독립변수와 잔차의 가중합(내적)은 0: $\mathbf{X}^T \mathbf{e} = \mathbf{0}$
        3. 적합값과 잔차의 가중합(내적)은 0: $\hat{\mathbf{Y}}^T \mathbf{e} = 0$
        4. 오차항 $\epsilon_i$는 서로 독립적이지만, 잔차 $e_i$들은 일반적으로 서로 상관관계를 가집니다.
            - $E(\mathbf{e}) = \mathbf{0}$
            - $Var(\mathbf{e}) = \sigma^2 (\mathbf{I} - \mathbf{H})$: 공분산 행렬이 대각행렬이 아니므로 잔차 간 공분산 존재.

### 3. 중회귀 분산분석표 (ANOVA Table)

- **회귀방정식의 신뢰성 측도**:
    1. **분산분석표에 의한 F-검정**: 모형 전체의 유의성 검정.
    2. **결정계수 ($R^2$)**: 모형의 설명력.
    3. **잔차평균제곱 (MSE)**: 모형의 적합도, 예측 오차의 크기.
    4. 추정된 회귀계수들의 분산 (표준오차).
    5. 종속변수 추정량($\hat{Y}$)의 분산.
- **변동의 분해**:
    - 총제곱합 (SST): $SST = \sum (Y_i - \bar{Y})^2 = \mathbf{Y}^T \mathbf{Y} - n \bar{Y}^2$
    - 회귀제곱합 (SSR): $SSR = \sum (\hat{Y}_i - \bar{Y})^2 = \hat{\mathbf{\beta}}^T \mathbf{X}^T \mathbf{Y} - n \bar{Y}^2$
    - 잔차제곱합 (SSE): $SSE = \sum (Y_i - \hat{Y}_i)^2 = \mathbf{Y}^T \mathbf{Y} - \hat{\mathbf{\beta}}^T \mathbf{X}^T \mathbf{Y}$
    - $SST = SSR + SSE$
- **분산분석표 (ANOVA Table)**:
    
    
    | 요인 (Source) | 자유도 (Df) | 제곱합 (Sum Sq) | 평균제곱 (Mean Sq) | F 값 (F value) | 유의확률 (Pr(>F)) |
    | --- | --- | --- | --- | --- | --- |
    | 회귀 (Regression) | k (독립변수 개수) | SSR | MSR = SSR / k | F = MSR / MSE | p-value |
    | 잔차 (Residuals) | n-(k+1) | SSE | MSE = SSE / (n-k-1) |  |  |
    | 계 (Total) | n-1 | SST |  |  |  |
- **R 활용 예제 (분산분석표, `market-2.csv`)**:
    - `market2_lm = lm(Y ~ X1+X2, data=market2)`: 중회귀모형 적합.
    - `summary(market2_lm)` 결과 중 일부:
        - `Coefficients`: 각 변수(X1, X2) 및 절편의 추정치, 표준오차, t-값, p-값.
            - X1과 X2 모두 p-값이 매우 작으므로 (***) 통계적으로 유의함.
        - `Residual standard error: 0.9318 on 12 degrees of freedom` ($\sqrt{MSE}$)
        - `Multiple R-squared: 0.9799` ($R^2$)
        - `Adjusted R-squared: 0.9765` (수정 $R^2$)
        - `F-statistic: 292.5 on 2 and 12 DF, p-value: 6.597e-11` (모형 전체의 F-검정 통계량 및 p-값)
    - `anova(market2_lm)`: 분산분석표 출력.
        
        
        |  | Df | Sum Sq | Mean Sq | F value | Pr(>F) |
        | --- | --- | --- | --- | --- | --- |
        | X1 | 1 | 485.57 | 485.57 | 559.283 | 1.955e-11 \*\*\* |
        | X2 | 1 | 22.30 | 22.30 | 25.691 | 0.0002758 \*\*\* |
        | Residuals | 12 | 10.42 | 0.87 |  |  |
        - **주의**: R의 `anova()` 함수는 Type I SS (Sequential Sum of Squares)를 기본으로 제공하므로, 변수 입력 순서에 따라 각 변수의 제곱합이 달라질 수 있습니다. 모형 전체의 F-검정은 `summary()` 결과를 참조하는 것이 더 일반적입니다.
- **결정계수 ($R^2$, Coefficient of Determination)**:
    - $R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$
    - 중회귀모형에서는 **중상관계수(Multiple Correlation Coefficient, R)**의 제곱과 같습니다.
    - 중상관계수 R은 종속변수 Y와 예측값 $\hat{Y}$ 사이의 단순상관계수와 같습니다.
    - R 코드: `cor(market2$Y, market2_lm$fitted.values)^2` $\approx$ `0.9799` (`summary` 결과의 $R^2$와 일치)
- **수정 결정계수 (Adjusted $R^2$)**:
    - $R^2_{adj} = 1 - \frac{SSE/(n-k-1)}{SST/(n-1)} = 1 - (1-R^2)\frac{n-1}{n-k-1}$
    - 독립변수의 수가 증가하면 $R^2$는 항상 증가하거나 최소한 감소하지 않으므로, 불필요한 변수가 추가되어도 $R^2$가 커지는 것을 보정하기 위해 사용됩니다.
    - 모형의 설명력을 비교할 때 유용합니다. (값이 클수록 좋음)
    - `summary(market2_lm)` 결과: `Adjusted R-squared: 0.9765`
- **잔차평균제곱 (MSE, Residual Mean Squares)**:
    - $MSE = \frac{SSE}{n-k-1}$
    - 오차 분산 $\sigma^2$의 비편향 추정량입니다.
    - 이 값의 제곱근($\sqrt{MSE}$)이 `summary()` 결과의 `Residual standard error`입니다.
    - R 코드: `sqrt(10.42/12)` (anova 결과의 SSE와 잔차 df 사용) $\approx$ `0.9318`

---
---

## II. 다중공선성과 모형 추정

### 4. 표준화된 중회귀분석

- **변수 표준화 (Variable Standardization)**:
    - 각 변수(독립변수 및 종속변수)에서 해당 변수의 평균을 빼고 표준편차로 나누어, 평균이 0이고 표준편차가 1인 변수로 변환하는 과정입니다.
    - $Z_X = \frac{X - \bar{X}}{s_X}$, $Z_Y = \frac{Y - \bar{Y}}{s_Y}$
- **표준화된 회귀모형 (Standardized Regression Model)**:
    - 표준화된 변수들을 사용하여 적합한 회귀모형입니다.
    - $Z_Y = \beta_1^* Z_{X1} + \beta_2^* Z_{X2} + \dots + \beta_k^* Z_{Xk} + \epsilon^*$
    - **표준화된 회귀계수 ($\beta_j^*$)**:
        - 다른 독립변수들이 일정할 때, 해당 표준화된 독립변수 $Z_{Xj}$가 1 표준편차만큼 변할 때 종속변수 $Z_Y$가 평균적으로 몇 표준편차만큼 변하는지를 나타냅니다.
        - **장점**: 독립변수들의 측정 단위가 서로 다를 때, 각 독립변수가 종속변수에 미치는 **상대적인 영향력(중요도)을 비교**하는 데 유용합니다. (절편항은 0이 됨)
- **R 활용: 표준화 회귀모형 (예제: `market-2.csv`)**:
    - `install.packages("lm.beta")` (필요시 패키지 설치)
    - `library(lm.beta)`
    - `market2_lm = lm(Y ~ X1+X2, data=market2)` (일반 회귀모형 적합)
    - `market2_beta = lm.beta(market2_lm)`: 표준화된 회귀계수 계산.
    - `print(market2_beta)` 또는 `summary(market2_beta)` 결과:
        
        
        | Coefficients | Estimate (원래) | Standardized (표준화) | Std. Error | t value | Pr(>|t|) |
        | --- | --- | --- | --- | --- | --- |
        | (Intercept) | 0.85041 | 0.00000 | 0.84624 | 1.005 | 0.334770 |
        | X1 | 1.55811 | 0.70156 | 0.14793 | 10.532 | 2.04e-07 \*\*\* |
        | X2 | 0.42736 | 0.33761 | 0.08431 | 5.069 | 0.000276 \*\*\* |
        - **해석**: X1(인테리어비)이 1 표준편차 증가할 때 Y(총판매액)는 약 0.70 표준편차 증가하고, X2(상점크기)가 1 표준편차 증가할 때 Y는 약 0.34 표준편차 증가합니다. 따라서 이 모형에서는 X1이 X2보다 Y에 미치는 상대적 영향력이 더 크다고 볼 수 있습니다.

### 5. 중회귀모형의 추정과 검정 (심화)

- **추정된 회귀계수의 분산-공분산 행렬**:
    - $Var(\hat{\mathbf{\beta}}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$
    - 이 행렬의 대각 원소는 각 회귀계수 추정치($\hat{\beta}_j$)의 분산을 나타내고, 비대각 원소는 회귀계수 추정치 간의 공분산을 나타냅니다.
    - $\sigma^2$은 보통 잔차평균제곱(MSE)으로 추정합니다: $\hat{\sigma}^2 = MSE$.
- **회귀계수의 구간 추정**:
    - 단순회귀와 마찬가지로 각 회귀계수 $\beta_j$에 대한 신뢰구간을 구할 수 있습니다.
    - $\hat{\beta}*j \pm t*{(\alpha/2, n-k-1)} \cdot SE(\hat{\beta}_j)$
    - 여기서 $n-k-1$은 잔차의 자유도입니다.
- **기댓값 및 예측값 신뢰구간**:
    - 중회귀에서도 특정 독립변수 값들의 조합($\mathbf{X}*h$)에 대한 평균 반응 $E(Y_h)$의 신뢰구간과 새로운 관측치 $Y*{h(new)}$의 예측구간을 구할 수 있습니다.
- **R 활용 예제: 신뢰구간 (예제: `market-2.csv`)**:
    - X1=10, X2=10일 때의 평균 반응($E(Y)$)에 대한 신뢰구간.
    - `pred_x = data.frame(X1=10, X2=10)`
    - **95% 기댓값 신뢰구간**:
        - `pc = predict(market2_lm, interval="confidence", newdata=pred_x)`
        - 결과: `fit = 20.70503`, `lwr = 19.95796`, `upr = 21.45209`
    - **99% 기댓값 신뢰구간**:
        - `pc99 = predict(market2_lm, interval="confidence", level=0.99, newdata=pred_x)` (level 옵션으로 신뢰수준 변경)
        - 결과: `fit = 20.70503`, `lwr = 19.65769`, `upr = 21.75236` (95%보다 구간 폭이 넓어짐)
- **회귀계수 가설검정**:
    - 각 회귀계수 $\beta_j$가 0인지 (즉, 해당 독립변수가 종속변수에 유의한 영향을 미치는지) 검정합니다.
    - $H_0: \beta_j = 0$ vs $H_1: \beta_j \neq 0$
    - 검정통계량: $t_0 = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)}$
    - `summary(market2_lm)` 결과의 `Coefficients` 테이블에서 각 변수에 대한 t-값과 p-값을 확인하여 유의성을 판단합니다. (이전 강의에서 다룬 내용과 동일)
- **일반적 모형 비교 (General Linear Hypothesis Test)**:
    - 두 개의 중첩된(nested) 회귀모형을 비교하여 특정 변수(들)의 집합이 통계적으로 유의한지 검정하는 일반적인 방법입니다.
    - **축소모형 (Reduced Model)**: 일부 변수들이 제외된 모형 (예: $Y \sim X_1$)
    - **완전모형 (Full Model)**: 축소모형에 추가 변수들이 포함된 모형 (예: $Y \sim X_1 + X_2 + X_3$)
    - 검정통계량 F는 두 모형의 잔차제곱합(SSE)의 차이를 이용하여 계산됩니다.
        - $F = \frac{(SSE_R - SSE_F) / (df_R - df_F)}{SSE_F / df_F}$
        - $SSE_R, df_R$: 축소모형의 잔차제곱합과 잔차 자유도
        - $SSE_F, df_F$: 완전모형의 잔차제곱합과 잔차 자유도

### 6. 변수 추가의 영향

- **추가제곱합 (Extra Sum of Squares)**:
    - 기존 모형에 새로운 변수를 추가했을 때, 회귀제곱합(SSR)이 추가적으로 얼마나 증가하는지를 나타내는 값입니다.
    - $SSR(X_j | X_1, \dots, X_{j-1}) = SSR(X_1, \dots, X_j) - SSR(X_1, \dots, X_{j-1})$
    - 이 값이 클수록 새로 추가된 변수 $X_j$가 종속변수의 변동을 설명하는 데 더 많이 기여한다는 의미입니다.
    - **R 활용: 추가제곱합 및 모형 비교 (예제: `health.csv`)**:
        - `health = read.csv("c:/data/reg/health.csv")`
        - 모형 1 (축소모형): `h1_lm = lm(Y ~ X1, data=health)`
        - 모형 2 (완전모형): `h2_lm = lm(Y ~ X1+X4, data=health)` (X4 변수 추가)
        - `anova(h1_lm, h2_lm)`: 두 모형 비교.
            
            
            | Model | Res.Df | RSS | Df | Sum of Sq (추가제곱합) | F | Pr(>F) |
            | --- | --- | --- | --- | --- | --- | --- |
            | 1 | 28 | 50795 |  |  |  |  |
            | 2 | 27 | 24049 | 1 | 26746 | 30.027 | 8.419e-06 \*\*\* |
            - **해석**: X4를 추가함으로써 설명되는 제곱합(추가제곱합)이 26746이고, 이에 대한 F-검정 결과 p-값이 매우 작으므로 X4는 Y를 설명하는 데 유의한 변수라고 할 수 있습니다.
- **추가변수그림 (Added Variable Plot 또는 Partial Regression Plot)**:
    - 특정 독립변수가 다른 독립변수들의 영향을 제거한 후에도 종속변수에 대해 얼마나 추가적인 설명력을 갖는지 시각적으로 보여주는 그래프입니다.
    - **그리는 절차**: (예: 변수 $X_k$의 효과를 보기 위해)
        1. $Y$를 $X_1, \dots, X_{k-1}$로 회귀시킨 후 잔차 $e(Y | X_1, \dots, X_{k-1})$를 구합니다.
        2. $X_k$를 $X_1, \dots, X_{k-1}$로 회귀시킨 후 잔차 $e(X_k | X_1, \dots, X_{k-1})$를 구합니다.
        3. 두 잔차 벡터 간의 산점도를 그립니다. 이 산점도의 기울기는 중회귀모형에서 $X_k$의 회귀계수 $\hat{\beta}_k$와 같습니다.
    - **R 활용: 추가변수그림 (예제: `health.csv`)**:
        - `library(car)` (패키지 필요)
        - `h4_lm = lm(Y ~ X1+X2+X3+X4, data=health)`
        - `avPlots(h4_lm)`: 각 독립변수에 대한 추가변수그림을 그려줍니다.
            - 그림에서 점들이 뚜렷한 선형 관계를 보이면 해당 변수가 유의미한 설명력을 가짐을 시사합니다.

## III. 모형 평가와 해석

### 7. 잔차의 검토 및 분석 사례

- **잔차의 검토**:
    - 중회귀모형의 가정(선형성, 등분산성, 정규성, 오차의 독립성)들이 타당한지 검토하는 데 중요합니다.
    - 주로 잔차 산점도(독립변수 대 잔차, 적합값 대 잔차)를 그려서 패턴을 확인합니다.
- **분석 사례 (chemical.csv)**:
    - **목표**: 화학 공정에서 속도(speed)와 온도(temp)가 손실률(loss)에 미치는 영향 분석.
    - **1) 자료파일 읽기**: `chemical = read.csv("c:/data/reg/chemical.csv")`
    - **2) 기술통계량 및 상관계수 보기**:
        - `summary(chemical[,-1])`: 각 변수의 기술 통계량 확인.
        - `cor(chemical[,-1])`: 변수 간 상관계수 행렬 확인.
            - 독립변수들과 종속변수 간 상관관계가 높음.
            - 독립변수들 간(speed와 temp)의 상관계수도 0.802로 다소 높음 (다중공선성 가능성 시사).
    - **3) 산점도 그리기**: `plot(chemical$speed, chemical$loss, ...)` 등으로 각 독립변수와 종속변수 간 관계 시각화.
    - **4) 회귀모형 적합하기**:
        - `chemical_lm = lm(loss ~ speed + temp, data=chemical)`
        - `summary(chemical_lm)`:
            
            
            | Coefficients | Estimate | Std. Error | t value | Pr(>|t|) |
            | --- | --- | --- | --- | --- |
            | (Intercept) | -47.6243 | 9.4580 | -5.035 | 0.000704 \*\*\* |
            | speed | 0.4216 | 0.2350 | 1.794 | 0.106360 |
            | temp | 1.9217 | 0.6977 | 2.754 | 0.022316 \* |
            - temp는 유의하지만(p < 0.05), speed는 유의수준 0.05에서 유의하지 않음(p > 0.1).
    - **추가변수그림**: `avPlots(chemical_lm)`를 통해 각 변수의 부분적인 영향력 시각화.
    - **5) 분산분석표 구하기**: `anova(chemical_lm)` (변수 순서에 따른 제곱합 변화 확인)
    - **6) 잔차 산점도 그리기**:
        - `(독립변수, 잔차)`: `plot(chemical$speed, chemical_lm$resid, ...)`
        - `(추정값, 잔차)`: `plot(chemical_lm$fitted, chemical_lm$resid, ...)`
        - `identify()` 함수를 사용하여 특정 잔차 값을 가지는 데이터 포인트 식별.
        - 잔차 그림을 통해 모형의 가정 위배 여부(예: 비선형성, 이분산성, 이상치)를 진단합니다.

---