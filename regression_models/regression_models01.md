# 회귀분석 01 - 단순회귀모형

---

## 목차

**I. 회귀분석의 기초 개념**
1. [회귀분석이란?](#1-회귀분석이란)
2. [회귀분석 관련 변수 분류](#2-회귀분석-관련-변수-분류)
3. [회귀분석의 용어 유래와 중요성](#3-회귀분석의-용어-유래와-중요성)

**II. 단순선형회귀모형**
1. [단순선형회귀모형의 기본 개념](#4-단순선형회귀모형의-기본-개념)
2. [단순회귀모형의 수학적 표현](#5-단순회귀모형의-수학적-표현)
3. [회귀직선의 적합 방법](#6-회귀직선의-적합-방법)

**III. 회귀계수 추정과 해석**
1. [최소제곱법과 계수 추정](#7-최소제곱법과-계수-추정)
2. [회귀계수의 통계적 유의성](#8-회귀계수의-통계적-유의성)
3. [회귀모형 분산분석표](#9-회귀모형-분산분석표)

**IV. 모형 평가와 해석**
1. [예측의 정확성 측정](#10-예측의-정확성-측정)
2. [결정계수와 상관계수](#11-결정계수와-상관계수)
3. [모형의 활용과 주의사항](#12-모형의-활용과-주의사항)

---

## I. 회귀분석의 기초 개념

### 1. 회귀분석이란?

- **회귀분석 관련 변수 분류**:
    - **설명변수 (Explanatory Variable)**: 다른 변수에 영향을 주는 변수.
        - **독립변수 (Independent Variable)**라고도 함.
        - 보통 **X**로 표시.
    - **반응변수 (Response Variable)**: 다른 변수에 의해 영향을 받는 변수.
        - **종속변수 (Dependent Variable)**라고도 함.
        - 보통 **Y**로 표시.
    - **예시**: 국민소득(X)이 증가하면 자동차 보유대수(Y)가 증가한다.
- **회귀분석 (Regression Analysis) 정의**:
    - 독립변수(들)과 종속변수 간의 **함수 관계를 규명**하는 통계적인 분석 방법.
- **"회귀(Regression)" 용어의 유래**:
    - "본래의 자기 자리로 돌아온다"라는 뜻.
    - 영국의 **프랜시스 골턴 (Francis Galton; 1822~1911)**이 처음 사용.
        - **완두콩 시험**: 부모콩(X)의 무게와 자식콩(Y)의 무게 간의 관계를 산점도로 분석.
        - 관계는 양의 직선 관계였으나, 기울기가 1보다 작아 자식콩의 무게는 전체 자식콩의 **평균 무게로 되돌아가려는 경향**을 발견. 이를 "회귀"로 명명.
    - *칼 피어슨 (Karl Pearson)**이 이를 계량적으로 처음 분석하여 발표.
    - (참고 문헌: Stanton, 2001; “Galton, Pearson, and the Peas: A Brief History of Linear Regression for Statistics Instructors”, Journal of Statistics Education)

### 2. 회귀분석 관련 변수 분류

- **산점도 (Scatterplot)**:
    - 한 변수를 x축, 다른 한 변수를 y축으로 하여 점을 찍어 그린 그림.
    - **두 연속형 변수들 간의 관계**를 파악하는 데 가장 널리 사용되는 그래프.
    - **예시 (R 코드)**: `market-1.csv` (상점 인테리어비와 총판매액 데이터)
        - `market = read.csv("c:/data/reg/market-1.csv")`
        - `plot(market$X, market$Y, xlab="인테리어비", ylab="총판매액", pch=19)`
        - `title("인테리어비와 판매액의 산점도")`
        - **산점도 해석**: 인테리어비가 증가하면 총판매액도 증가하는 **직선 관계**를 보임.
- **단순회귀모형의 수학적 표현**:
    - $Y_i = \beta_0 + \beta_1 X_i + \epsilon_i$
        - $Y_i$: i번째 관측치의 종속변수 값
        - $X_i$: i번째 관측치의 독립변수 값
        - $\beta_0$: **절편 (Intercept)** 계수 (모집단의 y절편)
        - $\beta_1$: **기울기 (Slope)** 계수 (모집단의 기울기, X가 한 단위 증가할 때 Y의 평균 변화량)
        - $\epsilon_i$: **오차항 (Error Term)**
            - 서로 독립이며, 평균이 0이고 분산이 $\sigma^2$인 정규분포를 따르는 확률변수로 가정 ($E(\epsilon_i) = 0$, $Var(\epsilon_i) = \sigma^2$).
            - 설명변수 X만으로는 설명되지 않는 Y의 변동 부분을 나타냄.
- **모형의 가정**:
    - n개의 관찰점 $(X_1, Y_1), (X_2, Y_2), ..., (X_n, Y_n)$에 대해 위 모형이 성립한다고 가정.

### 3. 회귀분석의 용어 유래와 중요성

## II. 단순선형회귀모형

### 4. 단순선형회귀모형의 기본 개념

- **회귀선 (Regression Line)**:
    - 단순회귀모형 $Y = \beta_0 + \beta_1 X + \epsilon$ 에서 오차항 $\epsilon$을 제외한 $E(Y|X) = \beta_0 + \beta_1 X$를 **모회귀선**이라고 함.
    - 표본 데이터를 이용하여 모회귀선을 추정한 직선을 **표본회귀선** 또는 **추정된 회귀선**이라고 함.
    - $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X$
        - $\hat{Y}$: 주어진 X 값에 대한 Y의 **기대값($E(Y)$)의 추정값**.
        - $\hat{\beta}_0$: **추정된 회귀절편 (Estimated Intercept)**. X=0일 때 $\hat{Y}$의 값.
        - $\hat{\beta}_1$: **추정된 회귀계수/기울기 (Estimated Slope)**. X가 한 단위 증가할 때 $\hat{Y}$의 평균 증가량.
- **최소제곱법 (Method of Least Squares, OLS)**:
    - 회귀계수 $\beta_0$와 $\beta_1$을 추정하는 가장 일반적인 방법.
    - 실제 관측값 $Y_i$와 회귀선에 의한 예측값 $\hat{Y}_i$ 사이의 **잔차(Residual, $e_i = Y_i - \hat{Y}_i$)의 제곱합을 최소화**하는 $\hat{\beta}_0$와 $\hat{\beta}_1$을 찾는 방법.
    - 즉, $\sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (Y_i - (\hat{\beta}_0 + \hat{\beta}_1 X_i))^2$ 을 최소화.
- **R을 활용한 회귀직선 추정 (예제: 인테리어비와 총판매액)**:
    - `market_lm = lm(Y ~ X, data=market)`: `lm` 함수 (linear model)를 사용하여 Y를 종속변수, X를 독립변수로 하는 회귀모형 적합.
    - `summary(market_lm)`: 회귀분석 결과 요약 출력.
        - `Coefficients:`
            - `(Intercept)` Estimate: `0.3282` ($\hat{\beta}_0$)
            - `X` Estimate: `2.1497` ($\hat{\beta}_1$)
        - **추정된 회귀식**: $\hat{Y} = 0.3282 + 2.1497 X$
    - **산점도 위에 회귀직선 그리기**:
        - `plot(market$X, market$Y, ...)`
        - `abline(market_lm)`: `market_lm` 객체에 저장된 회귀선을 기존 산점도에 추가.
        - `identify(market$X, market$Y)`: 그래프에서 특정 점을 클릭하여 인덱스 확인 가능 (이상치나 영향점 식별에 사용).
- **잔차 (Residual)의 특성**:
    - `resid = market_lm$residuals`: 잔차 값 추출.
    - `fitted = market_lm$fitted.values`: 적합된(예측된) Y 값 추출.
    - **잔차의 합은 0에 매우 가깝다**: `sum(resid)` 결과 $\approx 0$.
    - **실제 Y값의 합과 적합된 Y값의 합은 같다**: `sum(market$Y) == sum(fitted)`.
    - **독립변수 X와 잔차의 곱의 합은 0에 매우 가깝다**: `sum(market$X * resid)` 결과 $\approx 0$. (X와 잔차는 상관관계가 없음)
    - **적합된 Y값과 잔차의 곱의 합은 0에 매우 가깝다**: `sum(fitted * resid)` 결과 $\approx 0$. (적합값과 잔차는 상관관계가 없음)
- **산점도에 평균점 및 회귀식 텍스트 추가**:
    - `xbar = mean(market$X)`, `ybar = mean(market$Y)`
    - `points(xbar, ybar, pch=17, cex=2.0, col="RED")`: 평균점을 다른 모양과 색으로 표시.
    - `text(xbar, ybar, "(8.85, 19.36)")`: 평균점 좌표 표시.
    - `fx <- "Y-hat = 0.328+2.14*X"`
    - `text(locator(1), fx)`: 그래프의 원하는 위치에 회귀식 텍스트 추가 (클릭으로 위치 지정).

### 5. 회귀모형의 수학적 표현

- **회귀모형의 수학적 표현**:
    - $Y_i = \beta_0 + \beta_1 X_i + \epsilon_i$
        - $Y_i$: i번째 관측치의 종속변수 값
        - $X_i$: i번째 관측치의 독립변수 값
        - $\beta_0$: **절편 (Intercept)** 계수 (모집단의 y절편)
        - $\beta_1$: **기울기 (Slope)** 계수 (모집단의 기울기, X가 한 단위 증가할 때 Y의 평균 변화량)
        - $\epsilon_i$: **오차항 (Error Term)**
            - 서로 독립이며, 평균이 0이고 분산이 $\sigma^2$인 정규분포를 따르는 확률변수로 가정 ($E(\epsilon_i) = 0$, $Var(\epsilon_i) = \sigma^2$).
            - 설명변수 X만으로는 설명되지 않는 Y의 변동 부분을 나타냄.
- **모형의 가정**:
    - n개의 관찰점 $(X_1, Y_1), (X_2, Y_2), ..., (X_n, Y_n)$에 대해 위 모형이 성립한다고 가정.

### 6. 회귀직선의 적합 방법

- **회귀선 (Regression Line)**:
    - 단순회귀모형 $Y = \beta_0 + \beta_1 X + \epsilon$ 에서 오차항 $\epsilon$을 제외한 $E(Y|X) = \beta_0 + \beta_1 X$를 **모회귀선**이라고 함.
    - 표본 데이터를 이용하여 모회귀선을 추정한 직선을 **표본회귀선** 또는 **추정된 회귀선**이라고 함.
    - $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X$
        - $\hat{Y}$: 주어진 X 값에 대한 Y의 **기대값($E(Y)$)의 추정값**.
        - $\hat{\beta}_0$: **추정된 회귀절편 (Estimated Intercept)**. X=0일 때 $\hat{Y}$의 값.
        - $\hat{\beta}_1$: **추정된 회귀계수/기울기 (Estimated Slope)**. X가 한 단위 증가할 때 $\hat{Y}$의 평균 증가량.
- **최소제곱법 (Method of Least Squares, OLS)**:
    - 회귀계수 $\beta_0$와 $\beta_1$을 추정하는 가장 일반적인 방법.
    - 실제 관측값 $Y_i$와 회귀선에 의한 예측값 $\hat{Y}_i$ 사이의 **잔차(Residual, $e_i = Y_i - \hat{Y}_i$)의 제곱합을 최소화**하는 $\hat{\beta}_0$와 $\hat{\beta}_1$을 찾는 방법.
    - 즉, $\sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (Y_i - (\hat{\beta}_0 + \hat{\beta}_1 X_i))^2$ 을 최소화.
- **R을 활용한 회귀직선 추정 (예제: 인테리어비와 총판매액)**:
    - `market_lm = lm(Y ~ X, data=market)`: `lm` 함수 (linear model)를 사용하여 Y를 종속변수, X를 독립변수로 하는 회귀모형 적합.
    - `summary(market_lm)`: 회귀분석 결과 요약 출력.
        - `Coefficients:`
            - `(Intercept)` Estimate: `0.3282` ($\hat{\beta}_0$)
            - `X` Estimate: `2.1497` ($\hat{\beta}_1$)
        - **추정된 회귀식**: $\hat{Y} = 0.3282 + 2.1497 X$
    - **산점도 위에 회귀직선 그리기**:
        - `plot(market$X, market$Y, ...)`
        - `abline(market_lm)`: `market_lm` 객체에 저장된 회귀선을 기존 산점도에 추가.
        - `identify(market$X, market$Y)`: 그래프에서 특정 점을 클릭하여 인덱스 확인 가능 (이상치나 영향점 식별에 사용).
- **잔차 (Residual)의 특성**:
    - `resid = market_lm$residuals`: 잔차 값 추출.
    - `fitted = market_lm$fitted.values`: 적합된(예측된) Y 값 추출.
    - **잔차의 합은 0에 매우 가깝다**: `sum(resid)` 결과 $\approx 0$.
    - **실제 Y값의 합과 적합된 Y값의 합은 같다**: `sum(market$Y) == sum(fitted)`.
    - **독립변수 X와 잔차의 곱의 합은 0에 매우 가깝다**: `sum(market$X * resid)` 결과 $\approx 0$. (X와 잔차는 상관관계가 없음)
    - **적합된 Y값과 잔차의 곱의 합은 0에 매우 가깝다**: `sum(fitted * resid)` 결과 $\approx 0$. (적합값과 잔차는 상관관계가 없음)
- **산점도에 평균점 및 회귀식 텍스트 추가**:
    - `xbar = mean(market$X)`, `ybar = mean(market$Y)`
    - `points(xbar, ybar, pch=17, cex=2.0, col="RED")`: 평균점을 다른 모양과 색으로 표시.
    - `text(xbar, ybar, "(8.85, 19.36)")`: 평균점 좌표 표시.
    - `fx <- "Y-hat = 0.328+2.14*X"`
    - `text(locator(1), fx)`: 그래프의 원하는 위치에 회귀식 텍스트 추가 (클릭으로 위치 지정).

## III. 회귀계수 추정과 해석

### 7. 최소제곱법과 계수 추정

- **회귀선 (Regression Line)**:
    - 단순회귀모형 $Y = \beta_0 + \beta_1 X + \epsilon$ 에서 오차항 $\epsilon$을 제외한 $E(Y|X) = \beta_0 + \beta_1 X$를 **모회귀선**이라고 함.
    - 표본 데이터를 이용하여 모회귀선을 추정한 직선을 **표본회귀선** 또는 **추정된 회귀선**이라고 함.
    - $\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X$
        - $\hat{Y}$: 주어진 X 값에 대한 Y의 **기대값($E(Y)$)의 추정값**.
        - $\hat{\beta}_0$: **추정된 회귀절편 (Estimated Intercept)**. X=0일 때 $\hat{Y}$의 값.
        - $\hat{\beta}_1$: **추정된 회귀계수/기울기 (Estimated Slope)**. X가 한 단위 증가할 때 $\hat{Y}$의 평균 증가량.
- **최소제곱법 (Method of Least Squares, OLS)**:
    - 회귀계수 $\beta_0$와 $\beta_1$을 추정하는 가장 일반적인 방법.
    - 실제 관측값 $Y_i$와 회귀선에 의한 예측값 $\hat{Y}_i$ 사이의 **잔차(Residual, $e_i = Y_i - \hat{Y}_i$)의 제곱합을 최소화**하는 $\hat{\beta}_0$와 $\hat{\beta}_1$을 찾는 방법.
    - 즉, $\sum_{i=1}^{n} e_i^2 = \sum_{i=1}^{n} (Y_i - (\hat{\beta}_0 + \hat{\beta}_1 X_i))^2$ 을 최소화.
- **R을 활용한 회귀직선 추정 (예제: 인테리어비와 총판매액)**:
    - `market_lm = lm(Y ~ X, data=market)`: `lm` 함수 (linear model)를 사용하여 Y를 종속변수, X를 독립변수로 하는 회귀모형 적합.
    - `summary(market_lm)`: 회귀분석 결과 요약 출력.
        - `Coefficients:`
            - `(Intercept)` Estimate: `0.3282` ($\hat{\beta}_0$)
            - `X` Estimate: `2.1497` ($\hat{\beta}_1$)
        - **추정된 회귀식**: $\hat{Y} = 0.3282 + 2.1497 X$
    - **산점도 위에 회귀직선 그리기**:
        - `plot(market$X, market$Y, ...)`
        - `abline(market_lm)`: `market_lm` 객체에 저장된 회귀선을 기존 산점도에 추가.
        - `identify(market$X, market$Y)`: 그래프에서 특정 점을 클릭하여 인덱스 확인 가능 (이상치나 영향점 식별에 사용).
- **잔차 (Residual)의 특성**:
    - `resid = market_lm$residuals`: 잔차 값 추출.
    - `fitted = market_lm$fitted.values`: 적합된(예측된) Y 값 추출.
    - **잔차의 합은 0에 매우 가깝다**: `sum(resid)` 결과 $\approx 0$.
    - **실제 Y값의 합과 적합된 Y값의 합은 같다**: `sum(market$Y) == sum(fitted)`.
    - **독립변수 X와 잔차의 곱의 합은 0에 매우 가깝다**: `sum(market$X * resid)` 결과 $\approx 0$. (X와 잔차는 상관관계가 없음)
    - **적합된 Y값과 잔차의 곱의 합은 0에 매우 가깝다**: `sum(fitted * resid)` 결과 $\approx 0$. (적합값과 잔차는 상관관계가 없음)
- **산점도에 평균점 및 회귀식 텍스트 추가**:
    - `xbar = mean(market$X)`, `ybar = mean(market$Y)`
    - `points(xbar, ybar, pch=17, cex=2.0, col="RED")`: 평균점을 다른 모양과 색으로 표시.
    - `text(xbar, ybar, "(8.85, 19.36)")`: 평균점 좌표 표시.
    - `fx <- "Y-hat = 0.328+2.14*X"`
    - `text(locator(1), fx)`: 그래프의 원하는 위치에 회귀식 텍스트 추가 (클릭으로 위치 지정).

### 8. 회귀계수의 통계적 유의성

### 9. 회귀모형 분산분석표 (ANOVA Table)

- **분산분석표에 의한 F-검정**:
    - 회귀모형의 **유의성(Significance)**을 검정하는 방법. 즉, 독립변수 X가 종속변수 Y를 설명하는 데 통계적으로 유의한가? (기울기 $\beta_1 = 0$ 인가 아닌가?)
    - **총 변동 (Total Variation, SST)**: 종속변수 Y의 전체 변동. $SST = \sum (Y_i - \bar{Y})^2$
    - **회귀에 의한 변동 (Regression Variation, SSR)**: 회귀선에 의해 설명되는 Y의 변동. $SSR = \sum (\hat{Y}_i - \bar{Y})^2$
    - **오차에 의한 변동 (Error Variation, SSE)**: 회귀선에 의해 설명되지 않는 Y의 변동 (잔차 제곱합). $SSE = \sum (Y_i - \hat{Y}_i)^2 = \sum e_i^2$
    - **관계**: $SST = SSR + SSE$
    - **분산분석표 (ANOVA Table) 구성**:
        
        
        | 요인 (Source) | 자유도 (Df) | 제곱합 (Sum Sq) | 평균제곱 (Mean Sq) | F 값 (F value) | 유의확률 (Pr(>F)) |
        | --- | --- | --- | --- | --- | --- |
        | 회귀 (X) | 1 (독립변수 개수) | SSR | MSR = SSR / Df_R | F = MSR / MSE | p-value |
        | 잔차 (Residuals) | n-2 (n: 샘플 수) | SSE | MSE = SSE / Df_E |  |  |
        | 계 (Total) | n-1 | SST |  |  |  |
    - **R 활용 (분산분석표)**:
        - `anova(market_lm)`
        - `Response: Y`
        - `X` Df: 1, Sum Sq: 485.57 (SSR), Mean Sq: 485.57 (MSR)
        - `Residuals` Df: 13, Sum Sq: 32.72 (SSE), Mean Sq: 2.52 (MSE)
        - `F value`: 192.9
        - `Pr(>F)`: 3.554e-09 (p-value, 매우 작으므로 회귀모형이 유의함)
- **결정계수 ($R^2$, Coefficient of Determination)**:
    - 회귀선이 종속변수 Y의 변동을 **얼마나 잘 설명하는지** 나타내는 지표.
    - $R^2 = \frac{SSR}{SST} = 1 - \frac{SSE}{SST}$
    - 0과 1 사이의 값을 가지며, 1에 가까울수록 설명력이 높음.
    - `summary(market_lm)` 결과에서 확인:
        - `Multiple R-squared: 0.9369` (약 93.69%의 설명력)
        - `Adjusted R-squared: 0.932` (수정된 결정계수, 독립변수 개수를 고려)
- **추정값의 표준오차 (Standard Error of Estimates)**:
    - 회귀계수 추정치($\hat{\beta}_0, \hat{\beta}_1$)의 불확실성(표준편차)을 나타내는 값.
    - `summary(market_lm)` 결과에서 `Std. Error` 컬럼:
        - `(Intercept)` Std. Error: `1.4302`
        - `X` Std. Error: `0.1548`
    - **t 값 (t value)**: `Estimate / Std. Error`. 각 회귀계수가 0인지 검정.
    - **Pr(>|t|)**: t 값에 대한 p-value. (작을수록 해당 계수가 유의함)
- **잔차 표준오차 (Residual Standard Error)**:
    - $\sqrt{MSE}$ 값으로, 회귀모형의 예측 오차의 평균적인 크기를 나타냄.
    - `summary(market_lm)` 결과에서 확인:
        - `Residual standard error: 1.587 on 13 degrees of freedom`
- **상관계수(r)와 결정계수($R^2$)의 관계**:
    - 단순회귀분석에서는 **결정계수($R^2$)는 상관계수(r)의 제곱과 같다.** ($R^2 = r^2$)
    - 상관계수는 두 변수 간의 선형 관계의 강도와 방향을 나타내지만, 결정계수는 한 변수가 다른 변수를 설명하는 정도를 나타냅니다.

---

## IV. 모형 평가와 해석

---

### 10. 예측의 정확성 측정

- **기본 가정**:
    - 오차항 $\epsilon_i$는 서로 독립이며, 평균 0, 분산 $\sigma^2$인 정규분포를 따른다.
- **$\beta_1$ (기울기)의 신뢰구간**:
    - 모집단 기울기 $\beta_1$에 대한 (1-$\alpha$)100% 신뢰구간은 다음과 같이 계산됩니다.
        - $\hat{\beta}*1 \pm t*{(\alpha/2, n-2)} \cdot SE(\hat{\beta}_1)$
        - $\hat{\beta}_1$: 표본 기울기 추정치
        - $t_{(\alpha/2, n-2)}$: 자유도 n-2인 t-분포의 양쪽 꼬리 확률이 $\alpha/2$가 되는 t 값 (임계값)
        - $SE(\hat{\beta}_1)$: 기울기 추정치의 표준오차
- **$\beta_0$ (절편)의 신뢰구간**:
    - 모집단 절편 $\beta_0$에 대한 (1-$\alpha$)100% 신뢰구간은 다음과 같이 계산됩니다.
        - $\hat{\beta}*0 \pm t*{(\alpha/2, n-2)} \cdot SE(\hat{\beta}_0)$
        - $\hat{\beta}_0$: 표본 절편 추정치
        - $SE(\hat{\beta}_0)$: 절편 추정치의 표준오차
- **R 결과에서 신뢰구간 구하기 (예제: `market_lm` 인테리어비와 총판매액)**:
    - `summary(market_lm)` 결과:
        
        
        | Coefficients | Estimate | Std. Error | t value | Pr(>|t|) |
        | --- | --- | --- | --- | --- |
        | (Intercept) | 0.3282 | 1.4302 | 0.229 | 0.822 |
        | X | 2.1497 | 0.1548 | 13.889 | 3.55e-09 |
    - 자유도(df) = 13 (Residuals의 df)
    - 95% 신뢰구간을 위한 t 임계값 ($\alpha=0.05$): `q.val = qt(0.975, 13)`
    - **$\beta_1$의 95% 신뢰구간**:
        - 하한: `2.1497 - q.val * 0.1548` $\approx$ `1.815`
        - 상한: `2.1497 + q.val * 0.1548` $\approx$ `2.484`
    - **$\beta_0$의 95% 신뢰구간**:
        - 하한: `0.3282 - q.val * 1.4302` $\approx$ `2.762`
        - 상한: `0.3282 + q.val * 1.4302` $\approx$ `3.418`
- **기댓값 신뢰구간 (Confidence Interval for Mean Response)**:
    - 주어진 $X_h$ 값에 대한 **평균 반응 $E(Y_h)$의 신뢰구간**입니다.
    - $X_h$에서의 $\hat{Y}_h = \hat{\beta}_0 + \hat{\beta}_1 X_h$를 중심으로 일정 범위를 가집니다.
    - $X_h$가 표본 X들의 평균($\bar{X}$)에서 멀어질수록 구간의 폭이 넓어집니다.
- **예측값 신뢰구간 (Prediction Interval for New Observation)**:
    - 주어진 $X_h$ 값에 대한 **새로운 관측치 $Y_{h(new)}$의 신뢰구간**입니다.
    - 기댓값 신뢰구간보다 항상 넓습니다. 왜냐하면 개별 관측치는 평균 반응 주변에 오차항($\epsilon$)만큼의 추가적인 변동성을 가지기 때문입니다.
- **신뢰대 그리기 (Confidence Bands & Prediction Bands)**:
    - 다양한 X 값에 대한 기댓값 신뢰구간과 예측값 신뢰구간을 연결하여 띠 형태로 시각화한 것.
    - **R 코드 (기본 graphics)**:
        - `pred_frame = data.frame(X=seq(3.5, 14.5, 0.2))`: 예측을 위한 X 값 시퀀스 생성.
        - `pc = predict(market_lm, interval="confidence", newdata=pred_frame)`: 기댓값 신뢰구간 계산.
        - `pp = predict(market_lm, interval="prediction", newdata=pred_frame)`: 예측값 신뢰구간 계산.
        - `plot(market$X, market$Y, ylim=range(market$Y, pp))`: 실제 데이터 산점도.
        - `matlines(pred_X, pc, lty=c(1,2,2), col="BLUE")`: 기댓값 신뢰대 (실선: 적합선, 점선: 신뢰구간 경계).
        - `matlines(pred_X, pp, lty=c(1,3,3), col="RED")`: 예측값 신뢰대 (실선: 적합선, 점선: 신뢰구간 경계).
    - **R 코드 (ggplot2 활용)**:
        - `market_predict = predict(market_lm, interval="prediction")`
        - `all_data = cbind(market, market_predict)`: 원본 데이터와 예측값, 예측구간 결합.
        - `ggplot(all_data, aes(x=X, y=Y)) + geom_point() + stat_smooth(method=lm) + geom_line(aes(y = lwr), col = "coral2", linetype = "dashed") + geom_line(aes(y = upr), col = "coral2", linetype = "dashed")`
            - `stat_smooth(method=lm)`: 회귀선과 함께 기댓값 신뢰대를 자동으로 그려줌.
            - `geom_line(aes(y = lwr/upr), ...)`: 예측값 신뢰구간의 하한(lwr)과 상한(upr)을 점선으로 추가.
- **$\beta_1$ (기울기)의 검정**:
    - 귀무가설 $H_0: \beta_1 = 0$ (기울기는 0이다, 즉 X는 Y를 설명하지 못한다)
    - 대립가설 $H_1: \beta_1 \neq 0$ (기울기는 0이 아니다, 즉 X는 Y를 설명한다)
    - 검정통계량: $t_0 = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)}$
    - **R 결과 (`summary(market_lm)`)**:
        - `X` 행에서 `Estimate`($\hat{\beta}_1$), `Std. Error`($SE(\hat{\beta}_1)$), `t value`($t_0$), `Pr(>|t|)`(p-value) 확인.
        - `t value = 2.1497 / 0.1548 = 13.889`
        - `Pr(>|t|) = 3.55e-09` (p-value)
    - **기각역 및 p-값 구하기 (수동 계산)**:
        - 유의수준 $\alpha=0.05$일 때 기각역(양측검정): $|t_0| > qt(0.975, 13)$
        - `qt(0.975, 13)` $\approx$ `2.160`
        - $|13.889| > 2.160$ 이므로 귀무가설 기각.
        - p-값: `2 * (1 - pt(13.889, 13))` $\approx$ `3.55e-09`
        - p-값이 유의수준보다 매우 작으므로 귀무가설을 기각하고, $\beta_1$은 통계적으로 유의하다고 결론.

### 11. 결정계수와 상관계수

- **가중회귀의 필요성**:
    - 일반적인 최소제곱법(OLS)은 모든 관측치가 동일한 중요도(분산)를 가진다고 가정합니다 (등분산성 가정).
    - 그러나 실제 데이터에서는 **오차항의 분산이 일정하지 않은 경우 (이분산성, Heteroscedasticity)**가 발생할 수 있습니다.
    - 이분산성이 존재하면 OLS 추정량은 여전히 비편향적이지만, 가장 효율적인(분산이 가장 작은) 추정량이 아니게 되며, 표준오차 추정이 부정확해져 가설검정이나 신뢰구간의 신뢰도가 떨어집니다.
- **가중회귀 (Weighted Least Squares, WLS)**:
    - 각 관측치에 서로 다른 **가중치(weight)**를 부여하여 회귀모형을 적합하는 방법입니다.
    - 일반적으로 **오차 분산의 역수에 비례하는 가중치**를 사용합니다. 즉, 오차 분산이 작은 관측치에는 큰 가중치를, 오차 분산이 큰 관측치에는 작은 가중치를 부여합니다.
    - 가중 최소제곱법은 가중된 잔차 제곱합 $\sum w_i (Y_i - (\hat{\beta}_0 + \hat{\beta}_1 X_i))^2$을 최소화합니다.
- **R 활용 예제 (가중회귀)**:
    - `x = c(1,2,3,4,5)`
    - `y = c(2,3,5,8,7)`
    - `w = 1/x` (예시로 x값이 커질수록 오차 분산이 커진다고 가정하고, x의 역수를 가중치로 사용)
    - `w_lm = lm(y ~ x, weights=w)`: `lm` 함수에 `weights` 인자를 사용하여 가중회귀 수행.
    - `summary(w_lm)` 결과 해석:
        
        
        | Coefficients | Estimate | Std. Error | t value | Pr(>|t|) |
        | --- | --- | --- | --- | --- |
        | (Intercept) | 0.3784 | 0.6891 | 0.549 | 0.6212 |
        | x | 1.5405 | 0.2688 | 5.730 | 0.0106 \* |

### 12. 모형의 활용과 주의사항

- **목표**: 슈퍼마켓의 가격(price)과 계산대 대기시간(time) 간의 관계 분석.
- **1) 자료파일 만들기**: `supermarket.csv` (price, time 컬럼)
- **2) 자료 읽어 산점도 그리기**:
    - `super = read.csv("c:/data/reg/supermarket.csv")`
    - `plot(super$price, super$time, pch=19)`: 산점도를 통해 가격이 증가할수록 대기시간도 증가하는 경향 확인.
- **3) 회귀모형 적합하기**:
    - `super_lm = lm(time ~ price, data=super)`
    - `summary(super_lm)`:
        
        
        | Coefficients | Estimate | Std. Error | t value | Pr(>|t|) |
        | --- | --- | --- | --- | --- |
        | (Intercept) | 0.396460 | 0.191488 | 2.07 | 0.0722 . |
        | price | 0.115982 | 0.008979 | 12.92 | 1.22e-06 \*\*\* |
        - `Residual standard error: 0.3925 on 8 degrees of freedom`
        - `Multiple R-squared: 0.9542, Adjusted R-squared: 0.9485`
        - `F-statistic: 166.9 on 1 and 8 DF, p-value: 1.221e-06`
- **4) 분산분석표 구하기**:
    - `anova(super_lm)`: F-검정을 통해 회귀모형의 유의성 확인.
- **5) 잔차 및 추정값 보기**:
    - `cbind(super, super_lm$resid, super_lm$fitted)`: 원본 데이터, 잔차, 적합값 함께 출력.
- **6) 잔차 그림 그리기 (잔차 진단)**:
    - `plot(super$price, super_lm$resid, pch=19)`: 독립변수(price) 대 잔차 산점도.
    - `abline(h=0, lty=2)`: 잔차의 평균이 0인지 확인하기 위한 수평선 추가.
    - `plot(super_lm$fitted, super_lm$resid, pch=19)`: 적합값 대 잔차 산점도.
    - **잔차 그림 해석**: 잔차들이 0을 중심으로 무작위로 흩어져 있는지, 특정 패턴(예: 깔때기 모양 - 이분산성 의심)을 보이는지 확인하여 회귀모형의 가정을 검토.
- **7) 추정값의 신뢰대 그리기 (ggplot2 활용)**:
    - `super_predict = predict(super_lm, interval="predict")`
    - `super_data = cbind(super, super_predict)`
    - `library(ggplot2)`
    - `ggplot(super_data, aes(x=price, y=time)) + geom_point() + stat_smooth(method=lm) + geom_line(aes(y = lwr), ...) + geom_line(aes(y = upr), ...)`: 산점도, 회귀선, 기댓값 신뢰대, 예측값 신뢰대 함께 시각화.

---