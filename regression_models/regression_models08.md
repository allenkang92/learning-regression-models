# 회귀분석 08 - Python을 이용한 회귀모형 구현

---

## 목차

**I. 회귀분석을 위한 Python 환경 설정**
1. [파이썬 및 데이터 과학 환경 설정](#1-파이썬-및-데이터-과학-환경-설정)
2. [분석을 위한 주요 라이브러리 소개](#2-분석을-위한-주요-라이브러리-소개)
3. [프로그래밍 환경 설정 방법](#3-프로그래밍-환경-설정-방법)

**II. 데이터 준비와 탐색적 분석**
4. [데이터 로딩과 전처리](#4-데이터-로딩과-전처리)
5. [기초 통계량과 데이터 시각화](#5-기초-통계량과-데이터-시각화)

**III. 기본 회귀모형 구현**
6. [파이썬 중회귀모형 구축](#6-파이썬-중회귀모형-구축)
7. [추가제곱합 방법과 모형 선택](#7-추가제곱합-방법과-모형-선택)

**IV. 회귀모형 심화 구현**
8. [변수선택 기법 적용](#8-변수선택-기법-적용)
9. [고급 회귀모형 개발](#9-고급-회귀모형-개발)
10. [다항회귀모형 구현](#10-다항회귀모형-구현)
11. [가변수 회귀모형 구현](#11-가변수-회귀모형-구현)

**V. 회귀분석 진단 기법**
12. [특이점 진단과 처리](#12-특이점-진단과-처리)
13. [관측값 영향력 분석](#13-관측값-영향력-분석)

**VI. 회귀모형 가정 검정과 개선**
14. [등분산성 가정 검정](#14-등분산성-가정-검정)
15. [선형성 가정 검정](#15-선형성-가정-검정)
16. [오차 정규성 검정](#16-오차-정규성-검정)
17. [Box-Cox 변환과 모형 개선](#17-box-cox-변환과-모형-개선)

**VII. 일반화선형모형 구현**
18. [파이썬을 활용한 GLM 구현](#18-파이썬을-활용한-glm-구현)
19. [로지스틱 회귀모형 구현](#19-로지스틱-회귀모형-구현)
20. [로그선형모형 구현](#20-로그선형모형-구현)
21. [SAS를 활용한 GLM 구현](#21-sas를-활용한-glm-구현)
22. [SAS 로지스틱 회귀모형](#22-sas-로지스틱-회귀모형)
23. [SAS 로그선형모형](#23-sas-로그선형모형)

---

## I. 회귀분석을 위한 Python 환경 설정

### 1. 파이썬 및 데이터 과학 환경 설정

- **파이썬 설치**:
    - [Anaconda](https://www.anaconda.com/) 배포판을 통해 파이썬 및 주요 데이터 과학 라이브러리(NumPy, Pandas, Matplotlib, Scikit-learn, Statsmodels 등)를 쉽게 설치할 수 있습니다.
- **웹 기반 파이썬 실행 환경**:
    - **Google Colab**: 별도의 설치 없이 웹 브라우저에서 파이썬 코드를 실행하고 공유할 수 있는 환경입니다. (강의에서 주로 사용될 것으로 예상)
- **주요 라이브러리 임포트**:
    
    ```python
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf  # R-style formula를 사용한 모델링
    import statsmodels.api as sm         # 좀 더 세부적인 통계 모델링
    from statsmodels.formula.api import ols # Ordinary Least Squares (최소제곱법)
    from sklearn.linear_model import LinearRegression # Scikit-learn의 선형 회귀 모델
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    ```
    

### 2. 분석을 위한 주요 라이브러리 소개

### 3. 프로그래밍 환경 설정 방법

## II. 데이터 준비와 탐색적 분석

### 4. 데이터 로딩과 전처리

- **R 코드 복습 (p.5)**: `plot(market$X, market$Y, ...)`
- **파이썬 코드**:
    - 데이터 로드: `market = pd.read_csv("c:/data/reg/market-1.csv")`
    - 데이터 확인: `market.head()`
        
        ```
          ID     X     Y
        0   1   4.2   9.3
        1   2   8.5  18.5
        2   3   9.3  22.8
        3   4   7.5  17.7
        4   5   6.3  14.6
        
        ```
        
    - **Seaborn을 이용한 산점도**:
        
        ```python
        sns.scatterplot(x='X', y='Y', data=market)
        plt.title('Scatter Plot') # R의 title()과 유사
        plt.xlabel("인테리어비") # R의 xlab과 유사
        plt.ylabel("총판매액")   # R의 ylab과 유사
        plt.show()
        
        ```
        

### 5. 기초 통계량과 데이터 시각화

- **R 코드 복습 (p.11)**: `market_lm = lm(Y ~ X, data=market)`, `summary(market_lm)`, `abline(market_lm)`, `anova(market_lm)`
    - 추정된 회귀식: $\hat{Y} = 0.3282 + 2.1497 X$
- **파이썬 코드 (Method 1: `statsmodels.formula.api.ols`)**:
    - **회귀모형 적합**: `market_lm = ols('Y ~ X', data=market).fit()`
        - `'Y ~ X'`는 R의 formula와 동일한 방식으로 종속변수와 독립변수를 지정합니다.
        - `.fit()` 메서드를 호출하여 모델을 학습시킵니다.
    - **회귀분석 결과 요약**: `print(market_lm.summary())`
        - R의 `summary()`와 유사한 결과를 출력하며, 회귀계수, 표준오차, t-값, p-값, R-squared, F-통계량 등을 포함합니다.
    - **산점도와 회귀선 함께 그리기**:
        
        ```python
        sns.regplot(x='X', y='Y', data=market, ci=None) # ci=None은 신뢰구간 표시 안 함
        plt.title("Scatter Plot with Reg. line")
        plt.show()
        
        ```
        
    - **분산분석표 (ANOVA Table)**: `sm.stats.anova_lm(market_lm)`
        - R의 `anova()`와 유사한 결과를 출력하며, 각 변수의 제곱합(sum_sq), 평균제곱(mean_sq), F-값, 유의확률(PR(>F)) 등을 보여줍니다.
- **적합된 모델 객체(`market_lm`)의 주요 속성 및 메서드**:
    - `dir(market_lm)`: 객체가 가진 속성과 메서드 목록 확인.
    - `help(market_lm)`: 객체에 대한 상세 도움말 확인.
    - **잔차 (Residuals)**: `market_lm.resid`
        - `market_lm.resid.head()`: 처음 5개 잔차 값 확인.
    - **적합값 (Fitted values, 예측값)**: `market_lm.fittedvalues`
        - `market_lm.fittedvalues.head()`: 처음 5개 적합값 확인.
- **잔차 산점도 (적합값 vs. 잔차)**:
    
    ```python
    plt.scatter(market_lm.fittedvalues, market_lm.resid)
    plt.xlabel("fitted")
    plt.ylabel("residuals")
    plt.axhline(y=0, linestyle='--') # y=0에 수평 점선 추가 (잔차의 평균이 0인지 확인)
    plt.show()
    
    ```
    
- **신뢰대 (Confidence Interval Band) 시각화 (p.30)**:
    - `sns.regplot(x='X', y='Y', data=market)`: `ci` 인자를 지정하지 않으면 기본적으로 95% 신뢰구간이 함께 표시됩니다.
        
        ```python
        plt.title('Reg. line with CI')
        plt.show()
        
        ```
        
- **파이썬 코드 (Method 2: `statsmodels.api.OLS`)**:
    - 이 방법은 절편항(Intercept)을 명시적으로 추가해주어야 합니다.
    - `X_with_const = sm.add_constant(market.X)`: 기존 X 데이터에 상수항(절편을 위한 1로 채워진 열)을 추가합니다.
    - `Y = market.Y`
    - `market_lm2 = sm.OLS(Y, X_with_const).fit()`
    - `print(market_lm2.summary())`: 결과는 Method 1과 동일합니다.

## III. 기본 회귀모형 구현

### 6. 파이썬 중회귀모형 구축

- **데이터 (`market-2.csv`)**: 인테리어비(X1), 상점크기(X2), 총판매액(Y)
- **파이썬 코드 (Method 1: `statsmodels.formula.api.ols`)**:
    - 데이터 로드: `market2 = pd.read_csv("c:/data/reg/market-2.csv")`
    - 데이터 확인: `market2.head(3)`
        
        ```
          ID   X1    X2     Y
        0   1  4.2   4.5   9.3
        1   2  8.5  12.0  18.5
        2   3  9.3  15.0  22.8
        
        ```
        
    - **회귀모형 적합**: `market2_lm = ols('Y ~ X1 + X2', data=market2).fit()`
        - `'Y ~ X1 + X2'`는 Y를 종속변수로, X1과 X2를 독립변수로 하는 모형을 의미합니다.
    - **회귀분석 결과 요약**: `print(market2_lm.summary())`
        - 단순회귀와 마찬가지로 각 계수, R-squared, F-통계량 등을 확인할 수 있습니다.
    - **분산분석표**: `sm.stats.anova_lm(market2_lm)`
        - 각 독립변수(X1, X2)의 기여도와 잔차 정보를 보여줍니다. (Type III SS를 기본으로 할 수 있음, R과 다를 수 있음)
    - **결정계수 ($R^2$) 직접 확인**: `print(np.round(market2_lm.rsquared, 4))` (예: 0.9799)
- **표준화된 회귀계수 (Beta Coefficients)**:
    - 독립변수들의 상대적인 영향력을 비교하기 위해 사용합니다.
    - **방법**: 모든 변수(종속, 독립)를 각각 표준화(평균 0, 표준편차 1)한 후 회귀분석을 수행합니다.
    
    ```python
    from scipy import stats # zscore 계산을 위해 import
    # 숫자형 데이터만 선택하고 결측치 제거 후 표준화
    market2_z = market2.select_dtypes(include=[np.number]).dropna().apply(stats.zscore)
    market2_z_reg = smf.ols('Y ~ X1 + X2', data=market2_z).fit()
    print(market2_z_reg.summary())
    
    ```
    
    - `summary()` 결과의 `Coef.` 열에서 표준화된 계수를 확인할 수 있습니다. (절편은 0에 가까워짐)
- **파이썬 코드 (Method 2: `statsmodels.api.OLS`)**:
    - 독립변수 선택: `X = market2[['X1', 'X2']]`
    - 상수항 추가: `X = sm.add_constant(X)`
    - 종속변수 선택: `Y = market2['Y']` (강의자료에서는 `market.Y`로 오타가 있을 수 있음, `market2['Y']`가 맞음)
    - 회귀모형 적합: `market2_lm = sm.OLS(Y, X).fit()`
    - 결과 요약: `print(market2_lm.summary())`
- **파이썬 코드 (Method 3: `sklearn.linear_model.LinearRegression`)**:
    - Scikit-learn 라이브러리를 이용한 방법입니다. `statsmodels`와 결과 해석 방식이 다소 다릅니다. (통계적 유의성보다 예측 성능에 초점)
    - 독립변수 선택: `X = market2[['X1', 'X2']]`
    - 종속변수 선택: `Y = market2['Y']`
    - **주의**: Scikit-learn의 `LinearRegression`은 기본적으로 절편항을 포함하므로 `sm.add_constant`가 필요 없습니다. (만약 절편 없이 하려면 `fit_intercept=False` 옵션 사용)
    - 모델 생성 및 학습: `market3_lm = LinearRegression().fit(X, Y)`
    - **절편 확인**: `print(market3_lm.intercept_)` (예: 0.8504...)
    - **회귀계수 확인**: `print(market3_lm.coef_)` (예: [1.5581..., 0.4273...]) - X1, X2 순서
    - **예측**: `print(market3_lm.predict(X))`

### 7. 추가제곱합 방법과 모형 선택

- **R 코드 복습**: `anova(h3_lm, h4_lm)`
- **파이썬 코드 (Statsmodels 이용)**:
    - 데이터 로드: `health = pd.read_csv("c:/data/reg/health.csv")`
    - 데이터 확인: `health.head()`
    - **완전모형 (Full Model)**: `health_lm = ols('Y ~ X1 + X2 + X3 + X4', data=health).fit()`
    - **축소모형 (Restricted Model)**: `restricted_lm = ols('Y ~ X1 + X3 + X4', data=health).fit()` (X2 변수 제외)
    - **F-검정을 이용한 모형 비교**: `health_lm.compare_f_test(restricted_lm)`
        - 반환값: `(F-statistic, p-value, degrees_of_freedom_numerator)`
        - 예시 결과: `(0.3705..., 0.5481..., 1.0)`
        - **해석**: p-값(0.5481)이 유의수준(예: 0.05)보다 크므로, X2 변수를 추가하는 것이 통계적으로 유의미한 개선을 가져오지 않는다고 판단할 수 있습니다. (즉, X2는 모형에서 제외해도 무방)

---

이 강의는 R에서 다루었던 회귀분석의 주요 개념들을 파이썬 환경에서 어떻게 구현하고 해석할 수 있는지 보여줍니다. 특히 `statsmodels` 라이브러리는 R과 유사한 formula 방식과 상세한 통계적 결과 요약을 제공하여 통계적 모델링에 유용하며, `scikit-learn`은 예측 중심의 머신러닝 모델링에 강점을 가집니다. 표준화 계수를 통해 변수의 상대적 중요도를 파악하고, 추가제곱합 검정을 통해 변수 선택의 근거를 마련하는 방법도 다루고 있습니다.

## IV. 회귀모형 심화 구현

---

### 8. 변수선택 기법 적용

- **Scikit-learn과 단계적 회귀 (Stepwise Regression)**:
    - `scikit-learn` 라이브러리는 전통적인 통계적 유의성(p-value) 기반의 단계적 회귀(Forward, Backward, Stepwise) 기능을 **직접적으로 지원하지 않습니다**.
    - **이유**:
        - `scikit-learn`은 모델 학습에 있어 통계적 추론(유의성 검정 등)보다는 **예측 성능에 중점**을 둡니다.
        - 단계적 회귀는 선형 회귀의 계수 유의성에 기반하는데, `scikit-learn` 관점에서 OLS(최소제곱법)는 수많은 회귀 알고리즘 중 하나일 뿐이며, 항상 최선의 방법은 아니라고 보기 때문입니다.
    - **대안**:
        - `statsmodels` 라이브러리는 단계적 회귀와 유사한 기능을 제공하거나, 다른 기준(AIC, BIC 등)을 이용한 변수선택 방법을 지원합니다.
        - `mlxtend`와 같은 외부 라이브러리에서 단계적 회귀 기능을 찾을 수 있습니다.
        - 규제(Regularization) 기법 (Lasso, Ridge, ElasticNet)을 사용하여 변수 선택 효과를 간접적으로 얻을 수 있습니다. (Lasso는 일부 계수를 0으로 만들어 변수 선택 효과)

### 9. 고급 회귀모형 개발

### 10. 다항회귀모형 구현

- **라이브러리 임포트**:
    
    ```python
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    # from statsmodels.formula.api import ols (smf.ols로 사용 가능)
    # from sklearn.preprocessing import PolynomialFeatures (필요시 사용)
    
    ```
    
- **데이터 로드 및 준비**:
    - `tcrime = pd.read_csv("c:/data/reg/tcrime.csv")`
    - `tcrime.head(3)`: 데이터 확인.
    - **제곱항 직접 생성**: `tcrime["motorSq"] = tcrime["motor"]**2`
- **다항회귀모형 적합 (`statsmodels.formula.api.ols`)**:
    - `poly_reg = smf.ols("tcratio ~ motor + motorSq", data=tcrime).fit()`
        - Formula 문자열에 직접 `motorSq`와 같이 생성된 제곱항을 포함시킵니다.
        - (참고: `I(motor**2)` 형태로 formula 내에서 직접 계산도 가능합니다.)
    - `print(poly_reg.summary())`: 회귀분석 결과 확인.
- **`sklearn.preprocessing.PolynomialFeatures` 활용 (대안)**:
    - Scikit-learn을 사용할 경우, `PolynomialFeatures`를 이용하여 다항 특성을 생성한 후 `LinearRegression` 모델에 적용할 수 있습니다.

### 11. 가변수 회귀모형 구현

- **라이브러리 임포트**: 이전과 동일.
- **데이터 로드**: `soup = pd.read_csv("c:/data/reg/soup.csv")`
- **산점도 그리기 (가변수에 따른 구분, p.139)**:
    
    ```python
    import seaborn as sns
    sns.scatterplot(x='X', y='Y', hue='D', style='D', s=100, data=soup)
    plt.show() # matplotlib.pyplot as plt 임포트 필요
    
    ```
    
    - `hue='D'`: 'D' 컬럼 값에 따라 점의 색상을 다르게 표시.
    - `style='D'`: 'D' 컬럼 값에 따라 점의 모양을 다르게 표시.
- **교호작용을 고려하지 않은 모형 (p.140)**:
    - **Formula 작성**: `fm = 'Y ~ X + C(D)'`
        - `C(D)`: `statsmodels`에서 'D' 컬럼을 범주형 변수로 처리하라는 의미. 자동으로 가변수를 생성합니다. (기준 범주는 보통 가장 작은 값 또는 첫 번째 값)
    - 모형 적합: `soup_lm_res = smf.ols(formula=fm, data=soup).fit()`
    - 결과 확인: `print(soup_lm_res.summary())`
- **교호작용을 고려한 모형 (p.143)**:
    - **Formula 작성**: `fm2 = 'Y ~ X + C(D) + X:C(D)'`
        - `X:C(D)`: X와 C(D) 사이의 교호작용항을 의미.
        - (또는 `fm2 = 'Y ~ X * C(D)'` 로 주효과와 교호작용항을 모두 포함시킬 수 있음)
    - 모형 적합: `soup_lm2_res = smf.ols(formula=fm2, data=soup).fit()`
    - 결과 확인: `print(soup_lm2_res.summary())`

## V. 회귀분석 진단 기법

### 12. 특이점 진단과 처리

- **라이브러리 임포트**:
    
    ```python
    import numpy as np
    import pandas as pd
    import plotly.express as px # 동적 시각화를 위한 라이브러리
    from statsmodels.formula.api import ols
    import statsmodels.api as sm # anova_lm 등을 위해
    
    ```
    
- **데이터 로드 및 준비**:
    - `forbes = pd.read_csv("c:/data/reg/forbes.csv")`
    - `forbes["Lpress"] = 100*np.log10(forbes.press)`: 종속변수 변환.
- **산점도 (Plotly Express)**:
    
    ```python
    fig = px.scatter(forbes, x="temp", y="Lpress")
    fig.show()
    
    ```
    
- **회귀모형 적합 및 요약 (p.158)**:
    - `forbes_lm_res = ols('Lpress ~ temp', data=forbes).fit()`
    - `print(forbes_lm_res.summary())`
    - `sm.stats.anova_lm(forbes_lm_res)`: 분산분석표 확인.
- **특이값 검정 (`outlier_test`, p.159)**:
    - `outlier = forbes_lm_res.outlier_test()`: 각 관측값에 대한 스튜던트화 잔차, 보정 전 p-값, 본페로니 보정 p-값을 계산하여 특이점을 식별합니다.
    - `print(outlier)`

### 13. 관측값 영향력 분석

- **데이터 로드**: `soil = pd.read_csv("c:/data/reg/soil.csv")`
- **회귀모형 적합**: `soil_lm = ols('SL ~ SG+LOBS+PGC', data=soil).fit()`
- **영향력 진단 통계량 추출 (p.169)**:
    - `soil_influence = soil_lm.get_influence()`: 영향력 분석 객체 생성.
    - **스튜던트화 잔차 (내부)**: `soil_std_res = soil_influence.resid_studentized_internal`
    - **스튜던트화 잔차 (외부, 삭제 잔차 기반)**: `soil_stud_res = soil_influence.resid_studentized_external` (특이점 식별에 더 적합)
    - **햇 값 (레버리지)**: `soil_hat = soil_influence.hat_matrix_diag`
    - **Cook의 거리 (Cook's Distance)**: `soil_cooks = soil_influence.cooks_distance[0]` (Cook's D 값과 p-value 튜플 반환, 여기서는 값만 사용)
    - 결과를 Pandas DataFrame으로 정리:
        
        ```python
        diag_st = pd.concat([pd.DataFrame(soil_hat), pd.DataFrame(soil_std_res),
                             pd.DataFrame(soil_stud_res), pd.DataFrame(soil_cooks)], axis=1)
        diag_st.columns = ['Hii', 'ri', 'ti', 'Di']
        print(diag_st)
        
        ```
        
    - 특이값 검정: `outlier = soil_lm.outlier_test()`

## VI. 회귀모형 가정 검정과 개선

### 14. 등분산성 가정 검정

- **데이터 로드**: `goose = pd.read_csv("c:/data/reg/goose.csv")`
- **회귀모형 적합**: `goose_lm = ols("photo ~ obsA", data=goose).fit()`
- **잔차 및 적합값 추출**:
    - `goose_influence = goose_lm.get_influence()`
    - `goose_resid = goose_influence.resid` (또는 `goose_lm.resid`)
    - `goose_fitted = goose_lm.fittedvalues`
- **잔차 산점도 (적합값 vs. 잔차)**:
    
    ```python
    plt.scatter(goose_fitted, goose_resid)
    plt.xlabel("fitted")
    plt.ylabel("residuals")
    plt.axhline(y=0, linestyle='--')
    plt.show()
    
    ```
    
- **Goldfeld-Quandt 검정**:
    - `X_goose = goose['obsA']`
    - `X_goose_const = sm.add_constant(X_goose)` (절편항 추가)
    - `y_goose = goose['photo']`
    - `sm.stats.diagnostic.het_goldfeldquandt(y_goose, X_goose_const, drop=0.2)`
        - 반환값: `(GQ 통계량, p-value, 'increasing' 또는 'decreasing')`
        - 예시 결과: `(45.58..., 2.17e-08, 'increasing')` $\Rightarrow$ p-값이 매우 작으므로 등분산 가정을 기각. (분산이 증가하는 형태)

### 15. 선형성 가정 검정

- **데이터 로드**: `tree = pd.read_csv("c:/data/reg/tree.csv")`
- **회귀모형 적합**: `tree_lm = ols('V ~ D + H', data=tree).fit()`
- **잔차-설명변수 산점도**:
    
    ```python
    plt.scatter(tree.D, tree_lm.resid) # D에 대한 잔차 산점도
    plt.show()
    plt.scatter(tree.H, tree_lm.resid) # H에 대한 잔차 산점도
    plt.show()
    
    ```
    
    - 특정 설명변수에 대해 잔차가 곡선 패턴을 보이면 선형성 위배 의심.

### 16. 오차 정규성 검정

- **스튜던트화 잔차 (외부) 추출**:
    - `goose_influence = goose_lm.get_influence()`
    - `goose_stud_residual = goose_influence.resid_studentized_external`
- **정규확률그림 (Normal Q-Q Plot)**:
    
    ```python
    from scipy import stats
    stats.probplot(goose_stud_residual, dist="norm", plot=plt)
    plt.show()
    
    ```
    
    - 점들이 직선에서 벗어나면 정규성 위배 의심.
- **Shapiro-Wilk 검정**:
    
    ```python
    from scipy.stats import shapiro
    shapiro_result = shapiro(goose_stud_residual)
    print(shapiro_result)
    
    ```
    
    - 결과: `ShapiroResult(statistic=0.719..., pvalue=5.97e-08)` $\Rightarrow$ p-값이 매우 작으므로 정규성 가정을 기각.

### 17. Box-Cox 변환과 모형 개선

- **데이터 로드**: `energy = pd.read_csv("c:/data/reg/energy.csv")`
- **Box-Cox 변환 및 정규확률그림**:
    
    ```python
    from scipy import stats
    x_energy = energy['X'] # 강의자료에서는 X, Y 바로 사용
    y_energy = energy['Y']
    
    fig = plt.figure()
    ax1 = fig.add_subplot(211) # 2행 1열 중 첫 번째
    stats.probplot(y_energy, dist=stats.norm, plot=ax1) # 원본 Y의 정규확률그림
    ax1.set_title('Probability plot against normal distribution')
    
    ax2 = fig.add_subplot(212) # 2행 1열 중 두 번째
    yt_energy, lambda_optimal = stats.boxcox(y_energy) # Box-Cox 변환 및 최적 람다 값 반환
    stats.probplot(yt_energy, dist=stats.norm, plot=ax2) # 변환된 Y의 정규확률그림
    ax2.set_title('Probability plot after Box-Cox transformation')
    plt.show()
    
    print(f"Optimal lambda: {lambda_optimal}") # 예: 0.388... (0.5에 가까우므로 제곱근 변환 고려)
    
    ```
    

---
## VII. 일반화선형모형 구현

---

### 18. 파이썬을 활용한 GLM 구현

### 19. 로지스틱 회귀모형 구현

- **라이브러리 임포트**:
    
    ```python
    import pandas as pd
    import numpy as np
    import statsmodels.api as sm # GLM, add_constant 등 사용
    # import statsmodels.formula.api as smf (formula 방식 사용 시)
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # 분류 성능 평가
    
    ```
    
- **데이터 로드 및 준비 (날다람쥐 자료 `sugar_glider_binomial.csv`)**:
    - `glider = pd.read_csv("c:/data/reg/sugar_glider_binomial.csv")`
    - `y = glider.iloc[:, 1]` (종속변수 'occurr')
    - `X_vars = glider.iloc[:, 2:4]` (독립변수 'con_metric', 'p_size_km')
    - `X = sm.add_constant(X_vars)`: 절편항(const) 추가.
- **로지스틱 회귀모형 적합 (Model 1: 모든 변수 사용)**:
    - `logit_m1 = sm.GLM(y, X, family=sm.families.Binomial())`
        - `family=sm.families.Binomial()`: 이항분포 가족군 지정 (기본 연결함수는 로짓).
        - (참고: `sm.families.Binomial(link=sm.families.links.logit())`와 같이 명시적으로 연결함수 지정 가능)
    - `logit_m1_res = logit_m1.fit()`: 모델 학습.
    - `print(logit_m1_res.summary())`: 회귀분석 결과 요약 (계수, 표준오차, z-값, p-값, 이탈도, AIC 등).
- **결과 확인**:
    - `logit_m1_res.params`: 추정된 회귀계수 ($\hat{\beta}$).
    - `logit_m1_res.tvalues`: 각 계수에 대한 z-값 (Wald 검정 통계량, 정규분포 근사).
    - `logit_m1_res.pvalues`: 각 계수에 대한 p-값.
- **예측 (확률값)**:
    - `pred_res = logit_m1_res.predict(X)`: 학습된 모델을 사용하여 각 관측치에 대한 성공확률($\hat{\pi}$) 예측.
- **변수 선택 후 모형 적합 (Model 2: `p_size_km`만 사용, p.215)**:
    - (이전 강의에서 `stepAIC` 결과 `p_size_km`만 선택됨을 가정)
    - `X1 = X[["const", "p_size_km"]]` (또는 `X1 = X.drop(columns=["con_metric"])`)
    - `logit_m2 = sm.GLM(y, X1, family=sm.families.Binomial())`
    - `logit_m2_res = logit_m2.fit()`
    - `print(logit_m2_res.summary())`
- **승산비 (Odds Ratio) 계산 (p.224, p.225)**:
    - Model 2의 계수: `coef = logit_m2_res.params`
    - 승산비 ($\text{exp}(\hat{\beta})$): `np.exp(coef)`
        - `const`: $e^{\hat{\beta}_0}$
        - `p_size_km`: $e^{\hat{\beta}_1}$ (예: 1.021965) $\Rightarrow$ `p_size_km`이 1단위 증가할 때 성공 승산이 약 1.022배 증가.
    - 특정 $X$ 값에서의 예측 확률 (예: `p_size_km = 150`):
        - `imsi_X = [1, 150]` (절편항 포함)
        - `logit_m2_res.predict(imsi_X)` (예: 0.6749...)
- **로지스틱 회귀모형: 분류표 (Confusion Matrix) 및 정확도 (참고)**:
    - 예측 확률을 임계값(보통 0.5) 기준으로 0 또는 1로 변환: `pred_fitted = (pred_res > 0.5).astype(int)`
    - `cf_m = confusion_matrix(y, pred_fitted)`
    - `accuracy = accuracy_score(y, pred_fitted)`
    - `print(classification_report(y, pred_fitted))` (정밀도, 재현율, F1-점수 등 포함)

### 20. 로그선형모형 구현

- **데이터 준비 (R `MASS` 라이브러리 `Traffic` 데이터)**:
    - R에서 `write.csv(Traffic, file="c:/data/reg/Traffic.csv")`로 저장 후, 엑셀 등에서 첫 번째 불필요한 인덱스 열을 삭제하고 사용.
    - 파이썬에서 로드: `traffic = pd.read_csv("c:/data/reg/Traffic.csv")`
- **로그선형모형 적합 (`statsmodels.formula.api.glm`)**:
    - Formula 방식 사용 (p.270):
        - `formula = 'y ~ C(limit) + C(day) + C(year)'`
            - `C(변수명)`: 해당 변수를 범주형으로 처리 (가변수 자동 생성).
        - `log_m = smf.glm(formula=formula, data=traffic, family=sm.families.Poisson())`
            - `family=sm.families.Poisson()`: 포아송 분포 가족군 지정 (기본 연결함수는 로그).
        - `log_m_res = log_m.fit()`
        - `print(log_m_res.summary())`
    - 변수 선택 후 모형 적합 (예: `year` 변수 제외, p.272):
        - `formula_1 = 'y ~ C(limit) + C(day)'`
        - `log_m1 = smf.glm(formula=formula_1, data=traffic, family=sm.families.Poisson())`
        - `log_m1_res = log_m1.fit()`
        - `print(log_m1_res.summary())`
- **승산비(여기서는 Rate Ratio) 해석 (p.274)**:
    - (강의자료에서는 `log_m2_res.params`를 사용했는데, 문맥상 `log_m1_res.params`가 맞을 것으로 보임)
    - `params = log_m1_res.params`
    - `log_m1_odds = np.exp(params)`
    - 결과 예시:
        - `C(limit)[T.yes]`: 0.743590 $\Rightarrow$ 속도제한을 두는 경우(yes), 두지 않는 경우(no, 기준)에 비해 사고 발생률이 약 0.744배 (즉, 74.4% 수준으로 감소)함을 의미.

### 21. SAS를 활용한 GLM 구현

SAS는 통계 분석에 널리 사용되는 상용 소프트웨어로, GLM 분석을 위한 강력한 프로시저(PROC)를 제공합니다.

### 22. SAS 로지스틱 회귀모형

- **데이터 준비 (SAS `DATA` 스텝)**:
    - `DATA nsugar; SET sugar;`
    - `IF occurr = 0 THEN noccurr = 1; ELSE IF occurr = 1 THEN noccurr = 0;`
        - SAS `PROC LOGISTIC`은 기본적으로 반응변수의 **값이 작은 범주**를 성공(event)으로 간주하는 경우가 많으므로, `occurr=0` (미출현)을 사건으로 모델링하기 위해 `noccurr` 변수 생성. (또는 `DESCENDING` 옵션으로 `occurr=1`을 사건으로 지정 가능)
    - `RUN;`
- **로지스틱 회귀모형 적합 (`PROC LOGISTIC`)**:
    
    ```
    PROC LOGISTIC DATA=nsugar;
        MODEL noccurr = con_metric p_size_km; /* 또는 MODEL occurr(EVENT='1') = ... */
    RUN;
    
    ```
    
    - `MODEL 반응변수 = 독립변수들;`
- **출력 결과 비교 (R & SAS, p.209)**:
    - R의 `glm()` 결과와 SAS의 `PROC LOGISTIC` 결과에서 회귀계수 추정치, 표준오차, 검정통계량(z 또는 Wald Chi-Square), p-값 등이 유사하게 나타남을 보여줍니다.
    - 이탈도(Deviance), AIC 등의 적합도 지표도 함께 제공됩니다.
- **모형의 유의성 검정 (R, p.210)**:
    - `1 - pchisq(Null_Deviance - Residual_Deviance, df_diff)` (카이제곱 분포 이용)
    - p-값이 매우 작으면 모형이 통계적으로 유의함을 의미.
- **모형 선택: 변수선택방법 (R & SAS, p.216)**:
    - **R (`MASS` 패키지 `stepAIC`)**:
        - `library(MASS)`
        - `stepAIC(logit_m1, direction='both')` $\Rightarrow$ `p_size_km` 변수만 선택됨.
    - **SAS (`PROC LOGISTIC`의 `SELECTION` 옵션)**:
        - `PROC LOGISTIC DATA=nsugar; MODEL noccurr = con_metric p_size_km / SELECTION=stepwise; RUN;` $\Rightarrow$ `p_size_km` 변수만 선택됨.
- **승산비 (Odds Ratio) (R & SAS, p.224)**:
    - **R**: `exp(coef(logit_m2))`, `exp(confint(logit_m2, ...))`
    - **SAS**: `PROC LOGISTIC` 결과의 "Odds Ratio Estimates" 테이블에서 확인.
- **특정 X 값에서 성공확률($\hat{\pi}(x)$) 추정 (p.225)**:
    - **R**: `predict(logit_m2, newdata=list(p_size_km=150), type="response")`
    - **SAS**: `PROC LOGISTIC`의 `OUTPUT OUT=lresult P=pred;` 문으로 예측 확률을 데이터셋에 저장 후 확인.
- **정리된 자료 (Grouped Data)의 로지스틱 회귀모형 적합 (p.220, p.221)**:
    - 각 독립변수 조합에 대해 시행 횟수(count)와 성공 횟수(cases)가 주어진 경우.
    - **R**: `y <- cbind(glider_g$cases, glider_g$count - glider_g$cases)`, `glm(y ~ glider_g$p_size_med, family=binomial(link=logit))`
    - **SAS**: `PROC LOGISTIC DATA=sugar_g; MODEL cases/count = p_size_med; RUN;` (`cases/count` 형태로 반응변수 지정)

### 23. SAS 로그선형모형

- **데이터 준비 (SAS `DATA` 스텝, p.270)**:
    - 범주형 변수(limit, year, day)를 숫자형 가변수로 변환하는 과정. (예: `IF limit = "no" THEN nlimit=0; ELSE IF limit="yes" THEN nlimit=1;`)
- **로그선형모형 적합 (`PROC GENMOD`)**:
    
    ```
    PROC GENMOD DATA=ntraffic;
        MODEL y = nlimit day2-day92 /* day 가변수들 */ / DIST=POISSON LINK=LOG;
    RUN;
    
    ```
    
    - `DIST=POISSON`: 포아송 분포 지정.
    - `LINK=LOG`: 로그 연결함수 지정.
- **결과 비교 (R & SAS, p.272)**:
    - R의 `glm(..., family=poisson(link="log"))` 결과와 SAS의 `PROC GENMOD` 결과가 유사하게 나타남.
    - 최종 모형 (예: `year` 변수 제외)에 대한 결과 및 해석.

---