# 라이브러리 임포트
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# statsmodels 패키지가 설치되어 있지 않은 경우 주석 처리하고 아래 코드 실행하세요
# !pip install statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.stats.anova import anova_lm
    statsmodels_available = True
except ImportError:
    statsmodels_available = False
    print("statsmodels 패키지가 설치되어 있지 않습니다. 기본 분석만 진행합니다.")

# 한글 폰트 설정 
plt.rcParams['font.family'] = 'AppleGothic'  # macOS
# plt.rcParams['axes.unicode_minus'] = False   # 마이너스 기호 깨짐 방지

# 데이터 입력
age = np.array([3, 1, 5, 8, 1, 4, 2, 6, 9, 3, 5, 7, 2, 6])
cost = np.array([39, 24, 115, 105, 50, 86, 67, 90, 140, 112, 70, 186, 43, 126])

# 데이터프레임 생성
machine_data = pd.DataFrame({'age': age, 'cost': cost})
print("데이터프레임:")
print(machine_data)
print("\n")

# 1) 산점도 그리기 및 회귀직선 적합
plt.figure(figsize=(10, 6))
plt.scatter(age, cost, color='blue', marker='o')
plt.title('기계 사용연도와 정비비용의 관계')
plt.xlabel('사용연도 (년)')
plt.ylabel('정비비용 (천원)')

# 회귀모델 적합
slope, intercept, r_value, p_value, std_err = stats.linregress(age, cost)
regression_line = slope * age + intercept
plt.plot(age, regression_line, 'r-', linewidth=2)

# 회귀모델 상세 정보 출력
print("회귀모델 정보:")
print(f"기울기(slope): {slope:.4f}")
print(f"절편(intercept): {intercept:.4f}")
print(f"표준오차(std_err): {std_err:.4f}")
print(f"기울기 t-값: {slope/std_err:.4f}")
print(f"기울기 p-값: {p_value:.4f}")
print(f"해석: 기울기는 {slope:.4f}로, 사용연도가 1년 증가할 때마다 정비비용이 평균 {slope:.2f}천원 증가합니다.")
print(f"     p-값이 {p_value:.4f}로 0.05보다 작아 이 영향이 통계적으로 유의합니다.")
print("\n")

# 2) 상관계수와 결정계수 구하기
correlation = r_value
r_squared = r_value ** 2
print(f"상관계수(r): {correlation:.4f}")
print(f"결정계수(R²): {r_squared:.4f}")
print(f"해석: 상관계수가 0.7809로 사용연도와 정비비용 간 강한 양의 선형관계가 있습니다.")
print(f"     결정계수(R²)는 {r_squared:.4f}로 사용연도가 정비비용 변동의 약 {r_squared*100:.1f}%를 설명합니다.")
print("\n")

# 3) 분산분석표 및 회귀직선의 유의 여부 검정
# F-통계량과 p-value 직접 계산
n = len(age)
# 총 제곱합 (SST)
mean_y = np.mean(cost)
SS_total = np.sum((cost - mean_y)**2)
# 회귀 제곱합 (SSR)
SS_regression = np.sum((regression_line - mean_y)**2)
# 잔차 제곱합 (SSE)
SS_residual = np.sum((cost - regression_line)**2)

# 자유도
df_regression = 1  # 독립변수 1개
df_residual = n - 2  # n-2 (기울기와 절편으로 2개 모수 추정)
df_total = n - 1

# 평균 제곱
MS_regression = SS_regression / df_regression
MS_residual = SS_residual / df_residual

# F 통계량
f_statistic = MS_regression / MS_residual
# p-value
f_pvalue = 1 - stats.f.cdf(f_statistic, df_regression, df_residual)

print("분산분석표:")
print(f"{'출처':15} {'제곱합':>15} {'자유도':>10} {'평균제곱':>15} {'F':>10} {'p-value':>10}")
print(f"{'회귀':15} {SS_regression:15.2f} {df_regression:10d} {MS_regression:15.2f} {f_statistic:10.4f} {f_pvalue:10.4f}")
print(f"{'잔차':15} {SS_residual:15.2f} {df_residual:10d} {MS_residual:15.2f}")
print(f"{'총':15} {SS_total:15.2f} {df_total:10d}")
print("\n")
print(f"F-통계량: {f_statistic:.4f}")
print(f"p-value: {f_pvalue:.4f}")
print(f"유의수준 α=0.05에서 회귀직선이 {'유의함' if f_pvalue < 0.05 else '유의하지 않음'}")
print(f"해석: F-통계량({f_statistic:.4f})의 p-value({f_pvalue:.4f})가 유의수준 0.05보다 작으므로")
print("     사용연도는 정비비용을 예측하는 데 통계적으로 유의한 변수입니다.")
print("\n")

# 4) 사용연도가 4년인 기계의 평균 정비비용 추정
x_new = 4
predicted_cost = slope * x_new + intercept
print(f"사용연도가 4년인 기계의 추정 정비비용: {predicted_cost:.2f} 천원")
print(f"해석: 회귀모델에 따르면 사용연도가 4년인 기계의 평균 정비비용은 약 {predicted_cost:.2f}천원으로 예측됩니다.")
print("\n")

# 5) 잔차 계산 및 합이 0인지 확인
residuals = cost - regression_line
residuals_sum = np.sum(residuals)
print(f"잔차의 합: {residuals_sum:.10f}")  # 부동소수점 오차로 정확히 0이 아닐 수 있음
print(f"해석: 잔차의 합은 이론적으로 0이어야 합니다. 계산 결과는 부동소수점 오차로 인해")
print(f"     정확히 0이 아닐 수 있으나, {residuals_sum:.10f}로 0에 매우 가깝습니다.")
print("\n")

# 6) 잔차(e)와 x의 곱의 합 계산
x_residual_product = np.sum(age * residuals)
print(f"Σx_i·e_i의 값: {x_residual_product:.10f}")
print(f"해석: 독립변수(x)와 잔차(e)의 곱의 합은 이론적으로 0이어야 합니다.")
print(f"     계산 결과는 {x_residual_product:.10f}로 0에 매우 가깝습니다.")
print("\n")

# 7) 잔차(e)와 ŷ의 곱의 합 계산 (수정)
predicted_y = regression_line  # 예측값은 이미 regression_line에 저장되어 있음
y_hat_residual_product = np.sum(predicted_y * residuals)
print(f"Σŷ_i·e_i의 값: {y_hat_residual_product:.10f}")  # 이론적으로 0에 매우 가까움
print("\n")

# 참고: Σy_i·e_i 값은 잔차 제곱합(SSE)과 같습니다
print(f"[참고] Σy_i·e_i의 값 (SSE와 같음): {np.sum(cost * residuals):.4f}")
print(f"[참고] 잔차 제곱합 (SSE): {SS_residual:.4f}")
print("\n")
print("\n")

# 회귀식 표시
print(f"회귀식: cost = {intercept:.2f} + {slope:.2f} × age")
print(f"해석: 기계의 사용연도가 1년 증가할 때마다 정비비용은 평균적으로 {slope:.2f}천원 증가합니다.")
print("\n")
print("\n")

# 추가 분석: 잔차 플롯
plt.figure(figsize=(12, 10))

# 잔차 vs 예측값
plt.subplot(2, 2, 1)
plt.scatter(regression_line, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('예측값')
plt.ylabel('잔차')
plt.title('잔차 vs 예측값')
plt.grid(True, linestyle='--', alpha=0.7)

# 잔차 히스토그램
plt.subplot(2, 2, 2)
plt.hist(residuals, bins=7, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('잔차')
plt.ylabel('빈도')
plt.title('잔차 히스토그램')
plt.grid(True, linestyle='--', alpha=0.7)

# 가정 확인: 잔차 vs 독립변수
plt.subplot(2, 2, 3)
plt.scatter(age, residuals)
plt.axhline(y=0, color='r', linestyle='-')
plt.xlabel('사용연도')
plt.ylabel('잔차')
plt.title('잔차 vs 사용연도')
plt.grid(True, linestyle='--', alpha=0.7)

# 잔차의 정규성 (간단한 방법)
plt.subplot(2, 2, 4)
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('잔차 Q-Q 플롯')
plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('residual_plots.png', dpi=300, bbox_inches='tight')  # 고품질 이미지 저장
plt.show()

# 잔차 분석 결과 해석 출력
print("[잔차 분석 결과 해석]")
print("1. 예측값 vs 잔차: 패턴이 없고 무작위로 분포되어 있다면 선형성 가정이 충족됩니다.")
print("2. 잔차 히스토그램: 정규분포 형태를 보인다면 정규성 가정이 충족됩니다.")
print("3. 잔차 vs 독립변수: 패턴이 없어야 합니다. 패턴이 있다면 비선형 관계 가능성이 있습니다.")
print("4. Q-Q 플롯: 점들이 대각선에 가깝게 분포하면 정규성 가정이 충족됩니다.")
print("\n")

# statsmodels 패키지가 있을 경우 추가 분석 수행
if statsmodels_available:
    print("\n[statsmodels를 이용한 추가 분석]\n")
    # 독립변수에 상수항(절편) 추가하여 statsmodels로 분석
    X = sm.add_constant(age)
    sm_model = sm.OLS(cost, X).fit()
    
    # 상세한 회귀분석 결과 출력
    print(sm_model.summary())
    
    # ANOVA 테이블 출력
    print("\nANOVA 테이블:")
    anova_table = anova_lm(sm_model)
    print(anova_table)
    
    # 신뢰구간 출력
    print("\n회귀계수의 95% 신뢰구간:")
    print(sm_model.conf_int(alpha=0.05))
    
    # 결과 해석
    print("\n[회귀분석 결과 해석]")
    print(f"1. 모델 적합도: R² = {sm_model.rsquared:.4f}로, 사용연도가 정비비용 변동의 약 {sm_model.rsquared*100:.1f}%를 설명합니다.")
    print(f"2. 모델 유의성: F통계량 = {sm_model.fvalue:.4f}, p값 = {sm_model.f_pvalue:.4f}로, 유의수준 0.05에서 회귀모델이 통계적으로 유의합니다.")
    print(f"3. 기울기 해석: 사용연도가 1년 증가할 때마다 정비비용은 평균 {sm_model.params[1]:.2f}천원 증가합니다.")
    slope_pvalue = sm_model.pvalues[1]
    print(f"4. 기울기 유의성: p값 = {slope_pvalue:.4f}로, 사용연도 변수가 통계적으로 유의합니다.")
    print(f"5. 절편 해석: 사용연도가 0일 때 기대되는 정비비용은 약 {sm_model.params[0]:.2f}천원입니다.")
