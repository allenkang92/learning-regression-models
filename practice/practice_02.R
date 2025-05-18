# 공정온도, 공정압력, 강도 간의 중회귀분석
# 문제: 공정온도(X₁)와 공정압력(X₂)이 제품 강도(Y)에 미치는 영향 분석

# 데이터 입력
temp <- c(195, 179, 205, 204, 201, 184, 210, 209)  # 공정온도(X₁), 단위: °C
pressure <- c(57, 61, 60, 62, 61, 54, 58, 61)      # 공정압력(X₂), 단위: psi
strength <- c(81.4, 122.0, 101.7, 175.6, 150.3, 64.8, 92.1, 113.8)  # 강도(Y), 단위: kg/cm²

# 데이터프레임 생성
process_data <- data.frame(temp = temp, pressure = pressure, strength = strength)
print("데이터:")
print(process_data)
print("\n")

# 산점도 행렬 - 변수 간 관계 시각화
par(mfrow = c(1, 1))
pairs(process_data, 
      main = "변수 간 산점도 행렬", 
      pch = 16, 
      col = "blue")

# 상관계수 확인
cor_matrix <- cor(process_data)
print("상관계수 행렬:")
print(cor_matrix)
print("\n")

# 1) 선형회귀모형 추정
model <- lm(strength ~ temp + pressure, data = process_data)
model_summary <- summary(model)
print("중회귀분석 결과:")
print(model_summary)

# 모형 계수 출력
beta0 <- coef(model)[1]  # 절편
beta1 <- coef(model)[2]  # 공정온도(X₁)의 계수
beta2 <- coef(model)[3]  # 공정압력(X₂)의 계수

cat("\n추정된 회귀모형: Y =", round(beta0, 2), "+", round(beta1, 2), "X₁ +", round(beta2, 2), "X₂\n\n")
cat("해석:\n")
cat("- 공정온도(X₁)가 1°C 증가할 때, 다른 변수가 일정하면 강도는 평균적으로", round(beta1, 2), "kg/cm² 변화합니다.\n")
cat("- 공정압력(X₂)가 1psi 증가할 때, 다른 변수가 일정하면 강도는 평균적으로", round(beta2, 2), "kg/cm² 변화합니다.\n\n")

# 2) 오차분산(σ²)을 MSE로 추정하고, 계수의 분산 추정
mse <- sum(model$residuals^2) / model$df.residual
var_beta <- vcov(model)  # 계수의 분산-공분산 행렬

cat("오차분산(σ²) 추정값(MSE):", round(mse, 2), "\n")
cat("Var[β₀] 추정값:", round(var_beta[1,1], 4), "\n")
cat("Var[β₁] 추정값:", round(var_beta[2,2], 4), "\n")
cat("Var[β₂] 추정값:", round(var_beta[3,3], 4), "\n\n")

# 3) X₁=200°C, X₂=59psi에서 평균 제품 강도 추정 및 분산 계산
new_data <- data.frame(temp = 200, pressure = 59)
predicted_strength <- predict(model, newdata = new_data, interval = "confidence")

cat("X₁=200°C, X₂=59psi일 때:\n")
cat("평균 제품 강도 추정값(Ŷ):", round(predicted_strength[1], 2), "kg/cm²\n")

# 예측값의 분산 계산
X_new <- c(1, 200, 59)  # 절편, X₁, X₂
var_pred <- t(X_new) %*% var_beta %*% X_new
cat("추정된 평균 강도의 분산:", round(var_pred, 4), "\n")
cat("추정된 평균 강도의 표준오차:", round(sqrt(var_pred), 4), "\n\n")

# 4) 분산분석표 작성 및 F-검정 결과 해석
anova_result <- anova(model)
print("분산분석표:")
print(anova_result)

# 전체 모형의 F-검정
f_value <- model_summary$fstatistic[1]
p_value <- pf(f_value, model_summary$fstatistic[2], model_summary$fstatistic[3], lower.tail = FALSE)

cat("\nF-통계량:", round(f_value, 4), "\n")
cat("p-value:", format(p_value, digits = 4), "\n")
cat("유의수준 α=0.05에서 회귀모형이", ifelse(p_value < 0.05, "유의함", "유의하지 않음"), "\n\n")
cat("해석: F-통계량의 p-value가", ifelse(p_value < 0.05, "0.05보다 작으므로 회귀모형이 통계적으로 유의합니다.", "0.05보다 크므로 회귀모형이 통계적으로 유의하지 않습니다."), "\n")
cat("      즉, 공정온도와 공정압력 중 적어도 하나는 제품 강도에 유의한 영향을 미칩니다.\n\n")

# 5) 표준화된 중회귀방정식 구하기
# 각 변수 표준화
temp_std <- scale(temp)
pressure_std <- scale(pressure)
strength_std <- scale(strength)

# 표준화된 데이터로 회귀분석
model_std <- lm(strength_std ~ temp_std + pressure_std)
beta_std <- coef(model_std)

cat("표준화된 중회귀방정식: Z_Y =", round(beta_std[2], 4), "Z_X₁ +", round(beta_std[3], 4), "Z_X₂\n\n")
cat("해석:\n")
cat("- 공정온도(X₁)가 표준편차 1단위 증가할 때, 강도는 평균적으로", round(beta_std[2], 4), "표준편차만큼 변화합니다.\n")
cat("- 공정압력(X₂)가 표준편차 1단위 증가할 때, 강도는 평균적으로", round(beta_std[3], 4), "표준편차만큼 변화합니다.\n")
cat("- 표준화 계수의 크기를 비교하면, 강도에", ifelse(abs(beta_std[2]) > abs(beta_std[3]), "공정온도가 더 큰", "공정압력이 더 큰"), "영향을 미치는 것으로 볼 수 있습니다.\n\n")

# 잔차 분석 그래프
par(mfrow = c(2, 2))
plot(model)

# 추가 해석 및 결론
cat("=== 종합 해석 ===\n")
cat("1. 모형 적합도: 결정계수(R²)는", round(model_summary$r.squared, 4), "로, 공정온도와 공정압력이 강도 변동의 약", 
    round(model_summary$r.squared * 100, 1), "%를 설명합니다.\n")
cat("2. 유의성: 전체 모형은", ifelse(p_value < 0.05, "통계적으로 유의합니다.", "통계적으로 유의하지 않습니다."), "\n")
cat("   - 공정온도(X₁):", ifelse(summary(model)$coefficients[2,4] < 0.05, "유의함", "유의하지 않음"), 
    "(p-value =", round(summary(model)$coefficients[2,4], 4), ")\n")
cat("   - 공정압력(X₂):", ifelse(summary(model)$coefficients[3,4] < 0.05, "유의함", "유의하지 않음"), 
    "(p-value =", round(summary(model)$coefficients[3,4], 4), ")\n")
cat("3. 표준화 계수를 통해", ifelse(abs(beta_std[2]) > abs(beta_std[3]), "공정온도가", "공정압력이"), 
    "강도에 더 큰 영향을 미치는 것으로 볼 수 있습니다.\n")
