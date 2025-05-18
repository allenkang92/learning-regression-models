# 기계의 사용연도와 정비비용 관계 분석
# 문제: 사용연도(age of machines)와 정비비용(maintenance cost) 간의 관계 분석

# 데이터 입력
age <- c(3, 1, 5, 8, 1, 4, 2, 6, 9, 3, 5, 7, 2, 6)
cost <- c(39, 24, 115, 105, 50, 86, 67, 90, 140, 112, 70, 186, 43, 126)

# 데이터프레임 생성
machine_data <- data.frame(age = age, cost = cost)
print(machine_data)

# 1) 산점도 그리기 및 회귀직선 적합
plot(age, cost,
     main = "기계 사용연도와 정비비용의 관계",
     xlab = "사용연도 (년)",
     ylab = "정비비용 (천원)",
     pch = 16,
     col = "blue")

# 회귀모델 적합
model <- lm(cost ~ age, data = machine_data)
abline(model, col = "red", lwd = 2)

# 2) 상관계수와 결정계수 구하기
correlation <- cor(age, cost)
r_squared <- summary(model)$r.squared
cat("상관계수(r):", correlation, "\n")
cat("결정계수(R²):", r_squared, "\n")

# 3) 분산분석표 작성 및 회귀직선의 유의 여부 검정
anova_result <- anova(model)
summary_result <- summary(model)
print(anova_result)
print(summary_result)
# age 변수의 p-value (단순 선형 회귀에서는 모델 전체의 F-검정 p-value와 동일)
cat("유의확률(p-value):", summary_result$coefficients["age", "Pr(>|t|)"], "\n")
cat("유의수준 α=0.05에서 회귀직선이",
    ifelse(summary_result$coefficients["age", "Pr(>|t|)"] < 0.05, "유의함", "유의하지 않음"), "\n")

# 4) 사용연도가 4년인 기계의 평균 정비비용 추정
new_data <- data.frame(age = 4)
predicted_cost <- predict(model, newdata = new_data)
cat("사용연도가 4년인 기계의 추정 정비비용:", predicted_cost, "천원\n")

# 5) 잔차 계산 및 합이 0인지 확인
residuals <- residuals(model)
cat("잔차의 합:", sum(residuals), "\n") # 이론적으로 0에 매우 가까움

# 6) 잔차(e)와 x의 곱의 합 계산
x_residual_product <- sum(age * residuals)
cat("Σx_i·e_i의 값:", x_residual_product, "\n") # 이론적으로 0에 매우 가까움

# 7) 잔차(e)와 ŷ의 곱의 합 계산 
predicted_y <- fitted(model) # 회귀모델에 의해 예측된 y 값 (ŷ)
y_hat_residual_product <- sum(predicted_y * residuals)
cat("Σŷ_i·e_i의 값:", y_hat_residual_product, "\n") # 이론적으로 0에 매우 가까움

# 회귀식 표시
intercept <- coef(model)[1]
slope <- coef(model)[2]
cat("회귀식: cost =", round(intercept, 2), "+", round(slope, 2), "× age\n")
