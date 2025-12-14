## Load packages ----
library(tidyverse)
library(readxl)
library(caret)
library(pROC)
library(mfx)
library(corrplot)
library(randomForest)
library(e1071)
library(class)

## Load & basic cleaning
raw = read_excel("data/default of credit card clients.xls", skip = 1)

data = raw %>%
  rename(default = `default payment next month`) %>%
  mutate(
    default = factor(default, levels = c(0, 1),
                     labels = c("no_default", "default")),
    EDUCATION = factor(case_when(
      EDUCATION %in% c(1, 2, 3) ~ EDUCATION,
      TRUE ~ 4
    ))
  )

# Rename PAY_0 to PAY_1
if("PAY_0" %in% names(data)) {
  data = data %>% rename(PAY_1 = PAY_0)
}

# Repayment status columns
repay_cols = paste0("PAY_", 1:6)

# Recode non-positive repayment codes to 0 = "no delay"
data = data %>%
  mutate(across(all_of(repay_cols), ~ ifelse(.x <= 0, 0, .x)))

# Behavioral predictors (excluding SEX and MARRIAGE for bias concerns)
behavioral_vars = c(
  "LIMIT_BAL",
  "AGE",
  "EDUCATION",
  repay_cols,
  paste0("BILL_AMT", 1:6),
  paste0("PAY_AMT", 1:6)
)

## EDA

# Default distribution
ggplot(data, aes(x = default)) +
  geom_bar(fill = c("steelblue", "coral")) +
  geom_text(stat = "count", aes(label = after_stat(count)), vjust = -0.3)

# Repayment status distribution
pay_long = data[, repay_cols] %>%
  pivot_longer(cols = everything(),
               names_to = "pay_var",
               values_to = "status")

ggplot(pay_long, aes(x = factor(status))) +
  geom_bar(fill = "steelblue") +
  facet_wrap(~ pay_var, nrow = 2) +
  labs(title = "Repayment Status by Month",
       x = "Months Late (0 = on time)",
       y = "Count")

# Correlation heatmap
beh_num = data[, behavioral_vars] %>%
  mutate(across(everything(), as.numeric))

corr_mat = cor(beh_num, use = "pairwise.complete.obs")
corrplot(corr_mat,
         method = "color",
         type = "full",
         tl.cex = 0.7,
         tl.col = "black",
         mar = c(1, 1, 2, 1),
         title = "Correlation Heatmap: Behavioral Features")

## Train/Test split

set.seed(123)
idx = createDataPartition(data$default, p = 0.7, list = FALSE)
train_data = data[idx, ]
test_data  = data[-idx, ]

## Baseline regression: Logit & Probit

form_baseline = as.formula(
  paste("default ~", paste(behavioral_vars, collapse = " + "))
)

# Logistic regression
logit_baseline = glm(
  form_baseline,
  data   = train_data,
  family = binomial(link = "logit")
)

# Probit regression
probit_baseline = glm(
  form_baseline,
  data   = train_data,
  family = binomial(link = "probit")
)

# Compare AIC
AIC(logit_baseline, probit_baseline)

## ROC and AUC comparison

# Predictions on test set
test_data$logit_prob_baseline = predict(logit_baseline, test_data, type = "response")
test_data$probit_prob_baseline = predict(probit_baseline, test_data, type = "response")

# ROC curves
roc_logit_baseline = roc(test_data$default, test_data$logit_prob_baseline)
roc_probit_baseline = roc(test_data$default, test_data$probit_prob_baseline)

# AUC
auc(roc_logit_baseline)
auc(roc_probit_baseline)

# Plot
plot(roc_logit_baseline, col = "blue", main = "ROC: Logit vs Probit")
plot(roc_probit_baseline, col = "red", add = TRUE)
legend("bottomright", legend = c("Logit", "Probit"), col = c("blue", "red"), lwd = 2)

## Marginal effects for Logit

train_mfx = train_data %>%
  mutate(default_num = ifelse(default == "default", 1, 0))

form_mfx = as.formula(
  paste("default_num ~", paste(behavioral_vars, collapse = " + "))
)

logit_mfx = logitmfx(
  formula = form_mfx,
  data    = train_mfx
)

logit_mfx

## Baseline Logit performance on test set

test_data$logit_pred_baseline = ifelse(test_data$logit_prob_baseline > 0.5,
                                        "default", "no_default") %>% 
  factor(levels = levels(test_data$default))

cm_logit_baseline = confusionMatrix(test_data$logit_pred_baseline, test_data$default,positive = "default")
cm_logit_baseline


## Feature engineering

data_eng = data %>%
  mutate(
    # Payment behavior
    times_delayed = (PAY_1 > 0) + (PAY_2 > 0) + (PAY_3 > 0) + 
      (PAY_4 > 0) + (PAY_5 > 0) + (PAY_6 > 0),
    avg_pay_delay = (PAY_1 + PAY_2 + PAY_3 + PAY_4 + PAY_5 + PAY_6) / 6,
    max_pay_delay = pmax(PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6),
    pay_trend = PAY_1 - PAY_6,
    
    # Financial ratios
    util_ratio = pmin(BILL_AMT1 / (LIMIT_BAL + 1), 2),
    pay_to_bill = ifelse(BILL_AMT1 > 0, PAY_AMT1 / BILL_AMT1, 0),
    avg_bill = (BILL_AMT1 + BILL_AMT2 + BILL_AMT3 + BILL_AMT4 + BILL_AMT5 + BILL_AMT6) / 6,
    avg_payment = (PAY_AMT1 + PAY_AMT2 + PAY_AMT3 + PAY_AMT4 + PAY_AMT5 + PAY_AMT6) / 6
  ) %>%
  mutate(across(where(is.numeric), ~ replace(., is.na(.) | is.infinite(.), 0)))

# New feature set
engineered_vars = c(
  behavioral_vars,
  "times_delayed", "avg_pay_delay", "max_pay_delay", "pay_trend",
  "util_ratio", "pay_to_bill", "avg_bill", "avg_payment"
)

train_eng = data_eng[idx, ]
test_eng = data_eng[-idx, ]

## Logit with engineered features
form_eng = as.formula(
  paste("default ~", paste(engineered_vars, collapse = " + "))
)

logit_eng = glm(
  form_eng,
  data   = train_eng,
  family = binomial(link = "logit")
)

## Compare baseline vs engineered
AIC(logit_baseline, logit_eng)

# Predictions on test set
test_eng$logit_prob_eng = predict(logit_eng, test_eng, type = "response")
test_eng$logit_pred_eng = ifelse(test_eng$logit_prob_eng > 0.5,
                                  "default", "no_default") %>%
  factor(levels = levels(test_eng$default))

# Confusion matrix
cm_logit_eng = confusionMatrix(test_eng$logit_pred_eng, test_eng$default,positive = "default")
cm_logit_eng

# ROC and AUC
roc_logit_eng = roc(test_eng$default, test_eng$logit_prob_eng)
auc(roc_logit_eng)


## ML Classifiers

# Use engineered features for ML
train_ml = train_eng[, c("default", engineered_vars)]
test_ml = test_eng[, c("default", engineered_vars)]

# Simple oversampling
set.seed(123)
majority = train_ml[train_ml$default == "no_default", ]
minority = train_ml[train_ml$default == "default", ]

target_size = round(nrow(majority) * 0.7)
minority_oversampled = minority[sample(1:nrow(minority), target_size, replace = TRUE), ]

train_balanced = rbind(majority, minority_oversampled)
train_balanced = train_balanced[sample(1:nrow(train_balanced)), ]

## Random Forest
set.seed(123)
rf_model = randomForest(
  default ~ .,
  data = train_balanced,
  ntree = 300,
  mtry = 6,
  importance = TRUE,
  nodesize = 50
)

rf_pred = predict(rf_model, test_ml, type = "class")
rf_prob = predict(rf_model, test_ml, type = "prob")[, "default"]

cm_rf = confusionMatrix(rf_pred, test_ml$default, positive = "default")
cm_rf

roc_rf = roc(test_ml$default, rf_prob)
auc(roc_rf)

## SVM

set.seed(123)
svm_model = svm(
  default ~ .,
  data = train_balanced,
  kernel = "radial",
  cost = 1,
  gamma = 0.01,
  probability = TRUE
)

svm_pred = predict(svm_model, test_ml)
svm_prob = attr(predict(svm_model, test_ml, probability = TRUE), 
                 "probabilities")[, "default"]

cm_svm = confusionMatrix(svm_pred, test_ml$default, positive = "default")
cm_svm

roc_svm = roc(test_ml$default, svm_prob)
auc(roc_svm)

## KNN

# Scale features
preProc = preProcess(train_balanced[, engineered_vars], method = c("center", "scale"))
train_scaled = predict(preProc, train_balanced[, engineered_vars])
test_scaled = predict(preProc, test_ml[, engineered_vars])

set.seed(123)
knn_cv = train(
  x = train_scaled,
  y = train_balanced$default,
  method = "knn",
  tuneGrid = data.frame(k = seq(15, 45, by = 5)),
  trControl = trainControl(method = "cv", number = 5)
)

knn_pred = knn(train = train_scaled, 
                test = test_scaled,
                cl = train_balanced$default,
                k = knn_cv$bestTune$k,
                prob = TRUE)

knn_prob = attr(knn_pred, "prob")
knn_prob = ifelse(knn_pred == "default", knn_prob, 1 - knn_prob)

cm_knn = confusionMatrix(knn_pred, test_ml$default, positive = "default")
cm_knn

roc_knn = roc(test_ml$default, knn_prob)
auc(roc_knn)

## Final comparison

comparison = data.frame(
  Model = c("Logit Baseline", "Logit Engineered", "Random Forest", "SVM", "KNN"),
  Accuracy = c(cm_logit_baseline$overall['Accuracy'],
               cm_logit_eng$overall['Accuracy'],
               cm_rf$overall['Accuracy'],
               cm_svm$overall['Accuracy'],
               cm_knn$overall['Accuracy']),
  Sensitivity = c(cm_logit_baseline$byClass['Sensitivity'],
                  cm_logit_eng$byClass['Sensitivity'],
                  cm_rf$byClass['Sensitivity'],
                  cm_svm$byClass['Sensitivity'],
                  cm_knn$byClass['Sensitivity']),
  Precision = c(cm_logit_baseline$byClass['Precision'],
                cm_logit_eng$byClass['Precision'],
                cm_rf$byClass['Precision'],
                cm_svm$byClass['Precision'],
                cm_knn$byClass['Precision']),
  F1 = c(cm_logit_baseline$byClass['F1'],
         cm_logit_eng$byClass['F1'],
         cm_rf$byClass['F1'],
         cm_svm$byClass['F1'],
         cm_knn$byClass['F1']),
  AUC = c(auc(roc_logit_baseline),
          auc(roc_logit_eng),
          auc(roc_rf),
          auc(roc_svm),
          auc(roc_knn))
)

comparison

# Final ROC plot
plot(roc_logit_baseline, col = "blue", main = "ROC: All Models")
plot(roc_logit_eng, col = "darkgreen", add = TRUE)
plot(roc_rf, col = "red", add = TRUE)
plot(roc_svm, col = "purple", add = TRUE)
plot(roc_knn, col = "orange", add = TRUE)
legend("bottomright", 
       legend = c("Logit Baseline", "Logit Engineered", "RF", "SVM", "KNN"),
       col = c("blue", "darkgreen", "red", "purple", "orange"), 
       lwd = 2)
