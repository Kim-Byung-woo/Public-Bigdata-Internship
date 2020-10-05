# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:18:30 2020

@author: User
"""

import pandas as pd
import numpy as np

'''
클래스 레이블명은 TARGET이며 1이면 불만, 0이면 만족한 고객
'''
# 데이터 로드
import os
os.getcwd()

df_data = pd.read_csv('c:\\Users\\User\\Documents\\Python Scripts\\Public-Bigdata-Internship\\data\\dataframe.csv', index_col=None)
df_data = df_data.iloc[: , 1:]

# 피처/레이블 분리
df_data_x = df_data.drop(['Degree', 'X1', 'Y'], axis=1) # Degree = 종속변수(인구수)
df_data_y = df_data['Degree']

'''
# 데이터 전처리 - 이상치 제거
quartile_1 = x_train_df.quantile(0.25)
quartile_3 = x_train_df.quantile(0.75)
IQR = quartile_3 - quartile_1
condition = (x_train_df < (quartile_1 - 1.5 * IQR)) | (x_train_df > (quartile_3 + 1.5 * IQR))
condition = condition.any(axis=1)
condition = ~ condition
x_data_df = x_train_df[condition]
y_data_df = y_train_df[condition]
'''

# 데이터 전처리 - 정규화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_data = scaler.fit_transform(df_data_x)
y_data = df_data_y.to_numpy()

# 데이터 분할
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_data, 
                                                    y_data, 
                                                    test_size = 0.3, 
                                                    random_state = 777, 
                                                    stratify = y_data)

# 데이터 전처리 - 오버샘플링
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 0)
x_train, y_train = sm.fit_resample(x_train, y_train)



# 모델 선택
'''
params['n_estimators'] # 반복 수행하는 트리의 개수
params['objective'] = 'multiclass' # LGBMClassifier의 경우 'binary'또는 'multiclass'
params['metric'] = 'multi_logloss' # loss를 측정하기 위한 기준(multi_logloss: metric for multi-class)
params['num_leaves'] = 10 # 하나의 트리가 가질 수 있는 최대 리프 개수(높이면 정확도는 높지만 과적합 발생 가능)
params['max_depth'] = 10 # 트리의 최대 깊이
'''
from lightgbm import LGBMClassifier
model = LGBMClassifier(random_state = 0,  n_jobs=-1) # n_jobs: 사용가능 cpu core 개수
# 하이퍼 파라미터 찾기
params = { "n_estimators":[100, 200, 1000],
          "objective": ['multiclass'],
          "metric": ['multi_logloss'],
          "num_leaves": [10, 20, 50],
          "max_depth": [10, 20, 50],
          "learning_rate":[0.01, 0.05, 0.1]}

from sklearn.model_selection import GridSearchCV
grid_model = GridSearchCV(model,
                          param_grid = params,
                          cv = 5,
                          n_jobs = -1)

# 모델 러닝, eval_set 지정
grid_model.fit(x_train, 
          y_train,
          early_stopping_rounds = 50, # 50번 반복 후 오류가 변화가 없을 시 조기 중단합니다.
          eval_set = [(x_test, y_test)],
          eval_metric='error') # merror: Multiclass classification error rate


# 모델 검증
# cv결과표
cv_score = pd.DataFrame(grid_model.cv_results_)
print(f'Best Param: {grid_model.best_params_}')
print(f'Best Score: {grid_model.best_score_}')

# 모델 예측
estimator = grid_model.best_estimator_
y_predict = estimator.predict(x_test)
print("훈련 데이터셋 예측 결과: {:.3f}".format(estimator.score(x_train, y_train)))
print("테스트 데이터셋 예측 결과: {:.3f}".format(estimator.score(x_test, y_test)))

# F1 Score 확인
from sklearn.metrics import classification_report
rep = classification_report(y_test, y_predict)
print(rep)


# 피처 중요도를 그래프로 나타낸다.
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize = (10, 12))
from lightgbm import plot_importance
plot_importance(estimator, ax = ax)


