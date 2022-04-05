import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#Lendo arquivo

train = pd.read_csv("..train.csv", index_col=0)

#Separando a coluna alvo 

y = train['target']
features = train.drop(['target'], axis=1)

#Substituindo palavras e letras por números

object_cols = [col for col in features.columns if 'cat' in col]

X = features.copy()
X_test = test.copy()
ordinal_encoder = OrdinalEncoder()
X[object_cols] = ordinal_encoder.fit_transform(features[object_cols])
X_test[object_cols] = ordinal_encoder.transform(test[object_cols])

#Dividindo os dados para treinamento

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)

#Aplicando o treinamento por Random Forest

model = RandomForestRegressor(random_state=1)

model.fit(X_train, y_train)
preds_valid = model.predict(X_valid)

#Imprimindo o alor médio quadrático do erro

print(mean_squared_error(y_valid, preds_valid, squared=False))
