# Análise de Dados — Detecção de Fraudes em Cartões de Crédito

Para a análise de dados, foi utilizado um conjunto de dados disponibilizado pelo Google com transações de cartão de crédito e fraudes dentro dele.

Inicialmente, começamos importando os dados e tendo uma visualização superficial deles.

```python
import pandas as pd

url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

df.head()
```

## Visualização inicial dos dados

<img width="1578" height="218" alt="image" src="https://github.com/user-attachments/assets/9f9ac808-7d9d-4e69-9210-2ee5b7d49c30" />

Com essa visualização, não é possível notar a olho nu possíveis fraudes. Então, podemos ver uma proporção utilizando as classes.

O modelo diz que a classe `0` não tem fraude, enquanto a classe `1` representa uma fraude. Porém, isso ainda não é o suficiente para analisar como podemos identificar essas fraudes.

```python
df["Class"].value_counts(normalize=True)
```

## Proporção entre transações normais e fraudes

<img width="232" height="171" alt="image" src="https://github.com/user-attachments/assets/51f0d785-52dd-426c-9806-5e544ff707de" />

Para isso, um dos modelos utilizados foi o da biblioteca `scikit-learn`, junto com o `NumPy`.

Dessa forma, podemos verificar se determinadas transações são fraudes ou não com base no treinamento de um modelo utilizando **Regressão Logística**.

```python
import numpy as np

df["Amount_log"] = np.log1p(df["Amount"])

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

from sklearn.model_selection import train_test_split

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    stratify=y,
    test_size=0.3,
    random_state=42
)

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
```

## Resultado da Regressão Logística

<img width="478" height="236" alt="image" src="https://github.com/user-attachments/assets/fcaf38e6-2789-4df1-8bb4-3d9a4b9a8a73" />

É possível analisar que o modelo conseguiu detectar cerca de **84% das fraudes** durante os testes.

Esse resultado pode ser utilizado em situações reais de análise de dados para detecção de fraudes ou até mesmo em aplicações relacionadas à segurança.

# XGBoost

Existe outro método de classificação mais avançado, disponível através da biblioteca `XGBoost`.

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    scale_pos_weight=10, # ajuda com desbalanceamento
    use_label_encoder=False,
    eval_metric="logloss"
)

xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

print(classification_report(y_test, y_pred_xgb))
```

## Resultado do XGBoost

<img width="484" height="171" alt="image" src="https://github.com/user-attachments/assets/2d302bc0-1925-4a80-8e97-7ec88b7ad354" />

Analisando os resultados, podemos ver que o modelo conseguiu detectar cerca de **94% das fraudes em cartões de crédito**.

Nesse teste, o XGBoost apresentou um resultado melhor na detecção de fraudes quando comparado com a Regressão Logística.

# Conclusão

O importante sobre os modelos não são apenas os dados sendo extraídos, mas como utilizá-los para extrair informações de maneira correta, direta e com a maior qualidade e desempenho possível.

A análise mostra como diferentes modelos podem apresentar resultados diferentes utilizando os mesmos dados e como a escolha e configuração desses modelos influenciam diretamente na capacidade de identificar possíveis fraudes.

