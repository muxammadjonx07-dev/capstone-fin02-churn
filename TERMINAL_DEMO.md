# Как показать проект в терминале

## 1. Один раз — установить зависимости
```bash
pip install -r requirements.txt
```

## 2. Один раз — обучить и сохранить модель
```bash
python train.py
```
Выведет весь процесс: загрузку данных, метрики Logistic Regression и Random Forest,
финальные результаты на test set, сравнение с наивным baseline — и сохранит модель в
`models/churn_model.joblib`.

## 3. Живое демо на защите — предсказание в реальном времени
Самый простой вариант (без ввода вручную, ничего не может пойти не так на защите):
```bash
python predict_cli.py --example
```

Интерактивный вариант (сам вводишь данные клиента по вопросам):
```bash
python predict_cli.py
```

Вариант с JSON (если хочешь заранее подготовить 2-3 разных клиента для показа):
```bash
python predict_cli.py --json '{"gender": "Male", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No", "tenure": 60, "PhoneService": "Yes", "MultipleLines": "Yes", "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes", "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No", "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 95.0, "TotalCharges": 5700.0}'
```

## Что говорить на защите
1. Запусти `python train.py` заранее (до защиты), не при всех — это ~30 секунд, не страшно, но проще показать уже готовую модель
2. На самой защите просто запусти `python predict_cli.py --example` — покажет живое предсказание
3. При желании поменяй 1-2 значения в JSON-варианте, чтобы показать, что при других вводных ответ меняется (например, смени `Contract` на `"Month-to-month"` — вероятность churn вырастет)

## Файлы в этом наборе
- `train.py` — обучение модели (терминальная версия того же пайплайна, что в demo.ipynb)
- `predict_cli.py` — интерактивный/CLI-предиктор
- Положи их в корень своего репозитория, рядом с `demo.ipynb`
