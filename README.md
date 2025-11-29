# Система обнаружения вторжений на основе машинного обучения

## Описание проекта

Курсовой проект по разработке системы обнаружения вторжений (IDS), использующей алгоритмы машинного обучения для классификации сетевого трафика на нормальный и аномальный (атаки).

**Автор:** Тураев Сейит  
**Специальность:** 6-05-0533-12 Кибербезопасность  
**Курс:** 3  
**Научный руководитель:** Петров Сергей Валерьевич

## Цель проекта

Спроектировать и разработать программный прототип системы обнаружения вторжений, использующий модели машинного обучения (Random Forest и нейронную сеть) для классификации сетевого трафика.

## Структура проекта

```
ids_ml_course/
├── data/                    # Данные
│   ├── raw/                # Исходные данные CIC-IDS2017
│   └── processed/          # Обработанные данные
├── models/                  # Сохраненные модели
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_results_visualization.ipynb
├── src/                     # Исходный код
│   ├── data_preprocessing.py
│   ├── train_random_forest.py
│   ├── train_neural_network.py
│   ├── evaluation.py
│   └── realtime_ids.py
├── tests/                   # Тесты
├── docs/                    # Документация
├── results/                 # Результаты экспериментов
│   └── figures/            # Графики и визуализации
├── config.yaml             # Конфигурация
├── requirements.txt        # Зависимости
└── README.md              # Этот файл
```

## Установка и настройка

### 1. Клонирование репозитория

```bash
cd ids_ml_course
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # На Windows: venv\Scripts\activate
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Скачивание датасета CIC-IDS2017

Датасет доступен на сайте Canadian Institute for Cybersecurity:
https://www.unb.ca/cic/datasets/ids-2017.html

Распакуйте CSV файлы в директорию `data/raw/`

## Использование

### Предобработка данных

```bash
python src/data_preprocessing.py
```

### Обучение моделей

```bash
# Random Forest
python src/train_random_forest.py

# Нейронная сеть
python src/train_neural_network.py
```

### Оценка моделей

```bash
python src/evaluation.py
```

### Запуск Real-time IDS

```bash
python src/realtime_ids.py
```

## Датасет

**CIC-IDS2017** - современный датасет для исследования систем обнаружения вторжений, содержащий:
- Нормальный трафик
- Различные типы атак: Brute Force, DoS, DDoS, Web attacks, Infiltration, Botnet

## Модели машинного обучения

1. **Random Forest** - ансамблевый метод, демонстрирующий высокую точность и интерпретируемость
2. **Нейронная сеть (MLP)** - полносвязная сеть для сравнения с классическими методами

## Метрики оценки

- Accuracy (Точность)
- Precision (Прецизионность)
- Recall (Полнота)
- F1-score
- ROC-AUC
- Confusion Matrix

## Технологический стек

- **Python 3.8+**
- **pandas, numpy** - обработка данных
- **scikit-learn** - машинное обучение
- **TensorFlow/Keras** - нейронные сети
- **scapy** - анализ сетевых пакетов
- **matplotlib, seaborn** - визуализация

## Результаты

Результаты экспериментов, метрики и визуализации сохраняются в директории `results/`

## Лицензия

Учебный проект для Гродненского государственного университета имени Янки Купалы

## Контакты

Тураев Сейит - студент 3 курса, факультет математики и информатики

