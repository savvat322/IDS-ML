"""
Модуль обучения Random Forest классификатора для IDS
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report, accuracy_score, precision_score, 
    recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns


class RandomForestTrainer:
    """
    Класс для обучения Random Forest модели
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Инициализация тренера
        
        Args:
            config_path: путь к файлу конфигурации
        """
        # Автоматически определяем правильный путь к config.yaml
        if not os.path.exists(config_path):
            # Пробуем найти config.yaml относительно расположения скрипта
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            alternative_path = os.path.join(project_root, 'config.yaml')
            
            if os.path.exists(alternative_path):
                config_path = alternative_path
                print(f"✓ Найден config.yaml: {config_path}")
            else:
                raise FileNotFoundError(
                    f"Config file not found: {config_path}\n"
                    f"Also tried: {alternative_path}\n"
                    "Please ensure config.yaml exists in project root."
                )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Определяем корневую директорию проекта
        self.project_root = Path(config_path).parent
        self.processed_dir = self.project_root / self.config['data']['processed_dir']
        self.model_dir = self.project_root / self.config['models']['save_dir']
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.best_params = None
        self.training_time = 0
    
    def load_data(self, include_test=False):
        """
        Загрузка предобработанных данных
        
        Args:
            include_test: загружать ли тестовые данные
        
        Returns:
            X_train, X_val, y_train, y_val (и опционально X_test, y_test)
        """
        print("Загрузка предобработанных данных...")
        
        try:
            X_train = np.load(self.processed_dir / 'X_train.npy')
            X_val = np.load(self.processed_dir / 'X_val.npy')
            y_train = np.load(self.processed_dir / 'y_train.npy')
            y_val = np.load(self.processed_dir / 'y_val.npy')
            
            print(f"  ✓ Train: {X_train.shape}")
            print(f"  ✓ Val:   {X_val.shape}")
            
            if include_test:
                X_test = np.load(self.processed_dir / 'X_test.npy')
                y_test = np.load(self.processed_dir / 'y_test.npy')
                print(f"  ✓ Test:  {X_test.shape}")
                return X_train, X_val, X_test, y_train, y_val, y_test
            
            return X_train, X_val, y_train, y_val
        
        except FileNotFoundError:
            print("✗ Ошибка: обработанные данные не найдены!")
            print("Запустите сначала: python src/data_preprocessing.py")
            sys.exit(1)
    
    def train_basic_model(self, X_train, y_train):
        """
        Обучение базовой модели Random Forest
        
        Args:
            X_train, y_train: обучающие данные
            
        Returns:
            Обученная модель
        """
        print("\nОбучение базовой модели Random Forest...")
        
        rf_config = self.config['random_forest']
        
        # Используем очень консервативные параметры по умолчанию для сильной регуляризации
        model = RandomForestClassifier(
            n_estimators=150,  # Уменьшено для уменьшения сложности
            max_depth=8,  # Сильно уменьшено для сильной регуляризации
            min_samples_split=30,  # Сильно увеличено для сильной регуляризации
            min_samples_leaf=15,  # Сильно увеличено для сильной регуляризации
            max_features='sqrt',  # Ограничение признаков для регуляризации
            random_state=rf_config['random_state'],
            n_jobs=rf_config['n_jobs'],
            verbose=1
        )
        
        start_time = time.time()
        model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        print(f"  ✓ Обучение завершено за {self.training_time:.2f} секунд")
        
        return model
    
    def tune_hyperparameters(self, X_train, y_train):
        """
        Подбор гиперпараметров с помощью GridSearchCV
        
        Args:
            X_train, y_train: обучающие данные
            
        Returns:
            Лучшая модель и параметры
        """
        print("\nПодбор гиперпараметров (GridSearchCV)...")
        print("Это может занять продолжительное время...")
        
        rf_config = self.config['random_forest']
        
        param_grid = {
            'n_estimators': rf_config['n_estimators'],
            'max_depth': rf_config['max_depth'],
            'min_samples_split': rf_config['min_samples_split']
        }
        
        # Добавляем min_samples_leaf, если он указан в конфиге
        if 'min_samples_leaf' in rf_config:
            param_grid['min_samples_leaf'] = rf_config['min_samples_leaf']
        
        # Добавляем max_features, если он указан в конфиге
        if 'max_features' in rf_config:
            param_grid['max_features'] = rf_config['max_features']
        
        print("Сетка параметров:")
        for param, values in param_grid.items():
            print(f"  {param}: {values}")
        
        base_model = RandomForestClassifier(
            random_state=rf_config['random_state'],
            n_jobs=rf_config['n_jobs']
        )
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=rf_config['cv_folds'],
            scoring='f1',
            verbose=2,
            n_jobs=rf_config['n_jobs']
        )
        
        start_time = time.time()
        grid_search.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        
        self.best_params = grid_search.best_params_
        
        print(f"\n  ✓ GridSearch завершен за {self.training_time:.2f} секунд")
        print(f"  ✓ Лучший F1-score: {grid_search.best_score_:.4f}")
        print(f"  ✓ Лучшие параметры:")
        for param, value in self.best_params.items():
            print(f"      {param}: {value}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model, X_val, y_val, dataset_name='Validation'):
        """
        Оценка модели на выборке
        
        Args:
            model: обученная модель
            X_val, y_val: данные для оценки
            dataset_name: название набора данных
            
        Returns:
            Словарь с метриками
        """
        print(f"\nОценка модели на {dataset_name.lower()} выборке...")
        
        y_pred = model.predict(X_val)
        
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        
        print(f"\n  Accuracy:  {accuracy:.6f}")
        print(f"  Precision: {precision:.6f}")
        print(f"  Recall:    {recall:.6f}")
        print(f"  F1-Score:  {f1:.6f}")
        print("\n  Classification Report:")
        print(classification_report(y_val, y_pred, 
                                   target_names=['BENIGN', 'ATTACK'],
                                   digits=4))
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': classification_report(y_val, y_pred, 
                                                          target_names=['BENIGN', 'ATTACK'],
                                                          output_dict=True)
        }
        
        return metrics
    
    def compare_train_test_metrics(self, model, X_train, y_train, X_test, y_test):
        """
        Сравнение метрик на train и test для выявления переобучения
        
        Args:
            model: обученная модель
            X_train, y_train: обучающие данные
            X_test, y_test: тестовые данные
        """
        print("\n" + "=" * 60)
        print("СРАВНЕНИЕ МЕТРИК: TRAIN VS TEST")
        print("=" * 60)
        
        # Метрики на train
        train_metrics = self.evaluate_model(model, X_train, y_train, 'Train')
        
        # Метрики на test
        test_metrics = self.evaluate_model(model, X_test, y_test, 'Test')
        
        # Разница
        print("\n" + "-" * 60)
        print("РАЗНИЦА (Train - Test):")
        print("-" * 60)
        diff_accuracy = train_metrics['accuracy'] - test_metrics['accuracy']
        diff_precision = train_metrics['precision'] - test_metrics['precision']
        diff_recall = train_metrics['recall'] - test_metrics['recall']
        diff_f1 = train_metrics['f1_score'] - test_metrics['f1_score']
        
        print(f"  Accuracy:  {diff_accuracy:+.6f}")
        print(f"  Precision: {diff_precision:+.6f}")
        print(f"  Recall:    {diff_recall:+.6f}")
        print(f"  F1-Score:  {diff_f1:+.6f}")
        
        # Предупреждения
        if diff_f1 > 0.05:
            print("\n  ⚠ СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ: разница F1 > 0.05")
            print("     Рекомендуется увеличить регуляризацию")
        elif diff_f1 > 0.02:
            print("\n  ⚠ УМЕРЕННОЕ ПЕРЕОБУЧЕНИЕ: разница F1 > 0.02")
        elif diff_f1 > 0.01:
            print("\n  ✓ СЛАБОЕ ПЕРЕОБУЧЕНИЕ: разница F1 > 0.01")
        else:
            print("\n  ✓ Переобучение не обнаружено")
        
        if test_metrics['f1_score'] > 0.999:
            print("\n  ⚠ ПОДОЗРИТЕЛЬНО ВЫСОКИЙ F1-SCORE на тесте (>0.999)")
            print("     Возможна утечка данных или дубликаты между выборками")
        
        return {
            'train': train_metrics,
            'test': test_metrics,
            'differences': {
                'accuracy': diff_accuracy,
                'precision': diff_precision,
                'recall': diff_recall,
                'f1_score': diff_f1
            }
        }
    
    def analyze_feature_importance(self, model, top_n=20):
        """
        Анализ важности признаков
        
        Args:
            model: обученная модель
            top_n: количество топ признаков для отображения
        """
        print(f"\nАнализ важности признаков (топ-{top_n})...")
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:top_n]
        
        print("\n  Топ признаки:")
        for i, idx in enumerate(indices[:10], 1):
            print(f"    {i}. Feature {idx}: {importances[idx]:.4f}")
        
        # Визуализация
        plt.figure(figsize=(12, 6))
        plt.title(f'Top {top_n} Feature Importances - Random Forest')
        plt.bar(range(top_n), importances[indices])
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.xticks(range(top_n), indices, rotation=90)
        plt.tight_layout()
        
        # Сохраняем относительно корня проекта
        output_path = self.project_root / 'results/figures/rf_feature_importance.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ График сохранен: {output_path}")
        plt.close()
        
        return importances
    
    def save_model(self, model):
        """
        Сохранение обученной модели
        
        Args:
            model: обученная модель
        """
        model_path = self.project_root / self.config['models']['random_forest_path']
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_path)
        print(f"\n✓ Модель сохранена: {model_path}")
        
        # Сохраняем метаданные
        metadata = {
            'model_type': 'RandomForestClassifier',
            'training_time': self.training_time,
            'best_params': self.best_params if self.best_params else 'default',
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
        }
        
        import json
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Метаданные сохранены: {metadata_path}")
    
    def train_and_evaluate(self, use_grid_search=True, compare_train_test=True):
        """
        Полный цикл обучения и оценки
        
        Args:
            use_grid_search: использовать ли GridSearchCV для подбора параметров
            compare_train_test: сравнивать ли метрики на train и test
        """
        print("=" * 60)
        print("ОБУЧЕНИЕ RANDOM FOREST КЛАССИФИКАТОРА")
        print("=" * 60)
        
        # 1. Загрузка данных
        if compare_train_test:
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_data(include_test=True)
        else:
            X_train, X_val, y_train, y_val = self.load_data(include_test=False)
        
        # 2. Обучение модели
        if use_grid_search:
            model = self.tune_hyperparameters(X_train, y_train)
        else:
            model = self.train_basic_model(X_train, y_train)
        
        self.model = model
        
        # 3. Оценка модели на валидационной выборке
        val_metrics = self.evaluate_model(model, X_val, y_val, 'Validation')
        
        # 4. Сравнение train/test метрик (для выявления переобучения)
        comparison_metrics = None
        if compare_train_test:
            comparison_metrics = self.compare_train_test_metrics(
                model, X_train, y_train, X_test, y_test
            )
        
        # 5. Анализ важности признаков
        self.analyze_feature_importance(model)
        
        # 6. Сохранение модели
        self.save_model(model)
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ RANDOM FOREST ЗАВЕРШЕНО!")
        print("=" * 60)
        
        return model, val_metrics, comparison_metrics


def main():
    """
    Главная функция для запуска из командной строки
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение Random Forest для IDS')
    parser.add_argument('--no-grid-search', action='store_true',
                       help='Не использовать GridSearchCV (быстрое обучение)')
    parser.add_argument('--no-train-test-compare', action='store_true',
                       help='Не сравнивать метрики на train и test')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    trainer = RandomForestTrainer(config_path=args.config)
    trainer.train_and_evaluate(
        use_grid_search=not args.no_grid_search,
        compare_train_test=not args.no_train_test_compare
    )


if __name__ == '__main__':
    main()