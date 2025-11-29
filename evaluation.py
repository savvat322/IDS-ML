"""
Модуль оценки и сравнения моделей IDS
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


class ModelEvaluator:
    """
    Класс для оценки и сравнения моделей
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Инициализация оценщика
        
        Args:
            config_path: путь к файлу конфигурации
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.results_dir = Path(self.config['results']['save_dir'])
        self.figures_dir = Path(self.config['results']['figures_dir'])
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.rf_model = None
        
    def load_models(self):
        """
        Загрузка обученных моделей
        """
        print("Загрузка обученных моделей...")
        
        # Random Forest
        rf_path = Path(self.config['models']['random_forest_path'])
        if rf_path.exists():
            self.rf_model = joblib.load(rf_path)
            print(f"  ✓ Random Forest загружен: {rf_path}")
        else:
            print(f"  ✗ Random Forest не найден: {rf_path}")
            print("\n✗ Ошибка: модель не найдена!")
            print("Сначала обучите модель:")
            print("  python src/train_random_forest.py")
            sys.exit(1)
    
    def load_test_data(self):
        """
        Загрузка тестовых данных
        
        Returns:
            X_test, y_test
        """
        print("\nЗагрузка тестовых данных...")
        
        try:
            X_test = np.load(self.processed_dir / 'X_test.npy')
            y_test = np.load(self.processed_dir / 'y_test.npy')
            
            print(f"  ✓ Test: {X_test.shape}")
            
            return X_test, y_test
        
        except FileNotFoundError:
            print("✗ Ошибка: тестовые данные не найдены!")
            print("Запустите сначала: python src/data_preprocessing.py")
            sys.exit(1)
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """
        Оценка одной модели
        
        Args:
            model: модель для оценки
            X_test, y_test: тестовые данные
            model_name: имя модели
            
        Returns:
            Словарь с метриками и предсказаниями
        """
        print(f"\nОценка модели: {model_name}")
        
        # Предсказания
        start_time = time.time()
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        prediction_time = time.time() - start_time
        
        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        print(f"  Время предсказания: {prediction_time:.2f} сек")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'prediction_time': prediction_time,
            'confusion_matrix': cm,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
        }

    def plot_confusion_matrices(self, results, y_test):
        """
        Визуализация Confusion Matrix для всех моделей
        
        Args:
            results: список словарей с результатами
            y_test: истинные метки
        """
        print("\nСоздание Confusion Matrices...")
        
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, result in enumerate(results):
            cm = result['confusion_matrix']
            model_name = result['model_name']
            
            # Нормализация для процентов
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       ax=axes[idx], cbar=False,
                       xticklabels=['BENIGN', 'ATTACK'],
                       yticklabels=['BENIGN', 'ATTACK'])
            
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'confusion_matrices.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Confusion matrices сохранены: {output_path}")
        plt.close()
    
    def plot_roc_curves(self, results, y_test):
        """
        Визуализация ROC кривых для всех моделей
        
        Args:
            results: список словарей с результатами
            y_test: истинные метки
        """
        print("\nСоздание ROC кривых...")
        
        plt.figure(figsize=(10, 8))
        
        for result in results:
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, 
                    label=f'{result["model_name"]} (AUC = {roc_auc:.4f})')
        
        # Diagonal line (random classifier)
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)
        
        output_path = self.figures_dir / 'roc_curves.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ ROC кривые сохранены: {output_path}")
        plt.close()
    
    def plot_metrics_comparison(self, results):
        """
        Визуализация сравнения метрик
        
        Args:
            results: список словарей с результатами
        """
        print("\nСоздание сравнительных графиков...")
        
        # Подготовка данных
        model_names = [r['model_name'] for r in results]
        metrics = {
            'Accuracy': [r['accuracy'] for r in results],
            'Precision': [r['precision'] for r in results],
            'Recall': [r['recall'] for r in results],
            'F1-Score': [r['f1_score'] for r in results],
        }
        
        # График
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(model_names))
        width = 0.2
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            offset = width * idx - width * 1.5
            bars = ax.bar(x + offset, values, width, label=metric_name)
            
            # Добавляем значения на столбцы
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names)
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = self.figures_dir / 'metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Сравнение метрик сохранено: {output_path}")
        plt.close()
    
    def create_comparison_table(self, results):
        """
        Создание таблицы сравнения моделей
        
        Args:
            results: список словарей с результатами
            
        Returns:
            DataFrame с результатами
        """
        print("\nСоздание сравнительной таблицы...")
        
        data = []
        for result in results:
            data.append({
                'Model': result['model_name'],
                'Accuracy': f"{result['accuracy']:.4f}",
                'Precision': f"{result['precision']:.4f}",
                'Recall': f"{result['recall']:.4f}",
                'F1-Score': f"{result['f1_score']:.4f}",
                'Prediction Time (s)': f"{result['prediction_time']:.2f}",
            })
        
        df = pd.DataFrame(data)
        
        print("\n" + "=" * 80)
        print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА МОДЕЛЕЙ")
        print("=" * 80)
        print(df.to_string(index=False))
        print("=" * 80)
        
        # Сохранение в CSV
        output_path = self.results_dir / 'model_comparison.csv'
        df.to_csv(output_path, index=False)
        print(f"\n✓ Таблица сохранена: {output_path}")
        
        return df
    
    def save_results(self, results):
        """
        Сохранение результатов в JSON
        
        Args:
            results: список словарей с результатами
        """
        output_data = []
        
        for result in results:
            # Конвертируем numpy типы в Python типы
            output_data.append({
                'model_name': result['model_name'],
                'accuracy': float(result['accuracy']),
                'precision': float(result['precision']),
                'recall': float(result['recall']),
                'f1_score': float(result['f1_score']),
                'prediction_time': float(result['prediction_time']),
                'confusion_matrix': result['confusion_matrix'].tolist(),
            })
        
        output_path = self.results_dir / 'evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✓ Результаты сохранены: {output_path}")
    
    def evaluate_all(self):
        """
        Полная оценка всех моделей
        """
        print("=" * 60)
        print("ОЦЕНКА И СРАВНЕНИЕ МОДЕЛЕЙ")
        print("=" * 60)
        
        # 1. Загрузка моделей и данных
        self.load_models()
        X_test, y_test = self.load_test_data()
        
        # 2. Оценка каждой модели
        results = []
        
        if self.rf_model:
            rf_results = self.evaluate_model(self.rf_model, X_test, y_test, 
                                            'Random Forest')
            results.append(rf_results)
        
        # 3. Визуализация
        self.plot_confusion_matrices(results, y_test)
        self.plot_roc_curves(results, y_test)
        self.plot_metrics_comparison(results)
        
        # 4. Сравнительная таблица
        comparison_df = self.create_comparison_table(results)
        
        # 5. Сохранение результатов
        self.save_results(results)
        
        print("\n" + "=" * 60)
        print("ОЦЕНКА ЗАВЕРШЕНА!")
        print(f"Результаты сохранены в: {self.results_dir}")
        print(f"Графики сохранены в: {self.figures_dir}")
        print("=" * 60)
        
        return results, comparison_df


def main():
    """
    Главная функция для запуска из командной строки
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Оценка и сравнение моделей IDS')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(config_path=args.config)
    evaluator.evaluate_all()


if __name__ == '__main__':
    main()