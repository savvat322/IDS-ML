"""
Анализ разделимости данных - проверка, действительно ли задача простая
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns


class DataSeparabilityAnalyzer:
    """
    Анализ разделимости данных для понимания, почему метрики такие высокие
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Инициализация
        
        Args:
            config_path: путь к файлу конфигурации
        """
        # Автоматически определяем правильный путь к config.yaml
        if not os.path.exists(config_path):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            alternative_path = os.path.join(project_root, 'config.yaml')
            
            if os.path.exists(alternative_path):
                config_path = alternative_path
            else:
                raise FileNotFoundError(
                    f"Config file not found: {config_path}\n"
                    f"Also tried: {alternative_path}"
                )
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.project_root = Path(config_path).parent
        self.processed_dir = self.project_root / self.config['data']['processed_dir']
        self.results_dir = self.project_root / self.config['results']['save_dir']
        self.figures_dir = self.project_root / self.config['results']['figures_dir']
        self.figures_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """
        Загрузка данных
        """
        print("Загрузка данных...")
        
        X_train = np.load(self.processed_dir / 'X_train.npy')
        X_test = np.load(self.processed_dir / 'X_test.npy')
        y_train = np.load(self.processed_dir / 'y_train.npy')
        y_test = np.load(self.processed_dir / 'y_test.npy')
        
        print(f"  Train: {X_train.shape}")
        print(f"  Test:  {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def test_simple_models(self, X_train, y_train, X_test, y_test):
        """
        Тестирование простых моделей для оценки сложности задачи
        
        Args:
            X_train, y_train: обучающие данные
            X_test, y_test: тестовые данные
        """
        print("\n" + "=" * 70)
        print("ТЕСТИРОВАНИЕ ПРОСТЫХ МОДЕЛЕЙ")
        print("=" * 70)
        
        models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                n_jobs=-1
            ),
            'Decision Tree (max_depth=5)': DecisionTreeClassifier(
                max_depth=5,
                random_state=42
            ),
            'Decision Tree (max_depth=10)': DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),
            'Random Forest (weak)': RandomForestClassifier(
                n_estimators=10,
                max_depth=5,
                min_samples_split=50,
                min_samples_leaf=20,
                random_state=42,
                n_jobs=-1
            ),
        }
        
        results = []
        
        # Используем выборку для быстрого тестирования
        sample_size = min(50000, len(X_train))
        sample_indices = np.random.choice(len(X_train), sample_size, replace=False)
        X_train_sample = X_train[sample_indices]
        y_train_sample = y_train[sample_indices]
        
        print(f"\nИспользуется выборка: {sample_size} образцов для обучения")
        
        for name, model in models.items():
            print(f"\nОбучение {name}...")
            
            try:
                model.fit(X_train_sample, y_train_sample)
                
                # Предсказания на train и test
                y_train_pred = model.predict(X_train_sample)
                y_test_pred = model.predict(X_test)
                
                train_f1 = f1_score(y_train_sample, y_train_pred)
                test_f1 = f1_score(y_test, y_test_pred)
                train_acc = accuracy_score(y_train_sample, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)
                
                results.append({
                    'model': name,
                    'train_f1': train_f1,
                    'test_f1': test_f1,
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'f1_diff': train_f1 - test_f1
                })
                
                print(f"  Train F1: {train_f1:.4f}, Test F1: {test_f1:.4f}")
                print(f"  Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
                
            except Exception as e:
                print(f"  ✗ Ошибка: {e}")
        
        # Анализ результатов
        print("\n" + "=" * 70)
        print("АНАЛИЗ РЕЗУЛЬТАТОВ")
        print("=" * 70)
        
        df_results = pd.DataFrame(results)
        print("\nСводная таблица:")
        print(df_results.to_string(index=False))
        
        # Выводы
        print("\n" + "-" * 70)
        print("ВЫВОДЫ:")
        print("-" * 70)
        
        avg_test_f1 = df_results['test_f1'].mean()
        max_test_f1 = df_results['test_f1'].max()
        min_test_f1 = df_results['test_f1'].min()
        
        print(f"\nСредний F1-Score на тесте: {avg_test_f1:.4f}")
        print(f"Максимальный F1-Score: {max_test_f1:.4f}")
        print(f"Минимальный F1-Score: {min_test_f1:.4f}")
        
        if avg_test_f1 > 0.95:
            print("\n⚠ ЗАДАЧА ДЕЙСТВИТЕЛЬНО ПРОСТАЯ!")
            print("  Даже простые модели показывают очень высокие метрики.")
            print("  Это означает, что данные хорошо разделимы.")
            print("  Высокие метрики могут быть нормальными для этой задачи.")
        elif avg_test_f1 > 0.85:
            print("\n✓ Задача средней сложности")
            print("  Простые модели показывают хорошие, но не идеальные результаты.")
        else:
            print("\n✓ Задача сложная")
            print("  Простые модели показывают умеренные результаты.")
        
        # Визуализация
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # F1-Score
        axes[0].bar(df_results['model'], df_results['test_f1'], alpha=0.7)
        axes[0].set_title('F1-Score на тесте для разных моделей', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('F1-Score')
        axes[0].set_ylim([0, 1.1])
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, axis='y', alpha=0.3)
        
        # Accuracy
        axes[1].bar(df_results['model'], df_results['test_acc'], alpha=0.7, color='orange')
        axes[1].set_title('Accuracy на тесте для разных моделей', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_ylim([0, 1.1])
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.figures_dir / 'model_complexity_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ График сохранен: {output_path}")
        plt.close()
        
        return df_results
    
    def analyze_class_distribution(self, y_train, y_test):
        """
        Анализ распределения классов
        """
        print("\n" + "=" * 70)
        print("АНАЛИЗ РАСПРЕДЕЛЕНИЯ КЛАССОВ")
        print("=" * 70)
        
        train_class_0 = (y_train == 0).sum()
        train_class_1 = (y_train == 1).sum()
        test_class_0 = (y_test == 0).sum()
        test_class_1 = (y_test == 1).sum()
        
        train_total = len(y_train)
        test_total = len(y_test)
        
        print(f"\nTrain:")
        print(f"  Class 0 (BENIGN): {train_class_0:,} ({train_class_0/train_total*100:.2f}%)")
        print(f"  Class 1 (ATTACK): {train_class_1:,} ({train_class_1/train_total*100:.2f}%)")
        
        print(f"\nTest:")
        print(f"  Class 0 (BENIGN): {test_class_0:,} ({test_class_0/test_total*100:.2f}%)")
        print(f"  Class 1 (ATTACK): {test_class_1:,} ({test_class_1/test_total*100:.2f}%)")
        
        # Проверка на дисбаланс
        train_ratio = train_class_1 / train_class_0 if train_class_0 > 0 else 0
        test_ratio = test_class_1 / test_class_0 if test_class_0 > 0 else 0
        
        print(f"\nСоотношение классов:")
        print(f"  Train: {train_ratio:.3f} (ATTACK/BENIGN)")
        print(f"  Test:  {test_ratio:.3f} (ATTACK/BENIGN)")
        
        if abs(train_ratio - test_ratio) > 0.1:
            print("\n⚠ Внимание: значительное различие в распределении классов между train и test")
        else:
            print("\n✓ Распределение классов схоже между train и test")
    
    def run_analysis(self):
        """
        Запуск полного анализа
        """
        print("=" * 70)
        print("АНАЛИЗ РАЗДЕЛИМОСТИ ДАННЫХ")
        print("=" * 70)
        
        # Загрузка данных
        X_train, X_test, y_train, y_test = self.load_data()
        
        # Анализ распределения классов
        self.analyze_class_distribution(y_train, y_test)
        
        # Тестирование простых моделей
        results = self.test_simple_models(X_train, y_train, X_test, y_test)
        
        # Итоговые рекомендации
        print("\n" + "=" * 70)
        print("РЕКОМЕНДАЦИИ")
        print("=" * 70)
        
        avg_f1 = results['test_f1'].mean()
        
        if avg_f1 > 0.95:
            print("\n1. ЗАДАЧА ДЕЙСТВИТЕЛЬНО ПРОСТАЯ")
            print("   Высокие метрики (F1 > 0.99) могут быть нормальными.")
            print("   Рекомендуется:")
            print("   - Принять высокие метрики как нормальные для этой задачи")
            print("   - Проверить модель на реальных данных из продакшена")
            print("   - Использовать более строгие метрики (Precision@K, Recall@K)")
            print("   - Мониторить метрики в реальном времени")
        else:
            print("\n1. ЗАДАЧА СРЕДНЕЙ СЛОЖНОСТИ")
            print("   Сильная регуляризация должна помочь снизить метрики.")
            print("   Рекомендуется:")
            print("   - Применить сильную регуляризацию: python src/apply_strong_regularization.py")
            print("   - Использовать более простую модель")
            print("   - Проверить на других данных")
        
        print("\n2. ВАЖНО ПРОВЕРИТЬ:")
        print("   - Работает ли модель на реальных данных?")
        print("   - Нет ли утечки данных в продакшене?")
        print("   - Как модель ведет себя на новых типах атак?")
        
        print("\n" + "=" * 70)
        print("АНАЛИЗ ЗАВЕРШЕН")
        print("=" * 70)
        
        return results


def main():
    """
    Главная функция
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Анализ разделимости данных для понимания сложности задачи'
    )
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    analyzer = DataSeparabilityAnalyzer(config_path=args.config)
    analyzer.run_analysis()


if __name__ == '__main__':
    main()


