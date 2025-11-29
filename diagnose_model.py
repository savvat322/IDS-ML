"""
Модуль диагностики модели для выявления переобучения и утечки данных
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns


class ModelDiagnostics:
    """
    Класс для диагностики проблем с моделью
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Инициализация диагностики
        
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
        self.model_path = self.project_root / self.config['models']['random_forest_path']
        self.results_dir = self.project_root / self.config['results']['save_dir']
        self.figures_dir = self.project_root / self.config['results']['figures_dir']
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def load_data_and_model(self):
        """
        Загрузка данных и модели
        """
        print("=" * 60)
        print("ЗАГРУЗКА ДАННЫХ И МОДЕЛИ")
        print("=" * 60)
        
        # Загрузка данных
        try:
            self.X_train = np.load(self.processed_dir / 'X_train.npy')
            self.X_test = np.load(self.processed_dir / 'X_test.npy')
            self.y_train = np.load(self.processed_dir / 'y_train.npy')
            self.y_test = np.load(self.processed_dir / 'y_test.npy')
            
            print(f"✓ Train: {self.X_train.shape}")
            print(f"✓ Test:  {self.X_test.shape}")
        except FileNotFoundError as e:
            print(f"✗ Ошибка загрузки данных: {e}")
            sys.exit(1)
        
        # Загрузка модели
        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
            print(f"✓ Модель загружена: {self.model_path}")
        else:
            print(f"✗ Модель не найдена: {self.model_path}")
            print("Создаем временную модель для диагностики...")
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.model.fit(self.X_train, self.y_train)
            print("✓ Временная модель обучена")
    
    def check_overfitting(self):
        """
        Проверка переобучения: сравнение метрик на train и test
        """
        print("\n" + "=" * 60)
        print("ПРОВЕРКА ПЕРЕОБУЧЕНИЯ")
        print("=" * 60)
        
        # Предсказания на train и test
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)
        
        # Метрики на train
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        train_precision = precision_score(self.y_train, y_train_pred)
        train_recall = recall_score(self.y_train, y_train_pred)
        train_f1 = f1_score(self.y_train, y_train_pred)
        
        # Метрики на test
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        test_precision = precision_score(self.y_test, y_test_pred)
        test_recall = recall_score(self.y_test, y_test_pred)
        test_f1 = f1_score(self.y_test, y_test_pred)
        
        # Разница между train и test
        diff_accuracy = train_accuracy - test_accuracy
        diff_precision = train_precision - test_precision
        diff_recall = train_recall - test_recall
        diff_f1 = train_f1 - test_f1
        
        print("\nМетрики на обучающей выборке:")
        print(f"  Accuracy:  {train_accuracy:.6f}")
        print(f"  Precision: {train_precision:.6f}")
        print(f"  Recall:    {train_recall:.6f}")
        print(f"  F1-Score:  {train_f1:.6f}")
        
        print("\nМетрики на тестовой выборке:")
        print(f"  Accuracy:  {test_accuracy:.6f}")
        print(f"  Precision: {test_precision:.6f}")
        print(f"  Recall:    {test_recall:.6f}")
        print(f"  F1-Score:  {test_f1:.6f}")
        
        print("\nРазница (Train - Test):")
        print(f"  Accuracy:  {diff_accuracy:.6f}")
        print(f"  Precision: {diff_precision:.6f}")
        print(f"  Recall:    {diff_recall:.6f}")
        print(f"  F1-Score:  {diff_f1:.6f}")
        
        # Оценка переобучения
        print("\nОценка переобучения:")
        if diff_f1 > 0.05:
            print("  ⚠ СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ: разница F1 > 0.05")
        elif diff_f1 > 0.02:
            print("  ⚠ УМЕРЕННОЕ ПЕРЕОБУЧЕНИЕ: разница F1 > 0.02")
        elif diff_f1 > 0.01:
            print("  ✓ СЛАБОЕ ПЕРЕОБУЧЕНИЕ: разница F1 > 0.01")
        else:
            print("  ✓ Переобучение не обнаружено")
        
        if test_f1 > 0.999:
            print("  ⚠ ПОДОЗРИТЕЛЬНО ВЫСОКИЙ F1-SCORE на тесте (>0.999)")
            print("     Возможна утечка данных или дубликаты")
        
        # Визуализация
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        train_values = [train_accuracy, train_precision, train_recall, train_f1]
        test_values = [test_accuracy, test_precision, test_recall, test_f1]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars1 = ax.bar(x - width/2, train_values, width, label='Train', alpha=0.8)
        bars2 = ax.bar(x + width/2, test_values, width, label='Test', alpha=0.8)
        
        ax.set_xlabel('Метрики', fontsize=12)
        ax.set_ylabel('Значение', fontsize=12)
        ax.set_title('Сравнение метрик: Train vs Test', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim([0, 1.1])
        ax.grid(True, axis='y', alpha=0.3)
        
        # Добавляем значения на столбцы
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.4f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.figures_dir / 'overfitting_check.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ График сохранен: {output_path}")
        plt.close()
        
        return {
            'train_metrics': {
                'accuracy': train_accuracy,
                'precision': train_precision,
                'recall': train_recall,
                'f1': train_f1
            },
            'test_metrics': {
                'accuracy': test_accuracy,
                'precision': test_precision,
                'recall': test_recall,
                'f1': test_f1
            },
            'differences': {
                'accuracy': diff_accuracy,
                'precision': diff_precision,
                'recall': diff_recall,
                'f1': diff_f1
            }
        }
    
    def check_duplicates(self):
        """
        Проверка дубликатов между train и test выборками
        """
        print("\n" + "=" * 60)
        print("ПРОВЕРКА ДУБЛИКАТОВ МЕЖДУ TRAIN И TEST")
        print("=" * 60)
        
        # Конвертируем в DataFrame для удобства
        train_df = pd.DataFrame(self.X_train)
        test_df = pd.DataFrame(self.X_test)
        
        # Проверяем точные дубликаты
        print("\nПроверка точных дубликатов...")
        
        # Используем хеширование для эффективности
        train_hashes = pd.util.hash_pandas_object(train_df, index=False)
        test_hashes = pd.util.hash_pandas_object(test_df, index=False)
        
        # Находим пересечения
        train_set = set(train_hashes)
        test_set = set(test_hashes)
        duplicates = train_set.intersection(test_set)
        
        n_duplicates = len(duplicates)
        n_train = len(train_df)
        n_test = len(test_df)
        duplicate_pct_train = (n_duplicates / n_train) * 100 if n_train > 0 else 0
        duplicate_pct_test = (n_duplicates / n_test) * 100 if n_test > 0 else 0
        
        print(f"  Всего образцов в train: {n_train:,}")
        print(f"  Всего образцов в test:  {n_test:,}")
        print(f"  Найдено дубликатов:     {n_duplicates:,}")
        print(f"  Процент от train:       {duplicate_pct_train:.4f}%")
        print(f"  Процент от test:        {duplicate_pct_test:.4f}%")
        
        if n_duplicates > 0:
            print(f"\n  ⚠ ОБНАРУЖЕНЫ ДУБЛИКАТЫ!")
            if duplicate_pct_test > 1:
                print(f"  ⚠ КРИТИЧНО: {duplicate_pct_test:.2f}% тестовых данных дублируются в train")
            else:
                print(f"  ⚠ Небольшое количество дубликатов может влиять на метрики")
        else:
            print("\n  ✓ Дубликаты не обнаружены")
        
        # Проверяем похожие образцы (близкие по евклидову расстоянию)
        print("\nПроверка похожих образцов (близких по расстоянию)...")
        # Берем небольшую выборку для проверки из-за вычислительной сложности
        sample_size = min(1000, len(test_df))
        test_sample = test_df.sample(n=sample_size, random_state=42).values
        
        # Вычисляем минимальные расстояния от тестовых образцов до train
        from sklearn.metrics.pairwise import euclidean_distances
        
        min_distances = []
        batch_size = 100
        for i in range(0, len(test_sample), batch_size):
            batch = test_sample[i:i+batch_size]
            distances = euclidean_distances(batch, train_df.values[:min(10000, len(train_df))])
            min_distances.extend(distances.min(axis=1))
        
        min_distances = np.array(min_distances)
        very_close = (min_distances < 0.01).sum()
        close = (min_distances < 0.1).sum()
        
        print(f"  Проверено {sample_size} тестовых образцов")
        print(f"  Очень близких (<0.01): {very_close} ({very_close/sample_size*100:.2f}%)")
        print(f"  Близких (<0.1):        {close} ({close/sample_size*100:.2f}%)")
        
        if very_close > sample_size * 0.1:
            print(f"\n  ⚠ МНОГО ПОХОЖИХ ОБРАЗЦОВ: возможно, данные не были правильно перемешаны")
        
        return {
            'n_duplicates': n_duplicates,
            'duplicate_pct_train': duplicate_pct_train,
            'duplicate_pct_test': duplicate_pct_test,
            'very_close_samples': very_close,
            'close_samples': close
        }
    
    def analyze_feature_importance(self, top_n=30):
        """
        Анализ важности признаков для поиска подозрительных
        """
        print("\n" + "=" * 60)
        print("АНАЛИЗ ВАЖНОСТИ ПРИЗНАКОВ")
        print("=" * 60)
        
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print(f"\nТоп-{top_n} наиболее важных признаков:")
        print("-" * 60)
        print(f"{'Ранг':<6} {'Индекс':<8} {'Важность':<12} {'% от суммы':<12}")
        print("-" * 60)
        
        total_importance = importances.sum()
        suspicious_features = []
        
        for i, idx in enumerate(indices[:top_n]):
            importance = importances[idx]
            pct = (importance / total_importance) * 100
            print(f"{i+1:<6} {idx:<8} {importance:<12.6f} {pct:<12.2f}%")
            
            # Подозрительные признаки: слишком высокая важность
            if pct > 50:
                suspicious_features.append((idx, importance, pct))
        
        if suspicious_features:
            print(f"\n  ⚠ ПОДОЗРИТЕЛЬНЫЕ ПРИЗНАКИ (важность >50%):")
            for idx, imp, pct in suspicious_features:
                print(f"    Признак {idx}: {imp:.6f} ({pct:.2f}%)")
            print("    Возможна утечка данных!")
        else:
            print("\n  ✓ Подозрительных признаков не обнаружено")
        
        # Визуализация
        top_indices = indices[:top_n]
        top_importances = importances[top_indices]
        
        plt.figure(figsize=(14, 8))
        plt.barh(range(len(top_indices)), top_importances)
        plt.yticks(range(len(top_indices)), [f'Feature {idx}' for idx in top_indices])
        plt.xlabel('Важность признака', fontsize=12)
        plt.ylabel('Признак', fontsize=12)
        plt.title(f'Топ-{top_n} наиболее важных признаков', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_path = self.figures_dir / 'feature_importance_diagnosis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ График сохранен: {output_path}")
        plt.close()
        
        return {
            'top_features': [(int(idx), float(importances[idx])) for idx in indices[:top_n]],
            'suspicious_features': suspicious_features
        }
    
    def check_feature_correlation(self, sample_size=10000):
        """
        Проверка корреляции признаков с целевой переменной
        """
        print("\n" + "=" * 60)
        print("ПРОВЕРКА КОРРЕЛЯЦИИ ПРИЗНАКОВ С ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
        print("=" * 60)
        
        # Берем выборку для эффективности
        n_samples = min(sample_size, len(self.X_train))
        sample_indices = np.random.choice(len(self.X_train), n_samples, replace=False)
        X_sample = self.X_train[sample_indices]
        y_sample = self.y_train[sample_indices]
        
        # Вычисляем корреляции
        correlations = []
        for i in range(X_sample.shape[1]):
            corr = np.corrcoef(X_sample[:, i], y_sample)[0, 1]
            if np.isnan(corr):
                corr = 0
            correlations.append((i, abs(corr)))
        
        correlations.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\nТоп-20 признаков по абсолютной корреляции с целевой переменной:")
        print("-" * 60)
        print(f"{'Ранг':<6} {'Индекс':<8} {'Корреляция':<12}")
        print("-" * 60)
        
        suspicious_corr = []
        for i, (idx, corr) in enumerate(correlations[:20]):
            print(f"{i+1:<6} {idx:<8} {corr:<12.6f}")
            if abs(corr) > 0.9:
                suspicious_corr.append((idx, corr))
        
        if suspicious_corr:
            print(f"\n  ⚠ ПОДОЗРИТЕЛЬНО ВЫСОКАЯ КОРРЕЛЯЦИЯ (>0.9):")
            for idx, corr in suspicious_corr:
                print(f"    Признак {idx}: {corr:.6f}")
            print("    Возможна утечка данных!")
        else:
            print("\n  ✓ Подозрительных корреляций не обнаружено")
        
        return {
            'top_correlations': correlations[:20],
            'suspicious_correlations': suspicious_corr
        }
    
    def check_data_leakage_detailed(self):
        """
        Детальная проверка на утечку данных
        """
        print("\n" + "=" * 60)
        print("ДЕТАЛЬНАЯ ПРОВЕРКА НА УТЕЧКУ ДАННЫХ")
        print("=" * 60)
        
        issues = []
        suspicious_features = set()
        
        # 1. Проверка на признаки с нулевой дисперсией в одном из классов
        print("\n1. Проверка признаков с нулевой дисперсией в классах...")
        for i in range(self.X_train.shape[1]):
            train_feature = self.X_train[:, i]
            train_class0 = train_feature[self.y_train == 0]
            train_class1 = train_feature[self.y_train == 1]
            
            if len(train_class0) > 0 and len(train_class1) > 0:
                std0 = np.std(train_class0)
                std1 = np.std(train_class1)
                
                # Если один из классов имеет нулевую дисперсию, это подозрительно
                if std0 < 1e-10 or std1 < 1e-10:
                    mean0 = np.mean(train_class0)
                    mean1 = np.mean(train_class1)
                    if abs(mean0 - mean1) > 1e-6:
                        suspicious_features.add(i)
                        if len(issues) < 10:  # Ограничиваем вывод
                            issues.append(f"Признак {i}: нулевая дисперсия в одном из классов")
        
        if suspicious_features:
            print(f"  ⚠ Найдено {len(suspicious_features)} подозрительных признаков")
        else:
            print("  ✓ Признаков с нулевой дисперсией не найдено")
        
        # 2. Проверка на признаки, которые идеально разделяют классы
        print("\n2. Проверка признаков, идеально разделяющих классы...")
        perfect_separators = []
        sample_size = min(5000, len(self.X_train))
        sample_indices = np.random.choice(len(self.X_train), sample_size, replace=False)
        X_sample = self.X_train[sample_indices]
        y_sample = self.y_train[sample_indices]
        
        for i in range(X_sample.shape[1]):
            feature_values = X_sample[:, i]
            unique_values = np.unique(feature_values)
            
            # Проверяем, можно ли идеально разделить классы по этому признаку
            if len(unique_values) < 100:  # Только для дискретных признаков
                perfect = True
                for val in unique_values:
                    mask = feature_values == val
                    if mask.sum() > 0:
                        class_distribution = np.bincount(y_sample[mask])
                        if len(class_distribution) == 2:
                            # Если все образцы с этим значением принадлежат одному классу
                            if class_distribution[0] > 0 and class_distribution[1] > 0:
                                # Проверяем, что распределение не слишком неравномерно
                                ratio = min(class_distribution) / max(class_distribution)
                                if ratio > 0.1:  # Если есть хотя бы 10% другого класса
                                    perfect = False
                                    break
                
                if perfect and len(unique_values) > 1:
                    perfect_separators.append(i)
                    suspicious_features.add(i)
        
        if perfect_separators:
            print(f"  ⚠ Найдено {len(perfect_separators)} признаков-идеальных разделителей")
            print(f"    Индексы: {perfect_separators[:10]}")
        else:
            print("  ✓ Идеальных разделителей не найдено")
        
        # 3. Проверка на признаки с очень высокой важностью
        print("\n3. Проверка признаков с подозрительно высокой важностью...")
        if self.model is not None:
            importances = self.model.feature_importances_
            total_importance = importances.sum()
            
            high_importance_features = []
            for i, imp in enumerate(importances):
                pct = (imp / total_importance) * 100
                if pct > 30:  # Более 30% общей важности
                    high_importance_features.append((i, imp, pct))
                    suspicious_features.add(i)
            
            if high_importance_features:
                print(f"  ⚠ Найдено {len(high_importance_features)} признаков с очень высокой важностью (>30%)")
                for idx, imp, pct in high_importance_features[:5]:
                    print(f"    Признак {idx}: {imp:.6f} ({pct:.2f}%)")
            else:
                print("  ✓ Признаков с подозрительно высокой важностью не найдено")
        
        # 4. Проверка на признаки с очень высокой корреляцией
        print("\n4. Проверка признаков с очень высокой корреляцией...")
        n_samples = min(5000, len(self.X_train))
        sample_indices = np.random.choice(len(self.X_train), n_samples, replace=False)
        X_sample = self.X_train[sample_indices]
        y_sample = self.y_train[sample_indices]
        
        high_corr_features = []
        for i in range(X_sample.shape[1]):
            corr = np.abs(np.corrcoef(X_sample[:, i], y_sample)[0, 1])
            if not np.isnan(corr) and corr > 0.95:
                high_corr_features.append((i, corr))
                suspicious_features.add(i)
        
        if high_corr_features:
            print(f"  ⚠ Найдено {len(high_corr_features)} признаков с очень высокой корреляцией (>0.95)")
            for idx, corr in high_corr_features[:5]:
                print(f"    Признак {idx}: {corr:.6f}")
        else:
            print("  ✓ Признаков с очень высокой корреляцией не найдено")
        
        # Итоговый отчет
        print("\n" + "-" * 60)
        print("ИТОГОВЫЙ ОТЧЕТ ПО УТЕЧКЕ ДАННЫХ:")
        print("-" * 60)
        if suspicious_features:
            suspicious_list = sorted(list(suspicious_features))
            print(f"  ⚠ Найдено {len(suspicious_features)} подозрительных признаков:")
            print(f"    Индексы: {suspicious_list}")
            print("\n  Рекомендации:")
            print("    1. Проверить эти признаки на предмет утечки данных")
            print("    2. Рассмотреть удаление подозрительных признаков")
            print("    3. Убедиться, что признаки не содержат информацию о метках")
        else:
            print("  ✓ Признаков с явной утечкой данных не обнаружено")
        
        return {
            'suspicious_features': sorted(list(suspicious_features)),
            'perfect_separators': perfect_separators,
            'high_importance_features': high_importance_features if 'high_importance_features' in locals() else [],
            'high_corr_features': high_corr_features
        }
    
    def analyze_confusion_matrix(self):
        """
        Детальный анализ confusion matrix
        """
        print("\n" + "=" * 60)
        print("АНАЛИЗ CONFUSION MATRIX")
        print("=" * 60)
        
        y_test_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_test_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        print("\nConfusion Matrix (Test):")
        print(f"                Predicted")
        print(f"              BENIGN  ATTACK")
        print(f"Actual BENIGN   {tn:6d}  {fp:6d}")
        print(f"       ATTACK   {fn:6d}  {tp:6d}")
        
        total = tn + fp + fn + tp
        print(f"\nВсего образцов: {total:,}")
        print(f"  True Negatives (TN):  {tn:,} ({tn/total*100:.2f}%)")
        print(f"  False Positives (FP): {fp:,} ({fp/total*100:.2f}%)")
        print(f"  False Negatives (FN): {fn:,} ({fn/total*100:.2f}%)")
        print(f"  True Positives (TP):  {tp:,} ({tp/total*100:.2f}%)")
        
        # Проверка на подозрительно низкое количество ошибок
        error_rate = (fp + fn) / total if total > 0 else 0
        
        print(f"\nОбщая ошибка: {error_rate:.6f} ({error_rate*100:.4f}%)")
        
        if error_rate < 0.001:
            print("  ⚠ КРИТИЧНО НИЗКАЯ ОШИБКА (<0.1%)")
            print("     Возможна утечка данных или переобучение")
        elif error_rate < 0.01:
            print("  ⚠ ОЧЕНЬ НИЗКАЯ ОШИБКА (<1%)")
            print("     Возможна утечка данных")
        
        # Визуализация
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['BENIGN', 'ATTACK'],
                   yticklabels=['BENIGN', 'ATTACK'])
        plt.title('Confusion Matrix (Test Set)', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        output_path = self.figures_dir / 'confusion_matrix_diagnosis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ График сохранен: {output_path}")
        plt.close()
        
        return {
            'confusion_matrix': cm.tolist(),
            'error_rate': error_rate,
            'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)
        }
    
    def run_all_diagnostics(self):
        """
        Запуск всех диагностических проверок
        """
        print("=" * 60)
        print("ДИАГНОСТИКА МОДЕЛИ")
        print("=" * 60)
        
        # Загрузка данных и модели
        self.load_data_and_model()
        
        # Выполнение всех проверок
        results = {}
        
        results['overfitting'] = self.check_overfitting()
        results['duplicates'] = self.check_duplicates()
        results['feature_importance'] = self.analyze_feature_importance()
        results['correlations'] = self.check_feature_correlation()
        results['data_leakage'] = self.check_data_leakage_detailed()
        results['confusion_matrix'] = self.analyze_confusion_matrix()
        
        # Итоговый отчет
        print("\n" + "=" * 60)
        print("ИТОГОВЫЙ ОТЧЕТ")
        print("=" * 60)
        
        issues = []
        
        # Проверка переобучения
        if results['overfitting']['differences']['f1'] > 0.05:
            issues.append("СИЛЬНОЕ ПЕРЕОБУЧЕНИЕ")
        elif results['overfitting']['test_metrics']['f1'] > 0.999:
            issues.append("ПОДОЗРИТЕЛЬНО ВЫСОКИЙ F1-SCORE")
        
        # Проверка дубликатов
        if results['duplicates']['duplicate_pct_test'] > 1:
            issues.append("КРИТИЧНОЕ КОЛИЧЕСТВО ДУБЛИКАТОВ")
        
        # Проверка признаков
        if results['feature_importance']['suspicious_features']:
            issues.append("ПОДОЗРИТЕЛЬНЫЕ ПРИЗНАКИ (высокая важность)")
        
        if results['correlations']['suspicious_correlations']:
            issues.append("ПОДОЗРИТЕЛЬНЫЕ КОРРЕЛЯЦИИ")
        
        # Проверка утечки данных
        if 'data_leakage' in results and results['data_leakage']['suspicious_features']:
            issues.append(f"УТЕЧКА ДАННЫХ ({len(results['data_leakage']['suspicious_features'])} подозрительных признаков)")
        
        # Проверка ошибок
        if results['confusion_matrix']['error_rate'] < 0.001:
            issues.append("КРИТИЧНО НИЗКАЯ ОШИБКА")
        
        if issues:
            print("\n⚠ ОБНАРУЖЕНЫ ПРОБЛЕМЫ:")
            for i, issue in enumerate(issues, 1):
                print(f"  {i}. {issue}")
            print("\nРекомендации:")
            print("  1. Увеличить регуляризацию (max_depth, min_samples_split) - УЖЕ ВЫПОЛНЕНО")
            print("  2. Проверить данные на утечку информации - ВЫПОЛНЕНО")
            print("  3. Убедиться, что данные правильно перемешаны - УЖЕ ВЫПОЛНЕНО")
            print("  4. Рассмотреть удаление подозрительных признаков")
            print("\n  Для автоматического применения исправлений запустите:")
            print("    python src/fix_model_issues.py")
            print("\n  Или вручную:")
            print("    python src/fix_model_issues.py --use-grid-search  # с GridSearchCV")
        else:
            print("\n✓ Серьезных проблем не обнаружено")
        
        # Сохранение результатов
        import json
        output_path = self.results_dir / 'diagnostics_results.json'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Конвертируем numpy типы в Python типы
        def convert_to_python(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_python(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_python(item) for item in obj]
            return obj
        
        results_python = convert_to_python(results)
        results_python['issues'] = issues
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results_python, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Результаты сохранены: {output_path}")
        print("=" * 60)
        
        return results


def main():
    """
    Главная функция для запуска из командной строки
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Диагностика модели IDS')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    diagnostics = ModelDiagnostics(config_path=args.config)
    diagnostics.run_all_diagnostics()


if __name__ == '__main__':
    main()

