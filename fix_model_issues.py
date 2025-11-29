"""
Скрипт для исправления проблем модели: удаление подозрительных признаков и переобучение
"""

import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import json
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import argparse


class ModelFixer:
    """
    Класс для исправления проблем модели
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
        self.model_dir = self.project_root / self.config['models']['save_dir']
        self.results_dir = self.project_root / self.config['results']['save_dir']
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_diagnostics_results(self):
        """
        Загрузка результатов диагностики
        
        Returns:
            Словарь с результатами диагностики или None
        """
        diagnostics_path = self.results_dir / 'diagnostics_results.json'
        
        if not diagnostics_path.exists():
            print("⚠ Файл диагностики не найден!")
            print("Запустите сначала: python src/diagnose_model.py")
            return None
        
        with open(diagnostics_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"✓ Результаты диагностики загружены: {diagnostics_path}")
        return results
    
    def get_suspicious_features(self, diagnostics_results=None):
        """
        Получение списка подозрительных признаков из результатов диагностики
        
        Args:
            diagnostics_results: результаты диагностики (если None, загружаются из файла)
            
        Returns:
            Список индексов подозрительных признаков
        """
        if diagnostics_results is None:
            diagnostics_results = self.load_diagnostics_results()
        
        if diagnostics_results is None:
            return []
        
        suspicious_features = set()
        
        # Собираем подозрительные признаки из разных проверок
        if 'data_leakage' in diagnostics_results:
            if 'suspicious_features' in diagnostics_results['data_leakage']:
                suspicious_features.update(diagnostics_results['data_leakage']['suspicious_features'])
        
        if 'feature_importance' in diagnostics_results:
            if 'suspicious_features' in diagnostics_results['feature_importance']:
                for item in diagnostics_results['feature_importance']['suspicious_features']:
                    suspicious_features.add(item[0])  # item[0] - это индекс признака
        
        if 'correlations' in diagnostics_results:
            if 'suspicious_correlations' in diagnostics_results['correlations']:
                for item in diagnostics_results['correlations']['suspicious_correlations']:
                    suspicious_features.add(item[0])  # item[0] - это индекс признака
        
        return sorted(list(suspicious_features))
    
    def remove_features(self, features_to_remove, save_backup=True):
        """
        Удаление подозрительных признаков из обработанных данных
        
        Args:
            features_to_remove: список индексов признаков для удаления
            save_backup: сохранять ли резервные копии оригинальных данных
            
        Returns:
            True если успешно, False иначе
        """
        if not features_to_remove:
            print("✓ Нет признаков для удаления")
            return True
        
        print(f"\nУдаление {len(features_to_remove)} подозрительных признаков...")
        print(f"Индексы: {features_to_remove}")
        
        try:
            # Загрузка данных
            X_train = np.load(self.processed_dir / 'X_train.npy')
            X_val = np.load(self.processed_dir / 'X_val.npy')
            X_test = np.load(self.processed_dir / 'X_test.npy')
            
            print(f"  Исходная форма train: {X_train.shape}")
            
            # Сохранение резервных копий
            if save_backup:
                backup_dir = self.processed_dir / 'backup'
                backup_dir.mkdir(exist_ok=True)
                
                print("  Сохранение резервных копий...")
                np.save(backup_dir / 'X_train_backup.npy', X_train)
                np.save(backup_dir / 'X_val_backup.npy', X_val)
                np.save(backup_dir / 'X_test_backup.npy', X_test)
                print("  ✓ Резервные копии сохранены")
            
            # Удаление признаков (используем инвертированную маску)
            keep_features = [i for i in range(X_train.shape[1]) if i not in features_to_remove]
            
            X_train_cleaned = X_train[:, keep_features]
            X_val_cleaned = X_val[:, keep_features]
            X_test_cleaned = X_test[:, keep_features]
            
            print(f"  Новая форма train: {X_train_cleaned.shape}")
            print(f"  Удалено признаков: {len(features_to_remove)}")
            
            # Сохранение очищенных данных
            print("  Сохранение очищенных данных...")
            np.save(self.processed_dir / 'X_train.npy', X_train_cleaned)
            np.save(self.processed_dir / 'X_val.npy', X_val_cleaned)
            np.save(self.processed_dir / 'X_test.npy', X_test_cleaned)
            
            # Сохранение информации об удаленных признаках
            removal_info = {
                'removed_features': features_to_remove,
                'kept_features': keep_features,
                'original_n_features': X_train.shape[1],
                'new_n_features': X_train_cleaned.shape[1]
            }
            
            info_path = self.processed_dir / 'feature_removal_info.json'
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(removal_info, f, indent=2)
            
            print(f"  ✓ Информация об удалении сохранена: {info_path}")
            print("  ✓ Очищенные данные сохранены")
            
            return True
            
        except Exception as e:
            print(f"  ✗ Ошибка при удалении признаков: {e}")
            return False
    
    def retrain_model(self, use_grid_search=False):
        """
        Переобучение модели с очищенными данными
        
        Args:
            use_grid_search: использовать ли GridSearchCV
        """
        print("\n" + "=" * 60)
        print("ПЕРЕОБУЧЕНИЕ МОДЕЛИ")
        print("=" * 60)
        
        # Импортируем тренер
        sys.path.insert(0, str(self.project_root / 'src'))
        from train_random_forest import RandomForestTrainer
        
        trainer = RandomForestTrainer(config_path=str(self.project_root / 'config.yaml'))
        
        # Обучение с более консервативными параметрами
        model, val_metrics, comparison_metrics = trainer.train_and_evaluate(
            use_grid_search=use_grid_search,
            compare_train_test=True
        )
        
        print("\n" + "=" * 60)
        print("ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО")
        print("=" * 60)
        
        return model, val_metrics, comparison_metrics
    
    def remove_duplicates(self):
        """
        Удаление дубликатов между train и test
        
        Returns:
            True если успешно, False иначе
        """
        print("\n" + "=" * 60)
        print("УДАЛЕНИЕ ДУБЛИКАТОВ МЕЖДУ TRAIN И TEST")
        print("=" * 60)
        
        try:
            # Импортируем DuplicateFixer
            sys.path.insert(0, str(self.project_root / 'src'))
            from fix_duplicates import DuplicateFixer
            
            dup_fixer = DuplicateFixer(config_path=str(self.project_root / 'config.yaml'))
            n_removed = dup_fixer.find_and_remove_duplicates(save_backup=True)
            
            if n_removed > 0:
                print(f"\n✓ Удалено {n_removed} дубликатов")
                return True
            else:
                print("\n✓ Дубликаты не найдены")
                return True
                
        except Exception as e:
            print(f"\n✗ Ошибка при удалении дубликатов: {e}")
            return False
    
    def apply_fixes(self, remove_features=True, remove_duplicates=True, retrain=True, use_grid_search=False):
        """
        Применение всех исправлений
        
        Args:
            remove_features: удалять ли подозрительные признаки
            remove_duplicates: удалять ли дубликаты между train и test
            retrain: переобучать ли модель
            use_grid_search: использовать ли GridSearchCV при переобучении
        """
        print("=" * 60)
        print("ПРИМЕНЕНИЕ ИСПРАВЛЕНИЙ")
        print("=" * 60)
        
        # Загрузка результатов диагностики
        diagnostics_results = self.load_diagnostics_results()
        if diagnostics_results is None:
            print("\n✗ Невозможно применить исправления без результатов диагностики")
            return False
        
        # Удаление дубликатов (приоритетная задача)
        if remove_duplicates:
            success = self.remove_duplicates()
            if not success:
                print("\n✗ Ошибка при удалении дубликатов")
                return False
        
        # Получение подозрительных признаков
        suspicious_features = self.get_suspicious_features(diagnostics_results)
        
        if suspicious_features:
            print(f"\nНайдено {len(suspicious_features)} подозрительных признаков:")
            print(f"  Индексы: {suspicious_features}")
        else:
            print("\n✓ Подозрительных признаков не найдено")
        
        # Удаление признаков
        if remove_features and suspicious_features:
            success = self.remove_features(suspicious_features)
            if not success:
                print("\n✗ Ошибка при удалении признаков")
                return False
        elif remove_features:
            print("\n✓ Нет признаков для удаления")
        
        # Переобучение модели
        if retrain:
            model, val_metrics, comparison_metrics = self.retrain_model(use_grid_search=use_grid_search)
            
            # Проверка улучшения
            if comparison_metrics:
                test_f1 = comparison_metrics['test']['f1_score']
                train_f1 = comparison_metrics['train']['f1_score']
                diff_f1 = comparison_metrics['differences']['f1_score']
                
                print("\n" + "=" * 60)
                print("РЕЗУЛЬТАТЫ ПЕРЕОБУЧЕНИЯ")
                print("=" * 60)
                print(f"Train F1-Score: {train_f1:.6f}")
                print(f"Test F1-Score:  {test_f1:.6f}")
                print(f"Разница:        {diff_f1:.6f}")
                
                if test_f1 < 0.999:
                    print("\n✓ F1-Score на тесте стал более реалистичным (<0.999)")
                else:
                    print("\n⚠ F1-Score все еще очень высокий, возможно нужны дополнительные меры")
                
                if diff_f1 < 0.05:
                    print("✓ Переобучение уменьшено (разница <0.05)")
                else:
                    print("⚠ Переобучение все еще присутствует (разница >=0.05)")
        
        print("\n" + "=" * 60)
        print("ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ")
        print("=" * 60)
        
        return True


def main():
    """
    Главная функция
    """
    parser = argparse.ArgumentParser(
        description='Исправление проблем модели: удаление признаков и переобучение'
    )
    parser.add_argument('--no-remove-features', action='store_true',
                       help='Не удалять подозрительные признаки')
    parser.add_argument('--no-remove-duplicates', action='store_true',
                       help='Не удалять дубликаты между train и test')
    parser.add_argument('--no-retrain', action='store_true',
                       help='Не переобучать модель')
    parser.add_argument('--use-grid-search', action='store_true',
                       help='Использовать GridSearchCV при переобучении')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    fixer = ModelFixer(config_path=args.config)
    fixer.apply_fixes(
        remove_features=not args.no_remove_features,
        remove_duplicates=not args.no_remove_duplicates,
        retrain=not args.no_retrain,
        use_grid_search=args.use_grid_search
    )


if __name__ == '__main__':
    main()

