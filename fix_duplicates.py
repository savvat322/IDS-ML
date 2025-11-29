"""
Скрипт для удаления дубликатов между train и test выборками
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


class DuplicateFixer:
    """
    Класс для исправления проблемы дубликатов между train и test
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
    
    def find_and_remove_duplicates(self, save_backup=True):
        """
        Поиск и удаление дубликатов между train и test
        
        Args:
            save_backup: сохранять ли резервные копии
            
        Returns:
            Количество удаленных дубликатов
        """
        print("=" * 60)
        print("УДАЛЕНИЕ ДУБЛИКАТОВ МЕЖДУ TRAIN И TEST")
        print("=" * 60)
        
        # Загрузка данных
        print("\nЗагрузка данных...")
        X_train = np.load(self.processed_dir / 'X_train.npy')
        X_test = np.load(self.processed_dir / 'X_test.npy')
        y_train = np.load(self.processed_dir / 'y_train.npy')
        y_test = np.load(self.processed_dir / 'y_test.npy')
        
        print(f"  Train: {X_train.shape}")
        print(f"  Test:  {X_test.shape}")
        
        # Сохранение резервных копий
        if save_backup:
            backup_dir = self.processed_dir / 'backup'
            backup_dir.mkdir(exist_ok=True)
            
            print("\nСохранение резервных копий...")
            np.save(backup_dir / 'X_train_backup_dupfix.npy', X_train)
            np.save(backup_dir / 'X_test_backup_dupfix.npy', X_test)
            np.save(backup_dir / 'y_train_backup_dupfix.npy', y_train)
            np.save(backup_dir / 'y_test_backup_dupfix.npy', y_test)
            print("  ✓ Резервные копии сохранены")
        
        # Конвертация в DataFrame для эффективного поиска дубликатов
        print("\nПоиск дубликатов...")
        train_df = pd.DataFrame(X_train)
        test_df = pd.DataFrame(X_test)
        
        # Используем хеширование для эффективности
        train_hashes = pd.util.hash_pandas_object(train_df, index=False)
        test_hashes = pd.util.hash_pandas_object(test_df, index=False)
        
        # Создаем множества хешей
        train_hash_set = set(train_hashes)
        test_hash_set = set(test_hashes)
        
        # Находим дубликаты
        duplicate_hashes = train_hash_set.intersection(test_hash_set)
        n_duplicates = len(duplicate_hashes)
        
        print(f"  Найдено дубликатов: {n_duplicates:,}")
        print(f"  Процент от test:    {n_duplicates/len(test_df)*100:.4f}%")
        
        if n_duplicates == 0:
            print("\n✓ Дубликаты не найдены!")
            return 0
        
        # Удаляем дубликаты из test (оставляем их в train)
        print("\nУдаление дубликатов из test выборки...")
        
        # Создаем маску для test: True = оставить, False = удалить
        test_mask = ~test_hashes.isin(duplicate_hashes)
        n_test_before = len(test_df)
        
        X_test_cleaned = X_test[test_mask.values]
        y_test_cleaned = y_test[test_mask.values]
        
        n_test_after = len(X_test_cleaned)
        n_removed = n_test_before - n_test_after
        
        print(f"  Test до удаления:   {n_test_before:,}")
        print(f"  Test после удаления: {n_test_after:,}")
        print(f"  Удалено образцов:   {n_removed:,}")
        
        # Проверяем, что дубликаты действительно удалены
        test_df_cleaned = pd.DataFrame(X_test_cleaned)
        test_hashes_cleaned = pd.util.hash_pandas_object(test_df_cleaned, index=False)
        remaining_duplicates = len(set(test_hashes_cleaned).intersection(train_hash_set))
        
        print(f"\nПроверка: оставшихся дубликатов: {remaining_duplicates}")
        
        if remaining_duplicates > 0:
            print("  ⚠ Внимание: остались дубликаты, возможно из-за округления")
        else:
            print("  ✓ Все дубликаты успешно удалены")
        
        # Сохранение очищенных данных
        print("\nСохранение очищенных данных...")
        np.save(self.processed_dir / 'X_test.npy', X_test_cleaned)
        np.save(self.processed_dir / 'y_test.npy', y_test_cleaned)
        print("  ✓ Данные сохранены")
        
        # Сохранение информации об удалении
        removal_info = {
            'n_duplicates_found': int(n_duplicates),
            'n_test_before': int(n_test_before),
            'n_test_after': int(n_test_after),
            'n_removed': int(n_removed),
            'duplicate_pct_before': float(n_duplicates / n_test_before * 100),
            'duplicate_pct_after': float(remaining_duplicates / n_test_after * 100) if n_test_after > 0 else 0
        }
        
        import json
        info_path = self.processed_dir / 'duplicate_removal_info.json'
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(removal_info, f, indent=2)
        
        print(f"  ✓ Информация сохранена: {info_path}")
        
        print("\n" + "=" * 60)
        print("УДАЛЕНИЕ ДУБЛИКАТОВ ЗАВЕРШЕНО")
        print("=" * 60)
        
        return n_removed


def main():
    """
    Главная функция
    """
    parser = argparse.ArgumentParser(
        description='Удаление дубликатов между train и test выборками'
    )
    parser.add_argument('--no-backup', action='store_true',
                       help='Не сохранять резервные копии')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    fixer = DuplicateFixer(config_path=args.config)
    fixer.find_and_remove_duplicates(save_backup=not args.no_backup)


if __name__ == '__main__':
    main()


