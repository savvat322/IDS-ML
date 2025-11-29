"""
Комплексный скрипт для исправления всех обнаруженных проблем модели
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from fix_duplicates import DuplicateFixer
from fix_model_issues import ModelFixer


def main():
    """
    Главная функция для исправления всех проблем
    """
    print("=" * 70)
    print("КОМПЛЕКСНОЕ ИСПРАВЛЕНИЕ ПРОБЛЕМ МОДЕЛИ")
    print("=" * 70)
    
    config_path = 'config.yaml'
    
    # Шаг 1: Удаление дубликатов
    print("\n" + "=" * 70)
    print("ШАГ 1: УДАЛЕНИЕ ДУБЛИКАТОВ МЕЖДУ TRAIN И TEST")
    print("=" * 70)
    dup_fixer = DuplicateFixer(config_path=config_path)
    n_removed = dup_fixer.find_and_remove_duplicates(save_backup=True)
    
    if n_removed == 0:
        print("\n✓ Дубликаты не найдены, пропускаем этот шаг")
    else:
        print(f"\n✓ Удалено {n_removed} дубликатов")
    
    # Шаг 2: Применение остальных исправлений
    print("\n" + "=" * 70)
    print("ШАГ 2: ПРИМЕНЕНИЕ ОСТАЛЬНЫХ ИСПРАВЛЕНИЙ")
    print("=" * 70)
    model_fixer = ModelFixer(config_path=config_path)
    
    # Не удаляем дубликаты повторно, так как уже сделали это
    success = model_fixer.apply_fixes(
        remove_features=True,
        remove_duplicates=False,  # Уже удалили выше
        retrain=True,
        use_grid_search=False  # Можно изменить на True для лучших результатов
    )
    
    if success:
        print("\n" + "=" * 70)
        print("✓ ВСЕ ИСПРАВЛЕНИЯ ПРИМЕНЕНЫ УСПЕШНО!")
        print("=" * 70)
        print("\nРекомендуется:")
        print("  1. Запустить диагностику снова: python src/diagnose_model.py")
        print("  2. Проверить метрики: python src/evaluation.py")
        print("  3. Если проблемы остались, запустить с GridSearchCV:")
        print("     python fix_all_issues.py --use-grid-search")
    else:
        print("\n" + "=" * 70)
        print("✗ ОШИБКИ ПРИ ПРИМЕНЕНИИ ИСПРАВЛЕНИЙ")
        print("=" * 70)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Комплексное исправление всех проблем модели'
    )
    parser.add_argument('--use-grid-search', action='store_true',
                       help='Использовать GridSearchCV при переобучении')
    
    args = parser.parse_args()
    
    # Обновляем логику для использования GridSearchCV
    if args.use_grid_search:
        # Переопределяем функцию main для использования GridSearchCV
        original_main = main
        
        def main():
            original_main()
            # После основного исправления переобучаем с GridSearchCV
            print("\n" + "=" * 70)
            print("ПЕРЕОБУЧЕНИЕ С GRIDSEARCHCV")
            print("=" * 70)
            model_fixer = ModelFixer(config_path='config.yaml')
            model_fixer.retrain_model(use_grid_search=True)
    
    main()


