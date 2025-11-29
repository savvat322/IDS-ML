"""
Скрипт для применения сильной регуляризации к модели
"""

import sys
from pathlib import Path

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from train_random_forest import RandomForestTrainer


def main():
    """
    Применение сильной регуляризации и переобучение модели
    """
    print("=" * 70)
    print("ПРИМЕНЕНИЕ СИЛЬНОЙ РЕГУЛЯРИЗАЦИИ")
    print("=" * 70)
    
    print("\nТекущие параметры регуляризации:")
    print("  - max_depth: [5, 8, 10] (было [10, 15, 20])")
    print("  - min_samples_split: [20, 30, 50] (было [5, 10, 20])")
    print("  - min_samples_leaf: [10, 15, 20] (было [2, 4, 8])")
    print("  - max_features: ['sqrt', 'log2'] (добавлено)")
    print("  - n_estimators: [100, 150, 200] (было [100, 200, 300])")
    
    print("\n" + "=" * 70)
    print("ПЕРЕОБУЧЕНИЕ МОДЕЛИ С СИЛЬНОЙ РЕГУЛЯРИЗАЦИЕЙ")
    print("=" * 70)
    
    trainer = RandomForestTrainer(config_path='config.yaml')
    
    # Переобучение с GridSearchCV для подбора оптимальных параметров
    print("\nИспользуется GridSearchCV для подбора параметров...")
    print("Это может занять продолжительное время (30+ минут)...")
    
    model, val_metrics, comparison_metrics = trainer.train_and_evaluate(
        use_grid_search=True,
        compare_train_test=True
    )
    
    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ПЕРЕОБУЧЕНИЯ")
    print("=" * 70)
    
    if comparison_metrics:
        test_f1 = comparison_metrics['test']['f1_score']
        train_f1 = comparison_metrics['train']['f1_score']
        diff_f1 = comparison_metrics['differences']['f1_score']
        
        print(f"\nTrain F1-Score: {train_f1:.6f}")
        print(f"Test F1-Score:  {test_f1:.6f}")
        print(f"Разница:        {diff_f1:.6f}")
        
        print("\n" + "-" * 70)
        if test_f1 < 0.99:
            print("✓ F1-Score на тесте стал более реалистичным (<0.99)")
        elif test_f1 < 0.995:
            print("⚠ F1-Score все еще высокий, но улучшился (<0.995)")
        else:
            print("⚠ F1-Score все еще очень высокий (>0.995)")
            print("  Возможно, задача действительно очень простая для этих данных")
        
        if diff_f1 < 0.02:
            print("✓ Переобучение минимальное (разница <0.02)")
        elif diff_f1 < 0.05:
            print("⚠ Переобучение умеренное (разница <0.05)")
        else:
            print("⚠ Переобучение все еще присутствует (разница >=0.05)")
        
        print("\nРекомендуется запустить диагностику:")
        print("  python src/diagnose_model.py")
    
    print("\n" + "=" * 70)
    print("ПЕРЕОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 70)


if __name__ == '__main__':
    main()


