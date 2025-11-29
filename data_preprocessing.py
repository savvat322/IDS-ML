import os
import sys
import yaml
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import joblib
from tqdm import tqdm


class DataPreprocessor:
    """
    Класс для предобработки данных CIC-IDS2017
    """

    def __init__(self, config_path='config.yaml'):
        """
        Инициализация препроцессора

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
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
    def load_data(self, sample_size=None):
        """
        Загрузка CSV файлов датасета

        Args:
            sample_size: если указано, загружается только sample_size строк из каждого файла

        Returns:
            DataFrame с объединенными данными
        """
        print("Загрузка данных из директории:", self.raw_dir)

        csv_files = list(self.raw_dir.glob('*.csv'))

        if not csv_files:
            print("Файлы CSV не найдены в", self.raw_dir)
            print("Запустите src/download_dataset.py для получения данных")
            sys.exit(1)
        
        print(f"Найдено {len(csv_files)} CSV файлов")
        
        dataframes = []
        for csv_file in tqdm(csv_files, desc="Загрузка файлов"):
            try:
                if sample_size:
                    df = pd.read_csv(csv_file, nrows=sample_size, encoding='utf-8')
                else:
                    df = pd.read_csv(csv_file, encoding='utf-8')
                dataframes.append(df)
                print(f"  ✓ {csv_file.name}: {len(df)} записей")
            except Exception as e:
                print(f"  ✗ Ошибка загрузки {csv_file.name}: {e}")
        
        if not dataframes:
            print("Ошибка: не удалось загрузить ни одного файла")
            sys.exit(1)
        
        df = pd.concat(dataframes, ignore_index=True)
        print(f"\n✓ Всего загружено: {len(df)} записей")
        print(f"  Признаков: {len(df.columns)}")
        
        return df

    def clean_data(self, df):
        """
        Очистка данных
        """
        print("\nОчистка данных...")
        initial_shape = df.shape

        # Создаем копию для безопасной модификации
        df_clean = df.copy()

        # Удаление пробелов в названиях колонок
        df_clean.columns = df_clean.columns.str.strip()

        # Проверка наличия колонки Label
        if 'Label' not in df_clean.columns:
            label_candidates = [col for col in df_clean.columns if 'label' in col.lower()]
            if label_candidates:
                df_clean = df_clean.rename(columns={label_candidates[0]: 'Label'})
                print(f"  Переименована колонка {label_candidates[0]} -> Label")
            else:
                print("  ⚠ Колонка Label не найдена!")
                return df_clean

        # Замена бесконечных значений на NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)

        # Статистика пропусков
        missing_before = df_clean.isnull().sum().sum()
        print(f"  Пропущенных значений: {missing_before}")

        # Удаление строк с NaN в целевой переменной
        df_clean = df_clean.dropna(subset=['Label'])

        # ЭФФЕКТИВНОЕ заполнение NaN в числовых колонках
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) > 0:
            # Вычисляем медианы только для колонок с пропусками
            cols_with_na = [col for col in numeric_columns if df_clean[col].isnull().any()]
            if cols_with_na:
                medians = df_clean[cols_with_na].median()
                df_clean[cols_with_na] = df_clean[cols_with_na].fillna(medians)

        # Удаление дубликатов
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            print(f"  Найдено дубликатов: {duplicates}")
            df_clean = df_clean.drop_duplicates()

        # Удаление константных колонок
        constant_cols = [col for col in df_clean.columns if df_clean[col].nunique() <= 1 and col != 'Label']
        if constant_cols:
            print(f"  Удаление константных колонок: {len(constant_cols)}")
            df_clean = df_clean.drop(columns=constant_cols)

        print(f"  ✓ Форма данных: {initial_shape} -> {df_clean.shape}")

        return df_clean
    
    def encode_labels(self, df):
        """
        Кодирование меток: бинарная классификация (BENIGN vs ATTACK)
        
        Args:
            df: DataFrame с колонкой Label
            
        Returns:
            DataFrame с бинарными метками
        """
        print("\nКодирование меток...")
        
        # Показать распределение оригинальных меток
        print("\n  Оригинальное распределение:")
        label_counts = df['Label'].value_counts()
        for label, count in label_counts.items():
            print(f"    {label}: {count} ({count/len(df)*100:.2f}%)")
        
        # Бинарная классификация: BENIGN = 0, все остальное = 1 (атака)
        df['Label'] = df['Label'].apply(
            lambda x: 0 if x.strip().upper() == 'BENIGN' else 1
        )
        
        print("\n  Бинарное распределение:")
        print(f"    BENIGN (0): {(df['Label']==0).sum()} ({(df['Label']==0).sum()/len(df)*100:.2f}%)")
        print(f"    ATTACK (1): {(df['Label']==1).sum()} ({(df['Label']==1).sum()/len(df)*100:.2f}%)")
        
        return df
    
    def normalize_features(self, X_train, X_val, X_test):
        """
        Нормализация признаков с помощью StandardScaler
        
        Args:
            X_train, X_val, X_test: обучающая, валидационная и тестовая выборки
            
        Returns:
            Нормализованные выборки
        """
        print("\nНормализация признаков...")
        
        # Обучаем scaler только на train данных
        self.scaler.fit(X_train)
        
        X_train_scaled = self.scaler.transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Сохраняем scaler
        scaler_path = Path(self.config['models']['scaler_path'])
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, scaler_path)
        print(f"  ✓ Scaler сохранен: {scaler_path}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled
    
    def balance_classes(self, X_train, y_train, method='smote'):
        """
        Балансировка классов
        
        Args:
            X_train, y_train: обучающие данные
            method: метод балансировки ('smote', 'undersample', 'none')
            
        Returns:
            Сбалансированные X_train, y_train
        """
        print(f"\nБалансировка классов (метод: {method})...")
        
        print(f"  До балансировки: {X_train.shape[0]} образцов")
        print(f"    Class 0: {(y_train==0).sum()}")
        print(f"    Class 1: {(y_train==1).sum()}")
        
        if method == 'none':
            return X_train, y_train
        
        if method == 'smote':
            # SMOTE может быть медленным на больших данных
            # Используем sampling_strategy для ограничения oversample
            try:
                smote = SMOTE(random_state=42, sampling_strategy=0.5, n_jobs=-1)
                X_train, y_train = smote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"  ⚠ Ошибка SMOTE: {e}")
                print("  Пропускаем балансировку")
                return X_train, y_train
        
        elif method == 'undersample':
            rus = RandomUnderSampler(random_state=42)
            X_train, y_train = rus.fit_resample(X_train, y_train)
        
        print(f"  После балансировки: {X_train.shape[0]} образцов")
        print(f"    Class 0: {(y_train==0).sum()}")
        print(f"    Class 1: {(y_train==1).sum()}")
        
        return X_train, y_train
    
    def split_data(self, df):
        """
        Разделение данных на train/val/test
        
        Args:
            df: подготовленный DataFrame
            
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test, network_info
        """
        print("\nРазделение данных...")
        
        # Сохраняем сетевую информацию перед удалением нечисловых колонок
        network_cols = []
        network_info_cols = ['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 
                            'Protocol', 'Flow Duration', 'Total Length of Fwd Packets',
                            'Total Length of Bwd Packets']
        
        # Ищем колонки с сетевой информацией (разные варианты названий)
        # CIC-IDS2017 использует различные форматы названий
        for col in df.columns:
            col_lower = col.lower().replace(' ', '').replace('_', '')
            # IP адреса
            if any(keyword in col_lower for keyword in ['sourceip', 'srcip', 'ipsrc', 'source ip', 'src ip']):
                network_cols.append(col)
            elif any(keyword in col_lower for keyword in ['destinationip', 'dstip', 'ipdst', 'destination ip', 'dst ip']):
                network_cols.append(col)
            # Порты
            elif any(keyword in col_lower for keyword in ['sourceport', 'srcport', 'sport', 'source port', 'src port']):
                network_cols.append(col)
            elif any(keyword in col_lower for keyword in ['destinationport', 'dstport', 'dport', 'destination port', 'dst port']):
                network_cols.append(col)
            # Протокол
            elif 'protocol' in col_lower:
                network_cols.append(col)
        
        # Сохраняем сетевую информацию
        network_info = None
        if network_cols:
            available_network_cols = [col for col in network_cols if col in df.columns]
            if available_network_cols:
                network_info = df[available_network_cols].copy()
                print(f"  Найдено сетевых колонок: {len(available_network_cols)}")
        
        # Отделяем признаки от целевой переменной
        X = df.drop('Label', axis=1)
        y = df['Label']
        
        # Удаляем нечисловые колонки (если остались)
        X = X.select_dtypes(include=[np.number])
        
        print(f"  Признаков: {X.shape[1]}")
        print(f"  Образцов: {X.shape[0]}")
        
        # ВАЖНО: Явное перемешивание данных перед разделением
        # Это гарантирует случайное распределение образцов
        print("  Перемешивание данных...")
        random_state = self.config['preprocessing']['random_state']
        
        # Удаляем дубликаты ПЕРЕД разделением, чтобы избежать их попадания в обе выборки
        print("  Удаление дубликатов перед разделением...")
        df_combined = pd.concat([X, y], axis=1)
        n_before = len(df_combined)
        df_combined = df_combined.drop_duplicates()
        n_after = len(df_combined)
        n_removed = n_before - n_after
        
        if n_removed > 0:
            print(f"    Удалено {n_removed} дубликатов ({n_removed/n_before*100:.2f}%)")
        
        X = df_combined.drop('Label', axis=1)
        y = df_combined['Label']
        
        # Используем sample для перемешивания (более надежно для DataFrame)
        X_shuffled = X.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        y_shuffled = y.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
        
        test_size = self.config['preprocessing']['test_size']
        val_size = self.config['preprocessing']['validation_size']
        
        # Сначала отделяем test (train_test_split уже делает shuffle, но мы уже перемешали)
        X_temp, X_test, y_temp, y_test = train_test_split(
            X_shuffled, y_shuffled, test_size=test_size, 
            random_state=random_state, stratify=y_shuffled, shuffle=True
        )
        
        # Затем делим оставшееся на train и val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=random_state, stratify=y_temp, shuffle=True
        )
        
        print(f"  ✓ Данные перемешаны и разделены")
        print(f"  Train: {X_train.shape[0]} ({X_train.shape[0]/len(df)*100:.1f}%)")
        print(f"  Val:   {X_val.shape[0]} ({X_val.shape[0]/len(df)*100:.1f}%)")
        print(f"  Test:  {X_test.shape[0]} ({X_test.shape[0]/len(df)*100:.1f}%)")
        
        # Разделяем сетевую информацию аналогично
        network_info_train = None
        network_info_val = None
        network_info_test = None
        
        if network_info is not None:
            # Используем те же индексы для разделения
            network_info_shuffled = network_info.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
            
            # Разделяем аналогично X и y
            network_temp, network_info_test = train_test_split(
                network_info_shuffled, test_size=test_size,
                random_state=random_state, shuffle=True
            )
            
            network_info_train, network_info_val = train_test_split(
                network_temp, test_size=val_size_adjusted,
                random_state=random_state, shuffle=True
            )
        
        return X_train, X_val, X_test, y_train, y_val, y_test, network_info_train, network_info_val, network_info_test
    
    def save_processed_data(self, X_train, X_val, X_test, y_train, y_val, y_test,
                           network_info_train=None, network_info_val=None, network_info_test=None):
        """
        Сохранение обработанных данных
        
        Args:
            X_train, X_val, X_test, y_train, y_val, y_test: подготовленные данные
            network_info_train, network_info_val, network_info_test: сетевая информация
        """
        print("\nСохранение обработанных данных...")
        
        # Сохранение в формате numpy для эффективности
        np.save(self.processed_dir / 'X_train.npy', X_train)
        np.save(self.processed_dir / 'X_val.npy', X_val)
        np.save(self.processed_dir / 'X_test.npy', X_test)
        np.save(self.processed_dir / 'y_train.npy', y_train)
        np.save(self.processed_dir / 'y_val.npy', y_val)
        np.save(self.processed_dir / 'y_test.npy', y_test)
        
        # Сохранение сетевой информации
        if network_info_test is not None:
            network_info_test.to_csv(self.processed_dir / 'network_info_test.csv', index=False)
            print("  ✓ Сетевая информация сохранена для test")
        if network_info_train is not None:
            network_info_train.to_csv(self.processed_dir / 'network_info_train.csv', index=False)
        if network_info_val is not None:
            network_info_val.to_csv(self.processed_dir / 'network_info_val.csv', index=False)
        
        print(f"  ✓ Данные сохранены в {self.processed_dir}")
        
        # Также сохраняем информацию о признаках
        feature_info = {
            'n_features': X_train.shape[1],
            'n_train': X_train.shape[0],
            'n_val': X_val.shape[0],
            'n_test': X_test.shape[0],
        }
        
        import json
        with open(self.processed_dir / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        print("  ✓ Информация о признаках сохранена")
    
    def preprocess_all(self, sample_size=None, balance_method='smote'):
        """
        Выполнение полного цикла предобработки
        
        Args:
            sample_size: ограничение размера выборки (для тестирования)
            balance_method: метод балансировки классов
        """
        print("=" * 60)
        print("ПРЕДОБРАБОТКА ДАННЫХ CIC-IDS2017")
        print("=" * 60)
        
        # 1. Загрузка
        df = self.load_data(sample_size=sample_size)
        
        # 2. Очистка
        df = self.clean_data(df)
        
        # 3. Кодирование меток
        df = self.encode_labels(df)
        
        # 4. Разделение данных
        X_train, X_val, X_test, y_train, y_val, y_test, network_info_train, network_info_val, network_info_test = self.split_data(df)
        
        # 5. Нормализация
        X_train, X_val, X_test = self.normalize_features(X_train, X_val, X_test)
        
        # 6. Балансировка (только train)
        X_train, y_train = self.balance_classes(X_train, y_train, method=balance_method)
        # Примечание: после балансировки индексы network_info_train могут не совпадать
        # В реальной системе лучше сохранять маппинг
        
        # 7. Сохранение
        self.save_processed_data(X_train, X_val, X_test, y_train, y_val, y_test,
                                network_info_train, network_info_val, network_info_test)
        
        print("\n" + "=" * 60)
        print("ПРЕДОБРАБОТКА ЗАВЕРШЕНА УСПЕШНО!")
        print("=" * 60)
        
        return X_train, X_val, X_test, y_train, y_val, y_test


def main():
    """
    Главная функция для запуска из командной строки
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Предобработка данных CIC-IDS2017')
    parser.add_argument('--sample', type=int, default=None,
                       help='Использовать только sample строк из каждого файла (для тестирования)')
    parser.add_argument('--balance', type=str, default='smote',
                       choices=['smote', 'undersample', 'none'],
                       help='Метод балансировки классов')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor(config_path=args.config)
    preprocessor.preprocess_all(sample_size=args.sample, balance_method=args.balance)


if __name__ == '__main__':
    main()

