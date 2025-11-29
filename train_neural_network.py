"""
Модуль обучения нейронной сети для IDS
"""

import os
import sys
import yaml
import numpy as np
import json
from pathlib import Path
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt


class NeuralNetworkTrainer:
    """
    Класс для обучения нейронной сети
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Инициализация тренера
        
        Args:
            config_path: путь к файлу конфигурации
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.processed_dir = Path(self.config['data']['processed_dir'])
        self.model_dir = Path(self.config['models']['save_dir'])
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.history = None
        self.training_time = 0
        
        # Установка seed для воспроизводимости
        tf.random.set_seed(42)
        np.random.seed(42)
        
    def load_data(self):
        """
        Загрузка предобработанных данных
        
        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        print("Загрузка предобработанных данных...")
        
        try:
            X_train = np.load(self.processed_dir / 'X_train.npy')
            X_val = np.load(self.processed_dir / 'X_val.npy')
            X_test = np.load(self.processed_dir / 'X_test.npy')
            y_train = np.load(self.processed_dir / 'y_train.npy')
            y_val = np.load(self.processed_dir / 'y_val.npy')
            y_test = np.load(self.processed_dir / 'y_test.npy')
            
            print(f"  ✓ Train: {X_train.shape}")
            print(f"  ✓ Val:   {X_val.shape}")
            print(f"  ✓ Test:  {X_test.shape}")
            
            return X_train, X_val, X_test, y_train, y_val, y_test
        
        except FileNotFoundError:
            print("✗ Ошибка: обработанные данные не найдены!")
            print("Запустите сначала: python src/data_preprocessing.py")
            sys.exit(1)
    
    def build_model(self, input_dim):
        """
        Построение архитектуры нейронной сети
        
        Args:
            input_dim: размерность входных данных
            
        Returns:
            Скомпилированная модель
        """
        print(f"\nПостроение нейронной сети (input_dim={input_dim})...")
        
        nn_config = self.config['neural_network']
        
        model = Sequential([
            # Входной слой
            Dense(nn_config['hidden_layer_1'], 
                  activation='relu', 
                  input_shape=(input_dim,),
                  name='dense_1'),
            Dropout(nn_config['dropout_rate'], name='dropout_1'),
            
            # Второй скрытый слой
            Dense(nn_config['hidden_layer_2'], 
                  activation='relu',
                  name='dense_2'),
            Dropout(nn_config['dropout_rate'], name='dropout_2'),
            
            # Выходной слой
            Dense(1, activation='sigmoid', name='output')
        ])
        
        # Компиляция модели
        optimizer = Adam(learning_rate=nn_config['learning_rate'])
        
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        print("\n  Архитектура модели:")
        model.summary()
        
        return model
    
    def create_callbacks(self):
        """
        Создание callbacks для обучения
        
        Returns:
            Список callbacks
        """
        nn_config = self.config['neural_network']
        model_path = Path(self.config['models']['neural_network_path'])
        
        # EarlyStopping - остановка при отсутствии улучшений
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=nn_config['patience'],
            restore_best_weights=True,
            verbose=1
        )
        
        # ModelCheckpoint - сохранение лучшей модели
        model_checkpoint = callbacks.ModelCheckpoint(
            filepath=str(model_path),
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        
        # ReduceLROnPlateau - уменьшение learning rate при застое
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        return [early_stopping, model_checkpoint, reduce_lr]
    
    def train_model(self, X_train, y_train, X_val, y_val):
        """
        Обучение нейронной сети
        
        Args:
            X_train, y_train: обучающие данные
            X_val, y_val: валидационные данные
            
        Returns:
            Обученная модель и история обучения
        """
        print("\nОбучение нейронной сети...")
        
        nn_config = self.config['neural_network']
        
        # Построение модели
        model = self.build_model(input_dim=X_train.shape[1])
        
        # Создание callbacks
        model_callbacks = self.create_callbacks()
        
        # Обучение
        start_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            batch_size=nn_config['batch_size'],
            epochs=nn_config['epochs'],
            validation_data=(X_val, y_val),
            callbacks=model_callbacks,
            verbose=1
        )
        
        self.training_time = time.time() - start_time
        
        print(f"\n  ✓ Обучение завершено за {self.training_time:.2f} секунд")
        
        self.model = model
        self.history = history
        
        return model, history
    
    def evaluate_model(self, model, X_test, y_test):
        """
        Оценка модели на тестовой выборке
        
        Args:
            model: обученная модель
            X_test, y_test: тестовые данные
            
        Returns:
            Словарь с метриками
        """
        print("\nОценка модели на тестовой выборке...")
        
        # Предсказания
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        # Метрики
        test_loss, test_acc, test_precision, test_recall = model.evaluate(
            X_test, y_test, verbose=0
        )
        
        print(f"\n  Test Loss:      {test_loss:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall:    {test_recall:.4f}")
        
        print("\n  Classification Report:")
        print(classification_report(y_test, y_pred,
                                   target_names=['BENIGN', 'ATTACK'],
                                   digits=4))
        
        metrics = {
            'loss': float(test_loss),
            'accuracy': float(test_acc),
            'precision': float(test_precision),
            'recall': float(test_recall),
            'classification_report': classification_report(y_test, y_pred,
                                                          target_names=['BENIGN', 'ATTACK'],
                                                          output_dict=True)
        }
        
        return metrics
    
    def plot_training_history(self, history):
        """
        Визуализация истории обучения
        
        Args:
            history: объект History из model.fit()
        """
        print("\nСоздание графиков обучения...")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(history.history['precision'], label='Train Precision')
        axes[1, 0].plot(history.history['val_precision'], label='Val Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(history.history['recall'], label='Train Recall')
        axes[1, 1].plot(history.history['val_recall'], label='Val Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        output_path = Path('results/figures/nn_training_history.png')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ График сохранен: {output_path}")
        plt.close()
    
    def save_metadata(self, metrics):
        """
        Сохранение метаданных модели
        
        Args:
            metrics: словарь с метриками
        """
        model_path = Path(self.config['models']['neural_network_path'])
        
        metadata = {
            'model_type': 'Neural Network (MLP)',
            'architecture': {
                'input_dim': int(self.model.input_shape[1]),
                'hidden_layer_1': self.config['neural_network']['hidden_layer_1'],
                'hidden_layer_2': self.config['neural_network']['hidden_layer_2'],
                'dropout_rate': self.config['neural_network']['dropout_rate'],
                'output_dim': 1
            },
            'training_params': {
                'epochs': self.config['neural_network']['epochs'],
                'batch_size': self.config['neural_network']['batch_size'],
                'learning_rate': self.config['neural_network']['learning_rate'],
            },
            'training_time': self.training_time,
            'final_metrics': metrics
        }
        
        metadata_path = model_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n✓ Метаданные сохранены: {metadata_path}")
    
    def train_and_evaluate(self):
        """
        Полный цикл обучения и оценки
        """
        print("=" * 60)
        print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
        print("=" * 60)
        
        # 1. Загрузка данных
        X_train, X_val, X_test, y_train, y_val, y_test = self.load_data()
        
        # 2. Обучение модели
        model, history = self.train_model(X_train, y_train, X_val, y_val)
        
        # 3. Визуализация обучения
        self.plot_training_history(history)
        
        # 4. Оценка на тестовой выборке
        metrics = self.evaluate_model(model, X_test, y_test)
        
        # 5. Сохранение метаданных
        self.save_metadata(metrics)
        
        print("\n" + "=" * 60)
        print("ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ ЗАВЕРШЕНО!")
        print(f"Модель сохранена: {self.config['models']['neural_network_path']}")
        print("=" * 60)
        
        return model, metrics


def main():
    """
    Главная функция для запуска из командной строки
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Обучение нейронной сети для IDS')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    trainer = NeuralNetworkTrainer(config_path=args.config)
    trainer.train_and_evaluate()


if __name__ == '__main__':
    main()

