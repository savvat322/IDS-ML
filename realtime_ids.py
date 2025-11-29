import os
import sys
import yaml
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import time
from datetime import datetime
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Подавление сообщений о параллельных задачах от joblib
import os
# Отключаем verbose вывод от joblib (используется в RandomForest)
os.environ['JOBLIB_START_METHOD'] = 'threading'  # Может помочь уменьшить вывод

try:
    from scapy.all import sniff, IP, TCP, UDP, get_if_list, get_if_addr
    SCAPY_AVAILABLE = True
except ImportError:
    print("⚠ Scapy не установлен. Real-time захват пакетов недоступен.")
    print("Установите: pip install scapy")
    SCAPY_AVAILABLE = False
    get_if_list = None
    get_if_addr = None


class RealtimeIDS:
    """
    Класс для обнаружения вторжений в реальном времени
    """
    
    def __init__(self, config_path='config.yaml'):
        """
        Инициализация IDS
        
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
        
        # Сохраняем корневую директорию проекта для разрешения относительных путей
        self.project_root = Path(config_path).parent
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.scaler = None
        self.feature_info = None
        
        # Определяем пути к директориям (разрешаем относительно корня проекта)
        processed_dir_str = self.config['data']['processed_dir']
        if not os.path.isabs(processed_dir_str):
            self.processed_dir = self.project_root / processed_dir_str
        else:
            self.processed_dir = Path(processed_dir_str)
        
        # Параметры real-time
        self.rt_config = self.config['realtime']
        self.window_size = self.rt_config['window_size']
        self.alert_threshold = self.rt_config['alert_threshold']
        self.simulation_mode = self.rt_config.get('simulation_mode', True)
        
        # Буфер для накопления потоков
        self.flow_buffer = defaultdict(list)
        self.packet_count = 0
        self.attack_count = 0
        self.checked_flows = 0  # Счетчик проверенных потоков
        self.verbose = self.rt_config.get('verbose', False)  # Режим подробного вывода
        self.checked_flows = 0  # Счетчик проверенных потоков
        self.verbose = self.rt_config.get('verbose', False)  # Режим подробного вывода
        
        # Логирование (разрешаем путь относительно корня проекта)
        log_file_path = self.rt_config['log_file']
        if not os.path.isabs(log_file_path):
            self.log_file = self.project_root / log_file_path
        else:
            self.log_file = Path(log_file_path)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
    def load_model(self):
        """
        Загрузка обученной модели и scaler
        """
        print("Загрузка модели...")
        
        # Загрузка Random Forest модели (разрешаем путь относительно корня проекта)
        model_path_str = self.config['models']['random_forest_path']
        if not os.path.isabs(model_path_str):
            model_path = self.project_root / model_path_str
        else:
            model_path = Path(model_path_str)
        
        if not model_path.exists():
            error_msg = f"Модель не найдена: {model_path}\nОбучите модель: python src/train_random_forest.py"
            print(f"✗ {error_msg}")
            raise FileNotFoundError(error_msg)
        
        try:
            self.model = joblib.load(model_path)
            print(f"  ✓ Модель загружена: {model_path}")
        except Exception as e:
            error_msg = f"Ошибка при загрузке модели: {e}"
            print(f"✗ {error_msg}")
            raise RuntimeError(error_msg)
        
        # Загрузка scaler (разрешаем путь относительно корня проекта)
        scaler_path_str = self.config['models']['scaler_path']
        if not os.path.isabs(scaler_path_str):
            scaler_path = self.project_root / scaler_path_str
        else:
            scaler_path = Path(scaler_path_str)
        
        if scaler_path.exists():
            self.scaler = joblib.load(scaler_path)
            print(f"  ✓ Scaler загружен: {scaler_path}")
        else:
            print(f"  ⚠ Scaler не найден: {scaler_path}")
        
        # Проверка соответствия количества признаков модели
        model_n_features = self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else None
        if model_n_features:
            print(f"  ✓ Модель ожидает: {model_n_features} признаков")
            self.expected_n_features = model_n_features
        else:
            # Пытаемся определить из feature_info
            feature_info_path = self.processed_dir / 'feature_info.json'
            if feature_info_path.exists():
                import json
                with open(feature_info_path, 'r') as f:
                    self.feature_info = json.load(f)
                if 'n_features' in self.feature_info:
                    self.expected_n_features = self.feature_info['n_features']
                    print(f"  ✓ Информация о признаках: {self.expected_n_features} признаков")
            else:
                self.expected_n_features = None
        
        # Загрузка информации об удаленных признаках (если есть)
        removal_info_path = self.processed_dir / 'feature_removal_info.json'
        self.feature_removal_info = None
        if removal_info_path.exists():
            import json
            with open(removal_info_path, 'r') as f:
                self.feature_removal_info = json.load(f)
            print(f"  ✓ Информация об удаленных признаках загружена")
            if 'kept_features' in self.feature_removal_info:
                kept_count = len(self.feature_removal_info['kept_features'])
                if model_n_features and kept_count == model_n_features:
                    print(f"    Используется {kept_count} признаков (после удаления)")
                    self.expected_n_features = kept_count
                elif model_n_features:
                    print(f"    ⚠ Несоответствие: модель ожидает {model_n_features}, "
                          f"но сохранено {kept_count} признаков")
        
        # Финальная проверка
        if self.expected_n_features is None:
            print("  ⚠ Не удалось определить ожидаемое количество признаков")
            print("    Будет использовано количество признаков из данных")
    
    def extract_packet_features(self, packet):
        """
        Извлечение признаков из сетевого пакета
        
        Args:
            packet: пакет от scapy
            
        Returns:
            Словарь с признаками или None
        """
        try:
            if IP not in packet:
                return None
            
            features = {}
            
            # Основные признаки IP
            features['ip_src'] = packet[IP].src
            features['ip_dst'] = packet[IP].dst
            features['packet_length'] = len(packet)
            features['ttl'] = packet[IP].ttl
            features['protocol'] = packet[IP].proto
            
            # TCP признаки
            if TCP in packet:
                features['sport'] = packet[TCP].sport
                features['dport'] = packet[TCP].dport
                features['tcp_flags'] = int(packet[TCP].flags)
                features['has_tcp'] = 1
                features['has_udp'] = 0
            # UDP признаки
            elif UDP in packet:
                features['sport'] = packet[UDP].sport
                features['dport'] = packet[UDP].dport
                features['tcp_flags'] = 0
                features['has_tcp'] = 0
                features['has_udp'] = 1
            else:
                return None
            
            features['timestamp'] = time.time()
            
            return features
        
        except Exception as e:
            return None
    
    def aggregate_flow_features(self, flow_packets):
        """
        Агрегация признаков из нескольких пакетов в flow
        
        Args:
            flow_packets: список признаков пакетов
            
        Returns:
            Вектор признаков для модели
        """
        if not flow_packets:
            return None
        
        # Вычисляем статистику
        packet_lengths = [p['packet_length'] for p in flow_packets]
        
        # Создаем базовый набор признаков (упрощенная версия CIC-IDS2017)
        features = {
            'total_packets': len(flow_packets),
            'total_length': sum(packet_lengths),
            'mean_length': np.mean(packet_lengths),
            'std_length': np.std(packet_lengths) if len(packet_lengths) > 1 else 0,
            'min_length': min(packet_lengths),
            'max_length': max(packet_lengths),
            'protocol': flow_packets[0]['protocol'],
            'has_tcp': flow_packets[0]['has_tcp'],
            'has_udp': flow_packets[0]['has_udp'],
        }
        
        # Флаги TCP
        if features['has_tcp']:
            tcp_flags = [p['tcp_flags'] for p in flow_packets]
            features['syn_count'] = sum(1 for f in tcp_flags if f & 0x02)
            features['fin_count'] = sum(1 for f in tcp_flags if f & 0x01)
            features['rst_count'] = sum(1 for f in tcp_flags if f & 0x04)
            features['psh_count'] = sum(1 for f in tcp_flags if f & 0x08)
            features['ack_count'] = sum(1 for f in tcp_flags if f & 0x10)
        else:
            features['syn_count'] = 0
            features['fin_count'] = 0
            features['rst_count'] = 0
            features['psh_count'] = 0
            features['ack_count'] = 0
        
        # Временные признаки
        if len(flow_packets) > 1:
            timestamps = [p['timestamp'] for p in flow_packets]
            durations = np.diff(timestamps)
            features['flow_duration'] = timestamps[-1] - timestamps[0]
            features['mean_iat'] = np.mean(durations) if len(durations) > 0 else 0
        else:
            features['flow_duration'] = 0
            features['mean_iat'] = 0
        
        return features
    
    def create_feature_vector(self, flow_features):
        """
        Создание вектора признаков для модели
        Создает вектор размером, ожидаемым scaler (102 признака), 
        затем удаляет признаки, если нужно
        
        Args:
            flow_features: словарь с признаками flow
            
        Returns:
            numpy array для модели (после нормализации и удаления признаков)
        """
        # Определяем размер для scaler (обычно 102 признака)
        scaler_n_features = 102
        if self.feature_info and 'n_features' in self.feature_info:
            # Если feature_info указывает на исходное количество признаков
            scaler_n_features = self.feature_info['n_features']
        
        # Создаем базовый вектор из доступных признаков
        base_features = [
            flow_features.get('total_packets', 0),
            flow_features.get('total_length', 0),
            flow_features.get('mean_length', 0),
            flow_features.get('std_length', 0),
            flow_features.get('min_length', 0),
            flow_features.get('max_length', 0),
            flow_features.get('protocol', 0),
            flow_features.get('has_tcp', 0),
            flow_features.get('has_udp', 0),
            flow_features.get('syn_count', 0),
            flow_features.get('fin_count', 0),
            flow_features.get('rst_count', 0),
            flow_features.get('psh_count', 0),
            flow_features.get('ack_count', 0),
            flow_features.get('flow_duration', 0),
            flow_features.get('mean_iat', 0),
        ]
        
        # Дополняем нулями до размера scaler (102 признака)
        if len(base_features) < scaler_n_features:
            base_features.extend([0.0] * (scaler_n_features - len(base_features)))
        elif len(base_features) > scaler_n_features:
            # Обрезаем до нужного размера
            base_features = base_features[:scaler_n_features]
        
        return np.array(base_features, dtype=np.float64).reshape(1, -1)
    
    def predict_attack(self, flow_key, flow_packets):
        """
        Предсказание атаки для flow
        
        Args:
            flow_key: идентификатор потока
            flow_packets: список пакетов в потоке
            
        Returns:
            Вероятность атаки
        """
        # Агрегация признаков
        flow_features = self.aggregate_flow_features(flow_packets)
        if flow_features is None:
            return 0.0
        
        # Создание вектора признаков (размер для scaler - обычно 102)
        X = self.create_feature_vector(flow_features)
        
        # Нормализация (scaler ожидает 102 признака)
        if self.scaler:
            try:
                X = self.scaler.transform(X)
            except Exception as e:
                print(f"  ⚠ Ошибка нормализации: {e}")
                print(f"     Размер вектора: {X.shape}, ожидается: {self.scaler.n_features_in_}")
                return 0.0
        
        # Удаление признаков, если модель обучена на меньшем количестве
        if self.feature_removal_info and 'kept_features' in self.feature_removal_info:
            kept_indices = self.feature_removal_info['kept_features']
            if len(kept_indices) < X.shape[1]:
                # Выбираем только нужные признаки
                X = X[:, kept_indices]
        
        # Предсказание
        if self.model is None:
            print("  ⚠ Ошибка: Модель не загружена!")
            return 0.0
        
        try:
            proba = self.model.predict_proba(X)[0][1]  # Вероятность класса ATTACK
            return proba
        except Exception as e:
            print(f"  ⚠ Ошибка предсказания: {e}")
            print(f"     Размер вектора: {X.shape}, модель ожидает: {self.model.n_features_in_ if hasattr(self.model, 'n_features_in_') else 'N/A'}")
            return 0.0
    
    def log_alert(self, flow_key, probability, flow_packets):
        """
        Логирование обнаруженной атаки
        
        Args:
            flow_key: идентификатор потока
            probability: вероятность атаки
            flow_packets: пакеты потока
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Извлечение информации о пакетах
        packet_info = self._extract_packet_info(flow_packets)
        
        # Форматированный вывод
        alert_header = "🚨 ALERT! ATTACK DETECTED"
        separator = "=" * 70
        
        log_entry = (
            f"\n{separator}\n"
            f"{alert_header}\n"
            f"{separator}\n"
            f"⏰ Время:        {timestamp}\n"
            f"🌐 Flow:         {flow_key}\n"
            f"🎯 Вероятность:  {probability:.4f} ({probability*100:.2f}%)\n"
            f"📦 Пакетов в flow: {len(flow_packets)}\n"
            f"{separator}\n"
            f"📦 ИНФОРМАЦИЯ О ПАКЕТАХ:\n"
            f"{separator}\n"
            f"{packet_info}\n"
            f"{separator}\n\n"
        )
        
        print(log_entry)
        
        # Упрощенная версия для лога
        log_entry_simple = (
            f"[{timestamp}] ALERT! Attack detected\n"
            f"  Flow: {flow_key}\n"
            f"  Probability: {probability:.4f} ({probability*100:.2f}%)\n"
            f"  Packets in flow: {len(flow_packets)}\n"
            f"  {packet_info}\n"
            f"  {'=' * 50}\n"
        )
        
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry_simple)
    
    def packet_handler(self, packet):
        """
        Обработчик пакета для scapy.sniff
        
        Args:
            packet: захваченный пакет
        """
        self.packet_count += 1
        
        # Извлечение признаков
        features = self.extract_packet_features(packet)
        if features is None:
            return
        
        # Идентификатор потока (по IP и портам)
        flow_key = (
            features['ip_src'],
            features['ip_dst'],
            features.get('sport', 0),
            features.get('dport', 0)
        )
        
        # Добавление в буфер
        self.flow_buffer[flow_key].append(features)
        
        # Если накопилось достаточно пакетов, анализируем
        if len(self.flow_buffer[flow_key]) >= 10:
            self.checked_flows += 1
            
            # Вывод информации о проверяемом потоке
            if self.verbose:
                print(f"\n🔍 Проверка потока #{self.checked_flows}:")
                print(f"   Flow: {flow_key[0]} → {flow_key[1]} : {flow_key[2]} → {flow_key[3]}")
                print(f"   Пакетов в потоке: {len(self.flow_buffer[flow_key])}")
            
            probability = self.predict_attack(flow_key, self.flow_buffer[flow_key])
            
            if self.verbose:
                print(f"   Вероятность атаки: {probability:.4f} ({probability*100:.2f}%)")
                if probability >= self.alert_threshold:
                    print(f"   ⚠ АТАКА ОБНАРУЖЕНА!")
                else:
                    print(f"   ✓ Нормальный трафик")
            
            if probability >= self.alert_threshold:
                self.attack_count += 1
                self.log_alert(flow_key, probability, self.flow_buffer[flow_key])
            
            # Очистка буфера
            self.flow_buffer[flow_key] = []
        
        # Статистика
        if self.packet_count % 100 == 0:
            print(f"📊 Обработано пакетов: {self.packet_count}, проверено потоков: {self.checked_flows}, обнаружено атак: {self.attack_count}")
    
    def _extract_packet_info(self, flow_packets):
        """
        Извлечение информации о пакетах для отображения
        
        Args:
            flow_packets: список признаков пакетов
            
        Returns:
            Строка с информацией о пакетах
        """
        if not flow_packets:
            return "  Нет информации о пакетах"
        
        info_lines = []
        
        # Первый и последний пакет
        first_packet = flow_packets[0]
        last_packet = flow_packets[-1]
        
        info_lines.append(f"  📍 Первый пакет:")
        info_lines.append(f"    IP Source:      {first_packet.get('ip_src', 'N/A')}")
        info_lines.append(f"    IP Destination: {first_packet.get('ip_dst', 'N/A')}")
        info_lines.append(f"    Protocol:       {first_packet.get('protocol', 'N/A')}")
        if 'sport' in first_packet:
            info_lines.append(f"    Source Port:     {first_packet.get('sport', 'N/A')}")
            info_lines.append(f"    Dest Port:       {first_packet.get('dport', 'N/A')}")
        info_lines.append(f"    Packet Length:   {first_packet.get('packet_length', 'N/A')} bytes")
        
        # Статистика по всем пакетам
        packet_lengths = [p.get('packet_length', 0) for p in flow_packets]
        total_length = sum(packet_lengths)
        
        info_lines.append(f"\n  📊 Статистика потока:")
        info_lines.append(f"    Всего пакетов:   {len(flow_packets)}")
        info_lines.append(f"    Общий размер:    {total_length} bytes ({total_length/1024:.2f} KB)")
        info_lines.append(f"    Средний размер:  {np.mean(packet_lengths):.2f} bytes")
        info_lines.append(f"    Мин. размер:     {min(packet_lengths)} bytes")
        info_lines.append(f"    Макс. размер:    {max(packet_lengths)} bytes")
        
        # Протоколы
        protocols = [p.get('protocol', 0) for p in flow_packets]
        unique_protocols = set(protocols)
        protocol_names = {1: 'ICMP', 6: 'TCP', 17: 'UDP'}
        protocol_str = ', '.join([protocol_names.get(p, f'Protocol {p}') for p in unique_protocols])
        info_lines.append(f"    Протоколы:       {protocol_str}")
        
        # TCP флаги (если есть TCP)
        if any(p.get('has_tcp', 0) for p in flow_packets):
            tcp_flags = [p.get('tcp_flags', 0) for p in flow_packets if p.get('has_tcp', 0)]
            syn_count = sum(1 for f in tcp_flags if f & 0x02)
            fin_count = sum(1 for f in tcp_flags if f & 0x01)
            rst_count = sum(1 for f in tcp_flags if f & 0x04)
            info_lines.append(f"\n  🔌 TCP Статистика:")
            info_lines.append(f"    SYN пакетов:     {syn_count}")
            info_lines.append(f"    FIN пакетов:     {fin_count}")
            info_lines.append(f"    RST пакетов:     {rst_count}")
        
        return "\n".join(info_lines)
    
    def _load_original_test_data(self):
        """
        Попытка загрузить исходные тестовые данные с IP адресами и портами
        
        Returns:
            DataFrame с исходными данными или None
        """
        try:
            # Пробуем найти исходные CSV файлы
            raw_dir = self.project_root / self.config['data']['raw_dir']
            csv_files = list(raw_dir.glob('*.csv'))
            
            if not csv_files:
                return None
            
            # Загружаем небольшой кусок для маппинга (если нужно)
            # В реальности лучше сохранять маппинг индексов при предобработке
            return None  # Пока возвращаем None, так как нужен маппинг
            
        except Exception:
            return None
    
    def _adjust_sample_features(self, sample):
        """
        Приведение образца к формату, ожидаемому моделью
        
        Args:
            sample: вектор признаков
            
        Returns:
            Вектор признаков с правильным количеством признаков
        """
        current_n_features = len(sample)
        
        # Если есть информация об удаленных признаках, используем её (приоритет)
        if self.feature_removal_info and 'kept_features' in self.feature_removal_info:
            kept_indices = self.feature_removal_info['kept_features']
            if len(kept_indices) <= current_n_features:
                # Выбираем только нужные признаки по индексам
                try:
                    return sample[kept_indices]
                except IndexError:
                    print(f"  ⚠ Ошибка: индексы признаков выходят за границы")
                    # Fallback: используем первые N признаков
                    if self.expected_n_features:
                        return sample[:self.expected_n_features]
        
        # Если количество признаков совпадает с ожидаемым, возвращаем как есть
        if self.expected_n_features and current_n_features == self.expected_n_features:
            return sample
        
        # Если количество признаков больше ожидаемого, обрезаем
        if self.expected_n_features and current_n_features > self.expected_n_features:
            if not (self.feature_removal_info and 'kept_features' in self.feature_removal_info):
                # Только если нет информации об удаленных признаках
                print(f"  ⚠ Предупреждение: данные имеют {current_n_features} признаков, "
                      f"модель ожидает {self.expected_n_features}. Используем первые {self.expected_n_features}.")
            return sample[:self.expected_n_features]
        
        # Если количество признаков меньше ожидаемого, дополняем нулями
        if self.expected_n_features and current_n_features < self.expected_n_features:
            print(f"  ⚠ Предупреждение: данные имеют {current_n_features} признаков, "
                  f"модель ожидает {self.expected_n_features}. Дополняем нулями.")
            padding = np.zeros(self.expected_n_features - current_n_features)
            return np.concatenate([sample, padding])
        
        # Если не удалось определить ожидаемое количество, возвращаем как есть
        return sample
    
    def _int_to_ip(self, ip_int):
        """
        Преобразование целого числа в IP адрес
        
        Args:
            ip_int: целое число (IP адрес в числовом формате)
            
        Returns:
            Строка IP адреса (например, "192.168.1.100")
        """
        try:
            # Если это уже строка, возвращаем как есть
            if isinstance(ip_int, str):
                # Проверяем, не является ли это уже IP адресом
                if '.' in ip_int and len(ip_int.split('.')) == 4:
                    return ip_int
                return ip_int
            
            # Если это число, преобразуем в IP
            if pd.isna(ip_int):
                return 'N/A'
            
            ip_int = int(ip_int)
            
            # Преобразуем целое число в IP адрес
            # Формат: a.b.c.d = (a << 24) | (b << 16) | (c << 8) | d
            a = (ip_int >> 24) & 0xFF
            b = (ip_int >> 16) & 0xFF
            c = (ip_int >> 8) & 0xFF
            d = ip_int & 0xFF
            
            return f"{a}.{b}.{c}.{d}"
        except Exception:
            return str(ip_int)
    
    def _load_network_info(self, sample_index):
        """
        Загрузка сетевой информации для конкретного образца
        
        Args:
            sample_index: индекс образца в тестовой выборке
            
        Returns:
            Словарь с сетевой информацией или None
        """
        try:
            network_info_path = self.processed_dir / 'network_info_test.csv'
            if network_info_path.exists():
                network_df = pd.read_csv(network_info_path)
                if sample_index < len(network_df):
                    row = network_df.iloc[sample_index]
                    
                    # Извлекаем информацию (разные варианты названий колонок)
                    info = {}
                    
                    # Ищем колонки по точным названиям (CIC-IDS2017 использует: 'Src IP dec', 'Src Port', 'Dst IP dec', 'Dst Port', 'Protocol')
                    for col in network_df.columns:
                        col_lower = col.lower()
                        
                        # IP адреса (ищем 'src ip' или 'dst ip' в названии)
                        if 'src' in col_lower and 'ip' in col_lower:
                            ip_val = row[col]
                            info['ip_src'] = self._int_to_ip(ip_val)
                        elif 'dst' in col_lower and 'ip' in col_lower:
                            ip_val = row[col]
                            info['ip_dst'] = self._int_to_ip(ip_val)
                        
                        # Порты (ищем 'src port' или 'dst port' в названии)
                        elif 'src' in col_lower and 'port' in col_lower:
                            port_val = row[col]
                            if pd.notna(port_val):
                                try:
                                    info['sport'] = int(float(port_val))
                                except (ValueError, TypeError):
                                    info['sport'] = 'N/A'
                            else:
                                info['sport'] = 'N/A'
                        elif 'dst' in col_lower and 'port' in col_lower:
                            port_val = row[col]
                            if pd.notna(port_val):
                                try:
                                    info['dport'] = int(float(port_val))
                                except (ValueError, TypeError):
                                    info['dport'] = 'N/A'
                            else:
                                info['dport'] = 'N/A'
                        
                        # Протокол
                        elif col_lower == 'protocol':
                            protocol_val = row[col]
                            if pd.notna(protocol_val):
                                try:
                                    protocol_map = {6: 'TCP', 17: 'UDP', 1: 'ICMP', 0: 'HOPOPT'}
                                    protocol_int = int(float(protocol_val))
                                    info['protocol'] = protocol_map.get(protocol_int, f'Protocol {protocol_int}')
                                except (ValueError, TypeError):
                                    info['protocol'] = 'N/A'
                            else:
                                info['protocol'] = 'N/A'
                    
                    # Устанавливаем значения по умолчанию, если не найдены
                    if 'ip_src' not in info:
                        info['ip_src'] = 'N/A'
                    if 'ip_dst' not in info:
                        info['ip_dst'] = 'N/A'
                    if 'sport' not in info:
                        info['sport'] = 'N/A'
                    if 'dport' not in info:
                        info['dport'] = 'N/A'
                    if 'protocol' not in info:
                        info['protocol'] = 'N/A'
                    
                    return info
        except Exception as e:
            # Если не удалось загрузить, возвращаем None
            pass
        
        return None
    
    def _extract_sample_info(self, sample, probability, sample_index=None):
        """
        Извлечение информации о пакете/потоке из образца для отображения
        
        Args:
            sample: вектор признаков
            probability: вероятность атаки
            sample_index: индекс образца (для загрузки исходных данных)
            
        Returns:
            Строка с информацией о пакете/потоке
        """
        info_lines = []
        
        # Попытка загрузить сетевую информацию из сохраненных данных
        network_info = None
        if sample_index is not None:
            network_info = self._load_network_info(sample_index)
        
        # Если есть исходные данные, показываем их
        if network_info and 'ip_src' in network_info:
            info_lines.append(f"  🌐 Source IP:       {network_info.get('ip_src', 'N/A')}")
            info_lines.append(f"  🌐 Destination IP:  {network_info.get('ip_dst', 'N/A')}")
            
            # Порты
            sport = network_info.get('sport', 'N/A')
            dport = network_info.get('dport', 'N/A')
            info_lines.append(f"  🔌 Source Port:    {sport}")
            info_lines.append(f"  🔌 Destination Port: {dport}")
            
            # Протокол
            protocol = network_info.get('protocol', 'N/A')
            info_lines.append(f"  📡 Protocol:        {protocol}")
            
            # Дополнительная информация, если доступна
            if 'packet_length' in network_info:
                info_lines.append(f"  📦 Packet Length:   {network_info.get('packet_length', 'N/A')} bytes")
            if 'tcp_flags' in network_info:
                info_lines.append(f"  🚩 TCP Flags:       {network_info.get('tcp_flags', 'N/A')}")
            
            # Если порты не найдены, пытаемся найти их в других колонках
            # Если порты не найдены, пытаемся найти их в других форматах
            if sport == 'N/A' or dport == 'N/A':
                # Пытаемся найти порты в других колонках (может быть в числовом формате)
                try:
                    network_info_path = self.processed_dir / 'network_info_test.csv'
                    if network_info_path.exists() and sample_index is not None:
                        network_df = pd.read_csv(network_info_path)
                        if sample_index < len(network_df):
                            row = network_df.iloc[sample_index]
                            # Ищем любые колонки, которые могут содержать порты
                            for col in network_df.columns:
                                col_lower = col.lower()
                                # Проверяем, может ли это быть порт (значение от 1 до 65535)
                                val = row[col]
                                if pd.notna(val):
                                    try:
                                        port_val = int(float(val))
                                        if 1 <= port_val <= 65535:
                                            if ('src' in col_lower or 'source' in col_lower) and 'port' in col_lower:
                                                if sport == 'N/A':
                                                    sport = port_val
                                                    info_lines[2] = f"  🔌 Source Port:    {sport}"  # Обновляем строку
                                            elif ('dst' in col_lower or 'destination' in col_lower) and 'port' in col_lower:
                                                if dport == 'N/A':
                                                    dport = port_val
                                                    info_lines[3] = f"  🔌 Destination Port: {dport}"  # Обновляем строку
                                    except (ValueError, TypeError):
                                        pass
                except Exception:
                    pass
                
                if sport == 'N/A' or dport == 'N/A':
                    info_lines.append(f"\n  ⚠ Примечание: Порты не найдены в сетевой информации")
                    info_lines.append(f"     (данные CIC-IDS2017 используют flow-based признаки)")
        else:
            # Если исходных данных нет, показываем информацию на основе признаков
            # и пытаемся восстановить примерные значения
            
            # Находим признаки с наибольшими значениями (могут быть связаны с потоком)
            abs_values = np.abs(sample)
            top_indices = np.argsort(abs_values)[::-1][:10]
            
            info_lines.append(f"  ⚠ Исходные данные недоступны (данные нормализованы)")
            info_lines.append(f"  📊 Статистика потока (на основе признаков):")
            info_lines.append(f"     Всего признаков: {len(sample)}")
            info_lines.append(f"     Среднее значение: {np.mean(sample):.4f}")
            info_lines.append(f"     Макс. значение: {np.max(sample):.4f}")
            
            # Пытаемся интерпретировать топ признаки
            # (это приблизительно, так как данные нормализованы)
            info_lines.append(f"\n  🔝 Наиболее значимые признаки:")
            for rank, idx in enumerate(top_indices[:5], 1):
                value = sample[idx]
                abs_val = abs_values[idx]
                # Попытка интерпретации (на основе типичной структуры CIC-IDS2017)
                feature_names = {
                    0: "Flow Duration",
                    1: "Total Fwd Packets", 
                    2: "Total Backward Packets",
                    # Добавьте больше по мере необходимости
                }
                feature_name = feature_names.get(idx, f"Feature #{idx}")
                info_lines.append(f"     {rank}. {feature_name}: {value:.4f}")
        
        # Оценка уровня угрозы
        threat_level = "КРИТИЧЕСКИЙ" if probability >= 0.95 else "ВЫСОКИЙ" if probability >= 0.8 else "СРЕДНИЙ"
        info_lines.append(f"\n  ⚠ Уровень угрозы:   {threat_level}")
        
        # Анализ причины уровня угрозы
        threat_reasons = self._analyze_threat_level(sample, probability)
        if threat_reasons:
            info_lines.append(f"\n  📋 ПРИЧИНЫ ОПРЕДЕЛЕНИЯ УРОВНЯ УГРОЗЫ:")
            info_lines.append(f"     {threat_reasons}")
        
        return "\n".join(info_lines)
    
    def _analyze_threat_level(self, sample, probability):
        """
        Анализ причин определения уровня угрозы
        
        Args:
            sample: вектор признаков
            probability: вероятность атаки
            
        Returns:
            Строка с объяснением причин
        """
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            return None
        
        reasons = []
        
        # Получаем важность признаков
        importances = self.model.feature_importances_
        top_indices = np.argsort(importances)[::-1][:5]
        
        # Анализируем топ-5 наиболее важных признаков
        abs_values = np.abs(sample)
        sample_normalized = sample  # Данные уже нормализованы
        
        # Проверяем значения топ признаков
        suspicious_features = []
        for idx in top_indices:
            if idx < len(sample):
                value = sample[idx]
                abs_val = abs_values[idx]
                importance = importances[idx]
                
                # Если значение признака сильно отклоняется от нуля (нормализованные данные)
                # и признак важен, это может указывать на атаку
                if abs_val > 2.0:  # Более 2 стандартных отклонений
                    suspicious_features.append((idx, value, importance, abs_val))
        
        # Формируем объяснение
        if probability >= 0.95:
            reasons.append("КРИТИЧЕСКИЙ уровень:")
            reasons.append(f"  • Вероятность атаки: {probability*100:.2f}% (очень высокая)")
            if suspicious_features:
                reasons.append(f"  • Обнаружено {len(suspicious_features)} подозрительных признаков с экстремальными значениями:")
                for idx, val, imp, abs_val in suspicious_features[:3]:
                    # Попытка интерпретации признака
                    feature_names_map = {
                        0: "Flow Duration",
                        2: "Flow Packets/s", 
                        49: "Fwd Packet Length Mean",
                        46: "Bwd Packet Length Mean",
                        17: "Flow Bytes/s"
                    }
                    feature_name = feature_names_map.get(idx, f"Признак #{idx}")
                    reasons.append(f"    - {feature_name}: значение {val:.2f} (отклонение {abs_val:.2f}σ, важность {imp:.3f})")
            reasons.append("  • Рекомендуется: НЕМЕДЛЕННАЯ БЛОКИРОВКА соединения")
            
        elif probability >= 0.8:
            reasons.append("ВЫСОКИЙ уровень:")
            reasons.append(f"  • Вероятность атаки: {probability*100:.2f}% (высокая)")
            if suspicious_features:
                reasons.append(f"  • Обнаружено {len(suspicious_features)} подозрительных признаков:")
                for idx, val, imp, abs_val in suspicious_features[:2]:
                    feature_names_map = {
                        0: "Flow Duration",
                        2: "Flow Packets/s",
                        49: "Fwd Packet Length Mean",
                        46: "Bwd Packet Length Mean",
                        17: "Flow Bytes/s"
                    }
                    feature_name = feature_names_map.get(idx, f"Признак #{idx}")
                    reasons.append(f"    - {feature_name}: значение {val:.2f} (отклонение {abs_val:.2f}σ)")
            reasons.append("  • Рекомендуется: детальный анализ трафика и мониторинг")
            
        else:
            reasons.append("СРЕДНИЙ уровень:")
            reasons.append(f"  • Вероятность атаки: {probability*100:.2f}% (умеренная)")
            if suspicious_features:
                reasons.append(f"  • Обнаружено {len(suspicious_features)} подозрительных признаков")
            reasons.append("  • Рекомендуется: мониторинг соединения")
        
        return "\n".join(reasons)
    
    def simulate_realtime(self, test_data_path=None):
        """
        Симуляция real-time анализа на тестовых данных
        
        Args:
            test_data_path: путь к тестовым данным (если None, используется X_test)
        """
        print("\n🔍 Запуск симуляции Real-time IDS...")
        print(f"   Порог обнаружения: {self.alert_threshold}")
        print(f"   Логирование: {self.log_file}")
        print()
        
        # Загрузка тестовых данных
        if test_data_path:
            X_test = np.load(test_data_path)
        else:
            X_test = np.load(self.processed_dir / 'X_test.npy')
        
        print(f"Загружено тестовых образцов: {len(X_test)}")
        print(f"Режим подробного вывода: {'ВКЛ' if self.verbose else 'ВЫКЛ'}")
        print("Начало симуляции (нажмите Ctrl+C для остановки)...\n")
        
        try:
            for i, sample in enumerate(X_test[:1000]):  # Ограничим для демо
                self.packet_count += 1
                
                # Проверка и приведение к нужному формату
                sample = self._adjust_sample_features(sample)
                
                # Проверка, что модель загружена
                if self.model is None:
                    print("✗ Ошибка: Модель не загружена!")
                    print("   Убедитесь, что модель обучена: python src/train_random_forest.py")
                    break
                
                # Вывод информации о проверяемом образце
                if self.verbose:
                    print(f"\n🔍 Проверка образца #{i+1}:")
                    print(f"   Размерность признаков: {len(sample)}")
                    # Пытаемся загрузить сетевую информацию
                    network_info = self._load_network_info(i)
                    if network_info:
                        print(f"   IP: {network_info.get('ip_src', 'N/A')} → {network_info.get('ip_dst', 'N/A')}")
                        print(f"   Порты: {network_info.get('sport', 'N/A')} → {network_info.get('dport', 'N/A')}")
                        print(f"   Протокол: {network_info.get('protocol', 'N/A')}")
                
                # Предсказание
                sample_reshaped = sample.reshape(1, -1)
                proba = self.model.predict_proba(sample_reshaped)[0][1]
                
                if self.verbose:
                    print(f"   Вероятность атаки: {proba:.4f} ({proba*100:.2f}%)")
                    if proba >= self.alert_threshold:
                        print(f"   ⚠ АТАКА ОБНАРУЖЕНА!")
                    else:
                        print(f"   ✓ Нормальный трафик")
                
                if proba >= self.alert_threshold:
                    self.attack_count += 1
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Извлечение информации о пакете/потоке
                    feature_info = self._extract_sample_info(sample, proba, sample_index=i)
                    
                    # Форматированный вывод
                    alert_header = "🚨 ALERT! ATTACK DETECTED"
                    separator = "=" * 70
                    
                    log_entry = (
                        f"\n{separator}\n"
                        f"{alert_header}\n"
                        f"{separator}\n"
                        f"⏰ Время:        {timestamp}\n"
                        f"📊 Образец:      #{i}\n"
                        f"🎯 Вероятность:  {proba:.4f} ({proba*100:.2f}%)\n"
                        f"{separator}\n"
                        f"{feature_info}\n"
                        f"{separator}\n\n"
                    )
                    
                    print(log_entry)
                    
                    # Упрощенная версия для лога
                    log_entry_simple = (
                        f"[{timestamp}] ALERT! Attack detected (sample #{i})\n"
                        f"  Probability: {proba:.4f} ({proba*100:.2f}%)\n"
                        f"  {feature_info}\n"
                        f"  {'=' * 50}\n"
                    )
                    
                    with open(self.log_file, 'a', encoding='utf-8') as f:
                        f.write(log_entry_simple)
                
                self.packet_count += 1
                
                # Статистика (без лишних сообщений о параллельных задачах)
                if (i + 1) % 100 == 0:
                    print(f"\n📈 Прогресс: обработано {i+1} образцов, обнаружено атак: {self.attack_count}")
                
                # Периодический вывод статистики
                if not self.verbose and (i + 1) % 50 == 0:
                    print(f"📊 Прогресс: обработано {i+1} образцов, обнаружено атак: {self.attack_count}")
                
                # Задержка для имитации реального времени
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n\n⏹ Остановка симуляции...")
        
        print(f"\n📊 ИТОГОВАЯ СТАТИСТИКА:")
        print(f"   Обработано образцов: {self.packet_count}")
        print(f"   Обнаружено атак: {self.attack_count}")
        print(f"   Процент атак: {self.attack_count/self.packet_count*100:.2f}%")
        print(f"   Нормальный трафик: {self.packet_count - self.attack_count} ({(self.packet_count - self.attack_count)/self.packet_count*100:.2f}%)")
    
    def get_available_interfaces(self):
        """
        Получение списка доступных сетевых интерфейсов
        
        Returns:
            Список доступных интерфейсов
        """
        if not SCAPY_AVAILABLE or get_if_list is None:
            return []
        
        try:
            interfaces = get_if_list()
            return interfaces if interfaces else []
        except Exception:
            return []
    
    def find_best_interface(self):
        """
        Поиск лучшего интерфейса для захвата пакетов
        
        Returns:
            Имя интерфейса или None
        """
        interfaces = self.get_available_interfaces()
        if not interfaces:
            return None
        
        # Исключаем loopback интерфейсы
        excluded_keywords = ['lo', 'lo0', 'Loopback', 'Loopback Pseudo-Interface']
        
        # Фильтруем интерфейсы
        valid_interfaces = []
        for iface in interfaces:
            # Пропускаем loopback
            if any(keyword.lower() in iface.lower() for keyword in excluded_keywords):
                continue
            
            # Проверяем IP адрес
            try:
                if get_if_addr:
                    addr = get_if_addr(iface)
                    # Пропускаем интерфейсы без IP или с loopback IP
                    if addr == "0.0.0.0" or addr == "127.0.0.1":
                        continue
                    # Предпочитаем интерфейсы с локальными IP (192.168.x.x, 10.x.x.x, 172.16-31.x.x)
                    if addr.startswith("192.168.") or addr.startswith("10.") or (addr.startswith("172.") and 16 <= int(addr.split(".")[1]) <= 31):
                        valid_interfaces.insert(0, iface)  # Добавляем в начало
                        continue
                    # Остальные активные интерфейсы добавляем в конец
                    valid_interfaces.append(iface)
            except Exception:
                # Если не удалось получить IP, все равно добавляем (может быть активным)
                valid_interfaces.append(iface)
        
        if valid_interfaces:
            return valid_interfaces[0]
        elif interfaces:
            # Если ничего не подошло, возвращаем первый не-loopback
            for iface in interfaces:
                if not any(keyword.lower() in iface.lower() for keyword in excluded_keywords):
                    return iface
        
        return None
    
    def start_live_capture(self, interface=None):
        """
        Запуск захвата пакетов в реальном времени
        
        Args:
            interface: сетевой интерфейс (если None, используется значение из конфига или автоопределение)
        """
        if not SCAPY_AVAILABLE:
            print("✗ Scapy недоступен. Используйте режим симуляции.")
            return
        
        # Проверяем, что модель загружена
        if self.model is None:
            print("✗ Ошибка: Модель не загружена!")
            print("   Загружаем модель...")
            try:
                self.load_model()
            except Exception as e:
                print(f"✗ Не удалось загрузить модель: {e}")
                print("   Обучите модель: python src/train_random_forest.py")
                return
        
        # Определяем интерфейс
        if interface is None:
            interface = self.rt_config.get('interface', None)
        
        # Если интерфейс не указан, пытаемся найти автоматически
        if not interface:
            interface = self.find_best_interface()
            if interface:
                print(f"✓ Автоматически выбран интерфейс: {interface}")
            else:
                print("✗ Не удалось автоматически определить сетевой интерфейс")
                self._show_available_interfaces()
                print("\nПопробуйте режим симуляции: --simulate")
                return
        
        # Проверяем доступность интерфейса
        available_interfaces = self.get_available_interfaces()
        if interface not in available_interfaces:
            print(f"✗ Ошибка: Интерфейс '{interface}' не найден!")
            self._show_available_interfaces()
            print(f"\n💡 Попробуйте указать один из доступных интерфейсов:")
            print(f"   python src/realtime_ids.py --live --interface <имя_интерфейса>")
            print(f"\nИли используйте режим симуляции: --simulate")
            return
        
        print("\n🔍 Запуск Real-time IDS (Live Capture)...")
        print(f"   Интерфейс: {interface}")
        print(f"   Порог обнаружения: {self.alert_threshold}")
        print(f"   Логирование: {self.log_file}")
        print()
        print("⚠ Для захвата пакетов могут потребоваться права администратора!")
        print("Нажмите Ctrl+C для остановки...\n")
        
        try:
            sniff(iface=interface, prn=self.packet_handler, store=False)
        except KeyboardInterrupt:
            print("\n\n⏹ Остановка IDS...")
        except PermissionError:
            print("\n✗ Ошибка: Недостаточно прав для захвата пакетов!")
            print("   Запустите программу от имени администратора/root")
            print("   Или используйте режим симуляции: --simulate")
        except Exception as e:
            print(f"\n✗ Ошибка: {e}")
            print("Попробуйте режим симуляции: --simulate")
        
        print(f"\n📊 Статистика:")
        print(f"   Обработано пакетов: {self.packet_count}")
        print(f"   Обнаружено атак: {self.attack_count}")
    
    def _show_available_interfaces(self):
        """Показать список доступных сетевых интерфейсов"""
        interfaces = self.get_available_interfaces()
        if interfaces:
            print("\n📋 Доступные сетевые интерфейсы:")
            print("-" * 70)
            
            recommended = None
            for i, iface in enumerate(interfaces, 1):
                try:
                    addr = get_if_addr(iface) if get_if_addr else "0.0.0.0"
                    
                    # Определяем тип интерфейса
                    iface_type = ""
                    if "Loopback" in iface or addr == "127.0.0.1":
                        iface_type = " [Loopback - не рекомендуется]"
                    elif addr.startswith("192.168.") or addr.startswith("10.") or addr.startswith("172."):
                        iface_type = " [Локальная сеть]"
                        if recommended is None and "Loopback" not in iface:
                            recommended = iface
                    elif addr.startswith("169.254."):
                        iface_type = " [Автоконфигурация - обычно неактивен]"
                    elif addr != "0.0.0.0":
                        iface_type = " [Активный интерфейс]"
                        if recommended is None:
                            recommended = iface
                    
                    marker = " ⭐ РЕКОМЕНДУЕТСЯ" if iface == recommended else ""
                    print(f"   {i}. {iface}")
                    print(f"      IP: {addr}{iface_type}{marker}")
                    print()
                except Exception:
                    print(f"   {i}. {iface}")
                    print()
            
            if recommended:
                print(f"\n💡 Рекомендуемый интерфейс: {recommended}")
                print(f"   Использование: python src/realtime_ids.py --live --interface \"{recommended}\"")
            else:
                print("\n💡 Выберите интерфейс с активным IP адресом (не 0.0.0.0 и не 127.0.0.1)")
        else:
            print("\n⚠ Не удалось определить доступные интерфейсы")
    
    def run(self, mode='simulate'):
        """
        Запуск IDS
        
        Args:
            mode: режим работы ('simulate' или 'live')
        """
        print("=" * 60)
        print("REAL-TIME INTRUSION DETECTION SYSTEM")
        print("=" * 60)
        
        self.load_model()
        
        if mode == 'simulate' or self.simulation_mode:
            self.simulate_realtime()
        else:
            self.start_live_capture()


def main():
    """
    Главная функция для запуска из командной строки
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Real-time IDS')
    parser.add_argument('--mode', type=str, default='simulate',
                       choices=['simulate', 'live'],
                       help='Режим работы: simulate или live')
    parser.add_argument('--live', action='store_true',
                       help='Запуск в режиме live мониторинга (альтернатива --mode live)')
    parser.add_argument('--simulate', action='store_true',
                       help='Запуск в режиме симуляции (альтернатива --mode simulate)')
    parser.add_argument('--interface', type=str, default=None,
                       help='Сетевой интерфейс для live режима')
    parser.add_argument('--list-interfaces', action='store_true',
                       help='Показать список доступных сетевых интерфейсов и выйти')
    parser.add_argument('--threshold', type=float, default=None,
                       help='Порог вероятности для обнаружения атаки')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод информации о проверяемых пакетах')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Путь к файлу конфигурации')
    
    args = parser.parse_args()
    
    ids = RealtimeIDS(config_path=args.config)
    
    # Устанавливаем verbose режим, если указан
    if args.verbose:
        ids.verbose = True
    
    # Если запрошен список интерфейсов, показываем и выходим
    if args.list_interfaces:
        print("=" * 60)
        print("ДОСТУПНЫЕ СЕТЕВЫЕ ИНТЕРФЕЙСЫ")
        print("=" * 60)
        ids._show_available_interfaces()
        print("\n💡 Использование:")
        print("   python src/realtime_ids.py --live --interface <имя_интерфейса>")
        return
    
    # Определяем режим работы (приоритет: --live/--simulate > --mode)
    if args.live:
        mode = 'live'
    elif args.simulate:
        mode = 'simulate'
    else:
        mode = args.mode
    
    if args.threshold:
        ids.alert_threshold = args.threshold
    
    if mode == 'live':
        if args.interface:
            ids.start_live_capture(interface=args.interface)
        else:
            ids.start_live_capture()  # Использует интерфейс по умолчанию
    else:
        ids.run(mode=mode)


if __name__ == '__main__':
    main()

