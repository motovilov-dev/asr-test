import threading
import time
import queue
import numpy as np
import sounddevice as sd
import torch
import gigaam
import tempfile
import os
from typing import Optional, Callable, List, Dict
from collections import deque


class AdvancedStreamingASR:
    """
    Продвинутое потоковое распознавание речи с микрофона используя GigaAM модель
    """
    
    def __init__(
        self,
        model_name: str = "ctc",
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.0,
        min_audio_duration: float = 0.5,
        max_audio_duration: float = 30.0,
        callback: Optional[Callable[[str], None]] = None,
        device: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Инициализация продвинутого потокового ASR
        
        Args:
            model_name: Название модели GigaAM
            sample_rate: Частота дискретизации
            chunk_duration: Длительность чанка в секундах
            silence_threshold: Порог тишины для определения конца речи
            silence_duration: Длительность тишины для завершения сегмента
            min_audio_duration: Минимальная длительность аудио для обработки
            max_audio_duration: Максимальная длительность аудио для обработки
            callback: Функция обратного вызова для результатов распознавания
            device: Устройство для модели (cuda/cpu)
            verbose: Подробный вывод
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_audio_duration = min_audio_duration
        self.max_audio_duration = max_audio_duration
        self.callback = callback
        self.verbose = verbose
        
        # Загрузка модели
        if self.verbose:
            print("Загрузка модели GigaAM...")
        
        self.model = gigaam.load_model(
            model_name,
            fp16_encoder=False,
            use_flash=False,
        )
        
        # Перемещаем модель на нужное устройство
        if device:
            self.model = self.model.to(device)
        
        if self.verbose:
            print(f"Модель загружена на устройство: {next(self.model.parameters()).device}")
        
        # Буферы для аудио
        self.audio_buffer = deque(maxlen=int(self.sample_rate * self.max_audio_duration))
        self.is_recording = False
        self.audio_queue = queue.Queue(maxsize=10)  # Ограничиваем очередь
        self.result_queue = queue.Queue()
        
        # Статистика
        self.stats = {
            'processed_chunks': 0,
            'total_audio_time': 0.0,
            'successful_transcriptions': 0,
            'failed_transcriptions': 0
        }
        
        # Потоки
        self.processing_thread = None
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback для получения аудио данных"""
        if status:
            if self.verbose:
                print(f"Статус аудио: {status}")
        
        # Преобразуем в float32 и нормализуем
        audio_data = indata.copy().astype(np.float32)
        audio_data = audio_data.flatten()
        
        # Добавляем в буфер
        self.audio_buffer.extend(audio_data)
        
        # Проверяем на тишину и длительность
        if len(self.audio_buffer) > self.sample_rate * self.silence_duration:
            recent_audio = list(self.audio_buffer)[-int(self.sample_rate * self.silence_duration):]
            if np.max(np.abs(recent_audio)) < self.silence_threshold:
                # Тишина обнаружена
                audio_length = len(self.audio_buffer) / self.sample_rate
                
                if self.min_audio_duration <= audio_length <= self.max_audio_duration:
                    # Копируем буфер и очищаем его
                    audio_chunk = np.array(list(self.audio_buffer), dtype=np.float32)
                    self.audio_buffer.clear()
                    
                    # Отправляем на обработку
                    try:
                        self.audio_queue.put_nowait(audio_chunk)
                    except queue.Full:
                        if self.verbose:
                            print("Очередь аудио переполнена, пропускаем чанк")
                elif audio_length > self.max_audio_duration:
                    # Слишком длинный сегмент, обрезаем
                    max_samples = int(self.sample_rate * self.max_audio_duration)
                    audio_chunk = np.array(list(self.audio_buffer)[-max_samples:], dtype=np.float32)
                    self.audio_buffer.clear()
                    
                    try:
                        self.audio_queue.put_nowait(audio_chunk)
                    except queue.Full:
                        if self.verbose:
                            print("Очередь аудио переполнена, пропускаем чанк")
    
    def _process_audio(self):
        """Поток для обработки аудио через модель"""
        while self.is_recording:
            try:
                # Получаем аудио из очереди с таймаутом
                audio_data = self.audio_queue.get(timeout=1.0)
                
                if audio_data is None:
                    break
                
                # Обновляем статистику
                audio_duration = len(audio_data) / self.sample_rate
                self.stats['processed_chunks'] += 1
                self.stats['total_audio_time'] += audio_duration
                
                if self.verbose:
                    print(f"Обрабатываю аудио чанк: {audio_duration:.2f}с")
                
                # Создаем временный файл
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                try:
                    # Сохраняем аудио
                    self._save_audio(audio_data, temp_filename)
                    
                    # Транскрибируем
                    result = self.model.transcribe(temp_filename)
                    
                    if result and result.strip():
                        if self.verbose:
                            print(f"🎤 Распознано: {result}")
                        
                        if self.callback:
                            self.callback(result)
                        
                        self.result_queue.put(result)
                        self.stats['successful_transcriptions'] += 1
                    else:
                        if self.verbose:
                            print("Пустой результат распознавания")
                        self.stats['failed_transcriptions'] += 1
                    
                except Exception as e:
                    if self.verbose:
                        print(f"Ошибка при транскрибации: {e}")
                    self.stats['failed_transcriptions'] += 1
                finally:
                    # Удаляем временный файл
                    try:
                        os.unlink(temp_filename)
                    except OSError:
                        pass
                        
            except queue.Empty:
                continue
            except Exception as e:
                if self.verbose:
                    print(f"Ошибка в потоке обработки: {e}")
    
    def _save_audio(self, audio_data: np.ndarray, filename: str):
        """Сохраняет аудио данные в файл"""
        import wave
        
        # Нормализуем аудио
        if np.max(np.abs(audio_data)) > 1.0:
            audio_data = audio_data / np.max(np.abs(audio_data))
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Моно
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    def start(self):
        """Запускает потоковое распознавание"""
        if self.is_recording:
            if self.verbose:
                print("Распознавание уже запущено!")
            return
        
        self.is_recording = True
        
        # Запускаем поток обработки
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Запускаем запись с микрофона
        if self.verbose:
            print("🎤 Начинаю запись с микрофона...")
            print("💬 Говорите в микрофон. Распознавание будет происходить автоматически.")
            print("⏹️  Нажмите Ctrl+C для остановки")
        
        try:
            with sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                dtype=np.float32,
                blocksize=int(self.sample_rate * 0.1)  # 100ms блоки
            ):
                while self.is_recording:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            if self.verbose:
                print("\n⏹️  Остановка записи...")
        except Exception as e:
            if self.verbose:
                print(f"❌ Ошибка при записи: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Останавливает потоковое распознавание"""
        self.is_recording = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.verbose:
            print("🛑 Потоковое распознавание остановлено.")
            self.print_stats()
    
    def get_results(self) -> List[str]:
        """Возвращает все результаты распознавания"""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results
    
    def print_stats(self):
        """Выводит статистику работы"""
        print("\n📊 Статистика работы:")
        print(f"   Обработано чанков: {self.stats['processed_chunks']}")
        print(f"   Общее время аудио: {self.stats['total_audio_time']:.2f}с")
        print(f"   Успешных транскрибаций: {self.stats['successful_transcriptions']}")
        print(f"   Неудачных транскрибаций: {self.stats['failed_transcriptions']}")
        
        if self.stats['processed_chunks'] > 0:
            success_rate = (self.stats['successful_transcriptions'] / 
                          (self.stats['successful_transcriptions'] + self.stats['failed_transcriptions'])) * 100
            print(f"   Процент успеха: {success_rate:.1f}%")


def print_result(text: str):
    """Функция для вывода результатов с эмодзи"""
    print(f"🎯 {text}")


def save_results_to_file(text: str, filename: str = "transcriptions.txt"):
    """Функция для сохранения результатов в файл"""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(filename, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")


if __name__ == "__main__":
    # Создаем экземпляр продвинутого потокового ASR
    streaming_asr = AdvancedStreamingASR(
        model_name="ctc",
        sample_rate=16000,
        chunk_duration=2.0,
        silence_threshold=0.01,
        silence_duration=1.0,
        min_audio_duration=0.5,
        max_audio_duration=30.0,
        callback=print_result,
        verbose=True
    )
    
    try:
        # Запускаем распознавание
        streaming_asr.start()
    except KeyboardInterrupt:
        print("\n👋 Завершение работы...")
    finally:
        streaming_asr.stop() 