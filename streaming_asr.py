import threading
import time
import queue
import numpy as np
import sounddevice as sd
import torch
import gigaam
from typing import Optional, Callable


class StreamingASR:
    """
    Потоковое распознавание речи с микрофона используя GigaAM модель
    """
    
    def __init__(
        self,
        model_name: str = "ctc",
        sample_rate: int = 16000,
        chunk_duration: float = 2.0,
        silence_threshold: float = 0.01,
        silence_duration: float = 1.0,
        callback: Optional[Callable[[str], None]] = None
    ):
        """
        Инициализация потокового ASR
        
        Args:
            model_name: Название модели GigaAM
            sample_rate: Частота дискретизации
            chunk_duration: Длительность чанка в секундах
            silence_threshold: Порог тишины для определения конца речи
            silence_duration: Длительность тишины для завершения сегмента
            callback: Функция обратного вызова для результатов распознавания
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.callback = callback
        
        # Загрузка модели
        print("Загрузка модели GigaAM...")
        self.model = gigaam.load_model(
            model_name,
            fp16_encoder=False,
            use_flash=False,
        )
        print("Модель загружена!")
        
        # Буферы для аудио
        self.audio_buffer = []
        self.is_recording = False
        self.audio_queue = queue.Queue()
        self.result_queue = queue.Queue()
        
        # Потоки
        self.recording_thread = None
        self.processing_thread = None
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback для получения аудио данных"""
        if status:
            print(f"Статус аудио: {status}")
        
        # Преобразуем в float32 и нормализуем
        audio_data = indata.copy().astype(np.float32)
        audio_data = audio_data.flatten()
        
        # Добавляем в буфер
        self.audio_buffer.extend(audio_data)
        
        # Проверяем на тишину
        if len(self.audio_buffer) > self.sample_rate * self.silence_duration:
            recent_audio = self.audio_buffer[-int(self.sample_rate * self.silence_duration):]
            if np.max(np.abs(recent_audio)) < self.silence_threshold:
                # Тишина обнаружена, отправляем буфер на обработку
                if len(self.audio_buffer) > self.sample_rate * 0.5:  # Минимум 0.5 секунды
                    audio_chunk = np.array(self.audio_buffer, dtype=np.float32)
                    self.audio_queue.put(audio_chunk)
                    self.audio_buffer = []
    
    def _process_audio(self):
        """Поток для обработки аудио через модель"""
        while self.is_recording:
            try:
                # Получаем аудио из очереди с таймаутом
                audio_data = self.audio_queue.get(timeout=1.0)
                
                if audio_data is None:
                    break
                
                # Сохраняем временный файл
                temp_file = f"temp_audio_{time.time()}.wav"
                self._save_audio(audio_data, temp_file)
                
                try:
                    # Транскрибируем
                    result = self.model.transcribe(temp_file)
                    
                    if result and result.strip():
                        print(f"Распознано: {result}")
                        if self.callback:
                            self.callback(result)
                        self.result_queue.put(result)
                    
                except Exception as e:
                    print(f"Ошибка при транскрибации: {e}")
                finally:
                    # Удаляем временный файл
                    import os
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Ошибка в потоке обработки: {e}")
    
    def _save_audio(self, audio_data: np.ndarray, filename: str):
        """Сохраняет аудио данные в файл"""
        import wave
        
        with wave.open(filename, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Моно
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes((audio_data * 32767).astype(np.int16).tobytes())
    
    def start(self):
        """Запускает потоковое распознавание"""
        if self.is_recording:
            print("Распознавание уже запущено!")
            return
        
        self.is_recording = True
        
        # Запускаем поток обработки
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Запускаем запись с микрофона
        print("Начинаю запись с микрофона...")
        print("Говорите в микрофон. Распознавание будет происходить автоматически.")
        
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
            print("\nОстановка записи...")
        except Exception as e:
            print(f"Ошибка при записи: {e}")
        finally:
            self.stop()
    
    def stop(self):
        """Останавливает потоковое распознавание"""
        self.is_recording = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        print("Потоковое распознавание остановлено.")
    
    def get_results(self):
        """Возвращает все результаты распознавания"""
        results = []
        while not self.result_queue.empty():
            try:
                results.append(self.result_queue.get_nowait())
            except queue.Empty:
                break
        return results


def print_result(text: str):
    """Простая функция для вывода результатов"""
    print(f"🎤 {text}")


if __name__ == "__main__":
    # Создаем экземпляр потокового ASR
    streaming_asr = StreamingASR(
        model_name="ctc",
        sample_rate=16000,
        chunk_duration=2.0,
        silence_threshold=0.01,
        silence_duration=1.0,
        callback=print_result
    )
    
    try:
        # Запускаем распознавание
        streaming_asr.start()
    except KeyboardInterrupt:
        print("\nЗавершение работы...")
    finally:
        streaming_asr.stop() 