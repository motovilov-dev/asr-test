#!/usr/bin/env python3
"""
Пример использования потокового распознавания речи с микрофона
"""

import time
from advanced_streaming_asr import AdvancedStreamingASR, print_result, save_results_to_file


def custom_callback(text: str):
    """Пользовательская функция обратного вызова"""
    print(f"🔊 Распознанная речь: {text}")
    # Сохраняем в файл
    save_results_to_file(text, "my_transcriptions.txt")


def main():
    """Основная функция с примерами использования"""
    
    print("🎤 Потоковое распознавание речи с GigaAM")
    print("=" * 50)
    
    # Пример 1: Базовое использование
    print("\n1️⃣ Базовое использование:")
    streaming_asr = AdvancedStreamingASR(
        model_name="ctc",
        sample_rate=16000,
        silence_threshold=0.01,
        silence_duration=1.0,
        callback=print_result,
        verbose=True
    )
    
    try:
        streaming_asr.start()
    except KeyboardInterrupt:
        print("\n⏹️ Остановка...")
    finally:
        streaming_asr.stop()
    
    # Пример 2: С сохранением в файл
    print("\n2️⃣ С сохранением результатов в файл:")
    streaming_asr2 = AdvancedStreamingASR(
        model_name="ctc",
        sample_rate=16000,
        silence_threshold=0.015,  # Более чувствительный к тишине
        silence_duration=0.8,     # Быстрее реагирует на тишину
        min_audio_duration=1.0,   # Минимум 1 секунда
        max_audio_duration=15.0,  # Максимум 15 секунд
        callback=custom_callback,
        verbose=True
    )
    
    try:
        streaming_asr2.start()
    except KeyboardInterrupt:
        print("\n⏹️ Остановка...")
    finally:
        streaming_asr2.stop()
    
    # Пример 3: Настройки для быстрого распознавания
    print("\n3️⃣ Быстрое распознавание:")
    streaming_asr3 = AdvancedStreamingASR(
        model_name="ctc",
        sample_rate=16000,
        silence_threshold=0.02,    # Менее чувствительный
        silence_duration=0.5,      # Очень быстрое завершение
        min_audio_duration=0.3,    # Короткие фразы
        max_audio_duration=10.0,   # Не очень длинные
        callback=lambda text: print(f"⚡ {text}"),
        verbose=False  # Меньше вывода
    )
    
    try:
        streaming_asr3.start()
    except KeyboardInterrupt:
        print("\n⏹️ Остановка...")
    finally:
        streaming_asr3.stop()


if __name__ == "__main__":
    main() 