#!/usr/bin/env python3
"""
Тестовый скрипт для потокового ASR
"""

import time
from advanced_streaming_asr import AdvancedStreamingASR


def test_callback(text: str):
    """Тестовая функция обратного вызова"""
    print(f"✅ Тест: {text}")


def main():
    """Основная функция тестирования"""
    print("🧪 Тестирование потокового ASR")
    print("=" * 40)
    
    # Создаем экземпляр с тестовыми настройками
    asr = AdvancedStreamingASR(
        model_name="ctc",
        sample_rate=16000,
        silence_threshold=0.02,      # Менее чувствительный для тестов
        silence_duration=0.8,        # Быстрее завершение
        min_audio_duration=0.5,      # Минимум 0.5 секунды
        max_audio_duration=10.0,     # Максимум 10 секунд
        callback=test_callback,
        verbose=True
    )
    
    print("🎤 Готов к тестированию!")
    print("💬 Говорите в микрофон для тестирования")
    print("⏹️  Нажмите Ctrl+C для остановки")
    print("-" * 40)
    
    try:
        # Запускаем на 30 секунд для тестирования
        asr.start()
        
    except KeyboardInterrupt:
        print("\n⏹️ Тестирование остановлено пользователем")
    except Exception as e:
        print(f"❌ Ошибка при тестировании: {e}")
    finally:
        asr.stop()
        
        # Выводим результаты
        results = asr.get_results()
        if results:
            print(f"\n📝 Результаты тестирования ({len(results)} записей):")
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result}")
        else:
            print("\n📝 Результатов не найдено")


if __name__ == "__main__":
    main() 