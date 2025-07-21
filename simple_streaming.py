#!/usr/bin/env python3
"""
Простой интерфейс для потокового распознавания речи
"""

import time
from advanced_streaming_asr import AdvancedStreamingASR


def main():
    """Простой интерфейс для потокового ASR"""
    
    print("🎤 Потоковое распознавание речи")
    print("=" * 40)
    print("Настройки по умолчанию:")
    print("  • Модель: GigaAM CTC")
    print("  • Частота: 16kHz")
    print("  • Чувствительность: средняя")
    print("  • Длительность сегментов: 0.5-30 сек")
    print()
    
    # Создаем ASR с настройками по умолчанию
    asr = AdvancedStreamingASR(
        model_name="ctc",
        sample_rate=16000,
        silence_threshold=0.01,
        silence_duration=1.0,
        min_audio_duration=0.5,
        max_audio_duration=30.0,
        callback=lambda text: print(f"🎯 {text}"),
        verbose=True
    )
    
    print("🎤 Начинаю запись...")
    print("💬 Говорите в микрофон")
    print("⏹️  Нажмите Ctrl+C для остановки")
    print("-" * 40)
    
    try:
        asr.start()
    except KeyboardInterrupt:
        print("\n⏹️ Остановка...")
    finally:
        asr.stop()


if __name__ == "__main__":
    main() 