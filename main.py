import gigaam


model = gigaam.load_model(
    "ctc",  # GigaAM-V2 CTC model
    fp16_encoder=False,  # to use fp16 encoder weights - GPU only
    use_flash=False,  # disable flash attention - colab does not support it
)

result = model.transcribe("file.ogg")
print(result)