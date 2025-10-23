import gradio as gr
import torch
from transformers import pipeline

#Audio-Texto
asr = pipeline(task="automatic-speech-recognition", model="openai/whisper-small",device="cuda" if torch.cuda.is_available() else "cpu")
audio_a_texto = None

# Traductor
pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")


def transcribe(audio):

    # Audio-Texto
    global audio_a_texto
    if audio is None:
        return "Por favor, sube un archivo de audio."

    audio_a_texto = asr(audio)["text"]

    # Traductor

    resultado = pipe(audio_a_texto)


    return resultado

gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.TextArea(label="Transcripción")
).launch()
