import gradio as gr
import torch
from transformers import pipeline

asr = pipeline(task="automatic-speech-recognition", model="openai/whisper-small",device="cuda" if torch.cuda.is_available() else "cpu")
audio_a_texto = None

def transcribe(audio):

    global audio_a_texto
    if audio is None:
        return "Por favor, sube un archivo de audio."

    audio_a_texto = asr(audio)["text"]
    print(audio_a_texto)
    return audio_a_texto

gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.TextArea(label="Transcripción")
).launch()
