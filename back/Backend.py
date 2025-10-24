import gradio as gr
import torch
from transformers import pipeline

#Audio-Texto
asr = pipeline(task="automatic-speech-recognition", model="openai/whisper-small",device="cuda" if torch.cuda.is_available() else "cpu")
audio_a_texto = None

# Resumen
summarizer = pipeline("summarization")


# Traductor
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")



def transcribe(audio):

    # Audio-Texto
    global audio_a_texto
    if audio is None:
        return "Por favor, sube un archivo de audio."

    audio_a_texto = asr(audio)["text"]
    print(audio_a_texto)

    # Resumen
    resumen = summarizer(audio_a_texto,min_length = 10,max_length=50)[0]["summary_text"]

    # Traductor
    traduccion = translator(resumen)[0]["translation_text"]




    return traduccion

gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),
    outputs=gr.TextArea(label="Transcripción")
).launch()
