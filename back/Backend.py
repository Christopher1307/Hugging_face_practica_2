import os
import smtplib
from email.message import EmailMessage

import torch
import gradio as gr
from transformers import pipeline

# ------------------------
#  Configuración / Cache
# ------------------------
_asr_cache = None
_traductores_cache = {}
_resumidores_cache = {}

def _dispositivo():
    # 0 -> GPU CUDA si existe, -1 -> CPU
    return 0 if torch.cuda.is_available() else -1

# ------------------------
#  Modelos HF (pipelines)
# ------------------------
def cargar_asr():
    # Voz -> Texto (ASR)
    global _asr_cache
    if _asr_cache is None:
        _asr_cache = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",   # Whisper Small (multilingüe)
            device=_dispositivo()
        )
    return _asr_cache

def cargar_traductor(idioma_origen, idioma_destino):
    # Traducción (MarianMT según par)
    clave = f"{idioma_origen}-{idioma_destino}"
    if clave not in _traductores_cache:
        mapa_modelos = {
            "es-en": "Helsinki-NLP/opus-mt-es-en",
            "en-es": "Helsinki-NLP/opus-mt-en-es",
            "fr-es": "Helsinki-NLP/opus-mt-fr-es",
            "es-fr": "Helsinki-NLP/opus-mt-es-fr",
        }
        modelo = mapa_modelos.get(clave, "Helsinki-NLP/opus-mt-en-es")
        _traductores_cache[clave] = pipeline(
            "translation",
            model=modelo,                  # MarianMT (Helsinki-NLP)
            device=_dispositivo()
        )
    return _traductores_cache[clave]

def cargar_resumidor(opcion_modelo):
    # Resumen (BART o DistilBART)
    if opcion_modelo not in _resumidores_cache:
        modelo = "facebook/bart-large-cnn" if opcion_modelo == "BART (mejor calidad)" else "sshleifer/distilbart-cnn-12-6"
        _resumidores_cache[opcion_modelo] = pipeline(
            "summarization",
            model=modelo,
            device=_dispositivo()
        )
    return _resumidores_cache[opcion_modelo]

# ------------------------
#  Funciones de negocio
# ------------------------
def fn_transcribir(archivo_audio, idioma_pista):
    if archivo_audio is None:
        return ""
    asr = cargar_asr()
    kwargs = {}
    if idioma_pista != "auto":
        kwargs["generate_kwargs"] = {"language": idioma_pista}
    resultado = asr(archivo_audio, return_timestamps=False, **kwargs)
    if isinstance(resultado, dict) and "text" in resultado:
        return resultado["text"]
    if isinstance(resultado, list) and resultado and "text" in resultado[0]:
        return resultado[0]["text"]
    return ""

def fn_traducir(texto, idioma_origen, idioma_destino):
    if not texto.strip():
        return ""
    traductor = cargar_traductor(idioma_origen, idioma_destino)
    salida = traductor(texto, max_length=2048)
    return salida[0]["translation_text"]

def fn_resumir(texto, opcion_modelo, max_tokens, min_tokens):
    if not texto.strip():
        return ""
    # Evita error si el usuario pone min >= max
    if max_tokens <= min_tokens:
        max_tokens = min_tokens + 16
    resumidor = cargar_resumidor(opcion_modelo)
    salida = resumidor(
        texto,
        max_length=int(max_tokens),
        min_length=int(min_tokens),
        do_sample=False
    )
    return salida[0]["summary_text"]

def fn_todo_en_uno(archivo_audio, idioma_reunion, idioma_destino, opcion_modelo, max_tokens, min_tokens):
    transcripcion = fn_transcribir(archivo_audio, idioma_reunion)
    if not transcripcion:
        return "", "", ""
    idioma_origen_para_trad = idioma_reunion if idioma_reunion != "auto" else "es"
    traduccion = fn_traducir(transcripcion, idioma_origen_para_trad, idioma_destino)
    resumen = fn_resumir(traduccion, opcion_modelo, max_tokens, min_tokens)
    return transcripcion, traduccion, resumen

# ------------------------
#  Envío por email (SMTP)
# ------------------------
def enviar_email_resumen(destinatarios_csv, asunto, cuerpo, smtp_host, smtp_port, usuario, password, use_tls=True):
    destinatarios = [d.strip() for d in destinatarios_csv.split(",") if d.strip()]
    if not destinatarios or not cuerpo.strip():
        return "Faltan destinatarios o cuerpo."
    msg = EmailMessage()
    msg["From"] = usuario
    msg["To"] = ", ".join(destinatarios)
    msg["Subject"] = asunto
    msg.set_content(cuerpo)
    try:
        if use_tls and int(smtp_port) == 587:
            with smtplib.SMTP(smtp_host, int(smtp_port), timeout=30) as s:
                s.starttls()
                s.login(usuario, password)
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(smtp_host, int(smtp_port), timeout=30) as s:
                s.login(usuario, password)
                s.send_message(msg)
        return " Enviado"
    except Exception as e:
        return f" Error al enviar: {e}"

def construir_cuerpo_resumen(transcripcion, traduccion, resumen):
    partes = []
    if (transcripcion or "").strip():
        partes.append("— Transcripción —\n" + transcripcion.strip())
    if (traduccion or "").strip():
        partes.append("\n— Traducción —\n" + traduccion.strip())
    if (resumen or "").strip():
        partes.append("\n— Resumen —\n" + resumen.strip())
    return "\n\n".join(partes).strip()

def fn_enviar_email(destinatarios, asunto, host, puerto, user, pwd, use_tls, transcripcion, traduccion, resumen):
    cuerpo = construir_cuerpo_resumen(transcripcion or "", traduccion or "", resumen or "")
    return enviar_email_resumen(destinatarios, asunto, cuerpo, host, int(puerto), user, pwd, use_tls)

# ------------------------
#  UI (Gradio 5.x)
# ------------------------
tema = gr.themes.Soft(primary_hue="violet", neutral_hue="slate")

css_personalizado = """
:root { --brand: linear-gradient(135deg,#6d28d9, #22d3ee); }
html, body, .gradio-container { width: 100%; }
.gradio-container {
  max-width: 100% !important;
  margin: 0 auto !important;
  padding: 0 16px;
  font-family: 'Segoe UI', sans-serif;
}
.header-card {
  background: radial-gradient(1200px 400px at 10% 0%, rgba(109,40,217,.12), transparent 60%),
              radial-gradient(800px 400px at 90% 0%, rgba(34,211,238,.12), transparent 60%);
  border: 1px solid rgba(255,255,255,.12);
  backdrop-filter: blur(8px);
  border-radius: 16px;
  overflow: hidden;
}
.brand-title { font-size: 28px; font-weight: 800; letter-spacing: .3px; color:#fff; }
.brand-badge { display:inline-block; padding:6px 10px; border-radius:999px; background:#0ea5e9; color:#fff; font-size:12px; margin-left:8px;}
.stat { display:flex; gap:10px; align-items:center; padding:8px 12px; border-radius:12px;
  border:1px solid rgba(255,255,255,.12); backdrop-filter:blur(6px); color:#fff; }
.control-card { border:1px solid rgba(148,163,184,.25); border-radius: 12px; padding:8px; }
footer { display:none !important; }
"""

with gr.Blocks(theme=tema, css=css_personalizado, fill_height=True, title="Asistente de Reuniones IA") as demo:
    with gr.Column(elem_classes=["header-card"], scale=0):
        with gr.Row():
            gr.HTML("""
            <div style="display:flex;align-items:center;justify-content:space-between;width:100%;padding:18px 16px;">
              <div>
                <div class="brand-title">Asistente de Reuniones IA <span class="brand-badge">Demo</span></div>
                <div style="opacity:.8;margin-top:6px;color:#fff">Transcribe • Traduce • Resume — en una interfaz rápida y clara.</div>
              </div>
              <div style="display:flex;gap:10px">
                 <div class="stat"> <b>Voz→Texto</b></div>
                 <div class="stat"> <b>Traducción</b></div>
                 <div class="stat"> <b>Resumen</b></div>
              </div>
            </div>
            """)

    with gr.Tabs():
        # --- Voz -> Texto
        with gr.Tab("Voz → Texto"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["control-card"]):
                    entrada_audio = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio de la reunión")
                    idioma_audio = gr.Dropdown(choices=["auto","es","en","fr","de","it","pt"], value="auto", label="Idioma del audio")
                    boton_transcribir = gr.Button("Transcribir", variant="primary")
                with gr.Column(scale=1):
                    salida_transcripcion = gr.Textbox(label="Transcripción", lines=12, show_copy_button=True)
            boton_transcribir.click(fn_transcribir, [entrada_audio, idioma_audio], salida_transcripcion)

        # --- Traducción
        with gr.Tab("Traducción"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["control-card"]):
                    idioma_origen = gr.Dropdown(choices=["es","en","fr"], value="es", label="Desde")
                    idioma_destino = gr.Dropdown(choices=["en","es","fr"], value="en", label="Hacia")
                    texto_entrada = gr.Textbox(label="Texto de entrada", lines=10, placeholder="Pega aquí el texto a traducir")
                    boton_traducir = gr.Button("Traducir", variant="primary")
                with gr.Column(scale=1):
                    texto_traducido = gr.Textbox(label="Traducción", lines=10, show_copy_button=True)
            boton_traducir.click(fn_traducir, [texto_entrada, idioma_origen, idioma_destino], texto_traducido)

        # --- Resumen
        with gr.Tab("Resumen"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["control-card"]):
                    opcion_modelo = gr.Radio(choices=["BART (mejor calidad)","DistilBART (rápido)"], value="BART (mejor calidad)", label="Modelo")
                    longitud_max = gr.Slider(64, 512, value=220, step=4, label="Longitud máxima")
                    longitud_min = gr.Slider(16, 256, value=80, step=4, label="Longitud mínima")
                    texto_a_resumir = gr.Textbox(label="Texto a resumir", lines=12)
                    boton_resumir = gr.Button("Resumir", variant="primary")
                with gr.Column(scale=1):
                    texto_resumen = gr.Textbox(label="Resumen", lines=12, show_copy_button=True)
            boton_resumir.click(fn_resumir, [texto_a_resumir, opcion_modelo, longitud_max, longitud_min], texto_resumen)

        # --- Todo en uno + Email
        with gr.Tab(" Todo-en-uno"):
            with gr.Row():
                with gr.Column(scale=1, elem_classes=["control-card"]):
                    audio_total = gr.Audio(sources=["upload","microphone"], type="filepath", label="Audio")
                    idioma_reunion = gr.Dropdown(choices=["auto","es","en","fr"], value="auto", label="Idioma del audio")
                    idioma_destino_total = gr.Dropdown(choices=["es","en","fr"], value="en", label="Traducir a")
                    modelo_resumen_total = gr.Radio(choices=["BART (mejor calidad)","DistilBART (rápido)"], value="BART (mejor calidad)", label="Modelo de resumen")
                    longitud_max_total = gr.Slider(64, 512, value=220, step=4, label="Longitud máx. resumen")
                    longitud_min_total = gr.Slider(16, 256, value=80, step=4, label="Longitud mín. resumen")
                    boton_todo = gr.Button("Procesar todo", variant="primary")
                with gr.Column(scale=1):
                    salida_total_transcripcion = gr.Textbox(label="Transcripción", lines=6, show_copy_button=True)
                    salida_total_traduccion = gr.Textbox(label="Traducción", lines=6, show_copy_button=True)
                    salida_total_resumen = gr.Textbox(label="Resumen final", lines=8, show_copy_button=True)
            boton_todo.click(
                fn_todo_en_uno,
                [audio_total, idioma_reunion, idioma_destino_total, modelo_resumen_total, longitud_max_total, longitud_min_total],
                [salida_total_transcripcion, salida_total_traduccion, salida_total_resumen]
            )

            # Controles de email
            with gr.Row():
                destinatarios_input = gr.Textbox(label="Destinatarios (coma separada)", placeholder="persona1@empresa.com, persona2@empresa.com")
                asunto_input = gr.Textbox(label="Asunto", value="Resumen de la reunión")
            with gr.Row():
                smtp_host = gr.Textbox(label="SMTP host", value=os.getenv("SMTP_HOST","smtp.gmail.com"))
                smtp_port = gr.Number(label="SMTP puerto", value=int(os.getenv("SMTP_PORT","465")))
            with gr.Row():
                smtp_user = gr.Textbox(label="SMTP usuario", value=os.getenv("SMTP_USER",""), type="text")
                smtp_pass = gr.Textbox(label="SMTP contraseña/app password", value=os.getenv("SMTP_PASS",""), type="password")
                usar_tls = gr.Checkbox(label="Usar STARTTLS (587)", value=False)
            boton_enviar = gr.Button("Enviar resumen por email", variant="secondary")
            estado_envio = gr.Markdown()

            boton_enviar.click(
                fn_enviar_email,
                [destinatarios_input, asunto_input, smtp_host, smtp_port, smtp_user, smtp_pass, usar_tls,
                 salida_total_transcripcion, salida_total_traduccion, salida_total_resumen],
                [estado_envio]
            )

    with gr.Accordion("Ejemplos y consejos", open=False):
        gr.HTML("""
        <ul>
          <li>Si usas CPU, los modelos pueden tardar más; con GPU CUDA irá más rápido.</li>
          <li>Para MarianMT necesitas <code>sentencepiece</code>. Para Whisper, <code>ffmpeg</code>.</li>
          <li>El flujo Todo-en-uno hace: transcribir → traducir → resumir → (opcional) enviar email.</li>
        </ul>
        """)

demo.queue(max_size=32).launch()
