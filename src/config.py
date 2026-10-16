'''
CLASE 2:
Este script sigue lo estipulado en la clase 2, en el jupiter notebook: clase_02.ipynb

'''





#%%

import os
from dotenv import load_dotenv

# Carga las variables definidas en .env
load_dotenv()

provider = os.environ.get("LLM_PROVIDER", "google")

print(f"Proveedor configurado: {provider}")


# Chequeo de credenciales según el proveedor elegido (ver .env.example)
credenciales_esperadas = {
    "bedrock": "AWS_BEARER_TOKEN_BEDROCK",
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "ollama": None,  # Ollama corre local, no necesita API key
}
var_esperada = credenciales_esperadas.get(provider)
if var_esperada:
    assert os.environ.get(var_esperada), (
        f"Falta {var_esperada} en el .env para el proveedor '{provider}' — "
        f"copiá .env.example a .env y completá esa sección."
    )
print("Credenciales OK." if var_esperada else "Ollama: no requiere API key, solo que esté corriendo localmente.")


# --- Google Gemini ---  (uso las librerías instaladas con "pip install google-genai")
from google import genai
from google.genai import types

def llamar_google(prompt, system=None, temperature=0.7, max_tokens=500):
    client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])  #crea el objeto con el que se comunica con la API
    config = types.GenerateContentConfig(  #así debe responder el Gemini
        system_instruction=system,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )
    resp = client.models.generate_content(  #acá se hace la llamada a la api
        model=os.environ.get("GOOGLE_MODEL_ID", "gemini-3.1-flash-lite"), #qué modelo usar o el modelo default
        contents=prompt, #mensaje a enviar a Gemini
        config=config, #para los datos de la configuración que creamo en "config"
    )
    return resp.text  #devuelve la respuesta en texto

load_dotenv()
print("Modelo cargado:", os.environ.get("GOOGLE_MODEL_ID"))


#********** PUNTO 3: PRIMER CLIENTE *****************
#%%
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

response = client.models.generate_content(
    model=os.environ.get("GOOGLE_MODEL_ID", "gemini-3.1-flash-lite"),
    contents="¿Qué es Retrieval-Augmented Generation, en una oración?"
)

texto = response.text

print(texto)
print()
print("Uso de tokens:", response.usage_metadata)
print("Motivo de fin:", response.candidates[0].finish_reason)
print("Tokens de entrada:", response.usage_metadata.prompt_token_count)
print("Tokens de salida:", response.usage_metadata.candidates_token_count)

#********** PUNTO 4: MENSAJE DEL SISTEMA Y USUARIO *****************
# %%
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
model_id = os.environ.get("GOOGLE_MODEL_ID", "gemini-3.1-flash-lite")

system_prompt = [
    {"text": "Sos un asistente técnico para una certificación de desarrollo de agentes de IA. "
             "Respondé en español, en un máximo de 3 oraciones, en tono directo y sin relleno."}
]

messages = [
    {"role": "user", "content": [{"text": "¿Para qué sirve la ventana de contexto de un LLM?"}]}
]

response = client.models.generate_content(
    model=model_id,
    contents=messages[0]["content"][0]["text"],
    config=types.GenerateContentConfig(
        system_instruction=system_prompt[0]["text"]
    ),
)

print(response.text)


#*****MULTITURNO*************************
#no tiene memoria propia

# %%
model_id = os.environ.get("GOOGLE_MODEL_ID", "gemini-3.1-flash-lite")
# Turno 1
messages = [
    {
        "role": "user",
        "parts": [
            {"text": "Estoy comparando Claude y un modelo abierto para mi TP. ¿Qué debería tener en cuenta?"}
        ]
    }
]

resp1 = client.models.generate_content(
    model=model_id,
    contents=messages,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt[0]["text"]
    ),
)

print("Turno 1:", resp1.text)

# Agregamos la respuesta del modelo al historial
# y el nuevo mensaje del usuario

messages.append({
    "role": "model",
    "parts": [{"text": resp1.text}]
})

messages.append({
    "role": "user",
    "parts": [
        {"text": "Y de esos criterios, ¿cuál pesa más si el corpus tiene datos sensibles?"}
    ]
})

# Turno 2: el modelo ve toda la conversación anterior
resp2 = client.models.generate_content(
    model=model_id,
    contents=messages,  #el modelo no “recuerda” mágicamente el turno anterior; nuestro programa le vuelve a enviar el historial de conversación como parte del contexto.
    config=types.GenerateContentConfig(
        system_instruction=system_prompt[0]["text"]
    ),
)

print("\nTurno 2:", resp2.text)


#OTRO EJEMPLO CON UNA CONVERSACIÓN EN TRES TRAMOS
#%%
# Turno 1
messages = [
    {
        "role": "user",
        "parts": [
            {"text": "Estoy comparando Claude y un modelo abierto para mi TP. ¿Qué debería tener en cuenta?"}
        ]
    }
]

resp1 = client.models.generate_content(
    model=model_id,
    contents=messages,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt[0]["text"]
    ),
)

print("Turno 1:", resp1.text)


# Agregamos respuesta 1 + pregunta 2
messages.append({
    "role": "user",
    "parts": [{"text": resp1.text}]
})

messages.append({
    "role": "user",
    "parts": [
        {"text": "Y de esos criterios, ¿cuál pesa más si el corpus tiene datos sensibles?"}
    ]
})


# Turno 2
resp2 = client.models.generate_content(
    model=model_id,
    contents=messages,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt[0]["text"]
    ),
)

print("\nTurno 2:", resp2.text)


# Agregamos respuesta 2 + pregunta 3
messages.append({
    "role": "user",
    "parts": [{"text": resp2.text}]
})

messages.append({
    "role": "user",
    "parts": [
        {"text": "¿Y qué alternativa recomendarías en ese caso?"}
    ]
})


# Turno 3
resp3 = client.models.generate_content(
    model=model_id,
    contents=messages,
    config=types.GenerateContentConfig(
        system_instruction=system_prompt[0]["text"]
    ),
)

print("\nTurno 3:", resp3.text)

#%%
#5. Zero-shot vs. few-shot prompting 
'''
- **Zero-shot**: se le pide la tarea al modelo sin ejemplos, confiando en que ya "sabe" hacerla por su entrenamiento.
- **Few-shot**: se incluyen 1 o más ejemplos de entrada→salida en el prompt, para fijar el formato o el criterio esperado — útil cuando la salida tiene un formato específico o el criterio es ambiguo en zero-shot.
'''

# Zero-shot: sin ejemplos
zero_shot_messages = [
    {"role": "user", "parts": [{"text": "Clasificá el sentimiento de este comentario como positivo, negativo o neutro: "
                                            "'La verdad no está mal, pero esperaba más por el precio.'"}]}
]

resp_zero = client.models.generate_content(
     model=model_id, 
     contents=zero_shot_messages
)
print("Zero-shot:", resp_zero.text)


# %%
# Few-shot: con ejemplos que fijan el formato de salida exacto
few_shot_prompt = '''Clasificá el sentimiento como POSITIVO, NEGATIVO o NEUTRO. Respondé solo con la etiqueta.

Comentario: "Excelente atención, volvería sin dudarlo."
Etiqueta: POSITIVO

Comentario: "Nunca más compro acá, un desastre."
Etiqueta: NEGATIVO

Comentario: "La verdad no está mal, pero esperaba más por el precio."
Etiqueta:'''   #al no poner etiqueta, es el modelo el que debe completar este patrón.

few_shot_messages = [{"role": "user", "parts": [{"text": few_shot_prompt}]}]  #es la pregunta. gemini pide diccionario

#solicitud de respuesta
resp_few = client.models.generate_content(
    model=model_id, 
    contents=few_shot_messages
)  #se hace la pregunta.  Ahora few_shot_messages tiene el formato gemini.

print("Few-shot:", resp_few.text)

#DELIMITACIÓN DE INSTRUCCIONES Y CONTEXTO
# %%
prompt_delimitado = '''Sos un asistente que responde preguntas usando SOLO la información del contexto.
Si la respuesta no está en el contexto, decí explícitamente que no tenés esa información.

<contexto>
El programa tiene 12 clases semanales de 3 horas cada una, modalidad virtual.
El Trabajo Práctico Final Integrador es único para toda la cohorte: un sistema RAG
más un flujo con agentes, desplegado vía API.
</contexto>

<pregunta>
¿Cuántas clases tiene el programa y de qué modalidad son?
</pregunta>

<pregunta>
¿Qué profesor dicta el programa?
</pregunta>'''

#solicitud de respuesta
resp = client.models.generate_content(
    model=model_id,
    contents=prompt_delimitado
)

print(resp.text)


#%%
#7. ALGUNOS PARÁMETROS
inference_config = types.GenerateContentConfig(
    temperature=0.2,          # bajo -> respuestas más consistentes
    top_p=0.9,
    max_output_tokens=200,
    stop_sequences=["\n\n"],
)

resp = client.models.generate_content(
    model=model_id,
    contents="Listá 3 ventajas de usar RAG en vez de fine-tuning, una por línea.",
    config=inference_config,
)

print(resp.text)
print("\nStop reason:", resp.candidates[0].finish_reason)

#%%
for temp in [0.0, 0.7, 1.5]:
    
    cfg = types.GenerateContentConfig(
        temperature=temp,
        max_output_tokens=150
    )

    r = client.models.generate_content(
        model=model_id,
        contents="Listá 3 ventajas de usar RAG en vez de fine-tuning, una por línea.",
        config=cfg,
    )

    print(f"--- temperature={temp} ---")
    print(r.text)
    print()


#%%
#CONTROL DE ERRORES

import time  #para evaluar el tiempo
from google.genai import errors

#recibe el cliente gemini, modelo, mensajes, configuración y cantidad máxima de intentos
def generar_con_reintentos(client, model_id, messages, system=None,
                            inference_config=None, max_reintentos=3):
    """Llama a Gemini con reintentos y backoff exponencial ante rate limit."""
    
    #va a llamar a Gemini como máximo el valor de max_reintentos
    for intento in range(max_reintentos):
        try:
            return client.models.generate_content( #intenta llamar a gemini
                model=model_id,
                contents=messages,
                config=inference_config
            )

        except errors.ClientError as err:  #se analiza el error en caso de encontrar uno
            if err.code == 429 and intento < max_reintentos - 1:
                espera = 2 ** intento  # backoff exponencial: 1s, 2s, 4s...
                print(f"Rate limit alcanzado, reintentando en {espera}s...")
                time.sleep(espera)
                continue

            if err.code == 400:  #hay algo mal en la solicitud, por lo que no es necesario hacer un reintetno
                raise ValueError(f"Request inválido: {err}") from err

            raise  # cualquier otro error de cliente, no lo tragamos silenciosamente

    raise RuntimeError("Se agotaron los reintentos por rate limit.") #solo si no se obtuvo una respuesta


# Prueba: esto debería funcionar sin necesitar ningún reintento
resp = generar_con_reintentos(
    client,
    model_id,
    messages=[{"role": "user", "parts": [{"text": "Decí 'ok' y nada más."}]}],
)

print(resp.text)

#%%
#Síncrono vs. Asíncrono
import time
from concurrent.futures import ThreadPoolExecutor

#tenemos 3 preguntas (diccionario con tres elementos)
preguntas = [
    "Definí 'token' en una oración.",
    "Definí 'embedding' en una oración.",
    "Definí 'temperatura' en una oración.",
]

#recibe la pregunta y se la manda a gemini
def preguntar(pregunta):
    resp = client.models.generate_content(
        model=model_id,
        contents=pregunta
    )
    return resp.text

# Secuencial
t0 = time.time()
resultados_seq = [preguntar(p) for p in preguntas]
t_seq = time.time() - t0

# Concurrente (3 llamadas en paralelo)
t0 = time.time()
with ThreadPoolExecutor(max_workers=3) as pool:  #se crean hasta 3 hilos de ejecución
    resultados_par = list(pool.map(preguntar, preguntas))  #3 llamadas en paralelo
t_par = time.time() - t0

print(f"Secuencial: {t_seq:.2f}s")
print(f"Concurrente: {t_par:.2f}s")

#%%
from google import genai

# Si usas google.genai:
response = client.models.generate_content(
    model="gemini-3.1-flash-lite",
    contents="Hola, prueba de conexión."
)
print(response.text)
        
# %%
