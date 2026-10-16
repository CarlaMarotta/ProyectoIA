#%%
#Librerías
import os    #acceder a las variables de entorno del sistema (ej: las del archivo .env)
import json   # trabajar con datos en formato json
import time   #manejar tiempos en espera
import random # generar valores aleatorios para implementar estrategias de reintento
import httpx
from dotenv import load_dotenv   #carga las variables del archivo .env (o el archivo de configuración)
from google import genai
from google.genai import types
from pydantic import BaseModel, Field, ValidationError  #pydantic permite definir y validar estructura de datos
from typing import Literal
from google import genai
from google.genai import types
from google.genai import errors
from abc import ABC, abstractmethod

#%%
#Levanto las variables de entorno
load_dotenv() #si el archivo no fuera .env, entonces tendríamos que usar load_doenv(nombre_archivo.env)
provider = os.environ.get("LLM_PROVIDER", "gemini")  #busca la variable LLM_PROVIDER en .env o usa gemini por default
print(f"Proveedor configurado: {provider}")


# %%
# Crear el cliente de Gemini y seleccionar el modelo
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
model_id = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-flash")

#%%
#Función para llamar a un modelo google gemini y obtener una respuesta
def llamar_gemini(prompt, system=None, temperature=0.7, max_tokens=500):
    #parámetros para la generación de la respuesta del modelo
    config = types.GenerateContentConfig(
        temperature=temperature,   #grado de aleatoriedad en la generación de respuestas
        max_output_tokens=max_tokens,
        system_instruction=system   #opcional =>define instrucciones generales sobre cómo debe comportarse el modelo 
    )
    #enviar la solicitud
    resp = client.models.generate_content(
        model=model_id,  #qué modelo tiene que usar
        contents=prompt, #cuál es el texto de la consulta
        config=config    #la configuración definida en config
    )

    return resp.text

#%%
'''
Un ejemplo de uso: 

respuesta = llamar_gemini(
    prompt="¿Qué es JSON?",
    system="Sos un profesor de programación. Respondé de forma breve y clara."
)

print(respuesta)


'''

#**************************
# 1. JSON SCHEMA Y PYDANTIC
#***************************

#%%
#Define una estructura para representar una consulta de soporte técnico o administrativo
'''
Primero definimos qué estructura deben tener los datos y qué restricciones deben cumplir. 
Después generamos un JSON Schema que describe esas reglas y lo mostramos en pantalla.
'''

#BaseModel => permite definir modelos de datos con reglas de validación

class ConsultaSoporte(BaseModel):  #se crea un nuevo tipo de dato llamado ConsultaSoporte con Pydantic
    categoria: Literal["facturacion", "tecnico", "cuenta", "otro"]  #Literal: restringe los valores aceptados por un campo
    urgencia: Literal["baja", "media", "alta"]
    resumen: str = Field(   #Field: permite agregar restricciones y descripciones a un campo
        ...,  #campo obligatorio
        max_length=200,  #máximo de caracteres
        description="Resumen breve del reclamo, en una oración"  #información que debe contener
    )

#JSON Schema => formato estandar que permite describir cómo debe estar estructurado un documento JSON.
schema = ConsultaSoporte.model_json_schema()  
print(json.dumps(schema, indent=2, ensure_ascii=False)) #muestra el esquema en formato JSON
# schema=> convierte el diccionario Python a texto "JSON"
#indent=2 indentación de dos espacios
#ensure_ascii=False permite mostrar caracteres especiales


#%%
#Validación de datos y detección de errores sin que una excepción no controlada interrumpa el programa

dato_valido = {"categoria": "tecnico", "urgencia": "alta", "resumen": "El servicio no responde desde hace 2 horas."}
consulta = ConsultaSoporte(**dato_valido) #con ** se desempaqueta un diccionario, valida los datos y pasa sus elementos como argumentos que se guardan en "consulta"
print(consulta)

#Ejemplo de validación incorrecta porque "categoría" no puede tomar valor "ivnentado"
try:
    ConsultaSoporte(categoria="inventado", urgencia="alta", resumen="x")
except ValidationError as e:
    print("\nError de validación (esperado):")
    print(e)


#%%
#lo anterior es lo mismo que hacer:
ConsultaSoporte(
    categoria="tecnico",
    urgencia="alta",
    resumen="El servicio no responde desde hace 2 horas."
)

#***********************************
#**🏠 Practicá (tarea, después de la clase)**: agregá un campo `requiere_escalamiento: bool` al modelo `ConsultaSoporte`. Regenerá el JSON Schema y confirmá que el nuevo campo aparece en `properties`. Probá instanciar el modelo sin ese campo — ¿qué pasa? (pista: Pydantic trata los campos sin default como requeridos).
#***********************************

#%%
#Agrego el campo "requiere_escalamiento"
class ConsultaSoporte(BaseModel):
    categoria: Literal["facturacion", "tecnico", "cuenta", "otro"]
    urgencia: Literal["baja", "media", "alta"]
    resumen: str = Field(
        ...,
        max_length=200,
        description="Resumen breve del reclamo, en una oración"
    )
    requiere_escalamiento: bool

#%%
#Regenero JSON Schema
schema = ConsultaSoporte.model_json_schema()
print(json.dumps(schema, indent=2, ensure_ascii=False))    
   
    
# %%
#Ejecución con error: no tenemos valor asignado para "requiere_esclarecimiento"
try:
    consulta = ConsultaSoporte(
        categoria="tecnico",
        urgencia="alta",
        resumen="El servicio no responde."
    )
except ValidationError as e:
    print(e)
    
#%%
#Ejecución corregida:
consulta = ConsultaSoporte(
    categoria="tecnico",
    urgencia="alta",
    resumen="El servicio no responde.",
    requiere_escalamiento=True
)

print(consulta)


#*************************
# 2. OUTPUTS ESTRUCTURADOS
#*************************

'''
Pedirle al modelo "devolveme JSON" en el prompt **no garantiza** que la salida sea JSON válido, ni que cumpla el schema — el modelo puede agregar texto antes/después, usar comillas simples, olvidar un campo, etc.
AWS Bedrock (igual que OpenAI y algunos otros proveedores) tiene un mecanismo **nativo** que fuerza al modelo a producir una salida que cumple exactamente el schema: en Bedrock, el campo `outputConfig.textFormat` de la Converse API.
'''

'''
se va a combina Gemini con Pydantic para generar una salida estructurada y validarla
Se va a pedir a Gemini que interprete un texto libre y que genere automáticamente los datos con
la estructura de "ConsultaSoporte"

En síntesis: el programa envía una consulta escrita en lenguaje natural a Gemini, le pide que la clasifique
en un JSON con una estructura determinada y luego utiliza Pydantic para validar el resultado.
'''

#%%
#Generación de salida estructurada con Gemini y Pydantic: clasificador de consultas a modo de ejemplo

# Cliente y modelo configurados anteriormente
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
model_id = os.environ.get("GEMINI_MODEL_ID", "gemini-3.1-flash-lite")

#variable texto_libre con la consulta que se quiere clasificar
texto_libre = (
    "Hace dos días que no puedo entrar a mi cuenta y ya perdí "
    "una reunión importante por esto, necesito que lo resuelvan YA."
)
#llamo al modelo de Gemini y guardo la respuesta en la variable "response"
response = client.models.generate_content(
    model=model_id,
    contents=f"Clasificá esta consulta de soporte: '{texto_libre}'", #prompt que se envía a Gemini
    config=types.GenerateContentConfig(  #define la configuración de la respuesta que esperamos del modelo
        response_mime_type="application/json",  #solicita respuesta con formato JSON
        response_schema=ConsultaSoporte, #JSON con la estructura de ConsultaSoporte
        temperature=0.1  #temperatura baja para respuestas más consistentes
    )
)

raw_json = response.text  #respuesta de Gemini en formato JSON
print("JSON crudo devuelto por el modelo:", raw_json)

consulta_validada = ConsultaSoporte.model_validate_json(raw_json) #interpreta el JSON devuelto por Gemini y comprueba si cumple con las restricciones
print("\nObjeto Python validado:", consulta_validada)



#***********************************
# 3. MANEJO DE RESPUESTAS INVÁLIDAS
#***********************************
'''
Este ejemplo introduce un concepto muy importante de la Clase 3: cómo construir una aplicación que pueda recuperarse cuando un modelo de IA devuelve una respuesta inválida.
'''

# %%
#Función de validación

#se pasa la función llamar_genini para que se ejecute dentro del ciclo con un máximo de tres reintentos.
#este código captura errores de validación del JSON, pero no errores de conexión, límites de uso de la API o respuestas vacías. Esos casos requerirían un manejo adicional.
'''
Este código define una función que llama a Gemini, valida que la respuesta cumpla con un esquema Pydantic y, 
si es inválida, vuelve a intentarlo hasta tres veces.

llamar_fn: función que realiza la llamada al modelo (por ejemplo, llamar_gemini).
prompt: consulta que queremos enviar.
schema_model: clase Pydantic que define la estructura esperada.
system=None: instrucciones generales opcionales.
max_reintentos=3: cantidad máxima de intentos.
'''
def generar_estructurado_con_reintentos(llamar_fn, prompt, schema_model, system=None, max_reintentos=3):
    """Patrón universal: sirve con cualquier función llamar_* (Bedrock, OpenAI, Anthropic, Google, Ollama)."""
    instruccion_base = (  #Construye un texto que contiene la consulta original y el JSON Schema generado por Pydantic. Le indica al modelo que responda exclusivamente con un JSON válido.
        f"{prompt}\n\nRespondé ÚNICAMENTE con un JSON válido que cumpla este schema "
        f"(sin texto adicional, sin markdown):\n{schema_model.model_json_schema()}"
    )
    ultimo_error = None  #guarda el error producido por la validación anterior

    for intento in range(max_reintentos):
        instruccion = instruccion_base #usa las instrucciones originales en cada uno de los 3 intentos
        if ultimo_error: #si hubo error
            instruccion += f"\n\nTu respuesta anterior no cumplió el schema. Error: {ultimo_error}\nCorregí y devolvé SOLO el JSON."
        #Ejecuta la función
        texto = llamar_fn(instruccion, system=system, temperature=0.0)  #recibe una función, no el resultado de ejecutarla

        try:
            return schema_model.model_validate_json(texto) #valida que el JSON cumpla con el esquema de Pydantic
        except (ValidationError, json.JSONDecodeError) as e:
            ultimo_error = str(e)
            #print(f"Intento {intento + 1} inválido, reintentando...")
            if intento + 1 < max_reintentos:
               print(f"Intento {intento + 1} inválido, reintentando...")
            else:
               print(f"Intento {intento + 1} inválido. Se agotaron los reintentos.")

    raise ValueError(  #si los tres intentos fallan, la función genera una excepción.
        f"No se obtuvo una respuesta válida tras {max_reintentos} intentos. "
        f"Último error: {ultimo_error}"
    )

#Ejemplo de uso de la función creada    
resultado = generar_estructurado_con_reintentos(
    llamar_gemini, #función para consultar al modelo
    f"Clasificá esta consulta de soporte: '{texto_libre}'", #consulta al modelo
    ConsultaSoporte, #estructura de la respuesta
)

print(resultado)


#*****************************************
# 4. TOOL CALLING => llamada a herramientas
#*****************************************
'''
## 
Tool calling es un caso particular de salida estructurada: en vez de pedirle al modelo un objeto de negocio, le das una lista de **herramientas disponibles** (funciones con su schema de argumentos) y el modelo decide *si* necesita alguna y con qué argumentos — vos ejecutás la función real y le devolvés el resultado.
El modelo **nunca ejecuta código** — solo pide "llamá a esta función con estos argumentos". La ejecución es responsabilidad de tu aplicación (esto es clave para seguridad, lo retomamos en la Clase 9 y la Clase 11).

Hasta ahora, Gemini recibía una consulta y generaba una respuesta. Con Tool Calling, el modelo puede identificar que necesita utilizar una función externa para responder.
Lo fundamental es entender que el modelo decide qué función necesita y con qué argumentos, pero nuestro programa Python es el que ejecuta esa función.

La idea central de Tool Calling es que el modelo aporta la capacidad de interpretar y decidir, mientras que las herramientas aportan la capacidad de actuar sobre sistemas y datos externos.

'''
#%%
#Llamada a herramientas
'''
Este código tiene dos partes principales: 
(1) definir una función Python que consulta el estado de un ticket y 
(2) describir esa función para que Gemini sepa que puede solicitar su ejecución.
'''

client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
model_id = "gemini-3.1-flash-lite"

# 1. La herramienta real: una función Python (sin IA) que consulta un diccionario, pero podría consultar una base de datos, API o sistema de gestión de tickets
def consultar_estado_ticket(ticket_id: str) -> dict:
    tickets_mock = {
        "T-001": "En progreso",
        "T-002": "Resuelto",
        "T-003": "Esperando al cliente"
    }
    return {
        "ticket_id": ticket_id,
        "estado": tickets_mock.get(ticket_id, "No encontrado")
    }


# 2. Descripción de la herramienta para Gemini
tool = types.Tool(#objeto que describe una o varias herramientas que podría utilizar Gemini
    function_declarations=[  #declaración que permite que Gemini conozca la existencia de la función de Python
        types.FunctionDeclaration(#crea la descripción de una de las funciones que puede usar el modelo
            name="consultar_estado_ticket", #nombre de la función
            description="Consulta el estado actual de un ticket de soporte por su ID.", #para qué sirve=> ayuda al modelo a decidir cuándo usarla
            parameters={ #descripción de los argumentos de la función
                "type": "OBJECT",
                "properties": {
                    "ticket_id": {
                        "type": "STRING",
                        "description": "ID del ticket, ej. T-001"
                    }
                },
                "required": ["ticket_id"]
            }
        )
    ]
)

'''
El SDK de Gemini admite ejecución automática de funciones Python en determinadas configuraciones.
La desactivamos porque queremos reproducir el ejemplo del profesor y controlar nosotros mismos cuándo se ejecuta la herramienta.
'''
#crea la configuración que se utiliza para llamar a Gemini
config = types.GenerateContentConfig(
    tools=[tool], #herramientas que tiene disponibles
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True  #si queremos o no la ejecución automática de funciones. En este caso se quiere controlar cuándo se ejecutan
    )
)

# 3. Mensaje del usuario
messages = [ #acá se guarda la conversación
    types.Content( #se crea un mensaje del usuario
        role="user",
        parts=[types.Part.from_text( #parts permite incorporar varios textos. Ahora hay solo uno
            text="¿Cuál es el estado del ticket T-002?"
        )]
    )
]

# 4. Primera llamada a Gemini y almacenamiento de la respuesta
response = client.models.generate_content(
    model=model_id,
    contents=messages,
    config=config
)

# Conservamos la respuesta del modelo en el historial de la conversación
messages.append(response.candidates[0].content)

# 5. Verificamos si Gemini solicita ejecutar una herramienta
if response.function_calls:
    for function_call in response.function_calls:  #recorre todas las llamadas a funciones solicitadas por el modelo
        print(
            f"El modelo pide llamar a: "
            f"{function_call.name}({function_call.args})"
        )

        if function_call.name == "consultar_estado_ticket":
            # Ejecutamos nosotros la función Python
            resultado = consultar_estado_ticket(**function_call.args)

            # Incorporamos el resultado a la conversación
            messages.append(
                types.Content(
                    role="tool",
                    parts=[
                        types.Part.from_function_response(
                            name=function_call.name,
                            response=resultado
                        )
                    ]
                )
            )

    # 6. Segunda llamada: Gemini genera la respuesta final
    response_final = client.models.generate_content(
        model=model_id,
        contents=messages,
        config=config
    )
'''
Realiza una segunda llamada a Gemini, ahora con el historial que incluye:
1. La pregunta original del usuario.
2. La solicitud de ejecución de la herramienta.
3. El resultado obtenido al ejecutar la función Python.
Con esa información, Gemini puede elaborar una respuesta en lenguaje natural.
'''
print("\nRespuesta final:", response_final.text) #respuesta final del modelo.

else: #si no se solicita ninguna herramienta
print(response.text)
 
'''
Una aclaración: este ejemplo contempla una sola ronda de uso de herramientas. 
En una aplicación más compleja, Gemini podría solicitar nuevas herramientas después de recibir el primer 
resultado, por lo que necesitaríamos un ciclo que continuara hasta obtener una respuesta final.

'''


#************************************
# 5. Timeouts, reintentos y backoff
#****************************************

'''
Este código implementa una llamada a Gemini con control de tiempos de espera (timeouts) 
y reintentos ante errores de límite de uso (rate limit).

- **Timeout explícito**: sin él, una llamada colgada puede bloquear tu aplicación indefinidamente.
Un timeout es el tiempo máximo que estamos dispuestos a esperar por una operación.
- **Jitter**: sumar un poco de aleatoriedad al backoff evita que muchos clientes reintenten exactamente al mismo tiempo tras un error masivo (efecto "estampida").
El backoff exponencial consiste en aumentar progresivamente el tiempo de espera entre intentos fallidos.
'''

# %%
# Cliente Gemini con timeout explícito

# Crea un cliente para comunicarse con Gemini
client_prod = genai.Client(
    api_key=os.environ["GOOGLE_API_KEY"],
    http_options=types.HttpOptions(
        timeout=15000,  #son milisegundos
        client_args={"timeout": httpx.Timeout(15.0, connect=5.0)}
    )
)

model_id = "gemini-3.1-flash-lite"

#genera la función que permite hasta 3 intentos
def generar_con_reintentos(
    client,
    model_id,
    prompt,
    system=None,
    temperature=0.7,
    max_reintentos=3
):
    config = types.GenerateContentConfig( #configuraciones generales del modelo
        temperature=temperature,
        system_instruction=system
    )

    for intento in range(max_reintentos): #recorre cada uno de los 3 intentos
        try: 
            return client.models.generate_content( #envia la consulta a Gemini
                model=model_id,
                contents=prompt,
                config=config
            )

        except errors.ClientError as err: #captura posibles herrores HTTP devueltos por la API de Gemini
            codigo = err.code #código del error

            # Rate limit: error HTTP 429
            if codigo == 429 and intento < max_reintentos - 1:  #código de límite de uso
                espera = (2 ** intento) + random.uniform(0, 1)  #tiempo de espera antes del otro intento: 2 elevado a la potencia intento.
                print(f"Rate limit, reintentando en {espera:.1f}s...")
                time.sleep(espera) #pausa la ejecución durante los segundos calculados
                continue

            # Request inválido: error HTTP 400
            if codigo == 400: #solicitud inválida
                raise ValueError(f"Request inválido: {err}") from err

            raise

    raise RuntimeError("Se agotaron los reintentos.")


# Prueba de funcionamiento
resp = generar_con_reintentos(
    client_prod,
    model_id,
    prompt="Decí 'ok'."
)

print(resp.text)

#***************************************************
# 6. Observabilidad básica: tokens, costo, latencia
#***************************************************

'''
Este código permite registrar el consumo de tokens, estimar el costo 
y medir la latencia de cada llamada a un modelo de IA.
'''


# Configuración de Gemini
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"]) #creación del cliente (ya está arriba)
model_id = "gemini-3.1-flash-lite"  #definición del modelo que también está más arriba

# Precios ilustrativos en USD por millón de tokens.
# Reemplazar por los precios vigentes del modelo utilizado.
PRECIO_POR_MILLON_USD = {"input": 3.0, "output": 15.0}

#Función de estimación de costos
def estimar_costo_usd(input_tokens: int, output_tokens: int) -> float:
    costo_input = input_tokens / 1_000_000 * PRECIO_POR_MILLON_USD["input"]
    costo_output = output_tokens / 1_000_000 * PRECIO_POR_MILLON_USD["output"]
    return costo_input + costo_output


registro_llamadas = []  #lista vacía para almacenar registros de llamadas

#Función 
def llamar_con_registro(client, model_id, prompt, **kwargs):
    inicio = time.perf_counter() #instante en que comienza la operación

    resp = generar_con_reintentos(client, model_id, prompt, **kwargs)  #llama a Gemini con la función creada anteriormente

    latencia_ms = (time.perf_counter() - inicio) * 1000  #tiempo transcurrido desde el inicio en milisegundos

    usage = resp.usage_metadata  #metadatos de consumo de tokens que devuelve Gemini
    input_tokens = usage.prompt_token_count or 0 #cantidad de tokens de entrada
    output_tokens = usage.candidates_token_count or 0 #cantidad de tokens de salida
    
    #agrega un diccionario a la lista de registros
    registro_llamadas.append({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "costo_usd": estimar_costo_usd(input_tokens, output_tokens),
        "latencia_ms": round(latencia_ms, 2)
    })

    return resp


# Prueba
resp = llamar_con_registro(
    client,
    model_id,
    prompt="Explicá qué es un embedding en 1 oración."
)

print(resp.text)
print("\nRegistro:", registro_llamadas[-1])


#***************************************************
# 7. Streaming
#***************************************************
'''
En síntesis: el streaming permite que el usuario comience a leer la respuesta mientras Gemini todavía está 
generándola. Es especialmente útil para respuestas largas, chatbots y agentes conversacionales, porque mejora 
la percepción de velocidad.
'''

#%%
#Solicitud a Gemini con streaming

#se crea un cliente y se define el modelo como se hizo anteriormente
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
model_id = "gemini-3.1-flash-lite"

#solicitud utilizando streaming para recibir la respuesta en fragmentos
stream_response = client.models.generate_content_stream(  #necesito modelo y prompt
    model=model_id,
    contents="Listá 5 buenas prácticas de seguridad para agentes de IA, con una oración de explicación cada una."
)

#Recorre los fragmentos (chunks) de la respuesta a medida que van llegando
for chunk in stream_response:
    if chunk.text: #muestra el texto recibido inmediatamente
        print(chunk.text, end="", flush=True) 
        #end="": evita salto de línea luego de cada fragmento.
        #flush=True: fuerza la visualizción inmediata del texto, sin esperar a que se llene el bufer de salida

    if chunk.usage_metadata: #si el fragmento tiene información sobre el consumo de tokens
        print(f"\n\n[tokens: {chunk.usage_metadata}]")  #muestra los metadatos
        
        
#***************************************************
# 8. Abstracción de Proveedores
#***************************************************

'''
Este ejemplo introduce un concepto importante en el desarrollo de agentes de IA: crear una interfaz 
común para trabajar con distintos proveedores de modelos de lenguaje.

'''

#%%
#Implementación solo Gemini

#from abc import ABC, abstractmethod => ABC permite definir una clase abstracta, abstractmethod: indica que un método debe ser implementado por las clases que heredan de ella

# 1. Interfaz común para proveedores de LLM
class LLMProvider(ABC): #se define una clase abstracta
    """Interfaz común para proveedores de modelos de lenguaje."""

    @abstractmethod #el siguiente método es obligatorio para las clases que hereden de LLMProvider
    #Se define un método que debe generar texto
    def generate(
        self, #instancia de la clase
        prompt: str, #consulta para el modelo
        system: str = None, #instrucciones generales opcionales
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str: 
        """Devuelve el texto generado por el modelo."""
        ...
    #Se define un método para generar respuestas estructuradas y sí se implementa en la clase base
    def generate_structured(self, prompt: str, schema_model, system: str = None):
        """Implementación genérica para generar respuestas estructuradas."""
        return generar_estructurado_con_reintentos(
            self.generate, #es el método de generación del proveedor concreto
            prompt,
            schema_model,
            system=system
        )


# 2. Implementación específica para Gemini
class GoogleProvider(LLMProvider): #clase que hereda de LLMProvider

    def __init__(self): #define un constructor
        self.client = genai.Client( #crea el cliente de Gemini como atributo del objeto
            api_key=os.environ["GOOGLE_API_KEY"]
        )
        self.model_id = os.environ.get( #obtiene el modelo
            "GOOGLE_MODEL_ID",
            "gemini-3.1-flash-lite"
        )
    #Implementa el método obligatorio definido en LLMProvider
    def generate(
        self,
        prompt,
        system=None,
        temperature=0.7,
        max_tokens=500
    ):
        response = self.client.models.generate_content( #llamada a Gemini y guardado de respuesta
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )
        return response.text  #texto generado por Gemini
    
    #Define una nueva implementación del método que ya existe en LLMProvider => sobre escritura de métodos o method overriding
    def generate_structured(self, prompt, schema_model, system=None):
        """Utiliza la generación estructurada nativa de Gemini."""
        response = self.client.models.generate_content( #llamada a Gemini 
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig( #solicita respuesta JSON con estrucutura de clase Pydantic schema_model.
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema_model,
                temperature=0.1
            )
        )
        return schema_model.model_validate_json(response.text)


# 3. Factory para seleccionar el proveedor
def get_provider() -> LLMProvider:
    proveedores = {
        "google": GoogleProvider
    }

    nombre = os.environ.get("LLM_PROVIDER", "google")

    if nombre not in proveedores:
        raise ValueError(f"Proveedor desconocido: {nombre}")

    return proveedores[nombre]()


# 4. Prueba
llm = get_provider()

print(llm.generate("Decí 'funciona' y nada más."))


#%%
#Implementación todos los proveedores

import os
import json
import time
import random

import boto3   #pip install boto3
import httpx
import anthropic  #pip install anthropic
import ollama     #pip install ollama

from abc import ABC, abstractmethod
from botocore.config import Config
from botocore.exceptions import ClientError
from openai import OpenAI
from google import genai
from google.genai import types
from pydantic import ValidationError
from dotenv import load_dotenv


load_dotenv()


# ============================================================
# 1. Función genérica para respuestas estructuradas
# ============================================================

def generar_estructurado_con_reintentos(
    llamar_fn,
    prompt,
    schema_model,
    system=None,
    max_reintentos=3
):
    instruccion_base = (
        f"{prompt}\n\n"
        "Respondé ÚNICAMENTE con un JSON válido que cumpla este schema "
        f"(sin texto adicional, sin markdown):\n"
        f"{json.dumps(schema_model.model_json_schema(), ensure_ascii=False)}"
    )

    ultimo_error = None

    for intento in range(max_reintentos):
        instruccion = instruccion_base

        if ultimo_error:
            instruccion += (
                "\n\nTu respuesta anterior no cumplió el schema. "
                f"Error: {ultimo_error}\n"
                "Corregí y devolvé SOLO el JSON."
            )

        texto = llamar_fn(
            instruccion,
            system=system,
            temperature=0.0
        )

        try:
            return schema_model.model_validate_json(texto)

        except (ValidationError, json.JSONDecodeError) as e:
            ultimo_error = str(e)

            if intento + 1 < max_reintentos:
                print(f"Intento {intento + 1} inválido, reintentando...")
            else:
                print("Se agotaron los reintentos.")

    raise ValueError(
        f"No se obtuvo una respuesta válida tras {max_reintentos} intentos. "
        f"Último error: {ultimo_error}"
    )


# ============================================================
# 2. Clase abstracta: interfaz común
# ============================================================

class LLMProvider(ABC):
    """Interfaz común para cualquier proveedor de LLM."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> str:
        """Devuelve el texto generado por el modelo."""
        ...

    def generate_structured(
        self,
        prompt: str,
        schema_model,
        system: str = None
    ):
        """Generación estructurada genérica con validación y reintentos."""

        return generar_estructurado_con_reintentos(
            self.generate,
            prompt,
            schema_model,
            system=system
        )


# ============================================================
# 3. Amazon Bedrock
# ============================================================

class BedrockProvider(LLMProvider):

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.environ.get("AWS_REGION", "us-east-1"),
            config=Config(
                connect_timeout=5,
                read_timeout=15,
                retries={"max_attempts": 0}
            )
        )

        self.model_id = os.environ.get(
            "BEDROCK_MODEL_ID_CLAUDE",
            "global.anthropic.claude-sonnet-4-6"
        )

    def generate(
        self,
        prompt,
        system=None,
        temperature=0.7,
        max_tokens=500
    ):
        messages = [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ]

        kwargs = {
            "modelId": self.model_id,
            "messages": messages,
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens
            }
        }

        if system:
            kwargs["system"] = [{"text": system}]

        ultimo_error = None

        for intento in range(3):
            try:
                resp = self.client.converse(**kwargs)

                return resp["output"]["message"]["content"][0]["text"]

            except ClientError as e:
                ultimo_error = e
                codigo = e.response["Error"]["Code"]

                if codigo in (
                    "ThrottlingException",
                    "TooManyRequestsException"
                ) and intento < 2:
                    espera = (2 ** intento) + random.uniform(0, 1)
                    time.sleep(espera)
                    continue

                raise

        raise RuntimeError(
            f"Se agotaron los reintentos: {ultimo_error}"
        )


# ============================================================
# 4. Anthropic
# ============================================================

class AnthropicProvider(LLMProvider):

    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"]
        )

        self.model_id = os.environ.get(
            "ANTHROPIC_MODEL_ID",
            "claude-sonnet-4-6"
        )

    def generate(
        self,
        prompt,
        system=None,
        temperature=0.7,
        max_tokens=500
    ):
        kwargs = {
            "model": self.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        }

        if system:
            kwargs["system"] = system

        resp = self.client.messages.create(**kwargs)

        return "".join(
            bloque.text
            for bloque in resp.content
            if bloque.type == "text"
        )


# ============================================================
# 5. OpenAI
# ============================================================

class OpenAIProvider(LLMProvider):

    def __init__(self):
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"]
        )

        self.model_id = os.environ.get(
            "OPENAI_MODEL_ID",
            "gpt-4o"
        )

    def generate(
        self,
        prompt,
        system=None,
        temperature=0.7,
        max_tokens=500
    ):
        messages = []

        if system:
            messages.append({
                "role": "system",
                "content": system
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return resp.choices[0].message.content or ""


# ============================================================
# 6. Google Gemini
# ============================================================

class GoogleProvider(LLMProvider):

    def __init__(self):
        self.client = genai.Client(
            api_key=os.environ["GOOGLE_API_KEY"]
        )

        self.model_id = os.environ.get(
            "GOOGLE_MODEL_ID",
            "gemini-3.1-flash-lite"
        )

    def generate(
        self,
        prompt,
        system=None,
        temperature=0.7,
        max_tokens=500
    ):
        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                temperature=temperature,
                max_output_tokens=max_tokens
            )
        )

        return resp.text or ""

    def generate_structured(
        self,
        prompt,
        schema_model,
        system=None
    ):
        """Utiliza las salidas estructuradas nativas de Gemini."""

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_schema=schema_model,
                temperature=0.1
            )
        )

        return schema_model.model_validate_json(resp.text)


# ============================================================
# 7. Ollama (modelo ejecutado localmente)
# ============================================================

class OllamaProvider(LLMProvider):

    def __init__(self):
        self.client = ollama.Client(
            host=os.environ.get(
                "OLLAMA_HOST",
                "http://localhost:11434"
            )
        )

        self.model_id = os.environ.get(
            "OLLAMA_MODEL_ID",
            "llama3.1"
        )

    def generate(
        self,
        prompt,
        system=None,
        temperature=0.7,
        max_tokens=500
    ):
        messages = []

        if system:
            messages.append({
                "role": "system",
                "content": system
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        resp = self.client.chat(
            model=self.model_id,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens
            }
        )

        return resp.message.content or ""


# ============================================================
# 8. Factory: seleccionar el proveedor
# ============================================================

def get_provider() -> LLMProvider:
    """Devuelve una instancia del proveedor configurado."""

    proveedores = {
        "bedrock": BedrockProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "google": GoogleProvider,
        "ollama": OllamaProvider
    }

    nombre = os.environ.get(
        "LLM_PROVIDER",
        "google"
    ).lower()

    if nombre not in proveedores:
        raise ValueError(
            f"Proveedor desconocido: {nombre}"
        )

    return proveedores[nombre]()


# ============================================================
# 9. Prueba
# ============================================================

llm = get_provider()

respuesta = llm.generate(
    "Decí 'funciona' y nada más."
)

print(respuesta)