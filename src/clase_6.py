#CLASE 6: ARQUITECTURA RAG

#%%
#LIBRERÍAS
import os  #variables de entorno
from dotenv import load_dotenv  #variables archivo .env
from google import genai  #SDK de Google para interacción con modelos Gemini
from google.genai import types  #clases para configurar solicitudes a Gemini
import chromadb  #creación y consultas de bases vectoriales

#Conexión con Google Gemini, generación de embeddings y consultas modelo generativo
#%%

load_dotenv()

# Configuración de Gemini
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])  #cliente para comunicación con API gemini

MODEL_ID = "gemini-3.1-flash-lite"    #modelo generativo
EMBED_MODEL_ID = "gemini-embedding-001"  #modelo de embeddings


# Función para generar embeddings
'''
Parte de una lista de cadenas de texto, con input_type indicando que los textos son documentos (podrían ser consultas)
Devuelve una lista de vectores (cada uno una lista de números decimales)
'''

def embed_gemini(textos: list[str], input_type: str = "search_document") -> list[list[float]]:
    task_type = {                                 #diccionario que traduce entrada en Cohere a utilizados en Gemini
        "search_document": "RETRIEVAL_DOCUMENT",
        "search_query": "RETRIEVAL_QUERY"
    }.get(input_type, "RETRIEVAL_DOCUMENT")
    #envía los textos a Gemini para generar los embeddings
    resp = client.models.embed_content(  #llama a la API de embeddings
        model=EMBED_MODEL_ID,
        contents=textos,  #los textos que se convierten a vectores
        config=types.EmbedContentConfig(task_type=task_type)  #si los textos son consultas o documentos de donde se extraen las respuestas del LLM
    )

    return [embedding.values for embedding in resp.embeddings]  #lista de los vectores. Tantos vectores como documentos.


# Función para consultar al modelo generativo
def llamar_gemini(prompt, system=None, temperature=0.0, max_tokens=500):
    config = types.GenerateContentConfig(  #objeto config para generación de texto
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system
    )
    #manda la consulta al modelo generativo de Gemini
    resp = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=config
    )

    return resp.text


#%%
# Mismo corpus de juguete de la Clase 5, reconstruido en ChromaDB


corpus = [
    {"id": "apis-001", "texto": "La API de Gemini permite interactuar con distintos modelos mediante el SDK google-genai.", "doc": "guia-apis"},
    {"id": "apis-002", "texto": "Los parámetros de inferencia como temperature y max_output_tokens se configuran mediante GenerateContentConfig en Gemini.", "doc": "guia-apis"},
    {"id": "seguridad-001", "texto": "La inyección de prompts ocurre cuando contenido no confiable manipula las instrucciones del modelo.", "doc": "guia-seguridad"},
    {"id": "seguridad-002", "texto": "El principio de mínimo privilegio limita qué herramientas puede invocar un agente según el contexto.", "doc": "guia-seguridad"},
    {"id": "rag-001", "texto": "Un pipeline RAG combina recuperación de evidencia con generación sustentada en esa evidencia.", "doc": "guia-rag"},
    {"id": "rag-002", "texto": "El top-k determina cuántos fragmentos recuperados se incluyen como contexto para el modelo generador.", "doc": "guia-rag"},
]
#creación de un cliente chromadb
cliente_chroma = chromadb.Client()  # en memoria, alcanza para la demo de hoy

coleccion = cliente_chroma.get_or_create_collection("corpus_clase6")  #crea o llama a una colección

coleccion.add(
    ids=[c["id"] for c in corpus],
    embeddings=embed_gemini([c["texto"] for c in corpus]),  #creación de embeddings
    documents=[c["texto"] for c in corpus],
    metadatas=[{"doc": c["doc"]} for c in corpus],
)

print("Chunks indexados:", coleccion.count())  #número de registros almacenados

#%%
#IMPLEMENTACIÓN BASE DEL PIPELINE
'''
Este código implementa el pipeline RAG completo en su versión más sencilla: 
1. recibe una pregunta,
2. recupera los fragmentos más relevantes de ChromaDB, 
3. construye un contexto
4. utiliza Gemini para generar la respuesta.
'''
#función recuperar: convierte la consulta en un embedding y busca los k fragmentos más cercanos en ChromaDB
def recuperar(consulta: str, k: int = 3) -> list[dict]:  #la consulta devuelve una lista de direcciones
    vec = embed_gemini([consulta], input_type="search_query")[0]  #embedding de la consulta. Extrae el primer vector de la lista devuelta (porque hay una única consulta)
    resultado = coleccion.query(query_embeddings=[vec], n_results=k)  #consulta la colección creada y devuelve los k vectores más cercanos
    return [
        {"chunk_id": id_, "texto": texto, "doc": meta["doc"], "distancia": dist}
        for id_, texto, meta, dist in zip(
            resultado["ids"][0], resultado["documents"][0], resultado["metadatas"][0], resultado["distances"][0]
        )
    ]
'''
la idea es construir un diccionario con los fragmentos y 4 campos para cada fragmento
con for recorre los resultados recuperados, tomando id, texto, metada y distancia (a menor distancia mayor cercanía).
con zip se agrupan los datos correspondientes a cada fragmento recuperado
'''

#función armar_contecto. organiza los k fragmentos recuperados en un texto que incluye sus fuentes para ser enviado a Gemini
def armar_contexto(chunks: list[dict]) -> str:  #recibe chunks y devuelve una cadena de texto
    bloques = [f"[{i+1}] (fuente: {c['doc']}) {c['texto']}" for i, c in enumerate(chunks)]  #lista de textos por cada fragmento
    return "\n\n".join(bloques)  #se unen todos los bloques en una sola cadena de texto
'''
bloques => con enumarate(chunks) recorre los fragmentos, luego los numera, obtiene el documento
de origen, obtiene el contenido del gramento
'''


#función rag_v1: Coordina las funciones anteriores, construye el prompt y solicita la respuesta a Gemini
def rag_v1(consulta: str, k: int = 3) -> str:
    chunks = recuperar(consulta, k)
    contexto = armar_contexto(chunks)
    prompt = f"Contexto:\n{contexto}\n\nPregunta: {consulta}\n\nRespondé usando el contexto."
    return llamar_gemini(prompt)


print(rag_v1("¿Qué controla la temperatura en una llamada al modelo?"))

'''
OJO!!! 
Esto ya "funciona", pero es frágil: no delimita claramente contexto/pregunta, no define qué hacer
si no hay evidencia, y no dice de dónde sacó la respuesta. Las próximas 4 secciones arreglan cada
uno de esos problemas, uno a la vez.

'''
#DISEÑO DEL PROMPT CON EVIDENCIA RECUPERADA
'''
Se mejora el pipeline RAG para utilizar exclusivamente la información de ChromaDB
'''

# %%
#Definición de las instrucciones del sistema (instrucciones que queremos darle al sistema)
SYSTEM_RAG = (
    "Sos un asistente que responde preguntas usando EXCLUSIVAMENTE la evidencia provista. "
    "No uses conocimiento propio ni inventes información que no esté en la evidencia."
)
#prompt que recibe Gemini separando la evidencia de la pregunta
def armar_prompt_delimitado(consulta: str, contexto: str) -> str:
    return f'''<evidencia>  
{contexto}
</evidencia> 

<pregunta>
{consulta}
</pregunta>

Respondé la pregunta usando solo la evidencia. Si la evidencia no alcanza, decilo explícitamente.'''

#Corrdinación del proceso RAG
def rag_v2(consulta: str, k: int = 3) -> str:
    chunks = recuperar(consulta, k)  #embeddings, búsqueda en ChromaDB. Resultados en chunks una lista de diccionarios
    contexto = armar_contexto(chunks) 
    prompt = armar_prompt_delimitado(consulta, contexto)
    return llamar_gemini(prompt, system=SYSTEM_RAG)


print(rag_v2("¿Qué controla la temperatura en una llamada al modelo?"))

#GESTIÓN DE LA VENTANA DE CONTEXTO
'''
El objetivo es controlar cuánta información recuperada de ChromaDB enviamos al modelo generativo
'''

#%%


def armar_contexto_con_presupuesto(chunks: list[dict], max_caracteres: int = 800) -> str:
    """Agrega chunks al contexto hasta agotar el presupuesto de caracteres (proxy simple de tokens)."""
    bloques, total = [], 0
    for i, c in enumerate(chunks):
        bloque = f"[{i+1}] (fuente: {c['doc']}) {c['texto']}"
        if total + len(bloque) > max_caracteres:
            print(f"Presupuesto agotado: se descartan {len(chunks) - i} chunk(s) de menor ranking.")
            break
        bloques.append(bloque)
        total += len(bloque)
    return "\n\n".join(bloques)


chunks_prueba = recuperar("¿Qué controla la temperatura?", k=6)  # pedimos todos los que hay y se devuelven ordenados por cercanía vectorial
contexto_acotado = armar_contexto_con_presupuesto(chunks_prueba, max_caracteres=150) #límite de 150 caracteres para el contenido de los bloques
print(contexto_acotado)

#RESPUESTAS SUSTENTADAS Y MANEJO DE AUSENCIA DE EVIDENCIA
'''
vamos a comparar tres versiones del pipeline RAG para observar qué sucede cuando el usuario realiza una pregunta cuya respuesta no está en los documentos.

El objetivo es que nuestro RAG reconozca cuándo no tiene evidencia suficiente para responder, en lugar de generar una respuesta potencialmente inventada.
'''

#%%
# Consulta sin evidencia en el corpus
consulta_sin_evidencia = "¿Cómo se hace un backup de la base de datos?"

print("--- rag_v1 (sin instrucción de abstenerse) ---")
print(rag_v1(consulta_sin_evidencia)) #utiliza en rag_v1 que recupera 3 fragmentos por defecto y los utiliza como contexto para la respuesta

print("\n--- rag_v2 (con instrucción explícita) ---")  #ejecuta la segunda versión del pipeline rag_v2 incorporando las instrucciones del sistema y el prompt delimitado
print(rag_v2(consulta_sin_evidencia))

#Defino un nuevo rag. Agrego una distancia máxima utilizada como criterio de relevancia
def rag_v3(consulta: str, k: int = 3, umbral_distancia: float = 1.0) -> str:
    chunks = recuperar(consulta, k)

    # Filtramos los fragmentos según su distancia
    chunks_relevantes = [
        c for c in chunks if c["distancia"] < umbral_distancia
    ]

    # Si no hay evidencia suficiente, no consultamos a Gemini
    if not chunks_relevantes:
        return "No encontré evidencia suficiente en el corpus para responder esta pregunta."

    contexto = armar_contexto(chunks_relevantes) #genera una única cadena de texto
    prompt = armar_prompt_delimitado(consulta, contexto) #construye el promp con la pregunta, la evidencia recuperada y las instrucciones

    return llamar_gemini(prompt, system=SYSTEM_RAG)  #envía el prompt a Gemini


print(rag_v3(consulta_sin_evidencia, umbral_distancia=0.9))
print()
print(rag_v3("¿Qué controla la temperatura en una llamada al modelo?"))

# IMPLEMENTACIÓN DE CITAS


#%%
'''
OJO! no vamos citar las fuentes utilizadas, sino las fuentes recuperadas más allá de que luego no sean utilizadas por gemini 
'''

#La siguiente función devuelve un diccionario en lugar de una cadena de texto
def rag_con_citas(consulta: str, k: int = 3, umbral_distancia: float = 1.0) -> dict:
    chunks = recuperar(consulta, k)  #k fragmentos más cercanos
    chunks_relevantes = [c for c in chunks if c["distancia"] < umbral_distancia] #filtra por distancia < umbral
    #comprueba que la lista no está vacía
    if not chunks_relevantes:  
        return {"respuesta": "No encontré evidencia suficiente para responder.", "fuentes": []}

    contexto = armar_contexto(chunks_relevantes)  #organiza los fragmentos relevantes en una cadena de texto, numerados e indicando documento de origen
    instruccion_citas = (  #instrucción adicional para Gemini
        "Además, citá la evidencia usada con el número entre corchetes correspondiente, "
        "ej. 'La temperatura se configura en GenerateContentConfig [1].'"
    )
    prompt = armar_prompt_delimitado(consulta, contexto) + "\n\n" + instruccion_citas #construcción del prompt completo concatenando tres elementos
    respuesta = llamar_gemini(prompt, system=SYSTEM_RAG) #generación de la respuesta
    #lista de fuentes
    fuentes = [{"ref": i + 1, "chunk_id": c["chunk_id"], "doc": c["doc"]} for i, c in enumerate(chunks_relevantes)]
    return {"respuesta": respuesta, "fuentes": fuentes} #diccionario con dos elementos

#Ejemplo
resultado = rag_con_citas("¿Qué controla la temperatura en una llamada al modelo?")
print(resultado["respuesta"])
print("\nFuentes citadas (todas las disponibles en el contexto, el modelo elige cuáles usar):")
for f in resultado["fuentes"]:
    print(f"  [{f['ref']}] {f['chunk_id']} ({f['doc']})")
    
    
## 8. Registro de documentos y fragmentos utilizados
'''
Más allá de las citas que ve el usuario final, para debugging (Clase 7) y evaluación (Clase 8) conviene loguear 
internamente *qué se recuperó*, no solo qué se citó — a veces el problema no es la generación, es que la recuperación
trajo lo que no correspondía.    
'''

#%%


registro_consultas = []  # en producción: log estructurado o base de datos, no una lista en memoria

def rag_con_registro(consulta: str, k: int = 3, umbral_distancia: float = 1.0) -> dict:
    chunks = recuperar(consulta, k)
    resultado = rag_con_citas(consulta, k, umbral_distancia)

    registro_consultas.append({
        "consulta": consulta,
        "chunks_recuperados": [c["chunk_id"] for c in chunks],
        "distancias": [round(c["distancia"], 3) for c in chunks],
        "chunks_usados_en_contexto": [f["chunk_id"] for f in resultado["fuentes"]],
    })
    return resultado

#Ejecución del pipeline
rag_con_registro("¿Qué es la inyección de prompts?")
print(registro_consultas[-1])
