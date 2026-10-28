#CLASE 5

#%%
#LIBRERÍAS
import os, json, re, unicodedata, hashlib  
#re: expresiones regulares, unicodedata: normalizar caracteres unicode, hashlib: generar identificadores con funciones hash
from dotenv import load_dotenv
from google import genai
from google.genai import types  #clases para configurar las solicitudes a los modelos
import numpy as np
import chromadb  #importa ChromaDB, base vectorial que almacena embeddings para realizar búsquedas por similitud
import faiss  #crea índices vectoriales y realiza búsquedas por similitud
import numpy as np

'''
objetivo es convertir fragmentos de texto (chunks) en vectores numéricos, que posteriormente se podrán almacenar y utilizar para realizar búsquedas semánticas.

Recordemos la diferencia:
- Búsqueda tradicional: busca coincidencias entre palabras.
- Búsqueda semántica: busca textos que tengan significados similares, aunque utilicen palabras diferentes.

Por ejemplo, si preguntamos:

¿Cómo puedo evitar que un agente ejecute herramientas para las que no tiene autorización?

Una búsqueda semántica podría recuperar este fragmento del corpus:
El principio de mínimo privilegio limita qué herramientas puede invocar un agente según el contexto.

Aunque la pregunta no utiliza exactamente las mismas palabras, ambos textos están relacionados conceptualmente.

Para lograrlo, necesitamos representar los textos mediante embeddings.
'''

#%%
#Generación de embeddings

load_dotenv()

# Configuración del cliente Gemini
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
EMBED_MODEL_ID = "gemini-embedding-001"  #modelos a utilizar para generar embeddings

# Función para generar embeddings 
'''
textos: lista de textos a convertir en embeddings
task_type: para qué se usan los embeddings => RETRIEVAL_DOCUMENT: procesamiento de documentos que luego se querrán recuperar mediante búsquedas
la función devuelve una lista de vectores, donde cada vector es una lista de números decimales
'''

def embed_gemini(textos: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    #solicitud para generar embeddings
    response = client.models.embed_content(
        model=EMBED_MODEL_ID,
        contents=textos,
        config=types.EmbedContentConfig(task_type=task_type)  #embeddings para recuperación de documentos
    )
    return [embedding.values for embedding in response.embeddings] #extrae kis vectires bynperucis de la respuesta de gemini y los devuelve como una lista de listas


# Corpus de juguete: 3 documentos cortos, ya separados en chunks con metadata de tema.
corpus = [
    {"id": "apis-001", "texto": "La Converse API de Bedrock unifica la forma de llamar a distintos modelos: cambia el modelId, no el código.", "tema": "apis", "doc": "guia-apis"},
    {"id": "apis-002", "texto": "Los parámetros de inferencia como temperature y maxTokens van en inferenceConfig en Bedrock.", "tema": "apis", "doc": "guia-apis"},
    {"id": "seguridad-001", "texto": "La inyección de prompts ocurre cuando contenido no confiable manipula las instrucciones del modelo.", "tema": "seguridad", "doc": "guia-seguridad"},
    {"id": "seguridad-002", "texto": "El principio de mínimo privilegio limita qué herramientas puede invocar un agente según el contexto.", "tema": "seguridad", "doc": "guia-seguridad"},
    {"id": "rag-001", "texto": "Un pipeline RAG combina recuperación de evidencia con generación sustentada en esa evidencia.", "tema": "rag", "doc": "guia-rag"},
    {"id": "rag-002", "texto": "El top-k determina cuántos fragmentos recuperados se incluyen como contexto para el modelo generador.", "tema": "rag", "doc": "guia-rag"},
] #lista de diccionarios con el conjunto de documentos
  # tenemos 6 chunks correspondientes a 3 documentos (apis, seguridad, rag) 

# Generación de embeddings
vectores = embed_gemini([c["texto"] for c in corpus])  
'''
recorre la lista corpus y extrae el texto.  Luego, envía los 6 textos al modelos embeddigs de Gemini y obtiene sus vectores numéricos
en "vectores" se almacenan los embeddings generados
'''
print(f"{len(vectores)} vectores de dimensión {len(vectores[0])}") #cantidad de vectores y componentes del primer vector

'''
Todavía no construimos un Vector Store. Por ahora, los vectores están almacenados en una lista de Python llamada vectores.

El próximo paso de la clase probablemente será utilizar estos vectores para realizar búsquedas por similitud semántica y, eventualmente, almacenarlos en un Vector Store.
'''

#*******************************************************************
# 1. Almacenamiento vectorial y recuperación por vecinos cercanos
#*******************************************************************

'''
Con 6 vectores, buscar el más parecido a una consulta es trivial: calcular similitud coseno contra cada uno y ordenar. 
Con 6 millones, esa búsqueda lineal (fuerza bruta) no escala — un **vector store** indexa los vectores (estructuras tipo HNSW, IVF) para encontrar los vecinos más cercanos sin comparar contra todos.

'''

# %%
#Búsqueda por similitud semántica a partir de los embeddings generados

#Función para calcular la similitud de coseno => qué tan parecida es la dirección de los vectores: 1: misma dirección; 0: son perpendiculares; -1: direcciones opuestas
#se busca que tiene a 1
def similitud_coseno(a, b):
    a, b = np.array(a), np.array(b)  #converte los vectores recibidos en arreglos NumPy
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))  #similitud del coseno

#Función de búsqueda por fuerza bruta (consulta_vec => embedding de la pregunta del usuario)
'''
se denomina búsqueda por fuerza bruta porque compara el vector de la consulta con todos 
los vectores almacenados, sin utilizar un índice especializado para acelerar la búsqueda.
'''
def buscar_fuerza_bruta(consulta_vec, vectores, k=3):  #vectores: lista de embeddings de los documentos, k: cantidad de resultados que queremos recuperar por defecto
    scores = [similitud_coseno(consulta_vec, v) for v in vectores]  #similitud de coseno entre el vector consulta y cada uno de los 6 vectores que tenemos.
    indices_ordenados = np.argsort(scores)[::-1][:k]  #ordena los 6 resultados según similitud de mayor a menor.
    return [(corpus[i]["id"], scores[i]) for i in indices_ordenados] #selecciona los primeros k índices: devuelve una lista de tuplas con identificador del chunk y su putuación de similitud

consulta = "¿Cómo se controla la temperatura de generación?"
consulta_vec = embed_gemini([consulta], task_type="RETRIEVAL_QUERY")[0] #RETRIEVAL_QUERY: embedding de una consulta

for id_chunk, score in buscar_fuerza_bruta(consulta_vec, vectores, k=3):
    print(f"{id_chunk}  (sim={score:.3f})")
    
'''
Arroja como resultado: las tres mejores respuestas indicando documento y coeficiente de similitud
apis-002  (sim=0.648)
rag-002  (sim=0.613)
rag-001  (sim=0.600)

El proceso completo es:
1. El usuario realiza una consulta en lenguaje natural.
2. Gemini convierte esa consulta en un embedding.
3. Calculamos la similitud coseno entre el embedding de la consulta y cada uno de los embeddings del corpus.
4. Ordenamos los resultados según su similitud.
5. Recuperamos los k fragmentos más relevantes.

Sin embargo, todavía tenemos dos limitaciones:

Primera: estamos realizando una búsqueda por fuerza bruta, comparando la consulta con todos los vectores. Esto funciona bien con seis documentos, pero puede resultar costoso cuando tenemos millones de fragmentos.

Segunda: los embeddings están almacenados en una lista de Python, no en una base de datos vectorial.

'''
#*************************************************
# 2. ChromaDB: índice, agregar, consultar
#*************************************************

'''
ChromaDB (base de datos vectorial que permite almacenar embeddings y realizar búsquedas por similitud semántica)
Es un vector store embebido (corre en el mismo proceso, sin servidor aparte) con metadata
y persistencia incorporadas — el punto de entrada más simple para RAG.
'''
'''
Ojo: Chroma devuelve **distancia**, no similitud — con la métrica default (L2), 
valores más chicos = más parecido (al revés que la similitud coseno, 
donde más alto = más parecido). Revisá qué métrica usa tu colección antes de interpretar los 
números.
'''

# %%
# Crear un cliente de ChromaDB => base de datos almacenada localmente. Se hace por única vez porque quedan almacenados en el disco
cliente_chroma = chromadb.PersistentClient(path="./chroma_data")  

# Crear o recuperar una colección (la llama "corpus_curso"): conjunto de documentos y embeddings almacenados en ChromaDB
coleccion = cliente_chroma.get_or_create_collection("corpus_curso")  #se asimila a una base de datos relacional

# Incorporar los documentos y sus embeddings a la colección
coleccion.upsert(  #agrega si no existe o actualiza si ya existe
    ids=[c["id"] for c in corpus],  #extrae los identificadores
    embeddings=vectores,  #son los embeddings generados anteriormente con Gemini
    documents=[c["texto"] for c in corpus],  #almacena el texto orginal de cada chunk
    metadatas=[{"tema": c["tema"], "doc": c["doc"]} for c in corpus],  #almacena los metadatos asociados al chunk
)

print("Chunks en la colección:", coleccion.count())  #cantidad de documentos almacenados

# Buscar los 3 documentos más similares a la consulta
resultado = coleccion.query(
    query_embeddings=[consulta_vec],  #vector que se utiliza para la búsqueda
    n_results=3
)

# Mostrar los resultados (zip recorre simultáneamente 3 listas)
for id_, texto, dist in zip(  #es [0] porque buscamos los resultados de la primera consulta., se pueden hacer varias consultas en una misma operación
    resultado["ids"][0],
    resultado["documents"][0],
    resultado["distances"][0]
):
    print(f"{id_}  (distancia={dist:.3f})  {texto[:70]}...") #identifica el chunk, la distancia con la consulta, los primeros 70 caracteres

'''
IMPORTANTE!!!

Similitud coseno:	
Cuanto mayor, más similares son los vectores.	
Una similitud de 1 indica la misma dirección.	
Utilizada en el ejercicio anterior.	

Distancia:
Cuanto menor, más cercanos son los vectores.
Una distancia de 0 indica vectores idénticos.
Utilizada en este ejercicio con ChromaDB.

Una aclaración: ChromaDB utiliza por defecto la distancia euclidiana cuadrática (l2) para las colecciones nuevas. 
También permite configurar otras métricas, como la distancia coseno.

Por lo tanto, las distancias que obtenemos ahora no son directamente comparables con las similitudes
coseno calculadas anteriormente.

Si quisiéramos utilizar la distancia coseno, podríamos crear una colección configurada específicamen
te para esa métrica.
'''

#*******************************
# 3 FILTRADO POR METADATA
#*******************************

'''
Además de similitud semántica, casi siempre hace falta filtrar por un campo exacto — por ejemplo,
restringir la búsqueda a un tema o una fuente específica.
'''

# %%
# Búsqueda semántica con filtro por metadatos
resultado_filtrado = coleccion.query(
    query_embeddings=[consulta_vec],  #vectores para la búsqueda (embedding generado por gemini)
    n_results=3,
    where={"tema": "apis"},  # Solo busca entre los chunks con tema="apis"
)

# Mostrar los resultados
for id_, texto in zip(
    resultado_filtrado["ids"][0],
    resultado_filtrado["documents"][0]
):
    print(f"{id_}  {texto[:70]}...")
    
    
'''
En el ejercicio anterior realizamos una búsqueda semántica sobre todos los chunks almacenados en la colección.

Ahora incorporamos una condición adicional: solamente queremos recuperar información de una categoría determinada.
'''
#**************************************
# 4. Persistencia del índice
#**************************************
'''
Como usamos `PersistentClient`, los datos ya están en disco en `./chroma_data` — no hace falta "guardar" explícitamente. Confirmemos abriendo una conexión nueva.

Este código tiene como objetivo demostrar la persistencia de los datos en ChromaDB. Es decir, comprobar que los documentos y embeddings que almacenamos
anteriormente siguen disponibles cuando volvemos a conectarnos a la base de datos.
'''
    
# %%
cliente_nuevo = chromadb.PersistentClient(path="./chroma_data") #nueva conexión usando la misma carpeta
coleccion_reabierta = cliente_nuevo.get_collection("corpus_curso") #recupera la colección llamada corpus_curso de ejercicios anteriores
print("Chunks recuperados tras reabrir:", coleccion_reabierta.count()) #número de chunks en la colección recuperada

'''
Diferencia importante:
- get_or_create_collection(): recupera la colección si existe o la crea si no existe.
- get_collection(): recupera una colección existente. Si no existe, genera un error.

¿Qué estamos comprobando?
En los ejercicios anteriores:
1. Generamos seis embeddings utilizando Gemini.
2. Los almacenamos en ChromaDB junto con sus textos y metadatos.
3. Ahora creamos un nuevo cliente y recuperamos la colección existente.
Lo importante es que no necesitamos volver a generar los embeddings ni cargar los documentos.

ChromaDB conserva esa información en el disco.

Esto es fundamental para desarrollar sistemas RAG, porque normalmente los documentos se procesan y almacenan una vez, mientras que las consultas se realizan muchas veces.

Una aclaración: este código crea un nuevo cliente, pero no cierra explícitamente el anterior

'''
#*************************************
# 5. Comparación con FAISS
#*************************************

'''
FAISS (Facebook AI Similarity Search) es una librería de bajo nivel: ultrarrápida para operaciones
vectoriales puras, pero **sin metadata ni persistencia incorporada** — eso hay que armarlo a mano.
'''

# %%
# Convertir los embeddings en una matriz NumPy
matriz = np.array(vectores, dtype="float32")  #matriz 6x3072: tranforma la lista de embeddings en una matriz de NumPy

# Normalizar los vectores para utilizar similitud coseno
faiss.normalize_L2(matriz)  #normaliza cada vector para que su longitud sea igual a 1

# Crear un índice FAISS
indice_faiss = faiss.IndexFlatIP(matriz.shape[1]) #crea un índice para almacenar los vectores y poder realizar las búsquedas
'''
IndexFlatIP: crea un índice que utiliza el producto interno (Inner Product) para comparar vectores.
matriz.shape[1]: indica la cantidad de dimensiones de cada vector.
'''

# Incorporar los vectores al índice FAISS
indice_faiss.add(matriz) #NO ALMACENA LOS TEXTOS ORIGINALES! SOLO LOS VECTORES

# Preparar el vector de la consulta
consulta_matriz = np.array([consulta_vec], dtype="float32") #convierte el embedding en NumPy
faiss.normalize_L2(consulta_matriz)  #normaliza el vector de la consulta

# Buscar los 3 vectores más similares
similitudes, posiciones = indice_faiss.search(consulta_matriz, k=3) #vectores de similitud y sus posiciones

# Mostrar los resultados
for pos, sim in zip(posiciones[0], similitudes[0]):
    print(f"{corpus[pos]['id']}  (sim={sim:.3f})")
    
    
#*************************************************************    
# 7. Diagnóstico de resultados irrelevantes y redundantes
#*************************************************************

'''
Dos problemas típicos de un buscador semántico recién armado:

- **Resultados irrelevantes**: el top-k siempre devuelve *algo*, incluso si nada del corpus es realmente relevante — la similitud coseno nunca es "cero", solo más baja. Hay que mirar el *valor* de la similitud, no solo confiar en que "está en el top-k".
- **Resultados redundantes**: varios chunks casi idénticos (por overlap de chunking, o porque el documento repite la misma idea) ocupan varios lugares del top-k, desplazando resultados distintos pero relevantes.
'''



# %%
# CASO 1: Consulta irrelevante al corpus

consulta_irrelevante = "¿Cuál es la receta de una tarta de manzana?"  #no tiene relación con el corpus

vec_irrelevante = embed_gemini(  #convierte la pregunta en un embedding
    [consulta_irrelevante],
    task_type="RETRIEVAL_QUERY"
)[0]

print("Consulta irrelevante al corpus:")

#Ejecuta la función de búsqueda por fuerza bruta (por similitud coseno)
for id_chunk, score in buscar_fuerza_bruta(vec_irrelevante, vectores, k=3):
    print(f"  {id_chunk}  (sim={score:.3f})")

print("-> Notá que la similitud máxima puede ser menor que en una consulta relevante (sección 1).")

#%%
# CASO 2: Información redundante

chunk_redundante = "Los parámetros como la temperatura y el máximo de tokens se configuran en inferenceConfig."

#generación del embedding
vec_redundante = embed_gemini(
    [chunk_redundante],
    task_type="RETRIEVAL_DOCUMENT"  #estamos procesando un documento que se quiere incorporar al conjunto de búsqueda
)[0]

#se incorpora el nuevo vector => se crea una lista nueva. No se toca la lista original.
vectores_con_redundante = vectores + [vec_redundante]  #6 vectores originales + el vector redundante

#se agrega el nuevo documento al corpus
corpus_con_redundante = corpus + [
    {"id": "apis-003-redundante", "texto": chunk_redundante}
]
#cálculo y ordenamiento de similitudes
for id_chunk, score in [ #recorre los tres resultados seleccionados
    (corpus_con_redundante[i]["id"], s) #obtiene el identificador del chunk correspondiente a la posición y lo asicia con su puntuación
    for i, s in sorted(  #ordena de mayor a menor
        enumerate([  #asocia cada similitud con la posición del vector correspondiente
            similitud_coseno(consulta_vec, v)  #calcula la similitud coseno entre la consulta original y los 7 embeddings
            for v in vectores_con_redundante
        ]),
        key=lambda x: -x[1]
    )[:3]  #tres mejores resultados
]:
    print(f"{id_chunk}  (sim={score:.3f})")

print("-> apis-002 y apis-003-redundante pueden competir por el mismo lugar del top-k con contenido casi igual.")

'''
¿Qué estamos intentando demostrar?
Que la existencia de documentos redundantes puede perjudicar la calidad de la recuperación de información.

En nuestro ejemplo, los chunks apis-002 y apis-003-redundante contienen prácticamente la misma información.

Si ambos aparecen entre los tres resultados más relevantes, estaremos utilizando dos de los tres lugares disponibles para recuperar información repetida.

Esto puede provocar que otros documentos, que podrían aportar información complementaria, queden fuera de los resultados.

Este problema es especialmente importante en sistemas RAG, porque los documentos recuperados se incorporan como contexto para el modelo generativo.

Si ese contexto contiene información repetida, estamos desperdiciando parte del espacio disponible para proporcionar evidencia útil.


'''
#********************************************
# 9. Evaluación manual de consultas
#********************************************
'''
Antes de automatizar métricas (Clase 8), vale la pena simplemente *mirar* los resultados de un puñado
de consultas representativas — es sorprendente cuántos problemas se detectan a ojo.

En este ejercicio vamos a realizar varias consultas de prueba para evaluar la calidad de la recuperación semántica.

La idea es comprobar si el sistema recupera los chunks adecuados para distintas preguntas y observar qué sucede cuando una consulta
no tiene relación con el corpus.
'''

# %%
# Definir las consultas de prueba (se crea una lista con 4 consutlas de prueba, las primeras 3 relacionadas con el corpus)
consultas_de_prueba = [
    "¿Qué controla la temperatura en una llamada al modelo?",
    "¿Qué es la inyección de prompts?",
    "¿Para qué sirve el top-k en RAG?",
    "¿Cómo se hace un backup de la base de datos?",  # Fuera de dominio, a propósito
]

# Ejecutar las consultas y mostrar los resultados
for q in consultas_de_prueba:  #recorro las consultas
    vec_q = embed_gemini([q], task_type="RETRIEVAL_QUERY")[0] #convierte consulta en embedding
    top1_id, top1_score = buscar_fuerza_bruta(vec_q, vectores, k=1)[0] #función de búsqueda semántica
    print(f"Q: {q}\n  -> {top1_id} (sim={top1_score:.3f})\n")
    
    
'''
Nuestro corpus no contiene información sobre backups.

Sin embargo, la función buscar_fuerza_bruta() igualmente devolverá el documento con mayor similitud.

Esto demuestra una limitación importante: el sistema no sabe cuándo debe abstenerse de recuperar información.

Simplemente devuelve el documento más parecido, aunque no sea suficientemente relevante para responder la pregunta.

En un sistema RAG real, podríamos incorporar un mecanismo que permita detectar consultas fuera de dominio y evitar que el
modelo generativo responda utilizando información inadecuada.
'''
    
    
# %%
