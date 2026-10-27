#CLASE 4

#%%
#LIBRERÍAS
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np
import re  #expresiones regulares
import unicodedata  #caracteres unicode y normalización de su representación
import hashlib  #permite calcular hashes: valores que funcionan como huellas digitales de un contenido.
from dataclasses import dataclass, field  #Importa herramientas que simplifican la creación de clases destinadas a almacenar datos.


#%%
# Cargar las variables de entorno, cliente y modelo
load_dotenv()

# Crear el cliente de Google
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])


#************************************************
# 2. EMBEDDINGS Y SIMILITUD SEMÁNTICA
#************************************************

# Seleccionar el modelo de embeddings
EMBED_MODEL_ID = os.environ.get(
    "GOOGLE_EMBED_MODEL_ID",
    "gemini-embedding-001"
)

#%%
# Función para generar embeddings (convierte textos en vectores)
'''
Define una función que:
- Recibe textos: una lista de cadenas de texto.
- Recibe input_type: indica para qué se utilizarán los embeddings.
- Devuelve una lista de embeddings, donde cada embedding es una lista de números decimales (float).

OBJETIVO: La utilidad de este código es convertir textos en números para que podamos buscar información 
por su significado y no solamente por las palabras que contienen.
El modelo está entrenado para que los textos semánticamente relacionados tengan representaciones cercanas en ese espacio numérico.
Después podemos utilizar una operación matemática, llamada similitud coseno, para medir esa cercanía.

'''
def embed_google(
    textos: list[str],
    input_type: str = "RETRIEVAL_DOCUMENT"
) -> list[list[float]]:

    """Devuelve un embedding por cada texto recibido."""
    #llama a la API de Google
    response = client.models.embed_content(
        model=EMBED_MODEL_ID,
        contents=textos,  #lista de textos que se quieren convertir en vectores
        config=types.EmbedContentConfig( #propósito de los embeddings
            task_type=input_type
        )
    )
    #Recorre los embeddings devueltos por Google, extrae los valores numéricos de cada uno y devuelve una lista de listas.
    return [embedding.values for embedding in response.embeddings] #LIST COMPREHENSION
'''
Ejemplo Embeddings:
- RETRIEVAL_DOCUMENT: para textos que vamos a almacenar y posteriormente buscar.
- RETRIEVAL_QUERY: para consultas que queremos comparar con esos documentos.
'''

# %%
#EJEMPLO
textos = [
    "El cliente solicita una copia de su factura.",
    "El usuario no puede acceder a su cuenta."
]

vectores = embed_google(textos)

print("Cantidad de embeddings:", len(vectores))
print("Dimensiones del primer embedding:", len(vectores[0]))
print("Primeros 5 valores:", vectores[0][:5])
# %%

'''
ahora vamos a medir qué tan similares son dos textos en cuanto a su significado,
utilizando los embeddings que genera Gemini.
'''

#%%
# Función para calcular la similitud coseno
'''
Calcula la similitud coseno mediante tres operaciones:
- np.dot(a, b): calcula el producto escalar de los dos vectores.
- np.linalg.norm(a) y np.linalg.norm(b): calculan la longitud de cada vector.
- Divide el producto escalar por el producto de ambas longitudes. float() convierte el resultado en un número decimal de Python.
'''

def similitud_coseno(a, b):  #recibe dos vectores y calcula su similitud
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Lista de frases que queremos comparar 
frases = [
    "La ventana de contexto limita cuánto texto puede procesar un modelo por llamada.",
    "El límite de tokens de entrada y salida define cuánta información cabe en una consulta.",
    "Ualá es una fintech argentina que ofrece una tarjeta prepaga y una cuenta digital.",
]

# Generar embeddings utilizando Gemini
vectores = embed_google(frases)

print(f"Dimensión de cada embedding: {len(vectores[0])}")

# Comparar los embeddings
print("\nSimilitud entre frase 1 y 2 (ambas sobre contexto/tokens):",
      round(similitud_coseno(vectores[0], vectores[1]), 3))

print("Similitud entre frase 1 y 3 (temas no relacionados):       ",
      round(similitud_coseno(vectores[0], vectores[2]), 3))
# %%


#**********************************
#LIMPIEZA Y NORMALIZACIÓN
#**********************************

'''
## 4. Limpieza y normalización

Antes de vectorizar, conviene limpiar el texto — no es cosmético, afecta directamente la calidad 
de la búsqueda:

- **Espacios y saltos de línea redundantes**: texto extraído de PDF/HTML suele traer basura de 
formato.
- **Boilerplate repetido** (headers/footers, "Página X de Y", menús de navegación): si aparece 
en todos los chunks, contamina la similitud — todo se parece un poco entre sí por el ruido compartido.
- **Normalización Unicode**: comillas tipográficas, acentos mal codificados, etc.
- **¿Lowercase?** Depende: para texto natural generalmente no hace falta 
(los embeddings modernos son robustos a mayúsculas), pero para código o identificadores técnicos,
bajar a minúsculas puede destruir información relevante (`API_KEY` ≠ `api_key` en un fragmento de
código). No lo hagas por default sin pensarlo.

'''

#%%
#Limpieza y normalización (función simple)

def limpiar_texto(texto: str) -> str:  #(de cadena de texto a cadena de texto)
    texto = unicodedata.normalize("NFKC", texto)
    texto = re.sub(r"[ \t]+", " ", texto)  #reemplaza secuencias de escpacios por un único espacio
    texto = re.sub(r"\n{3,}", "\n\n", texto) #reemplaza tres o + saltos de línea por dos.
    return texto.strip()  #devuelve el texto limpio

#Ejemplo de texto sucio con varios especios y líneas entre textos
texto_sucio = """   Página 1 de 3

Guía  de   configuración
Para configurar  las credenciales,   copiá .env.example a .env.\n\n\n\nCompletá la sección de tu proveedor.
"""

print(repr(limpiar_texto(texto_sucio)))

#***********************************************
# 5. Estrategias de chunking y solapamiento
#***********************************************
'''
Un documento entero casi nunca entra (ni conviene que entre) en un solo embedding — hay que partirlo
en **chunks**. Dos decisiones centrales:
- **Tamaño del chunk**: muy chico → pierde contexto (una oración sola puede ser ambigua). 
Muy grande → mezcla varios temas en un solo vector, diluyendo la similitud con cualquiera de ellos.
- **Solapamiento (overlap)**: sin overlap, una idea que cruza el límite entre dos chunks queda 
cortada en ambos. Con un poco de overlap (ej. 10-20% del tamaño del chunk), se mitiga ese corte 
a costa de un poco de redundancia.
'''

# %%
#Función de chunking por caracteres
'''
Toma un texto, lo divide en fragmentos de hasta 300 cracteres compartiendo 50 caracteres entre
fragmentos consecutivos (solapamiento).
'''

def chunking_por_caracteres(texto: str, tamano: int = 300, solapamiento: int = 50) -> list[str]:
    """Chunking de tamaño fijo con solapamiento, por caracteres (simple, no respeta oraciones)."""
    chunks = [] #lista vacía para guardar los fragmentos
    paso = tamano - solapamiento #salto en nro de caracteres para pasar al siguiente fragmento
    for inicio in range(0, len(texto), paso): #recorre el texto de 250 en 250.
        chunk = texto[inicio:inicio + tamano]  #extrae 300 caracteres
        if chunk.strip():  #fragmento que no esté vacío ni contenga solo espacios
            chunks.append(chunk)
        if inicio + tamano >= len(texto):  #si llega al final del texto, termina
            break
    return chunks

#Función de chunking por párrafo
'''
Defino una función que agrupa párrafos de menos de 500 caracteres por fragmento.
Una limitación: si un solo párrafo supera los 500 caracteres, esta función lo conserva entero. 
Por lo tanto, max_chars no es un límite estricto. Además, la comprobación utiliza + 1, aunque la 
separación real entre párrafos ocupa dos caracteres (\n\n).
'''
def chunking_por_parrafos(texto: str, max_chars: int = 500) -> list[str]:
    """Agrupa párrafos (separados por línea en blanco) hasta llegar al tamaño máximo."""
    parrafos = [p.strip() for p in texto.split("\n\n") if p.strip()]  #divide el texto en donde encuentra dos saltos de línea consecutivos, elimina sobrantes y descarta párrafos vacíos.
    chunks, actual = [], ""  #inicializa una lista
    for p in parrafos:   #recorre los párrafos
        if len(actual) + len(p) + 1 <= max_chars:  #si el párrafo se puede agregar al fracmento actual sin superar el límite de 500
            actual = (actual + "\n\n" + p).strip()  #agrega el párrafo al fragmento separado con línea en blanco
        else:
            if actual:
                chunks.append(actual)
            actual = p  #comienza un nuevo fragmento
    if actual:
        chunks.append(actual) #Al terminar el recorrido, guarda el último fragmento pendiente.
    return chunks  #Devuelve la lista completa.


#******************************************************
# 6. Metadata, identificadores y trazabilidad
#******************************************************
'''
Un chunk suelto, sin contexto de dónde vino, no sirve para citar fuentes (Clase 6) ni para 
debuggear por qué el sistema recuperó algo irrelevante (Clase 7). Cada chunk necesita viajar con 
su metadata.

Hasta ahora aprendimos a limpiar un documento y dividirlo en fragmentos. Pero si tenemos cientos 
de documentos y miles de fragmentos, necesitamos saber a qué documento pertenece cada fragmento, 
qué posición ocupa y cómo identificarlo de manera única.

Este código resuelve precisamente ese problema.
'''
#%%
#Creación de clase con metadata

@dataclass  #python debe generar automáticamente métodos para inicializar y representar los objetos de la clase
class Chunk:
    texto: str   #contenido del fragmento
    documento_id: str   #identificador del documento de origen
    posicion: int       # orden del fragmento dentro del documento
    chunk_id: str = field(init=False)  #identificador del fragmento

    def __post_init__(self):  #método especial que se ejecuta automáticamente después de crear e inicializar un objeto Chunk
        hash_contenido = hashlib.sha256(self.texto.encode()).hexdigest()[:8]
        '''
        Realiza tres operaciones:
        1. self.texto.encode() convierte el texto en bytes.
        2. hashlib.sha256(...) calcula un hash del contenido.
        3. .hexdigest()[:8] obtiene los primeros ocho caracteres de su representación hexadecimal.
        '''
        self.chunk_id = f"{self.documento_id}-{self.posicion:03d}-{hash_contenido}"

#recibe texto e identificador y devuelve una lista de objetos Chunk
def construir_chunks(
    texto: str,
    documento_id: str,
    tamano: int = 300,
    solapamiento: int = 50
) -> list[Chunk]:

    textos = chunking_por_caracteres(texto, tamano, solapamiento)
    #Esta línea utiliza una list comprehension para crear un objeto Chunk por cada fragmento.
    return [
        Chunk(texto=t, documento_id=documento_id, posicion=i)
        for i, t in enumerate(textos)
    ]

#%%
#Ejemplo
corpus_ejemplo = """La Converse API de Bedrock es una interfaz unificada para conversación. Funciona igual para
distintos modelos: cambia el modelId, no la forma de llamar. Soporta mensajes de sistema,
parámetros de inferencia como temperatura y top_p, y tool calling con un schema JSON de argumentos.

Para usarla se necesita un cliente de boto3 apuntando al servicio bedrock-runtime, y credenciales
configuradas vía variables de entorno o un perfil de AWS."""

#limpio el texto, divido en chunks y agrega identificadores y metadata
chunks = construir_chunks(
    limpiar_texto(corpus_ejemplo),
    documento_id="doc-converse-api"
)
#Muestra su identificador, posición y primeros 60 caracteres de contenido.
for c in chunks:
    print(f"[{c.chunk_id}] (pos {c.posicion}) {c.texto[:60]}...")

# %%
#Comparación de estrategias de chuning

documento_largo = corpus_ejemplo * 3  # simulamos un documento más largo repitiendo el de ejemplo

for tamano, overlap in [(150, 0), (150, 30), (400, 0), (400, 80)]:
    chunks_resultantes = chunking_por_caracteres(documento_largo, tamano=tamano, solapamiento=overlap)
    print(f"tamano={tamano:>4} overlap={overlap:>3} -> {len(chunks_resultantes)} chunks")