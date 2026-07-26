# ==============================
# 🤖 AGENTE IA - CHALLENGE ALURA
# ==============================

# Importamos librerías necesarias
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

# ==============================
# 🔐 CARGAR VARIABLES DE ENTORNO
# ==============================
load_dotenv()

# Verificamos que la API KEY exista
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: Falta la API KEY en el archivo .env")
    exit()

# ==============================
# 📄 CARGAR DOCUMENTO PDF
# ==============================

# Ruta del archivo PDF
ruta_pdf = "src/data/documento.pdf"

# Cargar PDF
print("📄 Cargando documento...")
loader = PyPDFLoader(ruta_pdf)
documents = loader.load()

print(f"✅ Documento cargado con {len(documents)} páginas")

# ==============================
# 🧠 CREAR EMBEDDINGS
# ==============================

print("🧠 Generando embeddings...")

embeddings = OpenAIEmbeddings()

# Guardamos en base vectorial (FAISS)
db = FAISS.from_documents(documents, embeddings)

print("✅ Embeddings listos")

# ==============================
# 🤖 CREAR AGENTE
# ==============================

print("🤖 Inicializando agente...")

qa = RetrievalQA.from_chain_type(
    llm=OpenAI(),
    retriever=db.as_retriever()
)

print("✅ Agente listo\n")

# ==============================
# 💬 INTERFAZ DE PREGUNTAS
# ==============================

print("💬 Puedes hacer preguntas sobre el PDF")
print("👉 Escribe 'salir' para terminar\n")

while True:
    pregunta = input("🧑 Tú: ")

    if pregunta.lower() == "salir":
        print("👋 Cerrando agente...")
        break

    try:
        respuesta = qa.run(pregunta)
        print("🤖 Agente:", respuesta, "\n")

    except Exception as e:
        print("❌ Error:", e)
