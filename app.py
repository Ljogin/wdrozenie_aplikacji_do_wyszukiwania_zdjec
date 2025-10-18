import streamlit as st
import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64
from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams
from openai import OpenAI


# === 1️⃣ Wczytaj zmienne środowiskowe (z .env lub z sekretów Streamlit Cloud) ===
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")  # np. "https://12345-abcd.eu-central.aws.cloud.qdrant.io"

if not all([OPENAI_API_KEY, QDRANT_API_KEY, QDRANT_URL]):
    st.error("❌ Brakuje kluczy środowiskowych. Ustaw OPENAI_API_KEY, QDRANT_API_KEY i QDRANT_URL.")
    st.stop()

# === 2️⃣ Inicjalizacja klientów ===
client = OpenAI(api_key=OPENAI_API_KEY)
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
COLLECTION_NAME = "image_descriptions"

# === 3️⃣ Upewnij się, że kolekcja istnieje ===
try:
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE)
        )
except Exception as e:
    st.error(f"❌ Nie udało się połączyć z Qdrant Cloud: {e}")
    st.stop()


# === 4️⃣ Funkcja generująca opis zdjęcia ===
def generate_description(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    jpeg_bytes = buffer.getvalue()

    base64_image = base64.b64encode(jpeg_bytes).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Jesteś pomocnym asystentem, który opisuje zdjęcia."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Opisz to zdjęcie."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        max_tokens=300
    )

    return response.choices[0].message.content


# === 5️⃣ Funkcja do generowania embeddingów tekstu ===
def get_embedding(text):
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=[text]
    )
    return response.data[0].embedding


# === 6️⃣ UI aplikacji Streamlit ===
st.title("📷 AI Opis i Wyszukiwanie Zdjęć (Qdrant Cloud + GPT-4o)")

menu = st.sidebar.radio("Wybierz opcję", ["📤 Dodaj zdjęcia", "🔍 Szukaj po opisie"])

# --- Dodawanie zdjęć ---
if menu == "📤 Dodaj zdjęcia":
    uploaded_files = st.file_uploader("Prześlij jedno lub więcej zdjęć", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

    if uploaded_files:
        for file in uploaded_files:
            # 🔹 Przesuń wskaźnik na początek pliku (na wypadek, gdyby był już otwarty)
            file.seek(0)

            # 🔹 Sprawdź, czy to naprawdę obraz
            if file.type not in ["image/jpeg", "image/png"]:
                st.warning(f"⚠️ Plik {file.name} ma nieobsługiwany format ({file.type}). Dozwolone: JPG, PNG.")
                continue

            try:
                # 🔹 Otwórz obraz przez PIL
                image = Image.open(file).convert("RGB")
    
                # 🔹 Wyświetl obraz
                st.image(image, caption=file.name, use_column_width=True)

                # 🔹 Zamień na bajty (do dalszego przetwarzania)
                buffer = io.BytesIO()
                image.save(buffer, format="JPEG")
                image_bytes = buffer.getvalue()

                with st.spinner("🧠 Generuję opis zdjęcia..."):
                    description = generate_description(image_bytes)
                    embedding = get_embedding(description)

                    qdrant.upsert(
                        collection_name=COLLECTION_NAME,
                        points=[
                            models.PointStruct(
                                id=int.from_bytes(os.urandom(8), "big"),
                                vector=embedding,
                                payload={"description": description, "filename": file.name}
                            )
                        ]
                    )

                    st.success(f"✅ Zdjęcie '{file.name}' zostało zapisane w Qdrant Cloud!")
                    st.write("**Opis:**", description)

            except Exception as e:
                st.error(f"Błąd podczas przetwarzania pliku {file.name}: {e}")

        points, _ = qdrant.scroll(collection_name=COLLECTION_NAME)
        st.info(f"📦 Liczba rekordów w kolekcji: {len(points)}")


# --- Szukanie po opisie ---
elif menu == "🔍 Szukaj po opisie":
    query = st.text_input("Wpisz opis lub słowa kluczowe")

    if query:
        with st.spinner("🔎 Wyszukuję w Qdrant Cloud..."):
            try:
                query_vector = get_embedding(query)
                results = qdrant.search(
                    collection_name=COLLECTION_NAME,
                    query_vector=query_vector,
                    limit=5
                )

                if not results:
                    st.warning("⚠️ Brak wyników wyszukiwania.")
                else:
                    st.subheader("🔍 Wyniki wyszukiwania")
                    for r in results:
                        st.markdown(f"**🖼️ Plik:** {r.payload['filename']}")
                        st.markdown(f"**📝 Opis:** {r.payload['description']}")
                        st.caption(f"📊 Score: {r.score:.3f}")
                        st.divider()
            except Exception as e:
                st.error(f"Błąd wyszukiwania: {e}")
