import streamlit as st
import subprocess
import os
from deepface import DeepFace
from qdrant_client.models import Filter
from qdrant_client import QdrantClient
from PIL.ExifTags import TAGS, GPSTAGS
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import os

# Initialisation du client Qdrant
client = QdrantClient(url="http://localhost:6333")


def get_exif_data(image_path):
    """
    Retrieves the EXIF data from the image.
    """
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data:
        return {TAGS.get(tag, tag): value for tag, value in exif_data.items()}
    return None


def get_geotagging(exif):
    """
    Extracts GPS information from the EXIF data.
    """
    if 'GPSInfo' in exif:
        gps_info = exif['GPSInfo']
        geotags = {GPSTAGS.get(t, t): gps_info[t] for t in gps_info}
        return geotags
    return None


def get_decimal_coordinates(geotags):
    """
    Converts EXIF GPS coordinates to decimal format.
    """

    def convert_to_degrees(value):
        d, m, s = value
        return d + (m / 60.0) + (s / 3600.0)

    lat = convert_to_degrees(geotags['GPSLatitude'])
    lon = convert_to_degrees(geotags['GPSLongitude'])

    if geotags['GPSLatitudeRef'] != "N":
        lat = -lat
    if geotags['GPSLongitudeRef'] != "E":
        lon = -lon

    return lat, lon


def get_datetime(exif):
    """
    Retrieves the date and time from the EXIF data.
    """
    if 'DateTime' in exif:
        return exif['DateTime']
    return None


def generate_embeddings_from_image(img_path: str, model_name: str = "Facenet512", detector_backend: str = "retinaface"):
    """
    Génère les embeddings pour tous les visages dans une image.

    Args:
        img_path (str): Chemin vers l'image.
        model_name (str): Modèle utilisé pour les embeddings.
        detector_backend (str): Backend pour la détection des visages.

    Returns:
        list: Liste des vecteurs d'embeddings.
    """
    try:
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            align=True,
            enforce_detection=True
        )
        return [embedding["embedding"] for embedding in embedding_objs]
    except Exception as e:
        print(f"Erreur lors de la génération des embeddings pour {img_path} : {e}")
        return []


def query_similar_profiles(embedding: list, collection_name: str, limit: int = 3, threshold: float = 0.5):
    """
    Requête dans Qdrant pour trouver les profils les plus similaires à un embedding donné.

    Args:
        embedding (list): L'embedding pour lequel rechercher des similitudes.
        collection_name (str): Nom de la collection dans Qdrant.
        limit (int): Nombre maximum de résultats à retourner.
        threshold (float): Seuil minimum de similarité.

    Returns:
        list: Résultats des points similaires au-dessus du seuil.
    """
    try:
        hits = client.search(
            collection_name=collection_name,
            query_vector=embedding,
            limit=limit,
        )
        # Filtrer les résultats en fonction du seuil
        filtered_hits = [hit for hit in hits if hit.score >= threshold]
        return filtered_hits
    except Exception as e:
        print(f"Erreur lors de la requête de recherche : {e}")
        return []


def run_face_reco(file_path):
    # Générer les embeddings pour l'image contenant potentiellement plusieurs visages
    img_path = file_path
    embeddings = generate_embeddings_from_image(img_path)
    found_names = []

    if embeddings:
        # Pour chaque embedding détecté dans l'image
        for i, embedding in enumerate(embeddings):
            print(f"\nRecherche pour le visage {i + 1}:")
            results = query_similar_profiles(embedding, collection_name="Profile", threshold=0.5)
            if results:
                for result in results:
                    name = result.payload.get("name")
                    found_names.append(name)
                    print(f"Profil similaire trouvé : {result.payload}, score : {result.score}")

            else:
                print("Aucun résultat au-dessus du seuil.")
        print("found_name", found_names[0])
        return (found_names[0])
    else:
        print("Aucun visage détecté ou erreur lors de la génération des embeddings.")
    return None


# Runn the model LLava
def run_model(image_path, question=None):
    """
    Function to run the vision-language model using a subprocess.
    """
    command = [
        "python3",
        "run_llavaphi.py",
        f"{image_path}",
    ]

    if question:
        command.append(f"--question={question}")

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error(f"Error: {result.stderr}")
            return None
    except Exception as e:
        st.error(f"Exception: {str(e)}")
        return None


def run_model_with_context(image_path, contexte):
    """
    Function to run the vision-language model using a subprocess.
    """
    command = [
        "python3",
        "run_llavaphi_with_context.py",
        f"{image_path}",
        f"{contexte}",
    ]
    print("command:", command)
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            return result.stdout
        else:
            st.error(f"Error: {result.stderr}")
            return None
    except Exception as e:
        st.error(f"Exception: {str(e)}")
        return None


def generate_description(image_description, additional_info=None):
    """
    Generates a scene description with additional contextual information.

    :param image_description: Basic description of the image.
    :param additional_info: Dictionary containing additional contextual information.
           - May include 'time', 'gps', 'similar_profile'.
    :return: Complete description adapted to the context.
    """
    description = image_description

    if additional_info:
        if 'time' in additional_info:
            description += f" The photo was taken at {additional_info[0]}."

        if 'gps' in additional_info:
            description += f" The photo was taken at latitude : {additional_info[1]}, longitude : {additional_info[2]}."

        if 'similar_profile' in additional_info:
            profiles = additional_info['similar_profile']
            if len(profiles) == 1:
                description += f" One person is likely present in the scene,{profiles[0]}"
            elif len(profiles) > 1:
                description += f" The people ({profiles}) are likely present in the scene."

    return description


st.title("Vision-Language Model Interface")
st.sidebar.header("Configuration")
model_path = st.sidebar.text_input("Model Path", value="llava.pte")
tokenizer_path = st.sidebar.text_input("Tokenizer Path", value="tokenizer.bin")
prompt = st.sidebar.text_input("Prompt", value="ASSISTANT:")
seq_len = st.sidebar.number_input("Sequence Length", value=768, step=1)
temperature = st.sidebar.number_input("Temperature", value=0.0, step=0.1)

# State to keep track of the last uploaded image
if "last_image_path" not in st.session_state:
    st.session_state.last_image_path = None

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    temp_image_path = os.path.join("temp_image.png")
    with open(temp_image_path, "wb") as f:
        f.write(uploaded_file.read())

    st.session_state.last_image_path = temp_image_path

    st.image(temp_image_path, caption="Uploaded Image", use_container_width=True)

# Run model on the uploaded or previously uploaded image
if st.session_state.last_image_path:
    if st.button("Run Model"):
        with st.spinner("Running the model..."):
            result_reco = run_face_reco(file_path=st.session_state.last_image_path)
            image_path = st.session_state.last_image_path  # Path to the photo
            if result_reco:
                context = result_reco
                print("contex:", context)
                result = run_model_with_context(image_path, context)
                result = result.rsplit('>', 1)[-1].strip()  # Strip removes any extra spaces
                synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
                txt = result
                speech = synthesiser(txt, forward_params={"speaker_embeddings": speaker_embedding})
                sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

                filename = "speech.wav"
                st.audio(filename, format="audio/wav", autoplay=True)
                st.success("Model ran successfully!")
                st.session_state.last_result = result
                st.text_area("Output", result, height=200)
            else:
                result = run_model(image_path)
                result = result.split('>')[1].strip()
                synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
                txt = result
                speech = synthesiser(txt, forward_params={"speaker_embeddings": speaker_embedding})
                sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])

                filename = "speech.wav"
                st.audio(filename, format="audio/wav", autoplay=True)
                st.success("Model ran successfully!")
                st.session_state.last_result = result
                st.text_area("Output", result, height=200)
                # Strip removes any extra spaces
                st.success("Model ran successfully!")
                st.session_state.last_result = result
                st.text_area("Output", result, height=200)

    if "last_result" in st.session_state and st.session_state.last_result:
        question = st.text_input("Ask a follow-up question about the image")

        if st.button("Submit Question"):
            with st.spinner("Processing your question..."):
                follow_up_result = run_model(image_path=st.session_state.last_image_path, question=question)
                if follow_up_result:
                    st.success("Follow-up processed successfully!")
                    st.text_area("Follow-up Output", follow_up_result, height=200)
