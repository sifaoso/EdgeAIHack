from deepface import DeepFace
from qdrant_client.models import PointStruct
from qdrant_client import QdrantClient

# Initialisation du client Qdrant
client = QdrantClient(url="http://localhost:6333")


def generate_embeddings(img_path: str, model_name: str = "Facenet512", detector_backend: str = "retinaface"):
    """
    Génère les embeddings à partir d'une image contenant un ou plusieurs visages.

    Args:
        img_path (str): Chemin vers l'image.
        model_name (str): Modèle utilisé pour les embeddings (par défaut : "Facenet512").
        detector_backend (str): Backend utilisé pour détecter les visages (par défaut : "retinaface").

    Returns:
        list: Liste des embeddings générés pour chaque visage détecté.
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


def add_profile_to_qdrant(collection_name: str, embedding: list, profile_id: int, name: str, surname: str):
    """
    Ajoute un profil à une collection Qdrant.

    Args:
        collection_name (str): Nom de la collection.
        embedding (list): Vecteur d'embedding du profil.
        profile_id (int): Identifiant unique du profil.
        name (str): Prénom de l'utilisateur.
        surname (str): Nom de famille de l'utilisateur.
    """
    try:
        operation_info = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[
                PointStruct(
                    id=profile_id,
                    vector=embedding,
                    payload={
                        "name": name,
                        "surname": surname
                    },
                ),
            ]
        )
        print(f"Profil ajouté avec succès : {name} {surname} (ID: {profile_id})")
        return operation_info
    except Exception as e:
        print(f"Erreur lors de l'ajout du profil {name} {surname} : {e}")


# Liste des profils avec chemins vers les photos
profiles = [
    {
        "id": 1,
        "name": "Clément",
        "surname": "Tableau",
        "img_path": "Clem.jpg"
    },
    {
        "id": 2,
        "name": "Jérémy",
        "surname": "Billuart",
        "img_path": "Jerem.jpg"
    }, {
        "id": 3,
        "name": "Mathis",
        "surname": "Champagne",
        "img_path": "Mathis.jpg"
    }, {
        "id": 4,
        "name": "Romeo",
        "surname": "Correc",
        "img_path": "Romeo.jpg"
    }
]

# Ajout des profils à Qdrant après génération des embeddings
collection_name = "Profile"
for profile in profiles:
    embeddings = generate_embeddings(profile["img_path"])
    if embeddings:
        for i, embedding in enumerate(embeddings):
            add_profile_to_qdrant(
                collection_name=collection_name,
                embedding=embedding,
                profile_id=int(f"{profile['id']}{i}"),  # ID unique pour chaque visage
                name=profile["name"],
                surname=profile["surname"]
            )
    else:
        print(f"Pas d'embeddings générés pour {profile['name']} {profile['surname']}")
