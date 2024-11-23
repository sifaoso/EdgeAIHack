from deepface import DeepFace
from qdrant_client.models import Filter
from qdrant_client import QdrantClient

# Initialisation du client Qdrant
client = QdrantClient(url="http://localhost:6333")


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

    if embeddings:
        # Pour chaque embedding détecté dans l'image
        for i, embedding in enumerate(embeddings):
            print(f"\nRecherche pour le visage {i + 1}:")
            results = query_similar_profiles(embedding, collection_name="Profile", threshold=0.5)
            if results:
                for result in results:
                    print(f"Profil similaire trouvé : {result.payload}, score : {result.score}")
                    return (result.payload, result.score)
            else:
                print("Aucun résultat au-dessus du seuil.")
    else:
        print("Aucun visage détecté ou erreur lors de la génération des embeddings.")
    return None
