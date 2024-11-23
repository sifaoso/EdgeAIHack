from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
import torch
import os

synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)


def create_wav(txt):
    i = 0
    file_name = "speech_" + str(i) + '.wav'
    while file_name in os.listdir("vocal"):
        i += 1
        file_name = "speech_" + str(i) + '.wav'
    speech = synthesiser(txt, forward_params={"speaker_embeddings": speaker_embedding})

    sf.write("./vocal/" + file_name, speech["audio"], samplerate=speech["sampling_rate"])

    return (file_name)
