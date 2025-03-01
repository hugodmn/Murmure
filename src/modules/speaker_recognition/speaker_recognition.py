from speechbrain.inference.speaker import EncoderClassifier
from .type import AudioSegment
import torch
from typing import List
import numpy as np 
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist, squareform

class SpeakerRecogntionModule():

    def __init__(self, 
                 min_speakers=2, 
                 max_speakers=10, 
                 threshold=0.5):

        self.model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir="models")

        self.embeddings_storage = []

        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.threshold = threshold

    def compute_embeddings(self, 
                           audio: np.ndarray, 
                           whisper_segments: list, 
                           sample_rate: int = 16000) -> np.ndarray:
        """
        Extrait les embeddings des segments audio en batch.
        """
        audio = torch.tensor(audio)
        embeddings_list = []

        # Traitement segment par segment
        with torch.no_grad():
            for whisper_segment in whisper_segments:
                start_idx = int(whisper_segment['start'] * sample_rate)
                end_idx = int(whisper_segment['end'] * sample_rate)
                audio_slice = audio[start_idx:end_idx]  # Ajout d'une dimension batch

                # Calcul de l'embedding du segment
                speaker_embedding = self.model.encode_batch(audio_slice).cpu().numpy()
                print(speaker_embedding.size)
                embeddings_list.append(speaker_embedding)

        return np.array(embeddings_list)
        


    def cluster_speakers(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Clusterise les embeddings avec Agglomerative Clustering.
        :param embeddings: Liste des embeddings des segments.
        :return: Liste des labels des speakers assignés à chaque segment.
        """
        # Calcul de la matrice de distances entre les embeddings
        # distance_matrix = squareform(pdist(embeddings, metric='euclidean'))

        # Définition du nombre de clusters basé sur un seuil de distance
        clustering_model = AgglomerativeClustering(n_clusters=None, 
                                                   linkage='ward', 
                                                   distance_threshold=self.threshold)

        speaker_labels = clustering_model.fit_predict(embeddings)

        return speaker_labels
    

    def process_audio(self, audio: np.ndarray, whisper_segments: list, sample_rate: int = 16000) -> List[AudioSegment]:
     

        embeddings = self.compute_embeddings(audio, whisper_segments, sample_rate)
        speaker_labels = self.cluster_speakers(embeddings)

        audio_segments_list = []
        for idx, whisper_segment in enumerate(whisper_segments):
            audio_segments_list.append(AudioSegment(
                speaker_id=int(speaker_labels[idx]),
                starting_time=whisper_segment['start'],
                ending_time=whisper_segment['end']
            ))

        return audio_segments_list


                



if __name__ == '__main__' :

    Module = SpeakerRecogntionModule()

