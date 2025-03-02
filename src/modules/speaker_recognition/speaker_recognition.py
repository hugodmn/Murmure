from speechbrain.inference.speaker import EncoderClassifier
from .type import AudioSegment
import torch
from typing import List, Tuple
import numpy as np 
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
from ..types import TranscriptionOutput


class SpeakerRecogntionModule():

    def __init__(self, 
                 min_speakers=2, 
                 max_speakers=5, 
                 threshold=0.5):

        self.model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=".models")

        self.embeddings_storage = []

        self.similarity = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)

        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.threshold = threshold

    def compute_embeddings(self, 
                           audio: np.ndarray, 
                           segments : List[TranscriptionOutput], 
                           sample_rate: int = 16000) -> np.ndarray:
        """
        Extrait les embeddings des segments audio en batch.
        """

        audio = torch.tensor(audio)
        #audio_tensors_list = []
        embeddings_list = []

        for segment in segments:

                start_idx = int(segment.start * sample_rate)
                end_idx = int(segment.end * sample_rate)

          
                audio_slice = audio[start_idx:end_idx]  
            
                speaker_embedding = self.model.encode_batch(audio_slice).squeeze().cpu().numpy()
                embeddings_list.append(speaker_embedding)


        return np.array(embeddings_list)



    def pad_tensors(self, tensor_list : List[torch.tensor]) -> Tuple[torch.tensor,torch.tensor]:

        max_length = max(tensor.shape[-1] for tensor in tensor_list)


        padded_tensors = torch.stack([
        torch.nn.functional.pad(tensor.squeeze(), (-1, max_length - tensor.shape[-1]), 'constant', 0) 
        for tensor in tensor_list
    ])

        wav_lens = torch.tensor([tensor.shape[0] / max_length for tensor in tensor_list], dtype=torch.float32)

        return padded_tensors, wav_lens


    def cluster_speakers(self, 
                         embeddings: np.ndarray,
                         visualize_nb_speaker_probs : bool = False) -> np.ndarray:
        """
        Clusterise les embeddings avec Agglomerative Clustering.
        :param embeddings: Liste des embeddings des segments.
        :return: Liste des labels des speakers assignés à chaque segment.
        """
        # Calcul de la matrice de distances entre les embeddings
        # distance_matrix = squareform(pdist(embeddings, metric='euclidean'))

        # Définition du nombre de clusters basé sur un seuil de distance


        best_score = -1
        scores = []

   
        distance_matrix = cosine_distances(embeddings)



        for n_clusters in range(self.min_speakers, self.max_speakers + 1):
            
            clustering_model = AgglomerativeClustering(n_clusters=n_clusters, 
                                                    linkage= 'average',
                                                    metric='precomputed', 
                                                    )
            
            labels = clustering_model.fit_predict(distance_matrix)
    
            # Compute silhouette score (higher is better)
            if len(set(labels)) > 1:  # Avoid single-cluster case

                score = silhouette_score(distance_matrix, labels, metric='precomputed')
                scores.append(score)

                if score > best_score:
                    best_score = score
                    speaker_labels = labels

        # Plot silhouette scores to visualize best number of clusters

        if visualize_nb_speaker_probs : 

            plt.plot(range(self.min_speakers, self.max_speakers + 1), scores, marker='o')
            plt.xlabel("Number of clusters")
            plt.ylabel("Silhouette Score")
            plt.title("Silhouette Score vs. Number of Clusters")
            plt.show()

        return speaker_labels
    

    def process_audio(self, audio: np.ndarray, 
                      transcription_segments: List[TranscriptionOutput], 
                      sample_rate: int = 16000) -> List[TranscriptionOutput]:
     
      
        embeddings = self.compute_embeddings(audio, 
                                             transcription_segments, 
                                             sample_rate)
        
        speaker_labels = self.cluster_speakers(embeddings,
                                               visualize_nb_speaker_probs = False)



        for idx, segment in enumerate(transcription_segments):
            segment.speaker_id = speaker_labels[idx]


        return transcription_segments


                



if __name__ == '__main__' :

    Module = SpeakerRecogntionModule()

