import os 
from tqdm import tqdm 
from speechbrain.inference.speaker import EncoderClassifier
import torch
from typing import List, Tuple
import numpy as np 
import logging 
from pathlib import Path

from murmure.utils.logger import get_logger
from murmure.types.core import TranscriptionOutput

from .clustering import ClusterModule

logging.getLogger("speechbrain").setLevel(logging.WARNING)
logger = get_logger(__name__)


class SpeakerDiarizationModule():

    def __init__(self, 
                 model_dir_path : Path,
                 min_speakers=2, 
                 max_speakers=8,
                 ):

        self.model_dir_path = os.path.join(str(model_dir_path), 'speaker_embeddings')
        os.makedirs(self.model_dir_path, exist_ok=True)

        self.model = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb", savedir=self.model_dir_path)

        self.embeddings_storage = []

        self.min_speakers = min_speakers
        self.max_speakers = max_speakers


        self.cluster_module = ClusterModule(
            min_speaker=min_speakers,
            max_speaker=max_speakers
        )
   

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

        for segment in tqdm(segments, desc="computing embeddings"):

                
                start_idx = int(segment.start * sample_rate)
                end_idx = int(segment.end * sample_rate)

                audio_slice = audio[start_idx:end_idx]  

                audio_slice = audio_slice / torch.max(torch.abs(audio_slice))

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


    def process_audio(self, audio: np.ndarray, 
                      transcription_segments: List[TranscriptionOutput], 
                      sample_rate: int = 16000) -> List[TranscriptionOutput]:
     
        logger.info(' [STARTING] ')

        embeddings = self.compute_embeddings(audio, 
                                             transcription_segments, 
                                             sample_rate)
        
        # speaker_labels = self.cluster_speakers(embeddings,
        #                                        visualize_nb_speaker_probs = True)

        self.cluster_module.do_spec_clust_row_wise(
            X=embeddings,
            k_oracle=None,
            p_val=None
        )

        speaker_labels = self.cluster_module.labels_


        for idx, segment in enumerate(transcription_segments):
            segment.speaker_id = speaker_labels[idx]


        logger.info(' [FINISHING] ')


        return transcription_segments






    # def cluster_speakers(self, 
    #                      embeddings: np.ndarray,
    #                      visualize_nb_speaker_probs : bool = True) -> np.ndarray:
    #     """
    #     Clusterise les embeddings avec Agglomerative Clustering.
    #     :param embeddings: Liste des embeddings des segments.
    #     :return: Liste des labels des speakers assignés à chaque segment.
    #     """
    #     # Calcul de la matrice de distances entre les embeddings
    #     # distance_matrix = squareform(pdist(embeddings, metric='euclidean'))

    #     # Définition du nombre de clusters basé sur un seuil de distance


    #     best_score = -1
    #     scores = []

   
    #     distance_matrix = cosine_distances(embeddings)
    #     cosine_dist_vector  = squareform(distance_matrix)

    #     Z = linkage(cosine_dist_vector, method='complete')

    #     for n_clusters in range(self.min_speakers, self.max_speakers + 1):
            
    #         clustering_model = AgglomerativeClustering(n_clusters=n_clusters, 
    #                                                 linkage= 'complete',
    #                                                 metric='precomputed', 
    #                                                 )
            
    #         labels = clustering_model.fit_predict(distance_matrix)
    
    #         # # Compute silhouette score (higher is better)
    #         if len(set(labels)) > 1:  # Avoid single-cluster case

    #             score = silhouette_score(distance_matrix, labels, metric='precomputed')
    #             scores.append(score)

    #             if score > best_score:
    #                 best_score = score
    #                 cluesters_nb = n_clusters




    #     labels = fcluster(Z, t=cluesters_nb, criterion='maxclust')



    #     # Plot silhouette scores to visualize best number of clusters
    #     if visualize_nb_speaker_probs : 

    #         plt.plot(range(self.min_speakers, self.max_speakers + 1), scores, marker='o')
    #         plt.xlabel("Number of clusters")
    #         plt.ylabel("Silhouette Score")
    #         plt.title("Silhouette Score vs. Number of Clusters")
    #         plt.show()

    #     return labels
    