"""
Cluster method implemented by nikhilraghav29
from :
"Raghav, Nikhil and Gupta, Avisek and Sahidullah, Md and Das, Swagatam, Self-Tuning Spectral Clustering for Speaker Diarization, to appear in Proc. of ICASSP 2025"
"""
import numpy as np 
import scipy
import sklearn
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import SpectralClustering
from sklearn.cluster._kmeans import k_means
from sklearn.cluster import KMeans


class ClusterModule():

    def __init__(self, 
            min_speaker : int = 2,
            max_speaker : int = 10):
        
        self.min_speaker = min_speaker
        self.max_speaker = max_speaker


    def getEigenGaps(self, eig_vals):
        """Returns the difference (gaps) between the Eigen values.

        Arguments
        ---------
        eig_vals : list
            List of eigen values

        Returns
        -------
        eig_vals_gap_list : list
            List of differences (gaps) between adjacent Eigen values.
        """

        eig_vals_gap_list = []
        for i in range(len(eig_vals) - 1):
            gap = float(eig_vals[i + 1]) - float(eig_vals[i])
            # eig_vals_gap_list.append(float(eig_vals[i + 1]) - float(eig_vals[i]))
            eig_vals_gap_list.append(gap)

        return eig_vals_gap_list

    def get_sim_mat(self, X):
        """Returns the similarity matrix based on cosine similarities.

        Arguments
        ---------
        X : array
            (n_samples, n_features).
            Embeddings extracted from the model.

        Returns
        -------
        M : array
            (n_samples, n_samples).
            Similarity matrix with cosine similarities between each pair of embedding.
        """

        # Cosine similarities
        M = sklearn.metrics.pairwise.cosine_similarity(X, X)
        return M
    


    def get_laplacian(self, M):
        """Returns the un-normalized laplacian for the given affinity matrix.

        Arguments
        ---------
        M : array
            (n_samples, n_samples)
            Affinity matrix.

        Returns
        -------
        L : array
            (n_samples, n_samples)
            Laplacian matrix.
        """

        M[np.diag_indices(M.shape[0])] = 0
        D = np.sum(np.abs(M), axis=1)
        D = np.diag(D)
        L = D - M
        return L

    def get_spec_embs(self, L, k_oracle=4):
        """Returns spectral embeddings and estimates the number of speakers
        using maximum Eigen gap.

        Arguments
        ---------
        L : array (n_samples, n_samples)
            Laplacian matrix.
        k_oracle : int
            Number of speakers when the condition is oracle number of speakers,
            else None.

        Returns
        -------
        emb : array (n_samples, n_components)
            Spectral embedding for each sample with n Eigen components.
        num_of_spk : int
            Estimated number of speakers. If the condition is set to the oracle
            number of speakers then returns k_oracle.
        """
 
        lambdas, eig_vecs = scipy.linalg.eigh(L)

        # if params["oracle_n_spkrs"] is True:
        if k_oracle is not None:
            num_of_spk = k_oracle
        else:
            lambda_gap_list = self.getEigenGaps(lambdas[1 : self.max_speaker])

            num_of_spk = (
                np.argmax(
                    lambda_gap_list[
                        : min(self.max_speaker, len(lambda_gap_list))
                    ]
                )
                if lambda_gap_list
                else 0
            ) + 2

            if num_of_spk < self.min_speaker:
                num_of_spk = self.min_speaker
        #print("The number of estimated speakers:", num_of_spk)
        emb = eig_vecs[:, 0:num_of_spk]

        return emb, num_of_spk

    def cluster_embs(self, emb, k):
        """Clusters the embeddings using kmeans.

        Arguments
        ---------
        emb : array (n_samples, n_components)
            Spectral embedding for each sample with n Eigen components.
        k : int
            Number of clusters to kmeans.

        Returns
        -------
        self.labels_ : self
            Labels for each sample embedding.
        """
        _, self.labels_, _ = k_means(emb, k)

    def optimal_threshold_top_fourty(self, affinity_matrix):
        """
        Calculate the optimal threshold for each row of the affinity matrix using KMeans clustering and find
        the score corresponding to the top 40% of the higher cluster center.
        
        For each row, perform k-means clustering to separate the elements into two clusters, compute cluster statistics, 
        and calculate an optimal threshold. Additionally, find the similarity score corresponding to the top 40% 
        of the higher cluster center and append it to the list.
        
        Parameters:
        affinity_matrix (list of list or np.array): A 2D square affinity matrix of shape N x N.
        
        Returns:
        list: A list of N tuples, each containing the higher cluster center and the top 40% score.
        list: A list of N optimal thresholds, one for each row of the affinity matrix.
        """
        n = len(affinity_matrix)
        print("The number of embeddings are:",n)
        thresholds = []
        top_40_scores = []
        single_clusters = 0

        # Iterate over each row of the affinity matrix
        for i in range(n):
            #print("We are inside top 40 percent")
            # Step 1: Extract all elements in the row
            #row_elements = affinity_matrix[i]
            
            # Extract all elements in the row, excluding the diagonal element
            row_elements = np.concatenate((affinity_matrix[i][:i], affinity_matrix[i][i+1:]))

            #print("The type of row lwmwntsis", type(row_elements))            
            """ row_elemenrs is an ndarray"""
            # Step 2: Reshape row elements into a 2D array for clustering
            row_elements_reshaped = np.atleast_2d(row_elements).T

            # Step 3: Perform k-means clustering with 2 clusters
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(row_elements_reshaped)
            labels = kmeans.labels_

            # Step 4: Identify cluster centers
            cluster_centers = kmeans.cluster_centers_

            # Step 5: Determine which cluster has higher and lower center
            if cluster_centers[0] > cluster_centers[1]:
                C_label, I_label = 0, 1  # 0th cluster is the higher cluster
            else:
                C_label, I_label = 1, 0  # 1st cluster is the higher cluster

            # Step 6: Retrieve elements belonging to the higher cluster (C_label)
            C_elements = row_elements[labels == C_label]
            #print("The length of the ndarray of higher cluster is", len(C_elements))
            """ C_elements is an ndarray"""
            #print("The type of row elements is", type(C_elements)) 

            # Step 7: Sort the C_elements in decreasing order
            sorted_C_elements = np.sort(C_elements)[::-1]

            if len(C_elements) == 1:
                single_clusters += 1

            # Step 8: Find the score corresponding to the top 40% of sorted elements
            top_40_index = int((len(sorted_C_elements)-1) * 0.20)  # Find index of the top 40%
            top_40_score = sorted_C_elements[top_40_index]  # Score at the 40% mark

            # Step 9: Append only the top 40% score to top_40_scores
            top_40_scores.append(top_40_score)

            # Step 10: Retrieve elements belonging to the lower cluster (I_label)
            I_elements = row_elements[labels == I_label]

            # Step 11: Compute means and standard deviations for both clusters
            mu_C = np.mean(C_elements)
            sigma_C = np.std(C_elements)
            mu_I = np.mean(I_elements)
            sigma_I = np.std(I_elements)

            # Step 12: Calculate the optimal threshold using F-ratio normalization
            threshold = (mu_I * sigma_C + mu_C * sigma_I) / (sigma_I + sigma_C)

            # Step 13: Store the threshold for this row
            thresholds.append(threshold)
        print("Count of the number of clusters with 1 element:", single_clusters)
        return top_40_scores


    def p_pruning_score_row_wise(self, A, thresholds):
            """
            Refine the affinity matrix by zeroing less similar values based on row-wise thresholds.

            Arguments
            ---------
            A : array (n_samples, n_samples)
                Affinity matrix.
            thresholds : list of floats
                List of threshold values for each row. Retain elements greater than the row's threshold
                and set all other elements to zero.

            Returns
            -------
            A : array (n_samples, n_samples)
                Pruned affinity matrix where values are zeroed based on row-wise thresholds.
            """
            # Iterate over each row and apply the corresponding threshold
            #print("The max threshold is:", np.max(thresholds))
            for i in range(len(A)):
                #A[i] = np.where(A[i] > np.mean(thresholds), A[i], 0)
                A[i] = np.where(A[i] > thresholds[i], A[i], 0)
                #A[i] = np.where(A[i] > threshold, A[i], 0) # Upper traingular 
            
            return A

    def do_spec_clust_row_wise(self, X, k_oracle, p_val):
            """Function for spectral clustering.

            Arguments
            ---------
            X : array
                (n_samples, n_features).
                Embeddings extracted from the model.
            k_oracle : int
                Number of speakers (when oracle number of speakers).
            p_val : float
                p percent value to prune the affinity matrix.
            """
            
            # Similarity matrix computation
            sim_mat = self.get_sim_mat(X)
            #print("The type of Sim-mat is:", type(sim_mat))
            # row-wise thresholds computation using F-ratio

            #Upper_A = self.lower_triangular_to_vector(sim_mat)
            #threshold = self.optimal_threshold(sim_mat)
            #thresholds = self.eer_delta(sim_mat)                                # EER-Delta
            thresholds = self.optimal_threshold_top_fourty(sim_mat)            # SC-pNA
            pruned_sim_mat = self.p_pruning_score_row_wise(sim_mat, thresholds) # SC-pNA, EER-Delta
            #pruned_sim_mat = self.p_pruning_score_row_wise(sim_mat, threshold) 

            # Symmetrization
            #sym_pruned_sim_mat = self.symmetrization(pruned_sim_mat)
            sym_pruned_sim_mat = 0.5 * (pruned_sim_mat + pruned_sim_mat.T)

            # Laplacian calculation
            laplacian = self.get_laplacian(sym_pruned_sim_mat)

            # Get Spectral Embeddings
            emb, num_of_spk = self.get_spec_embs(laplacian, k_oracle)

            # Perform clustering
            #print(num_of_spk)
            # I have commented the following line to apply the GMM for clustering instead of k-means
            self.cluster_embs(emb, num_of_spk)
            print("Estimated number of speakers:",num_of_spk)
            #self.do_GMM(emb,num_of_spk)
