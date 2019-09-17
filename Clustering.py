'''
@author: Erkan
'''
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.cluster as sc_alg
from mpl_toolkits.mplot3d import Axes3D
from dask.array.chunk import argmax

class clustering_methods():

    def __init__(self, n_bins=3, F=1-1e-5):
        """
            Read and then preprocess 1D radar echoes. The Echoes are 
            transformed to magnitudes and then they arevalidated by 
            removing clutters. The class provides 3 clustering methods: 
            kmeans++, spectra and agglemoretive hierarchical clustering 
            methods. Those clustering methods work on the l2 normalized 
            histograms of validated echoes.
            @Arguments:
            F is the percentile point, i.e., P(x<val_Th)<F is the total 
            probability under a Rayleigh distribution before threshold.
            n_bins is the number of bins of histograms. 
        """
        
        # available clustering methods
        self._kmeans = 1
        self._spectral = 2
        self._hac = 3
        
        # percentile point
        self.F = F
        
        self.n_bins = n_bins

        # directory of data samples
        self.dir_path = os.path.dirname(os.path.realpath(__file__))
    
        # path to data samples
        self.dataset_path = os.path.join(self.dir_path,"test")
        
        # list of data samples
        _, _, self.sample_files =  os.walk(self.dataset_path).__next__()
        
        self.preprocess()
        
    def preprocess(self):
        """
            Remove clutter/noise from echo. Clutter is modeled as
            Rayleigh distribution. Each sample is validated by removing
            clutter. The validation thresholds are data-dependent as 
            they are determined according to the given percentile point, 
            That is, the probability mass above the computed threshold 
            is equal to 1-F. After removing clutter/noise DBSCAN finds 
            a high density of the echoes connected to the highest peak.
        """

        # magnitudes of echos
        self.dataset = []
        # validation threshold to remove clutter
        self.val_Th = []
        
        # min and max magnitudes in the dataset
        self.min_mag = 1
        self.max_mag = -1
        
        for f in self.sample_files:
            
            f_path = "{0}/{1}".format(self.dataset_path,f)
                
            sample = np.load(f_path)
            
            # assumption: mean is not a robust estimator and
            # echo mostly consists of noise or clutter returns 
            mag_sample = np.abs(sample)
            
            self.dataset.append(mag_sample)
            
            # ML estimation of scale parameter 
            sigma = np.sqrt(np.mean(mag_sample**2)/2)
            
            # the upper tail probability Q=1-F of Rayleigh distribution
            val_Th = sigma*np.sqrt(-2*np.log(1-self.F))
            self.val_Th.append(val_Th)
            
            min_sample = min(mag_sample)
            max_sample = max(mag_sample)
            
            if (min_sample < self.min_mag):
                self.min_mag = min_sample
                
            if (max_sample > self.max_mag):
                self.max_mag = max_sample
                
        
        self.dataset = np.array(self.dataset)
        self.val_Th = np.array(self.val_Th)
        
        # remove clutter/noise and compute l2-norm histograms
        self.validation_histogram()
        
    def validation_histogram(self):
        """
            Validate input signals using the given thresholds
            and compute the weighted histograms of magnitudes
            of samples in the dataset.
        """
        
        # validated part of echoes
        self.val_sample = []
        # indices of validated samples
        self.val_idx = []
        # l2-norm histograms of validated signal
        self.val_hist = []
        
        for i,mag_samples in enumerate(self.dataset):
            
            # return validated samples with their indices 
            val_idx_sample = list(filter(lambda x: x[1] > self.val_Th[i], enumerate(mag_samples)))
            
            self.val_sample.append([])
            self.val_idx.append([])

            for v in val_idx_sample:
                self.val_sample[i].append(v[1])
                self.val_idx[i].append(v[0]) 
                
            #  focus in a high density region around the highest peak
            self.DBSCAN(i)
                
            # compute the weighted histograms of the validated parts of each echoes 
            hist_weights = (self.val_sample[i] - self.min_mag) / self.max_mag
            bins = np.histogram(self.val_sample[i], bins=self.n_bins, weights=hist_weights, 
                                range=(self.min_mag,self.max_mag), density=True)[0]
                   
            # l2 normalize histograms to convert them into unit vectors             
            l2_norm_bins = bins/np.sqrt(np.sum(bins*bins))
            
            self.val_hist.append(l2_norm_bins)
                
        self.val_hist = np.array(self.val_hist)
        
    def DBSCAN(self, iv):
        """
            Density-based spatial clustering of applications with 
            noise (DBSCAN) is used to separate interval of the highest 
            density radar echo (core sample and reachable samples
            in its neighborhood) due to an object from those reside
            in low density intervals. Those validated magnitudes in 
            low-density intervals are removed so that the remaining
            is only the samples around the highest peak.
            @Arguments:
            iv is the index of the validated echo
        """

        dbscan = sc_alg.DBSCAN(eps=2,min_samples=3).fit(np.reshape(self.val_idx[iv],(-1,1)))
        
        # find the label of the highest peak and remove other clusters and noise
        max_idx = argmax(self.val_sample[iv]) 
        
        labels = dbscan.labels_          
        
        # label of the highest validated magnitude
        max_l = labels[max_idx]
        
        # find indices of those validated magnitudes in low density intervals
        ld_idx = [i for i,l in enumerate(labels) if not(l==max_l)]
        
        # remove those validated magnitudes in density intervals
        self.val_sample[iv] = np.delete(self.val_sample[iv],ld_idx)
            
        self.val_idx[iv] = np.delete(self.val_idx[iv],ld_idx)
                   
    def k_means(self):
        """
           kmeans is an assignment based clustering which starts
           with assigning points to given centroids and then it 
           iteratively updates the centroids.
           As opposed to the standard kmeans, kmeans++ can select
           initial cluster centers. This accelerates the process
           of finding the optimum solution.
           Like kmeans, kmeans++ operates on euclidean distance, 
           and it assumes that clusters are compact convex and 
           isotropic (i.e.,same variances along each directions). 
           Therefore it works poorly on clusters that are not in 
           the shape of sphere. However, it results in tighter
           clusters compared to HAC and spectral clustering method. 
           Here, k_means++ to cluster l2 normalized histograms into
           3 classes. 
           @Returns: labels of data points.
        """
        self.kmeans = sc_alg.KMeans(n_clusters=3, init='k-means++', n_init=50,
                               max_iter=1000, tol=1-10, random_state=0).fit(self.val_hist)
                               
        self.visualization_clusters(self._kmeans)
        
        print("labels from kmeans++ clustering:\n{0}".format(self.kmeans.labels_)) 
        
        return self.kmeans.labels_
    
    def spectral_clustering(self):
        """
            Spectral clustering is top-down approach which starts with
            one big cluster and gradually divides it into subclusters
            by removing the weakest connections.
            Spectral clustering finds clusters based on the connectivity
            data points. In spectral clustering data points are treated
            as nodes in a graph. It partitions this graph into clusters
            using edges connecting them. 
            Spectral clustering involves 3 steps:
                1. Compute an affinity matrix (A) to represent data as 
                   a graph.
                2. Compute graph laplacian (D-A) where D is degree matrix.
                3. Create clusters by egienvalue decomposition of graph 
                   laplacian. Eigenvalues indicate how tightly clusters are
                   formed.
            As opposed to kmeans, no assumption is made about the shape 
            of clusters.It is fast for small datasets because of eigen
            value decomposition.
            @Returns: labels of data points.
        """
        self.spectral= sc_alg.SpectralClustering(n_clusters=3, assign_labels='discretize', 
                                                 affinity='chi2', random_state=0).fit(self.val_hist)
                                                 
        self.visualization_clusters(self._spectral)  
        
        print("labels from spectral clustering:\n{0}".format(self.spectral.labels_))                              
                                               
        return self.spectral.labels_
    
    def agglomerative_hierarchical(self):
        """
            Agglomerative Hierarchical clustering (HAC) is a bottom 
            up clustering method. Initially, each data points are
            treated as single clusters. Then, the HAC successively
            merge pairs of the closest clusters into larger one.
            The HAC forms a tree where the root is the global cluster
            that gathers all points. Thus, it allows us to see clusters
            inside clusters. 
            Hierarchical clustering does not require us to specify the 
            number of clusters. However, We can select the number of 
            clusters we want at the end or we can decide the number of
            clusters by looking at the tree. Shortcomings of the HAC
            are: i) the previous step cannot be undo, ii) it is not 
            efficient for large data.
           @Returns: labels of data points.
        """
        self.hac = sc_alg.AgglomerativeClustering(n_clusters=3, affinity='euclidean', 
                                                  linkage='ward', compute_full_tree=False).fit(self.val_hist)

                               
        self.visualization_clusters(self._hac)
        
        print("labels from HAC:\n{0}".format(self.hac.labels_)) 
        
        return self.kmeans.labels_
        
           
    def visualization_clusters(self, cmethod):
        """
           visualization of 3 clusters with labels of each data points.
           3 Different colors (red, blue and green) are used as labels.
           @Arguments:
           cmethod is the clustering method.
           @NOTE: For different clustering methods, colors may not indi-
           cate same clusters.  
        """
        
        # colors used to label data points
        colors = ["red","blue","yellow"]
        
        legends = [None, None, None]
        
        fig = plt.figure(cmethod)
        ax_cluster = Axes3D(fig)
        
        if (cmethod == self._kmeans):
            labels = self.kmeans.labels_
            smethod = "kmeans"
        elif (cmethod == self._spectral):
            labels = self.spectral.labels_
            smethod = "Spectral clustering"
        else:
            labels = self.hac.labels_
            smethod = "HAC"
            
        for i,v in enumerate(self.val_hist):
            # select cluster color/label
            l = labels[i]
            
            if (l == 0):
                cls = 'class 1'
            elif (l == 1):
                cls = 'class 2'
            else:
                cls = 'class 3'
                
            if (legends[l] == None):
                ax_cluster.scatter(v[0], v[1], v[2], marker='o', color=colors[l], s=20, label=cls)
                legends[l] = l
            else:
                ax_cluster.scatter(v[0], v[1], v[2], marker='o', color=colors[l], s=20)
                
            ax_cluster.text(v[0], v[1], v[2], '{0}'.format(i), size=5, zorder=1, color='k') 
            
        ax_cluster.set_title("{0}".format(smethod))    
        ax_cluster.set_xlabel('X')
        ax_cluster.set_ylabel('Y')
        ax_cluster.set_zlabel('Z')
        ax_cluster.legend()
                     
    def draw_save_valid(self):
        """
           draw validated data points of each echoes and
           their normalized histograms, and then save their
           figures.
        """
        
        fig_visual = plt.figure()
        
        ax_echo = fig_visual.add_subplot(211)
        ax_hist = fig_visual.add_subplot(212)
        
        save_path = os.path.join(self.dir_path,'vis_valid')
        
        if not(os.path.isdir(save_path)):
            os.makedirs(save_path)
        
        for i,f in enumerate(self.sample_files):
            
            ax_echo.set_title("Validated radar signal:{0}".format(i))
            ax_echo.set_xlabel("Sample No")
            ax_echo.set_ylabel("Magnitude")
            ax_echo.ticklabel_format(axis='y',style='sci', scilimits=(0,0))
            ax_echo.autoscale(True,'y',True)
            ax_echo.plot(self.val_idx[i], self.val_sample[i], marker='.')
            
            ax_hist.set_title("L2 normalized histogram")
            delta_x = (self.max_mag - self.min_mag)/self.n_bins
            x_ticks = np.arange(self.min_mag, self.max_mag, delta_x)
            ax_hist.bar(x_ticks, self.val_hist[i], width=delta_x)
            
            plt.tight_layout()
            
            plt.savefig(os.path.join(save_path,os.path.splitext(f)[0]))
            
            ax_echo.cla();
                        
            ax_hist.cla();
            

if __name__ == '__main__':
    
    c = clustering_methods()
    
    c.draw_save_valid()
    
    c.k_means()
    
    c.spectral_clustering()
    
    c.agglomerative_hierarchical()
    
    plt.show()
    

    
    
            

