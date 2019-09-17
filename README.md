# Clustering-1D-Radar-Signal

The problem at hand is to cluster a dataset of 1D radar signals into three different groups. The radar
signals were collected in the presence or absence of a weapon in front of a radar antenna. To solve the
given problem the hint stating that “focus on around the highest peak of the signal for the best
clustering” was used. Instead of defining a constant and intuitive interval around the highest peaks,
adjustable and data-dependent intervals were computed for each radar echoes independently. For this
purpose, noisy parts of radar echoes are identified by using Rayleigh distribution of clutter modeling.
That is, the amplitude of clutter echo is assumed to be Rayleigh distributed.

After removing clutter returns, the remaining is significant returned echoes. To label the interval around
the highest peaks the density-based spatial clustering of applications with noise (DBSCAN) is employed.
The DBSCAN identifies clutter echoes that could not be filtered out by the validation threshold and also
it results in a number of high density clusters. To focus around the highest peak, the label assigned to
the highest peak in each radar signal is found. The validated echoes belonging to that class are in the
neighborhood of the highest peak and used as meaningful input signal.

Similar to the bag of words modeling, the magnitudes of the highest peak and other relevant echoes in
its local neighborhood from the DBSCAN (i.e., features) are represented by l2-normalized histograms. In
clustering methods, those histograms are used as feature descriptors. That is, distance metrics like
Euclidean distance and chi-square distance use feature descriptors to group input signals.
