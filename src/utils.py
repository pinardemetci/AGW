import numpy as np
import scipy as sp
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier

def unit_normalize(data, norm="l2", bySample=True):
	"""
	Default norm used is l2-norm. Other options: "l1", and "max"
	If bySample==True, then we independently normalize each sample. If bySample==False, then we independently normalize each feature
	"""
	assert (norm in ["l1","l2","max"]), "Norm argument has to be either one of 'max', 'l1', or 'l2'."

	if bySample==True:
		axis=1
	else:
		axis=0

	return normalize(data, norm=norm, axis=axis) 

def zscore_standardize(data):
	scaler=StandardScaler()
	scaledData=scaler.fit_transform(data)
	return scaledData

def build_kNN(data, num_neighbors, mode="distance", metric="minkowski"):
	assert (mode in ["connectivity", "distance"]), "Norm argument has to be either one of 'connectivity', or 'distance'. "
	if mode=="connectivity":
		include_self=True
	else:
		include_self=False
	graph=kneighbors_graph(data, num_neighbors, mode=mode, metric=metric, include_self=include_self)
	return csr_matrix(graph)

def get_shorted_distances(graph):
	shortestPath_data= dijkstra(csgraph= graph, directed=False, return_predecessors=False)
	shortestPath_max= np.nanmax(shortestPath_data[shortestPath_data != np.inf])
	shortestPath_data[shortestPath_data > shortestPath_max] = shortestPath_max
	shortestPath_data=shortestPath_data/shortestPath_data.max()
	return shortestPath_data

def get_spatial_distance_matrix(data, metric="eucledian"):
	Cdata= sp.spatial.distance.cdist(data,data,metric=metric)
	return Cdata/Cdata.max()
