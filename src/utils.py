
# -*- coding: utf-8 -*-
import numpy as np
import ot
import optim
import argparse
import pickle
import os
import datetime,dateutil
import sys
import random 
from bregman import sinkhorn_scaling
import logging
from scipy.spatial.distance import cdist
# For computing graph distances:
from scipy.sparse.csgraph import dijkstra
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist

def barycentric_projection(source, target, couplingMatrix):
	"""
	Given: data in the target space, data in the source space, a coupling matrix learned via Gromow-Wasserstein OT
	Returns: source (target) matrix transported onto the target (source)
	"""
	P = (couplingMatrix.T/couplingMatrix.sum(1)).T
	transported_data= np.matmul(P, target)
	return transported_data

def compute_graph_distances(data, n_neighbors=5, mode="distance", metric="correlation"):
	graph=kneighbors_graph(data, n_neighbors=n_neighbors, mode=mode, metric=metric, include_self=True)
	shortestPath=dijkstra(csgraph= csr_matrix(graph), directed=False, return_predecessors=False)
	max_dist=np.nanmax(shortestPath[shortestPath != np.inf])
	shortestPath[shortestPath > max_dist] = max_dist

	return np.asarray(shortestPath)

class StopError(Exception):
	pass

def init_matrix_GW(C1,C2,p,q,loss_fun='square_loss'):
	""" 
	"""        
	if loss_fun == 'square_loss':
		def f1(a):
			return a**2 

		def f2(b):
			return b**2

		def h1(a):
			return a

		def h2(b):
			return 2*b

	constC1 = np.dot(np.dot(f1(C1), p.reshape(-1, 1)),
					 np.ones(len(q)).reshape(1, -1))
	constC2 = np.dot(np.ones(len(p)).reshape(-1, 1),
					 np.dot(q.reshape(1, -1), f2(C2).T))
	constC=constC1+constC2
	hC1 = h1(C1)
	hC2 = h2(C2)

	return constC,hC1,hC2

def tensor_product(constC,hC1,hC2,T):
	""" 
	"""
	A=-np.dot(hC1, T).dot(hC2.T)
	tens = constC+A

	return tens

def gwloss(constC,hC1,hC2,T):
	"""
	"""
	tens=tensor_product(constC,hC1,hC2,T) 
			  
	return np.sum(tens*T) 


def gwggrad(constC,hC1,hC2,T):
	"""
	"""
	return 2*tensor_product(constC,hC1,hC2,T) 

def gw_lp(C1,C2,p,q,loss_fun='square_loss',alpha=1,amijo=True,**kwargs): 
	"""
	"""
	constC,hC1,hC2=init_matrix_GW(C1,C2,p,q,loss_fun)
	M=np.zeros((C1.shape[0],C2.shape[0]))
	
	G0=p[:,None]*q[None,:]
	
	def f(G):
		return gwloss(constC,hC1,hC2,G)
	def df(G):
		return gwggrad(constC,hC1,hC2,G)
 
	return optim.cg(p,q,M,alpha,f,df,G0,amijo=amijo,constC=constC,C1=C1,C2=C2,**kwargs)
	
def fgw_lp(M,C1,C2,p,q,loss_fun='square_loss',alpha=1,amijo=True,G0=None,**kwargs): 
	"""
	"""
	constC,hC1,hC2=init_matrix_GW(C1,C2,p,q,loss_fun)
	
	if G0 is None:
		G0=p[:,None]*q[None,:]
	
	def f(G):
		return gwloss(constC,hC1,hC2,G)
	def df(G):
		return gwggrad(constC,hC1,hC2,G)
 
	return optim.cg(p,q,M,alpha,f,df,G0,amijo=amijo,C1=C1,C2=C2,constC=constC,**kwargs)


def create_log_dir(FLAGS):
	now = datetime.datetime.now(dateutil.tz.tzlocal())
	timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

	log_dir = FLAGS.log_dir + "/" + str(sys.argv[0][:-3]) + "_" + timestamp 
	print(log_dir)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)

	# save command line arguments
	with open(log_dir + "/hyperparameters_" + timestamp + ".csv", "w") as f:
		for arg in FLAGS.__dict__:
			f.write(arg + "," + str(FLAGS.__dict__[arg]) + "\n")

	return log_dir

def unique_repr(dictio,type_='normal'):
	"""Compute a hashable unique representation of a list of dict with unashable values"""
	if 'normal':
		t = tuple((k, dictio[k]) for k in sorted(dictio.keys()))
	if 'not_normal':
		t=()
		for k in sorted(dictio.keys()):
			if not isinstance(dictio[k],list):
				t=t+((k, dictio[k]),)
			else: #suppose list of dict
				listechanged=[]
				for x in dictio[k]:
					for k2 in sorted(x.keys()):
						if not isinstance(x[k2],dict):
							listechanged.append((k2,x[k2]))
						else:
							listechanged.append((k2,tuple((k3, x[k2][k3]) for k3 in sorted(x[k2].keys()))))
				tupletoadd=((k, tuple(listechanged)),)
				t=t+tupletoadd
	return t

def save_obj(obj, name,path='obj/' ):
	try:
		if not os.path.exists(path):
			print('Makedir')
			os.makedirs(path)
	except OSError:
		raise
	with open(path+ name + '.pkl', 'wb') as f:
		pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name,path):
	with open(path + name, 'rb') as f:
		return pickle.load(f)

def indices_to_one_hot(number, nb_classes,label_dummy=-1):
	"""Convert an iterable of indices to one-hot encoded labels."""
	
	if number==label_dummy:
		return np.zeros(nb_classes)
	else:
		return np.eye(nb_classes)[number]

def dist(x1, x2=None, metric='sqeuclidean'):
	"""Compute distance between samples in x1 and x2 using function scipy.spatial.distance.cdist
	Parameters
	----------
	x1 : np.array (n1,d)
		matrix with n1 samples of size d
	x2 : np.array (n2,d), optional
		matrix with n2 samples of size d (if None then x2=x1)
	metric : str, fun, optional
		name of the metric to be computed (full list in the doc of scipy),  If a string,
		the distance function can be 'braycurtis', 'canberra', 'chebyshev', 'cityblock',
		'correlation', 'cosine', 'dice', 'euclidean', 'hamming', 'jaccard', 'kulsinski',
		'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
		'sokalmichener', 'sokalsneath', 'sqeuclidean', 'wminkowski', 'yule'.
	Returns
	-------
	M : np.array (n1,n2)
		distance matrix computed with given metric
	"""
	if x2 is None:
		x2 = x1

	return cdist(x1, x2, metric=metric)

def reshaper(x):
	x=np.array(x)
	try:
		a=x.shape[1]
		return x
	except IndexError:
		return x.reshape(-1,1)

def hamming_dist(x,y):
	#print('x',len(x[-1]))
	#print('y',len(y[-1]))
	return len([i for i, j in zip(x, y) if i != j])   

def allnan(v): #fonctionne juste pour les dict de tuples
	from math import isnan
	import numpy as np
	return np.all(np.array([isnan(k) for k in list(v)]))
def dict_argmax(d):
	l={k:v for k, v in d.items() if not allnan(v)}
	return max(l,key=l.get)
def dict_argmin(d):
	return min(d, key=d.get)

def read_files(mypath):
	from os import listdir
	from os.path import isfile, join

	return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def per_section(it, is_delimiter=lambda x: x.isspace()):
	ret = []
	for line in it:
		if is_delimiter(line):
			if ret:
				yield ret  # OR  ''.join(ret)
				ret = []
		else:
			ret.append(line.rstrip())  # OR  ret.append(line)
	if ret:
		yield ret
		
def split_train_test(dataset,ratio=0.9, seed=None):
   idx_train = []
   X_train = []
   X_test = []
   random.seed(seed)
   for idx, val in random.sample(list(enumerate(dataset)),int(ratio*len(dataset))):
	   idx_train.append(idx)
	   X_train.append(val)
   idx_test=list(set(range(len(dataset))).difference(set(idx_train)))
   for idx in idx_test:
	   X_test.append(dataset[idx])
   x_train,y_train=zip(*X_train)
   x_test,y_test=zip(*X_test)
   return np.array(x_train),np.array(y_train),np.array(idx_train),np.array(x_test),np.array(y_test),np.array(idx_test)    

def setup_logger(name, log_file, level=logging.INFO):
	formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

	handler = logging.FileHandler(log_file)        
	handler.setFormatter(formatter)

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.addHandler(handler)

	return logger

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

