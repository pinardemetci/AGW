from fgcoot import *
from utils import *
import numpy as np
from sklearn.preprocessing import normalize

def barycentric_projection(source, target, couplingMatrix):
	"""
	Given: data in the target space, data in the source space, a coupling matrix learned via Gromow-Wasserstein OT
	Returns: source (target) matrix transported onto the target (source)
	"""
	P = (couplingMatrix.T/couplingMatrix.sum(1)).T
	transported_data= np.matmul(P, target)
	return transported_data

data1=normalize(np.genfromtxt("../data/scGEM_methylation.csv", delimiter=","))
data2=normalize(np.genfromtxt("../data/scGEM_expression.csv", delimiter=","))
D1=compute_graph_distances(data1)
D2=compute_graph_distances(data2)

print(type(D1), type(D2))

# G=fgw_lp(M,C1,C2,p,q,loss_fun='square_loss',alpha=0.5,amijo=True,G0=None)

# G=cot_numpy(data, data, w1 = None, w2 = None, v1 = None, v2 = None,
#               niter=10, algo='sinkhorn', reg=0.001,algo2='sinkhorn',
#               reg2=0.001, verbose=True, log=False, random_init=False, C_lin=None)

G= fgcoot_numpy(data1, data2, D1,D2, w1 = None, w2 = None, v1 = None, v2 = None, alpha=0.75,
			  niter=50,reg=1e-10,algo2='sinkhorn',reg2=1e-4)
print(G)
import matplotlib.pyplot as plt 
plt.imshow(G[0], cmap="Reds")
plt.show()
plt.clf()
plt.imshow(G[1], cmap="seismic")
plt.show()
plt.clf()

data1projected=barycentric_projection(data1, data2, G[0])
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
concatdata=np.concatenate((data1projected,data2))
concatDataPC=pca.fit_transform(concatdata)
data1pc=concatDataPC[0:data1.shape[0],:]
data2pc=concatDataPC[data1.shape[0]:,:]

data1_ctypes=np.loadtxt("../data/scGEM_typeExpression.txt")
data2_ctypes=np.loadtxt("../data/scGEM_typeMethylation.txt")


plt.scatter(data2pc[:,0], data2pc[:,1], c=data2_ctypes)
plt.scatter(data1pc[:,0], data1pc[:,1], c=data1_ctypes)

plt.show()




