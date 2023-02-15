import numpy as np
import pandas as pd
import ot 
from sklearn.preprocessing import normalize 
import optim

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

def init_matrix_coot(X1, X2, v1, v2):
    r"""Return loss matrices and tensors for COOT fast computation
    Returns the value of |X1-X2|^{2} \otimes T as done in [1] based on [2] for the Gromov-Wasserstein distance. 
    Where :
        - X1 : The source dataset of shape (n,d)
        - X2 : The target dataset of shape (n',d')
        - v1 ,v2 : weights (histograms) on the columns of resp. X1 and X2
        - T : Coupling matrix of shape (n,n')
    Parameters
    ----------
    X1 : numpy array, shape (n, d)
         Source dataset
    X2 : numpy array, shape (n', d')
         Target dataset
    v1 : numpy array, shape (d,)
        Weight (histogram) on the features of X1.
    v2 : numpy array, shape (d',)
        Weight (histogram) on the features of X2.    
    
    Returns
    -------
    constC : ndarray, shape (n, n')
        Constant C matrix (see paragraph 1.2 of supplementary material in [1])
    hC1 : ndarray, shape (n, d)
        h1(X1) matrix (see paragraph 1.2 of supplementary material in [1])
    hC2 : ndarray, shape (n', d')
        h2(X2) matrix (see paragraph 1.2 of supplementary material in [1])
    References
    ----------
    .. [1] Redko Ievgen, Vayer Titouan, Flamary R{\'e}mi and Courty Nicolas
          "CO-Optimal Transport"
    .. [2] Peyré, Gabriel, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    def f1(a):
        return (a ** 2)

    def f2(b):
        return (b ** 2)

    def h1(a):
        return a

    def h2(b):
        return 2 * b

    constC1 = np.dot(np.dot(f1(X1), v1.reshape(-1, 1)),
                     np.ones(f1(X2).shape[0]).reshape(1, -1))
    constC2 = np.dot(np.ones(f1(X1).shape[0]).reshape(-1, 1),
                     np.dot(v2.reshape(1, -1), f2(X2).T))

    constC = constC1 + constC2
    hX1 = h1(X1)
    hX2 = h2(X2)

    return constC, hX1, hX2


def init_matrix_GW(C1, C2, p, q, loss_fun='square_loss'):
    r"""Return loss matrices and tensors for Gromov-Wasserstein fast computation
    Returns the value of :math:`\mathcal{L}(\mathbf{C_1}, \mathbf{C_2}) \otimes \mathbf{T}` with the
    selected loss function as the loss function of Gromow-Wasserstein discrepancy.
    The matrices are computed as described in Proposition 1 in :ref:`[12] <references-init-matrix>`
    Where :
    - :math:`\mathbf{C_1}`: Metric cost matrix in the source space
    - :math:`\mathbf{C_2}`: Metric cost matrix in the target space
    - :math:`\mathbf{T}`: A coupling between those two spaces
    The square-loss function :math:`L(a, b) = |a - b|^2` is read as :
    .. math::
        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)
        \mathrm{with} \ f_1(a) &= a^2
                        f_2(b) &= b^2
                        h_1(a) &= a
                        h_2(b) &= 2b
    The kl-loss function :math:`L(a, b) = a \log\left(\frac{a}{b}\right) - a + b` is read as :
    .. math::
        L(a, b) = f_1(a) + f_2(b) - h_1(a) h_2(b)
        \mathrm{with} \ f_1(a) &= a \log(a) - a
                        f_2(b) &= b
                        h_1(a) &= a
                        h_2(b) &= \log(b)
    Parameters
    ----------
    C1 : array-like, shape (ns, ns)
        Metric cost matrix in the source space
    C2 : array-like, shape (nt, nt)
        Metric cost matrix in the target space
    p : array-like, shape (ns,)
        Probability distribution in the source space
    q : array-like, shape (nt,)
        Probability distribution in the target space
    loss_fun : str, optional
        Name of loss function to use: either 'square_loss' or 'kl_loss' (default='square_loss')
    Returns
    -------
    constC : array-like, shape (ns, nt)
        Constant :math:`\mathbf{C}` matrix in Eq. (6)
    hC1 : array-like, shape (ns, ns)
        :math:`\mathbf{h1}(\mathbf{C1})` matrix in Eq. (6)
    hC2 : array-like, shape (nt, nt)
        :math:`\mathbf{h2}(\mathbf{C2})` matrix in Eq. (6)
    .. _references-init-matrix:
    References
    ----------
    .. [12] Gabriel Peyré, Marco Cuturi, and Justin Solomon,
        "Gromov-Wasserstein averaging of kernel and distance matrices."
        International Conference on Machine Learning (ICML). 2016.
    """
    C1, C2, p, q = list_to_array(C1, C2, p, q)
    nx = get_backend(C1, C2, p, q)

    if loss_fun == 'square_loss':
        def f1(a):
            return (a**2)

        def f2(b):
            return (b**2)

        def h1(a):
            return a

        def h2(b):
            return 2 * b
    elif loss_fun == 'kl_loss':
        def f1(a):
            return a * nx.log(a + 1e-15) - a

        def f2(b):
            return b

        def h1(a):
            return a

        def h2(b):
            return nx.log(b + 1e-15)

    constC1 = nx.dot(
        nx.dot(f1(C1), nx.reshape(p, (-1, 1))),
        nx.ones((1, len(q)), type_as=q)
    )
    constC2 = nx.dot(
        nx.ones((len(p), 1), type_as=p),
        nx.dot(nx.reshape(q, (1, -1)), f2(C2).T)
    )
    constC = constC1 + constC2
    hC1 = h1(C1)
    hC2 = h2(C2)

    return constC, hC1, hC2

def fgcoot(X1,X2,D1,D2,w1,w2,v1,v2,niter=10,algo="sinkhorn",reg=0,algo2="sinkhorn", reg2=0, armijo=False,verbose=True,random_init=False, C_lin=(None, None)):
	if v1 is None:
        v1 = np.ones(X1.shape[1]) / X1.shape[1]  # is (d,)
    if v2 is None:
        v2 = np.ones(X2.shape[1]) / X2.shape[1]  # is (d',)
    if w1 is None:
        w1 = np.ones(X1.shape[0]) / X1.shape[0]  # is (n',)
    if w2 is None:
        w2 = np.ones(X2.shape[0]) / X2.shape[0]  # is (n,)

    # C_lin_samp, C_lin_feat = C_lin

	if not random_init:
        Ts = np.ones((X1.shape[0], X2.shape[0])) / (
            X1.shape[0] * X2.shape[0]
        )  # is (n,n')
        Tv = np.ones((X1.shape[1], X2.shape[1])) / (
            X1.shape[1] * X2.shape[1]
        )  # is (d,d')

    else:
        Ts = random_gamma_init(w1, w2)
        Tv = random_gamma_init(v1, v2)

    constC_s_gw, hC1_s_gw, hC2_s_gw=init_matrix_GW(D1, D2, w1, w2, loss_fun='square_loss')
    constC_s, hC1_s, hC2_s = init_matrix_COOT(X1, X2, v1, v2)
    constC_v, hC1_v, hC2_v = init_matrix_COOT(X1.T, X2.T, w1, w2)
    cost = np.inf

    def f(G):
        return gwloss(constC, hC1, hC2, G)

    def df(G):
        return gwggrad(constC, hC1, hC2, G)

	for i in range(niter):
	    # SAMPLES
	    Tsold = Ts
	    Tvold = Tv
	    costold = cost
	    M = constC_s - np.dot(hC1_s, Tv).dot(hC2_s.T)
	    Ts = optim.cg(w1, w2, (1 - alpha) * M, alpha, f, df, Tsold, armijo=armijo, C1=D1, C2=D2, constC=constC_s_gw, log=False)
	    #Note: Made a slight change in cg() function. It was solving with emd, added a sinkhorn option

	    # FEATURES
	    M = constC_v - np.dot(hC1_v, Ts).dot(hC2_v.T)
	    if algo2 == "emd":
            Tv = ot.emd(v1, v2, M, numItermax=1e7)
        elif algo2 == "sinkhorn":
            Tv = ot.sinkhorn(v1, v2, M, reg2)

        delta, logfgw = np.linalg.norm(Ts - Tsold) + np.linalg.norm(Tv - Tvold)
        # I think the cost would need to be updated to Fused COOT cost?
        #cost = np.sum(M * Tv) + np.sum(C_lin_samp * Ts) # CHECK THIS!
		res, log = cg(p, q, (1 - alpha) * M, alpha, f, df, G0, armijo=armijo, C1=C1, C2=C2, constC=constC, log=True, **kwargs)
        fgw_dist = log['loss'][-1]
        cost=np.sum(M * Tv) + fgw_dist

		if delta < 1e-16 or np.abs(costold - cost) < 1e-7:
            if verbose:
                print("converged at iter ", i)
            break

    return Ts, Tv

