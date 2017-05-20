import random
import time
import math
from scipy.stats import multivariate_normal
import numpy as np
import pandas


def k_means(data, k, eps=1e-4, mu=None):
    """ Run the k-means algorithm
    data - an nxd ndarray
    k - number of clusters to fit
    eps - stopping criterion tolerance
    mu - an optional KxD ndarray containing initial centroids

    returns: a tuple containing
        mu - a KxD ndarray containing the learned means
        cluster_assignments - an N-vector of each point's cluster index
    """
    def dist( x, y):
        return math.sqrt(sum([(i-j)**2 for i,j in zip(x,y)]))

    def cost(data, cluster_mu, return_vector):
        cost = 0
        for index, data_point in enumerate(data):
            cluster_mu_index = return_vector[index]
            centroid = cluster_mu[cluster_mu_index]
            cost += dist(data_point,centroid)
        return cost


    def find_close(point, mu):
        close = math.inf
        i = 0
        index = 0
        return_point = None
        for centroid in mu:
            #print(centroid,"centroid")
            if (dist(point, centroid) < close):
                return_point= centroid
                index = i
                close = dist(point,centroid)
                #print(close,"clos")
            i+=1
        return return_point, index
    #print("find_close [1,1]",find_close([1,1],[[1,1],[11,11]]))

    mu_freq = [0 for i in range(k)]

    n, d = data.shape
    if mu is None:
        # randomly choose k points as initial centroids
        mu = data[random.sample(range(data.shape[0]), k)]
        print(mu,"mu")


    prev_cost = math.inf
    cost1 = 0

    return_vector = [None for i in range(len(data))]
    while abs(prev_cost - cost1) > 1e-4 :

        for data_index ,data_point in enumerate(data):
            centroid, index = find_close(data_point, mu)
            return_vector[data_index] = index

        mu = np.array([np.zeros(len(data[0])) for i in range(k)])
        #print(return_vector.count(0), return_vector.count(1), return_vector.count(2))
        for i,j in enumerate(return_vector):
            mu[j] += data[i]
            mu_freq[j]+=1
        #print(mu)
        #print(mu,"mu" ,mu_freq,"fre mu")
        mu = np.array([mu[i]/mu_freq[i] for i in range(k)])

        #print(mu,mu_freq,"hhgkhgjkh")
        cost1 = cost(data, mu, return_vector)
        if cost1 < prev_cost:
            prev_cost = cost1

    return mu, return_vector



class MixtureModel(object):
    def __init__(self, k):
        self.k = k
        self.params = {
            'pi': np.random.dirichlet([1]*k),
        }

    def __getattr__(self, attr):
        if attr not in self.params:
            raise AttributeError()
        return self.params[attr]

    def __setstate__(self, state):
        for k, v in state.items():
            setattr(self, k, v)

    def e_step(self, data):
        """ Performs the E-step of the EM algorithm
        data - an NxD pandas DataFrame

        returns a tuple containing
            (float) the expected log-likelihood
            (NxK ndarray) the posterior probability of the latent variables
        """
        raise NotImplementedError()


    def m_step(self, data, p_z):
        """ Performs the M-step of the EM algorithm
        data - an NxD pandas DataFrame
        p_z - an NxK numpy ndarray containing posterior probabilities

        returns a dictionary containing the new parameter values
        """
        raise NotImplementedError()

    def fit(self, data, eps=1e-4, verbose=True, max_iters=100):
        """ Fits the model to data
        data - an NxD pandas DataFrame
        eps - the tolerance for the stopping criterion
        verbose - whether to print ll every iter
        max_iters - maximum number of iterations before giving up

        returns a boolean indicating whether fitting succeeded

        if fit was successful, sets the following properties on the Model object:
          n_train - the number of data points provided
          max_ll - the maximized log-likelihood
        """
        last_ll = np.finfo(float).min
        start_t = last_t = time.time()
        i = 0
        while True:
            i += 1
            if i > max_iters:
                return False
            ll, p_z = self.e_step(data)
            new_params = self.m_step(data, p_z)
            self.params.update(new_params)
            if verbose:
                dt = time.time() - last_t
                last_t += dt
                print('iter %s: ll = %.5f  (%.2f s)' % (i, ll, dt))
                last_ts = time.time()
            if abs((ll - last_ll) / ll) < eps:
                break
            last_ll = ll

        setattr(self, 'n_train', len(data))
        setattr(self, 'max_ll', ll)
        self.params.update({'p_z': p_z})

        print('max ll = %.5f  (%.2f min, %d iters)' %
              (ll, (time.time() - start_t) / 60, i))

        return True


class GMM(MixtureModel):
    def __init__(self, k, d):
        super(GMM, self).__init__(k)
        self.params['mu'] = np.random.randn(k, d)

    def e_step(self, data):
        n = len(data)
        return_sum = 0
        return_array = np.zeros((n,self.k))

        for i in range(self.k):
            return_array[:,i] = self.pi[i] * multivariate_normal.pdf(data,self.mu[i],self.sigsq[i])

        return np.sum(np.log(np.sum(return_array, axis = 1))), np.transpose(np.transpose(return_array)/np.sum(return_array,axis = 1))



    def m_step(self, data, pz_x):
        n,d = data.shape
        return_dict =  {
            'pi': np.array([0 for i in range(self.k)]),
            'mu': np.array([np.zeros(d) for i in range(self.k)]),
            'sigsq': np.zeros(self.sigsq.shape)
        }
        

        new_grouping =  np.sum(pz_x,axis=0)
        return_dict["pi"] = new_grouping/n
        
        for i in range(self.k):
            for j in range(n):
                return_dict["mu"][i]+=(pz_x[j][i]*data[j])/new_grouping[i]
            for k in range(n):
                return_dict["sigsq"][i]+=pz_x[k][i]*(np.linalg.norm(data[k]-return_dict["mu"][i])**2)/(2*new_grouping[i])
        return return_dict


    def fit(self, data, *args, **kwargs):
        self.params['sigsq'] = np.asarray([np.mean(data.var(0))] * self.k)
        return super(GMM, self).fit(data, *args, **kwargs)


class CMM(MixtureModel):
    def __init__(self, k, ds):
        """d is a list containing the number of categories for each feature"""
        super(CMM, self).__init__(k)
        self.params['alpha'] = [np.random.dirichlet([1]*d, size=k) for d in ds]

    def e_step(self, data):
        N, D = data.shape
        return_array = np.zeros((N,self.k))
        for i in range(D):
            column = data.as_matrix()[:,i]
            dummies = pandas.get_dummies(column).as_matrix() # n * alpha_i
            array = np.matmul(dummies,self.alpha[i].T)  # alphai * k , return-array = n *k
            #return_array[return_array ==0]= 1
            array[array==0]=1
            return_array += np.log(array)
            #print(return_array.shape)
        pi_matrix = np.tile(self.pi,(N,1))  #  n * k
        pi_matrix[pi_matrix==0]=1
        return_array  +=np.log(pi_matrix)
        exp_matrix = np.exp(return_array)
       # print(exp_matrix)
        normalized = np.sum(exp_matrix,axis= 1)
       # print(normalized)
        return_array_2 = np.transpose(np.transpose(exp_matrix)/ normalized)
        #print(return_array_2,"sdfa")
        log = np.multiply(return_array_2,return_array)
        log_likelihood = np.sum(log)

        return log_likelihood, return_array_2


    def m_step(self, data, p_z):
        return_dict = {}
        n,D = data.shape
        new_pi = np.sum(p_z,axis = 0)/n
        temp = []
        for i in range(D):
            column = data.as_matrix()[:,i]
            dummies = pandas.get_dummies(column).as_matrix()
            new_p_z = np.transpose(p_z)
            num = np.matmul(new_p_z, dummies)
            dem = num.sum(axis=1)
            temp.append(np.transpose(np.transpose(num)/ dem))
        return_dict['pi'] = new_pi
        return_dict["alpha"] = temp
        return return_dict

    @property
    def bic(self):
        param = self.k -1
        for i in range(len(self.alpha)):
            param+= self.k * (self.alpha[i].shape[1]-1)

        return self.max_ll - param *np.log(self.n_train)/2


