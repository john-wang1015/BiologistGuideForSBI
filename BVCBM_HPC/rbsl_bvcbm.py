import elfi
import numpy as np
import matlab.engine
import os
from elfi.methods.bsl import pre_sample_methods, pdf_methods
from scipy.io import loadmat
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt

def bvcbm_simulation(g_age, g_age_2, tau, eng=None, batch_size=1, random_state=None):
    t1 = 1.0
    t2 = 0.0
    t3 = 17.3
    t6 = 1.0
    t7 = 0.0
    t8 = 17.3

    if hasattr(g_age, 'shape'):
        g_age = g_age.flatten()[0]
        g_age_2 = g_age_2.flatten()[0]
        tau = tau.flatten()[0]
    theta = np.array([t1, t2, t3, g_age, tau, t6, t7, t8, g_age_2])

    length_of_simulation = 32
    start_matlab = False if eng is not None else True
    if start_matlab:
        eng = matlab.engine.start_matlab()
    bvcbm_path = os.path.abspath('BVCBM_HPC')
    model_path = os.path.join(bvcbm_path, 'Model')
    eng.addpath(bvcbm_path, nargout=0)
    eng.addpath(model_path, nargout=0)  # Explicitly add Model path
    theta_matlab = matlab.double(theta.tolist())

    sx_all = np.zeros((batch_size, length_of_simulation))

    for batch_ii in range(batch_size):
        sx = eng.simulator(theta_matlab, length_of_simulation,  nargout=1)
        sx = np.array(sx).flatten()
        sx_all[batch_ii, :] = sx

    if start_matlab:
        eng.quit()

    return sx_all


def sum_fn(x):
    """Identity."""
    return x


def log_sum_fn(x):
    return np.log(x)


# Build ELFI model...
def get_model(true_params=None, x_obs=None):
    if true_params is None:
        true_params = np.array([300.0, 100.0,  16.0])

    eng = matlab.engine.start_matlab()
    sim_fn = partial(bvcbm_simulation, eng=eng)
    y = sim_fn(*true_params)

    m = elfi.ElfiModel()
    # prior
    elfi.Prior('uniform', 2, (32*24)-3, model=m, name='g_age')
    elfi.Prior('uniform', 2, (32*24)-3, model=m, name='g_age2')
    elfi.Prior('uniform', 2, 29, model=m, name='tau')
    # simulator
    elfi.Simulator(sim_fn, m['g_age'], m['g_age2'],  m['tau'], observed=y, name='BVCBM')
    # summary
    elfi.Summary(sum_fn, m['BVCBM'], name='S')
    elfi.Summary(log_sum_fn, m['BVCBM'], name='log_S')
    # distance
    elfi.Distance('euclidean', m['S'], name='d')
    return m


def run_rbsl():
    m = get_model()
    likelihood = pdf_methods.robust_likelihood("variance")
    feature_names = ['log_S']
    seed = 1
    # print("cores: ", multiprocessing.cpu_count())
    # true_params = np.array([300.0, 100.0,  16.0])

    n_sim = 10000
    true_params = {'g_age': 300.0, 'g_age2': 100.0, 'tau': 16.0}
    # pre_sample_methods.plot_features(m, true_params, n_sim, feature_names,
    #                                 seed=seed)
    #plt.savefig('rbslv_bvcbm_features.png')
    #plt.clf()
    #n_sim = [100, 300, 500, 1000]
    #log_stdev = pre_sample_methods.log_SL_stdev(model=m,
    #                         theta=true_params,
    #                         n_sim=n_sim,
    #                         feature_names=feature_names,
                             # likelihood=likelihood,
    #                         M=20,
    #                         seed=123)
    #print('log_stdev: ', log_stdev)

    nsim_round = 350
    r_bsl_v = elfi.BSL(m,
                       nsim_round,
                       batch_size=50,
                       feature_names=feature_names,
                       likelihood=likelihood,
                       seed=seed
                       )
    mcmc_iterations = 1100
    est_posterior_cov = loadmat('cov_matrix_subset.mat')['subset_matrix']
    est_posterior_cov = np.array(est_posterior_cov).reshape((3,3))
    logit_transform_bound = [(2, (32*24)-1),
                             (2, (32*24)-1),
                             (2, 31)]
    
    elfi.set_client('multiprocessing')

    params0 = [300.0, 100.0, 16.0]
    res = r_bsl_v.sample(mcmc_iterations,
                         0.1*np.eye(3),
                         #est_posterior_cov,
                         burn_in=100,
                         param_names=['g_age', 'g_age2', 'tau'],
                         params0=params0,
                         logit_transform_bound=logit_transform_bound,
                         bar=False
                         )
    print(res)

    print(res.compute_ess())

    res.plot_traces()
    plt.savefig('rbslv_bvcbm_traces.png')


    mbins = 30
    res.plot_marginals(reference_value=true_params,
                       bins=mbins)
    plt.savefig('rbslv_bvcbm_marginals.png')

    res.plot_pairs(reference_value=true_params,
                   bins=mbins)
    plt.savefig('rbslv_bvcbm_pairs.png')

    gamma_dict = dict(zip(['gamma_{}'.format(index) for index in range(r_bsl_v.observed.size)],
                          np.transpose(res.outputs['gamma'])))
    elfi.visualization.visualization.plot_marginals(gamma_dict, bins=mbins)
    plt.savefig('rbslv_bvcbm_gamma.png')


if __name__ == '__main__':
    run_rbsl()
