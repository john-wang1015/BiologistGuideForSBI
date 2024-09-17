import elfi
import elfi.clients
import elfi.clients.multiprocessing
import numpy as np
# import matlab.engine
import os
from elfi.methods.bsl import pre_sample_methods, pdf_methods
import scipy.io as sio
# from functools import partial
import matplotlib.pyplot as plt
import pickle as pkl

import ctypes

# Load the shared library
lib = ctypes.CDLL('BVCBM/Model/ModelInterface.dylib')
lib.FullSimulation_biphasic_raw.restype = ctypes.POINTER(ctypes.c_double)
lib.FullSimulation_biphasic_raw.argtypes = [
    ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_double,
    ctypes.c_double, ctypes.c_int
]


def bvcbm_simulation(g_age, g_age_2, tau, random_state=None, batch_size=1):
    t1 = np.ones(batch_size)
    t2 = np.zeros(batch_size)
    t3 = 17.3 * np.ones(batch_size)

    if hasattr(g_age, 'shape'):
        g_age = g_age.flatten()
        g_age_2 = g_age_2.flatten()
        tau = tau.flatten()

    tau = np.atleast_1d(tau)
    if len(tau) == 1:  # TODO: hacky
        tau = np.repeat(tau, batch_size)

    theta = np.zeros([batch_size, 9])
    theta[:, 0], theta[:, 5] = t1, t1
    theta[:, 1], theta[:, 6] = t2, t2
    theta[:, 2], theta[:, 7] = t3, t3
    theta[:, 3], theta[:, 4], theta[:, 8] = g_age, tau, g_age_2

    length_of_simulation = 19
    sx_all = np.zeros((batch_size, length_of_simulation))

    for batch_ii in range(batch_size):
        result_ptr = lib.FullSimulation_biphasic_raw(
            theta[batch_ii, 0], theta[batch_ii, 1], int(theta[batch_ii, 2]), int(theta[batch_ii, 3]),
            int(theta[batch_ii, 4]), theta[batch_ii, 5], theta[batch_ii, 6], int(theta[batch_ii, 7]),
            int(theta[batch_ii, 8]), tau[batch_ii], 100.0, length_of_simulation
        )

        result = np.ctypeslib.as_array(result_ptr, shape=(length_of_simulation,))

        sx_all[batch_ii, :] = result

        lib.free_memory(result_ptr)

    return sx_all


def sum_fn(x):
    """Identity."""
    return x


def log_sum_fn(x):
    return np.log(x)


def get_model(true_params=None, x_obs=None):
    if true_params is None:
        true_params = np.array([200.0, 50.0,  12.0])

    # eng = matlab.engine.start_matlab()
    # sim_fn = partial(bvcbm_simulation, eng=eng)
    sim_fn = bvcbm_simulation
    y = x_obs  #sim_fn(*true_params)

    m = elfi.ElfiModel()
    # prior
    elfi.Prior('uniform', 2, (19*24)-3, model=m, name='g_age')
    elfi.Prior('uniform', 2, (19*24)-3, model=m, name='g_age2')
    elfi.Prior('uniform', 2, 16, model=m, name='tau')
    # simulator
    elfi.Simulator(sim_fn, m['g_age'], m['g_age2'],  m['tau'], observed=y, name='BVCBM')
    # summary
    elfi.Summary(sum_fn, m['BVCBM'], name='S')
    elfi.Summary(log_sum_fn, m['BVCBM'], name='log_S')
    # distance
    elfi.Distance('euclidean', m['S'], name='d')
    return m


def run_rbsl():
    pancreatic = sio.loadmat("CancerDatasets.mat")['Pancreatic_data']
    obs = pancreatic[0:19, 0]
    obs = obs.reshape((1, -1))
    m = get_model(x_obs=obs)
    likelihood = pdf_methods.robust_likelihood("variance")
    feature_names = ['log_S']
    seed = 1
    # print("cores: ", multiprocessing.cpu_count())
    # true_params = np.array([300.0, 100.0,  16.0])

    true_params = {'g_age': 200.0, 'g_age2': 50.0, 'tau': 12.0}
    # n_sim = 10_000
    # pre_sample_methods.plot_features(m, true_params, n_sim, feature_names,
    #                                  seed=seed)
    # plt.savefig('rbslv_bvcbm_features.png')
    # plt.clf()
    n_sim = [100, 300, 500, 1000]
    # NOTE: below won't wory without modification
    # TODO! FIX BUG IN ELFI FOR BELOW WITH ROBUST LIKELIHOOD
    log_stdev = pre_sample_methods.log_SL_stdev(model=m,
                                                theta=true_params,
                                                n_sim=n_sim,
                                                feature_names=feature_names,
                                                likelihood=likelihood,
                                                M=10,
                                                seed=123
                                                )
    print('log_stdev: ', log_stdev)
    # Assuming FullSimulation returns a pointer to an array of doubles
    # lib.FullSimulation_biphasic_rlib.FullSimulation_biphasic_rawaw.restype = ctypes.POINTER(ctypes.c_double)
    # .argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int]

    # # Call the FullSimulation function
    # result_ptr = lib.FullSimulation_biphasic_raw(0.1, 0.05, 10, 5, 2, 100.0, 30)
    # # Convert the result to a numpy array
    # result = np.ctypeslib.as_array(result_ptr, shape=(30,))
    # print(result)

    nsim_round = 400
    r_bsl_v = elfi.BSL(m,
                       nsim_round,
                       feature_names=feature_names,
                       likelihood=likelihood,
                       seed=seed
                       )
    mcmc_iterations = 2000
    est_posterior_cov = sio.loadmat("cov_matrix_pan1.mat")['cov_rm']
    est_posterior_cov = np.array(est_posterior_cov).reshape((3, 3))
    est_posterior_cov[[1, 2]] = est_posterior_cov[[2, 1]]
    logit_transform_bound = [(2, (19*24)-1),
                             (2, (19*24)-1),
                             (2, 18)]
    # elfi.set_client(elfi.clients.multiprocessing.Client(num_processes=6))

    params0 = [200.0, 50.0, 12.0]
    res = r_bsl_v.sample(mcmc_iterations,
                         #  0.1*np.eye(3),
                         est_posterior_cov,
                         burn_in=500,
                         param_names=['g_age', 'g_age2', 'tau'],
                         params0=params0,
                         logit_transform_bound=logit_transform_bound,
                         bar=True
                         )
    print(res)

    # sio.savemat("rbsl_bvcbm_3para_syn1.mat",{"posterior":res})

    print(res.compute_ess())

    res.plot_traces()
    plt.savefig('rbsl_bvcbm_traces_pan1.pdf')

    mbins = 30
    res.plot_marginals(reference_value=true_params,
                       bins=mbins)
    plt.savefig('rbsl_bvcbm_marginals_pan1.pdf')

    res.plot_pairs(reference_value=true_params,
                   bins=mbins)
    plt.savefig('rbsl_bvcbm_pairs_pan1.pdf')

    gamma_dict = dict(zip(['gamma_{}'.format(index) for index in range(r_bsl_v.observed.size)],
                          np.transpose(res.outputs['gamma'])))
    elfi.visualization.visualization.plot_marginals(gamma_dict, bins=mbins)
    plt.savefig('rbsl_bvcbm_gamma_pan1.pdf')

    folder_name = "res/rbsl/pan1"
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)
    with open(f'{folder_name}rbsl_thetas.pkl', 'wb') as f:
        pkl.dump(res.samples, f)


if __name__ == '__main__':
    run_rbsl()
