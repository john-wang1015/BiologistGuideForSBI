import numpy as np
import matlab.engine
import os
import scipy.io as sio
import multiprocessing
from functools import partial
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.pyplot import figure
import torch

from sbi.inference import SNPE_C, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
_ = torch.manual_seed(42)

def fucci_simulation(theta):
    eng = matlab.engine.start_matlab()
    #fucci_path = os.path.abspath('FUCCI')
    #eng.addpath(fucci_path, nargout=0)
    theta_matlab = matlab.double(theta.tolist())

    sx_all = np.zeros((len(theta.tolist()), 15))
    n = theta.shape[0]
    sx_all = eng.simulator_density(theta_matlab, n, 15,  nargout=1)
    eng.quit()
    sx_all = np.asarray(sx_all)
    print(sx_all)
    sx_sims = np.array(sx_all, copy=True)  
    return torch.from_numpy(sx_sims).to(torch.float32)

def simulators(theta):
    return fucci_simulation(theta)

def run_snpe_c():
    x_0 = sio.loadmat('CellDensity_synthetic_dataset.mat')['sy3']
    theta_true = sio.loadmat('CellDensity_synthetic_dataset.mat')['theta3']

    prior_min = torch.as_tensor([0., 0., 0., 0., 0., 0.])
    prior_max = torch.as_tensor([1., 1., 1., 10., 10., 10.])
    prior = utils.BoxUniform(low=prior_min, high=prior_max)
    simulator, prior = prepare_for_sbi(simulators, prior)

    inference = SNPE_C(prior=prior, density_estimator='nsf')

    num_rounds = 10
    num_simulations = 10000

    posteriors = []
    posterior_samples = []
    proposal = prior

    folder_name = "res/npe/fucci_syn3_density"
    is_exist = os.path.exists(folder_name)
    if not is_exist:
        os.makedirs(folder_name)

    for kk in range(num_rounds):
        print("*****************iteration {}**************************".format(kk))
        #theta, x = simulate_for_sbi(simulator, proposal, num_simulations=1000,simulation_batch_size = 16)
        
        theta = prior.sample([num_simulations])
        x = simulators(theta)

        density_estimator = inference.append_simulations(
            theta, x, proposal=proposal
        ).train()
        posterior = inference.build_posterior(density_estimator)

        posterior_sample = posterior.sample((num_simulations,), x=x_0)

        mat_path = os.path.join(folder_name, "snpe_c_nsf fucci syn3 density, round {}.mat".format(kk))
        sio.savemat(mat_path, {"posterior":posterior_sample})

        figure(figsize=(20, 10))
        for i in range(1,7):
            plt.subplot(2,3,i)
            sns.kdeplot(data=posterior_sample[:, i-1], color='blue')
            plt.axvline(x=theta_true[0,i-1], color='r', linestyle='--')
            if i == 1:
                plt.legend(labels=['Proposal', 'NPE', 'True'])
        pic_path = os.path.join(folder_name, 'snpc_fucci_syn3_density_iteration {}.png'.format(kk))
        plt.savefig(pic_path)
        plt.show()
        
        posterior_samples.append(posterior_sample)
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_0)
    

    # Iterate through each model in the list and save
    for idx, model in enumerate(posteriors):
        file_path = os.path.join(folder_name, 'fucci_syn3_density_model_{idx}.pth')
        torch.save(model.state_dict(), file_path)


if __name__ == '__main__':
    run_snpe_c()



