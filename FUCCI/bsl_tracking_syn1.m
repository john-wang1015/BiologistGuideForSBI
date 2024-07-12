clc;clear;

load('CellTracking_synthetic_dataset.mat');
load('bsl_cov_tracking_syn1.mat','cov_rm')

cov_rw = cov_rm;

Y = sy1; theta_true = theta1;

m = 200;
M = 10000;

% define prior
prior.num_params = 6;
prior.p1 = [0, 0, 0, 0, 0, 0];
prior.p2 = [1, 1, 1, 10, 10, 10];
prior.sampler = @() [unifrnd(prior.p1,prior.p2)]; 
prior.pdf = @(theta) prod(exp(theta)./(1 + exp(theta)).^2);
prior.trans_f = @(theta) [log((theta - prior.p1)./(prior.p2 - theta))];
prior.trans_finv = @(theta) [(prior.p2.*exp(theta) + prior.p1)./(1 + exp(theta))];

init = theta_true';   % initial value of chain NOT on transformed scale


tic;
[theta,dist] = bsl_cell_tracking(Y,init,m,M,cov_rw,prior,'res/bsl/syn1_tracking/tracking_syn1_');
finaltime=toc;


save('res/bsl/syn1_tracking/results_test_bsl.mat','theta','dist','finaltime');

