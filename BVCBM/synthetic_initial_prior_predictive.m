clc;clear;
addpath("Model/");
load("synthetic_dataset.mat")

%%
N = 10000; 

prior.num_params = 9; 
prior.alpha = [1, 1];
prior.beta = [1, 1e5];
prior.lower = [0, 1];
prior.upper = [50, 500];
prior.tau_low = 2;
prior.tau_upper = sim_params.max_time;

prior.sampler = @() [betarnd(prior.alpha,prior.beta),...
    unifrnd(prior.lower, prior.upper),...
    ceil(unifrnd(prior.tau_low, prior.tau_upper)),...
    betarnd(prior.alpha,prior.beta),...
    unifrnd(prior.lower, prior.upper),];

%% 
part_vals_prior = zeros(N,9);
prior_pred_sim = zeros(N,sim_params.max_time);

tic
parfor i = 1:N
    i
    part_vals_prior(i,:) = prior.sampler();
    prior_pred_sim(i,:) = simulator(part_vals_prior(i,:), sim_params, sim_params.max_time);
end
toc 

%%
save("Neural_method_initial.mat");

%%
figure
quant1_95CI = quantile(prior_pred_sim,[0.025 0.975]);
quant1_75CI = quantile(prior_pred_sim,[0.25 0.75]);
quant1_90CI = quantile(prior_pred_sim,[0.1 0.9]);
len1 = 1:sim_params.max_time;
cmap1 = [150,150,150;150,150,150;150,150,150]/255;
cmap2 = [107,174,214;107,174,214;107,174,214]/255;
cmap3 = [8,48,107;8,48,107;8,48,107]/255;
h175 = fill([len1, fliplr(len1)], [quant1_75CI(1,:), fliplr(quant1_75CI(2,:))],cmap3(1,:),'LineStyle','none');
hold on
h190 = fill([len1, fliplr(len1)], [quant1_90CI(1,:), fliplr(quant1_90CI(2,:))],cmap2(1,:),'LineStyle','none');
h195 = fill([len1, fliplr(len1)], [quant1_95CI(1,:), fliplr(quant1_95CI(2,:))],cmap1(1,:),'LineStyle','none');
set(h195,'facealpha',.4)
set(h190,'facealpha',.5)
set(h175,'facealpha',.8)
plot(1:length(syn1),syn1,'k','LineWidth',4);
plot(1:length(syn2),syn2,'k','LineWidth',4);
plot(1:length(syn3),syn3,'k','LineWidth',4);
plot(1:length(syn4),syn4,'k','LineWidth',4);
xlabel('time (days)','fontsize',24)
ylabel('Tumour size','fontsize',24)
xlim([1 32])
set(gca,'FontSize',20)