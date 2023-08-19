clc;clear;
addpath("Model\");
%%
theta1 = [0.01, 1e-5, 45, 400, 26, 0.99, 1e-5, 10, 50];
theta2 = [0.2, 1e-5, 40, 300, 15, 0.7, 1e-5, 5, 50];
theta3 = [0.9, 1e-5, 20, 1000, 45, 0.9, 1e-5, 20, 100];

sim_params1.page = 2;
sim_params1.max_time = 32;
sim_params1.startingvol = 100;

sim_params2.page = 2;
sim_params2.max_time = 25;
sim_params2.startingvol = 100;

sim_params3.page = 2;
sim_params3.max_time = 66;
sim_params3.startingvol = 100;

%%
syn1 = simulator(theta1,sim_params1,sim_params1.max_time);
syn2 = simulator(theta2,sim_params2,sim_params2.max_time);
syn3 = simulator(theta3,sim_params3,sim_params3.max_time);

%%
save("synthetic_dataset.mat")

%%
clc;clear
load("synthetic_dataset.mat")
figure
time1 = linspace(1,32,32);
time2 = linspace(1,25,25);
time3 = linspace(1,66,66);
plot(time1,syn1,'LineWidth',3)
hold on 
plot(time2,syn2,'LineWidth',3)
plot(time3,syn3,'LineWidth',3)
legend("syn1","syn2","syn3")
xlabel("Time (day)")
ylabel("Tumour Volume (mm^3)")
xlim([1 66])
set(gca,'FontSize',20)

%% generate the prior predictive to use as training data
clc;clear;
load("synthetic_dataset.mat")

N = 10000;
prior.num_params = 9; 

prior.alpha = [1, 1];
prior.beta = [1, 1e3];
prior.lower = [0, 1];
prior.upper1 = [50, 32*24];
prior.upper2 = [50, 25*24];
prior.upper3 = [50, 66*24];
prior.tau_low = 2;
prior.tau_upper1 = 31;
prior.tau_upper2 = 24;
prior.tau_upper3 = 65;

sim_params1.page = 2;
sim_params1.max_time = 32;
sim_params1.startingvol = 100;

sim_params2.page = 2;
sim_params2.max_time = 25;
sim_params2.startingvol = 100;

sim_params3.page = 2;
sim_params3.max_time = 66;
sim_params3.startingvol = 100;

prior.sampler1 = @() [betarnd(prior.alpha,prior.beta),...
    unifrnd(prior.lower, prior.upper1),...
    ceil(unifrnd(prior.tau_low, prior.tau_upper1)),...
    betarnd(prior.alpha,prior.beta),...
    unifrnd(prior.lower, prior.upper1),];

prior.sampler2 = @() [betarnd(prior.alpha,prior.beta),...
    unifrnd(prior.lower, prior.upper2),...
    ceil(unifrnd(prior.tau_low, prior.tau_upper2)),...
    betarnd(prior.alpha,prior.beta),...
    unifrnd(prior.lower, prior.upper2),];

prior.sampler3 = @() [betarnd(prior.alpha,prior.beta),...
    unifrnd(prior.lower, prior.upper3),...
    ceil(unifrnd(prior.tau_low, prior.tau_upper3)),...
    betarnd(prior.alpha,prior.beta),...
    unifrnd(prior.lower, prior.upper3),];

%% for syn 1 with length 32
part_vals1 = zeros(N,prior.num_params);
prior_pred_sims1 = zeros(N,32);

parfor i = 1:N
    i
    part_vals1(i, :) = prior.sampler1();
    prior_pred_sims1(i, :) = simulator(part_vals1(i, :),sim_params1,sim_params1.max_time);
end

%% for syn 2 with length 25
part_vals2 = zeros(N,prior.num_params);
prior_pred_sims2 = zeros(N,25);

parfor i = 1:N
    i
    part_vals2(i, :) = prior.sampler2();
    prior_pred_sims2(i, :) = simulator(part_vals2(i, :),sim_params2,sim_params2.max_time);
end

%% for syn 3 with length 66
part_vals3 = zeros(N,prior.num_params);
prior_pred_sims3 = zeros(N,66);

parfor i = 1:N
    i
    part_vals3(i, :) = prior.sampler3();
    prior_pred_sims3(i, :) = simulator(part_vals3(i, :),sim_params3,sim_params3.max_time);
end

%%
save('BVCBM_syn_dataset.mat')


%% SNPE for synthetic data 1
clc;clear;
addpath("Model\");
load("BVCBM_SNLE_syn1_theta20k.mat");

N = 10000;

sim_params1.page = 2;
sim_params1.max_time = 32;
sim_params1.startingvol = 100;

part_vals1 = double(theta_new);
part_vals1(:,5) = floor(part_vals1(:,5));
post_pred_sims1 = zeros(N,32);

parfor i = 1:N
    i
    post_pred_sims1(i, :) = simulator(part_vals1(i, :),sim_params1,sim_params1.max_time);
end

%%
save('BVCBM_SNLE_syn1_sims20k.mat')

%%
clc;clear
load('BVCBM_SNPE_syn1_sims20k.mat')
y = observation;

figure
quant1_95CI = quantile(post_pred_sims1,[0.025 0.975]);
quant1_75CI = quantile(post_pred_sims1,[0.25 0.75]);
quant1_90CI = quantile(post_pred_sims1,[0.1 0.9]);
len1 = 1:length(y);
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
plot(1:length(y),y,'k','LineWidth',4);
xlabel('time (days)','fontsize',24)
ylabel('Tumour size','fontsize',24)
xlim([1 32])
set(gca,'FontSize',24)

%%
load('BVCBM_syn_dataset.mat')

figure
subplot(1,3,1)
[f,xi] = ksdensity(double(theta_new(:,4)));
plot(xi,f,'color',[8,48,107]/255,'LineWidth',3);
hold on 
xline(theta1(4),'LineWidth',3)
title('g_{age}^1')
set(gca,'FontSize',24)

subplot(1,3,2)
[f,xi] = ksdensity(double(theta_new(:,9)));
plot(xi,f,'color',[8,48,107]/255,'LineWidth',3);
hold on 
xline(theta1(9),'LineWidth',3)
title('g_{age}^2')
set(gca,'FontSize',24)

subplot(1,3,3)
[f,xi] = ksdensity(double(theta_new(:,5)));
plot(xi,f,'color',[8,48,107]/255,'LineWidth',3);
hold on 
xline(theta1(5),'LineWidth',3)
title('\tau')
set(gca,'FontSize',24)