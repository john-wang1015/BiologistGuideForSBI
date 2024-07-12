clc;clear;
addpath("Model/")
load('paper2_BVCBM_synthetic_datasets.mat');

ncpus = 20; %number of CPUs to use
clust = parcluster('local');
clust.JobStorageLocation = tempdir;
par = parpool(clust,ncpus); 

Y = sim1; theta_true = param1;
n = 10000;
sx = zeros(n,32);

parfor k = 1:n
    sx(k,:) = simulator(param1,32);
end

save('results_bsl_pre_sample.mat');

figure
for i = 1:32
    subplot(4,8,i)
    histogram(log(sx(:,i)))
    xlabel(sprintf('\\theta_{%d}', i)); % Corrected the xlabel
    set(gca,'FontSize',24)
end

print(gcf, 'verfy_bsl_bvcbm.png', '-dpng', '-r300');

