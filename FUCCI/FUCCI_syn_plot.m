clc;clear;
load('SNPE_syn2_CellDensity_30k.mat')
npe_summary = summaries;
npe_theta = theta_new;

load('SNLE_syn2_CellDensity_30k.mat')
nle_summary = summaries;
nle_theta = theta_new;

%%
title_string = ["R_r","R_y","R_g","M_r","M_y","M_g"];

figure
for i = 1:6
    subplot(2,3,i)
    [f1,xi1] = ksdensity(npe_theta(:,i),'support',[0,1]);
    plot(xi1,f1,'color',[8,48,107]/255,'LineWidth',3);
    hold on 
    [f2,xi2] = ksdensity(nle_theta(:,i),'support',[0,1]);
    plot(xi2,f2,'--','color',[0,109,44]/255,'LineWidth',3);
    title(title_string(i))
    set(gca,'FontSize',24)
    if i ==1
        legend('npe','nle')
    end
end

%%
figure
for i = 1:15
    subplot(3,5,i)
    [f1,xi1] = ksdensity(npe_summary(:,i),'Bandwidth',4);
    plot(xi1,f1,'color',[8,48,107]/255,'LineWidth',3);
    hold on 
    [f2,xi2] = ksdensity(nle_summary(:,i),'Bandwidth',4);
    plot(xi2,f2,'--','color',[0,109,44]/255,'LineWidth',3);
    xline(observation(i),'LineWidth',3)
    title(sprintf("S_{%d}",i))
    set(gca,'FontSize',24)
    if i ==1
        legend('npe','nle','true')
    end
end