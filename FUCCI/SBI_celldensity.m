clc;clear;
load('SNLE_nsf_syn2_celldensity_20k.mat')
rng(1234);

N = 10000;
T_record = 48;

t_lb = 0; t_ub = 1;
m_lb = 0; m_ub = 1;
num_params = 6;

InitPosData = readmatrix("C:\Users\aufb\Dropbox\biologists' guide for sbi\FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
FinalPosData = readmatrix("C:\Users\aufb\Dropbox\biologists' guide for sbi\FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","FinalPos");
CellTrackingData = readmatrix("C:\Users\aufb\Dropbox\biologists' guide for sbi\FUCCI-main/Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");
ntrack = max(CellTrackingData(:,4));
    
Xmax = 1309.09; %Length of the domain
Ymax = 1745.35; %Width of the domain
    
%summary statistics: 
Nred = sum(FinalPosData(:,3) == 1);
Nyellow = sum(FinalPosData(:,3) == 2);
Ngreen = sum(FinalPosData(:,3) == 3);
CellTracking = false;

if CellTracking == true%if using cell tracking summary statistics
    %compute distance travelled through each cell phase by all tracked cells
    RedDistance = 0;
    YellowDistance = 0;
    GreenDistance = 0;
    j = 2;
    for i = 1:ntrack
       while CellTrackingData(j,4) == i
           if CellTrackingData(j,3) == 1 && CellTrackingData(j-1,3) == 1
               RedDistance = RedDistance + sqrt((CellTrackingData(j,1)-CellTrackingData(j-1,1))^2+(CellTrackingData(j,2)-CellTrackingData(j-1,2))^2);
           elseif CellTrackingData(j,3) == 2 
               YellowDistance = YellowDistance + sqrt((CellTrackingData(j,1)-CellTrackingData(j-1,1))^2+(CellTrackingData(j,2)-CellTrackingData(j-1,2))^2);
           elseif CellTrackingData(j,3) == 3 
               GreenDistance = GreenDistance + sqrt((CellTrackingData(j,1)-CellTrackingData(j-1,1))^2+(CellTrackingData(j,2)-CellTrackingData(j-1,2))^2);
           end
           j = j + 1;
           if (j > size(CellTrackingData,1))
               break
           end
       end
    end

   TotalDistance = [RedDistance/ntrack, YellowDistance/ntrack, GreenDistance/ntrack];
else %if using cell density summary statistics

    %compute median positon of cells on LHS and RHS of scratch
    RedDistanceMed = [median(FinalPosData(FinalPosData(:,3) == 1 & FinalPosData(:,1) <= Xmax/2, 1)), median(FinalPosData(FinalPosData(:,3) == 1 & FinalPosData(:,1) > Xmax/2, 1))];
    YellowDistanceMed = [median(FinalPosData(FinalPosData(:,3) == 2 & FinalPosData(:,1) <= Xmax/2, 1)), median(FinalPosData(FinalPosData(:,3) == 2 & FinalPosData(:,1) > Xmax/2, 1))];
    GreenDistanceMed = [median(FinalPosData(FinalPosData(:,3) == 3 & FinalPosData(:,1) <= Xmax/2, 1)), median(FinalPosData(FinalPosData(:,3) == 3 & FinalPosData(:,1) > Xmax/2, 1))];

    %compute IQR of cells on the LHS and RHS of scratch
    RedDistanceIQR = [iqr(FinalPosData(FinalPosData(:,3) == 1 & FinalPosData(:,1) <= Xmax/2, 1)), iqr(FinalPosData(FinalPosData(:,3) == 1 & FinalPosData(:,1) > Xmax/2, 1))];
    YellowDistanceIQR = [iqr(FinalPosData(FinalPosData(:,3) == 2 & FinalPosData(:,1) <= Xmax/2, 1)), iqr(FinalPosData(FinalPosData(:,3) == 2 & FinalPosData(:,1) > Xmax/2, 1))];
    GreenDistanceIQR = [iqr(FinalPosData(FinalPosData(:,3) == 3 & FinalPosData(:,1) <= Xmax/2, 1)), iqr(FinalPosData(FinalPosData(:,3) == 3 & FinalPosData(:,1) > Xmax/2, 1))];

    TotalDistance = [RedDistanceMed,YellowDistanceMed,GreenDistanceMed,RedDistanceIQR,YellowDistanceIQR,GreenDistanceIQR]; 

end

%%
load('syntheticData2_cellDensity.mat');
%sy = [Nred, Nyellow, Ngreen, TotalDistance]; 
s = SetupStruct(ntrack, Xmax, Ymax, InitPosData, CellTrackingData);

theta = zeros(N,num_params);
summaries = zeros(N,length(sy));
theta_temp = double(theta_new);

% creating initial sample
parfor i = 1:N
    i
    %while ~all(theta(i,:)) %repeat untill 1 proposed value is accepted
            % Draw theta ~ pi(theta)
            theta_prop = theta_temp(i,:);

            %simulate x ~ f(x|theta)
            [SummaryStatData, ExitSimStatus] = Main_simulate(theta_prop, s, T_record, CellTracking);

            if ExitSimStatus %check if simulation finished correctly
                continue;
            end 

            sx = GenerateSummaryStatistics(SummaryStatData, CellTracking, Xmax); %compute simulated summary stats
            if any(isnan(sx)) %check integrity of summary statistics
                continue  
            end 

            d = norm(sx - sy, 2); %discrepency function
            %store results
            theta(i,:) = theta_prop;
            summaries(i,:) = sx; %store simulated summary statistics
end

%%
dim_old = length(theta_old);
dim_new = N + dim_old;

theta_D = zeros(dim_new,num_params);
summaries_D = zeros(dim_new,length(sy));

theta_D(1:dim_old,:) = theta_old;
theta_D(dim_old+1:dim_new,:) = theta;

summaries_D(1:dim_old,:) = sims_old;
summaries_D(dim_old+1:dim_new,:) = summaries;

%%
save("SNLE_syn2_CellDensity_30k.mat");

%%
num_plots = length(sy);

figure
for i = 1:5
    subplot(6,5,i)
    [f,xi] = ksdensity(sims_old(:,i)); 
    plot(xi,f,'LineWidth',3);
    hold on 
    xline(sy(i),'LineWidth',3)
end

for i = 6:10
    subplot(6,5,i)
    [f,xi] = ksdensity(summaries(:,i-5)); 
    plot(xi,f,'LineWidth',3);
    hold on 
    xline(sy(i-5),'LineWidth',3)
end

for i = 11:15
    subplot(6,5,i)
    [f,xi] = ksdensity(sims_old(:,i-5)); 
    plot(xi,f,'LineWidth',3);
    hold on 
    xline(sy(i-5),'LineWidth',3)
end

for i = 16:20
    subplot(6,5,i)
    [f,xi] = ksdensity(summaries(:,i-10)); 
    plot(xi,f,'LineWidth',3);
    hold on 
    xline(sy(i-10),'LineWidth',3)
end

for i = 21:25
    subplot(6,5,i)
    [f,xi] = ksdensity(sims_old(:,i-10)); 
    plot(xi,f,'LineWidth',3);
    hold on 
    xline(sy(i-10),'LineWidth',3)
end

for i = 26:30
    subplot(6,5,i)
    [f,xi] = ksdensity(summaries(:,i-15)); 
    plot(xi,f,'LineWidth',3);
    hold on 
    xline(sy(i-15),'LineWidth',3)
end

