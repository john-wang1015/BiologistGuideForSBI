function sx = simulator_density(theta, n, len_obs)
    CellTracking = false;

    InitPosData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
    CellTrackingData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");
    ntrack = max(CellTrackingData(:,4));
    
    Xmax = 1309.09; %Length of the domain
    Ymax = 1745.35; %Width of the domain
    T_record = 48;

    s = SetupStruct(ntrack, Xmax, Ymax, InitPosData, CellTrackingData);

    sx = zeros(n,len_obs);
    
    % creating initial sample
    parfor i = 1:n
        [SummaryStatData, ExitSimStatus] = Main_simulate(theta(i,:), s, T_record, CellTracking);
                              
        if ExitSimStatus %check if simulation finished correctly
            continue;
        end 
                
        summaries = GenerateSummaryStatistics(SummaryStatData, CellTracking, Xmax); %compute simulated summary stats
        if any(isnan(summaries)) %check integrity of summary statistics
           continue  
        end 

        sx(i,:) = summaries; %store simulated summary statistics
    end
