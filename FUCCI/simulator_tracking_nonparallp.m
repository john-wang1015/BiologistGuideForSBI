function sx = simulator_tracking_nonparallp(theta, len_obs, InitPosData, CellTrackingData)
    CellTracking = true;

    InitPosData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","InitPos");
    CellTrackingData = readmatrix("Data/DataProcessing/FUCCI_processed.xlsx", "sheet","CellTracking");
    
    ntrack = max(CellTrackingData(:,4));
    
    Xmax = 1309.09; %Length of the domain
    Ymax = 1745.35; %Width of the domain
    T_record = 48;

    s = SetupStruct(ntrack, Xmax, Ymax, InitPosData, CellTrackingData);

    sx = zeros(1,len_obs);
    
    % creating initial sample
    [SummaryStatData, ExitSimStatus] = Main_simulate(theta, s, T_record, CellTracking);
                
    summaries = GenerateSummaryStatistics(SummaryStatData, CellTracking, Xmax); %compute simulated summary stats

    sx(1,:) = summaries; %store simulated summary statistics
