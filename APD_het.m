% APD Heterogeneity Calculation

% Define good APs
good_APs = [1,2,3,4,7,8,9,10,12,13];
H = [];

% start for loop to go through all the good APs

for i = 1:length(good_APs)
    % File Name
    filename = fullfile('C:\Users\Sofia\Desktop\C24006_OMFigs\S1_400ms_bin5', ...
                    ['APD_data_' num2str(good_APs(i)) '.mat']);

    B = load(filename);
    A = B.apdMap;
    A = A(:);
    % Get 5% percentile
    P = prctile(A, 5);
    % Get 95% percentile
    L = prctile(A,95);
    % Get median
    M = nanmedian(A);
    % Calculate heterogeneity
    het = (L-P)/M;
    H = [H; het];
end

H