
%% SNR mask

% data = uigetfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\data.mat'); %choose your clean data file (change to your directory)
% data = load(data);
% data = struct2cell(data);
start = start*Fs;
endp = endp*Fs;
data{1,1} = data{1,1}(:,:,start:endp);
data= cell2mat(data);
data = normalize_data(data);
length_of_data =(endp-start+1);

disp('Loaded data') % Display messages throughout just to know where we are.

%% Create initial variables

% initialize your apd map
apdMap = nan(size(data, 1), size(data, 2));
AP_storage = nan(size(data, 1), size(data, 2));
AP_start_points = nan(size(data, 1), size(data, 2));
AP_end_points = nan(size(data, 1), size(data, 2));
AP_baselines = nan(size(data, 1), size(data, 2));
                        
% Define your threshold for repolarization (assume normalized data)
AP_level = 1.0 - percentAPD / 100;
% Will only use if you select a small area instead of the whole FOV
area_coords = int16(area_coords);
j_min = 1 + area_coords(1);
i_min = 1 + area_coords(2);
j_max = area_coords(1) + area_coords(3);
i_max = area_coords(2) + area_coords(4);
 %% BG time points
        % Plot one good pixel to choose the background noise and actual
        % signal. Right now it just plots pixel 150,150, might need to
        % change depending on the data. Make sure to change the pixel for
        % both figures.
        figure('Name','fig1')
        
        plot(1:length_of_data,squeeze(data(150,150,:)));
        % plot(1:length_of_data,squeeze(data(174,172,:)));
        % plot(1:length_of_data,squeeze(data(172,70,:))); % nov5

        title('Choose background noise')
        % In this figure, select a range of values where there should be no signal
        [xi] = getpts;
        BG_start_point = round(xi(1));
        BG_end_point = round(xi(2));
        close fig1

        %% Signal time points
        figure('Name','fig2')
        plot(1:length_of_data,squeeze(data(150,150,:)))
        % plot(1:length_of_data,squeeze(data(174,172,:)));
        % plot(1:length_of_data,squeeze(data(172,70,:))); % nov5

        title('Choose signal')
        % In this figure, select a range of values where there is signal
        [xi] = getpts;
        signal_start_point = round(xi(1));
        signal_end_point = round(xi(2));
        close fig2

        %% Get SNR mask
        % For each pixel, find the standard deviation of the data when there is no
        % signal. Then find the signal level (roughly the maximum value minus the
        % minimum value when you have an AP). Divide this number by the std of the
        % noise to get a SNR. For each pixel, store this ratio.
        for i = 1: 256
            for j = 1:256
                pixel_noise = squeeze(data(i,j,BG_start_point:BG_end_point));
                dev_noise = std(pixel_noise);
                pixel_S2 = squeeze(data(i,j,signal_start_point:signal_end_point));
                MSV = max(pixel_S2)-min(pixel_S2);
                SNR = MSV/dev_noise;
                if isnan(SNR)
                    SNR_matrix(i,j) = 0;
                else
                    SNR_matrix(i,j) = SNR;
                end
            end
        end

        % Visualize the SNR distribution
        figure
        imagesc(SNR_matrix)
        colorbar


        % Create a mask with the pixels where SNR is larger than your selected
        % threshold. Here we are saying SNR>2 but can change as needed.
        SNR_mask = zeros(256,256);
        SNR_mask(SNR_matrix>20) = 1;

        % Visualize your final mask
        figure
        imagesc(SNR_mask)