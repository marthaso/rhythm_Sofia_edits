function [alternanceMap] = alternanceMap(data,...
                                         start, endp,...
                                         minapd, maxapd,...
                                         percentAPD,...
                                         area_coords,...
                                         Fs, cmap, movie_scrn, handles)
%% the function apdMap creates a visual representation of the alternance distribution 
%
% INPUTS
% data          = cmos data
% start         = start time
% endp          = end time
% minapd        = minimal APD
% maxapd        = maximal APD
% percentAPD    = percent repolarization
% area_coords   = area coordinates
%                 [xmin, ymin, width, height]
% Fs            = sampling frequency
%
% OUTPUT
% A figure that has a color repersentation of alternance appearence.
%
% METHOD
% We calculate difference between the first and the second AP durations (APD).
% APD is being calculated as difference between adjucent moments of time
% when AP is lower than a given level (e.g. 0.45 for APD55).
% If two regions have opposite sign of the difference then they undergo
% discordant alternance (otherwise - concordant).
%
% AUTHOR: Pikunov Andrey (pikunov@phystech.edu)
%
% Email optocardiography@gmail.com for any questions or concerns.
% Refer to efimovlab.org for more information.

%% Load data.
data = uigetfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\data.mat'); %choose your clean data file (change to your directory)
data = load(data);
data = struct2cell(data);
start = start*Fs;
endp = endp*Fs;
data{1,1} = data{1,1}(:,:,start:endp);
data= cell2mat(data);
data = normalize_data(data);
length_of_data =(endp-start+1);

disp('Loaded data') % Display messages throughout just to know where we are.

%% Create initial variables

% initialize your alternance maps
alternanceMap = nan(size(data, 1), size(data, 2));
oddAP = nan(size(data, 1), size(data, 2));
evenAP = nan(size(data, 1), size(data, 2));
alternanceMap_2 = nan(size(data, 1), size(data, 2));
% Define your threshold for repolarization (assume normalized data)
AP_level = 1.0 - percentAPD / 100;
% Will only use if you select a small area instead of the whole FOV
area_coords = int16(area_coords);
j_min = 1 + area_coords(1);
i_min = 1 + area_coords(2);
j_max = area_coords(1) + area_coords(3);
i_max = area_coords(2) + area_coords(4);


%% SNR mask
[SNR_mask, SNR_matrix] = SNR_mask_calc(data,length_of_data);
% SNR_mask = zeros(256,256);
% SNR_mask(160,222) = 1;
% SNR_mask = zeros(256,256);
% SNR_mask(89,190) = 1;

%% Get APs locations and baselines
% a=1;
% pixel_example2 = squeeze(data(89,190,:));
% figure
% plot(pixel_example2)
% pixel_example2 = squeeze(data(103,76,:));
% figure
% plot(pixel_example2)

% % In this figure, select start of each AP, end of each AP and baseline for
% % each AP. This should be done in order (start AP #1, end AP #1, start
% % baseline AP #1, end baseline AP #2, start AP #2, end AP #2, etc)
% [xi] = getpts;
% start_points=[];
% end_points=[];
% baseline_start_points=[];
% baseline_end_points=[];
% % Create four vectors with the start points, end points, and baselines.
% for peak = 1:4:length(xi)-3
%     start_points = [start_points, round(xi(peak))];
%     end_points = [end_points, round(xi(peak+1))];
%     baseline_start_points = [baseline_start_points, round(xi(peak+2))];
%     baseline_end_points = [baseline_end_points, round(xi(peak+3))];
% end
% Multiply our data times the SNR mask we generated previously to only
% compute APDs in the good pixels.
data = data.*SNR_mask;
counting = 0;

%% Now find the APDs for one pixel
AP_storage = cell(256,256);
AP_start_points = cell(256,256);
AP_end_points = cell(256,256);
AP_baselines = cell(256,256);
% for pix_x = i_min : i_max
%     % pix_x
%     for pix_y = j_min : j_max
for pix_x = 1 : i_max
    % pix_x
    for pix_y = 1 : j_max
        % Discard pixels with no signal
        if max(data(pix_x,pix_y,:))==0

        else
            % Check if this pixel has the right number of peaks. Otherwise,
            % discard.
            pixel_calc = data(pix_x,pix_y,:);
            pixel_calc = normalize_data(pixel_calc);
            pixel_calc = squeeze(pixel_calc);
            [pks, locs] =findpeaks(pixel_calc,'MinPeakDistance',50,'MinPeakProminence',0.05);
            % If it has the right number of peaks, continue with the analysis.
            if length(pks) == 4
                % Create vectors where you will store APDs, start points, end
                % points, and baselines for all APs in this pixel.
                pixel_APDs = [];
                AP_start_points_1 = [];
                AP_end_points_1 = [];
                AP_baselines_1 = [];
                % Now, go through each AP to compute these values.
                for k = 1:length(pks)
                    % k = 1: the first AP is the first peak +- 1/2 max APD. Or
                    % the start/end of signal if it's on the edges
                    if locs(k) > maxapd/2
                        start_of_window = locs(k) - (maxapd/2);
                    else
                        start_of_window = 1;
                    end
                    if (locs(k) + maxapd/2) < length(pixel_calc)
                        end_of_window = locs(k) + (maxapd/2);
                    else
                        end_of_window = length(pixel_calc);
                    end
                    AP = pixel_calc(round(start_of_window):round(end_of_window));
                    % Find start point
                    [start_of_AP] = start_of_AP_calc(AP);
                     if ~isnan(start_of_AP)
                        start_of_AP_pixel = start_of_AP + start_of_window;
                        
                    else
                        
                        start_of_AP_pixel = NaN;
                    end
                    
                    

                    % Find baseline
                    [baseline_of_AP] = baseline_of_AP_calc(AP);
                                  

                    % Find endpoint
                    [end_of_AP] = end_of_AP_calc(AP,baseline_of_AP,AP_level);
                    if any(~isnan(end_of_AP)) && any(~isnan(start_of_AP))
                        end_of_AP_pixel = end_of_AP + start_of_window;
                        % Find APD
                        apd = end_of_AP - start_of_AP;
                    else
                        apd = NaN;
                        end_of_AP_pixel = NaN;
                    end



                    if ~isscalar(apd)
                        disp('apd is NOT a scalar');
                    else
                        if ((minapd < apd) && (apd < maxapd))
                            pixel_APDs = [pixel_APDs,apd];
                            AP_start_points_1 = [AP_start_points_1,start_of_AP_pixel];
                            AP_end_points_1 = [AP_end_points_1,end_of_AP_pixel];
                            AP_baselines_1 = [AP_baselines_1,baseline_of_AP];
                        end
                    end
                    if isnan(apd)
                        pixel_APDs = [pixel_APDs,apd];
                        AP_start_points_1 = [AP_start_points_1,start_of_AP_pixel];
                        AP_end_points_1 = [AP_end_points_1,end_of_AP_pixel];
                        AP_baselines_1 = [AP_baselines_1,baseline_of_AP];
                    end

                end
                % For the pixel, store info for all APs
                AP_storage{pix_x,pix_y}=pixel_APDs;
                AP_start_points{pix_x,pix_y} = AP_start_points_1;
                AP_baselines{pix_x,pix_y} = AP_baselines_1;
                AP_end_points{pix_x,pix_y} = AP_end_points_1;

                % Compute alternans
                [alternan_value,odd_average, even_average] = alternan_value_calc(pixel_APDs);
                alternanceMap(pix_x,pix_y) = alternan_value;
                oddAP(pix_x,pix_y) = odd_average;
                evenAP(pix_x,pix_y) = even_average;

            end
            
        end
    end
end
% plot_alternans(alternanceMap)
% nanmean(nanmean(abs(alternanceMap)))
[clean_alternanceMap] = clean_Map(alternanceMap);
plot_alternans(clean_alternanceMap)
nanmean(nanmean(clean_alternanceMap))
% figure
% histogram(alternanceMap)
figure
histogram(clean_alternanceMap)
total_mean = nanmean(nanmean(abs(clean_alternanceMap)))
total_odd = nanmean(nanmean(oddAP))
total_even = nanmean(nanmean(evenAP))
total_std = nanstd(clean_alternanceMap(:))
% plot examples
% find pixels with signal
% [y,x] = find(SNR_mask==1);
% number of points you want
%k = 10;

% take absolute value
A = SNR_matrix.*SNR_mask;
A(A==0) = NaN;

a = [];
k=1;
mini = 0;
[~, mini] = mink(A(:), 1000);
while length(a) < 5
    
    if ~isnan(clean_alternanceMap(mini(k)))
        a = [a,mini(k)];
    end
    
    k = k+1;
end
        



% % get k largest values and their linear indices
% [~, a] = mink(A(:), k);

% convert linear indices to (row, col)
[y, x] = ind2sub(size(A), a);
for pix = 1:5
    [fig] = plot_examples_simple(data,y(pix),x(pix),AP_start_points{y(pix)...
        ,x(pix)},AP_baselines{y(pix),x(pix)},AP_end_points{y(pix),x(pix)},AP_storage{y(pix),x(pix)},...
        clean_alternanceMap(y(pix),x(pix)));
    SNR_matrix(y(pix),x(pix))
end

[fig] = plot_examples_simple(data,y,x,AP_start_points{y...
        ,x},AP_baselines{y,x},AP_end_points{y,x},AP_storage{y,x});
a=1;

    function [alternan_value, odd_average, even_average] = alternan_value_calc(pixel_APDs)
        odds = [];
        evens = [];
        for odds_even = 1:2:length(pixel_APDs)-1
            odds = [odds, pixel_APDs(odds_even)];
            evens = [evens, pixel_APDs(odds_even+1)];
        end
        odd_average = nanmean(odds);
        even_average = nanmean(evens);
        alternan_value = odd_average-even_average;

    end

    function plot_alternans(alternanceMap)
        handles.activeCamData.saveData = alternanceMap;
        alternanceMap = handles.activeCamData.saveData;
        figure;
        imagesc(alternanceMap, 'AlphaData', ~isnan(alternanceMap));
        set(gca, 'Color', 'k');
        colormap redblue(256);
        alternance_max = max(max(abs(alternanceMap(:))));
        alternance_min = -alternance_max;
        clim([alternance_min alternance_max]);
        cb = colorbar;
        cb_label = sprintf('Alternance for APD%d (ms)', int8(handles.percentAPD));
        ylabel(cb, cb_label);
    end

    function [clean_alternanceMap] = clean_Map(alternanceMap)
        p5  = prctile(alternanceMap(:), 2);
        p95 = prctile(alternanceMap(:), 98);

        % Replace values outside percentile range with NaN
        clean_alternanceMap = alternanceMap;
        clean_alternanceMap(alternanceMap < p5 | alternanceMap > p95) = NaN;
        % % Extract only the valid (non-NaN) values
        % vals = alternanceMap(~isnan(alternanceMap));
        % 
        % % Number of pixels to threshold (0.5% of all non-NaN values)
        % n = numel(vals);
        % kp = ceil(0.01 * n);   % 1%
        % 
        % % Sort the values
        % sortedVals = sort(vals);
        % 
        % % Determine lower and upper cutoff thresholds
        % lowerThresh = sortedVals(kp);
        % upperThresh = sortedVals(end-kp+1);
        % 
        % % Create a mask of values to NaN (both top and bottom 0.5%)
        % mask = (alternanceMap <= lowerThresh) | (alternanceMap >= upperThresh);
        % 
        % % Apply the mask
        % alternanceMap(mask) = NaN;
        % clean_alternanceMap = alternanceMap;
    end

    function [end_of_AP] = end_of_AP_calc(AP,AP_baseline,AP_level)
        averaged_AP = movmean(AP, 5, 'Endpoints', 'discard');
        % Find the peak of AP
        [~,max_place] = findpeaks(averaged_AP,'MinPeakDistance',50,'MinPeakProminence',0.05);
        max_place = max(max_place);
        if isempty(max_place)
            [~,max_place] = findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.05);
        end
        % Find location of peak
        norm_AP = max(AP,AP_baseline);
        norm_AP = (norm_AP - min(norm_AP)) / (max(norm_AP) - min(norm_AP));
        norm_AP = norm_AP(max_place:length(norm_AP));
        points_lower_than_threshold = find(norm_AP <= AP_level);
        if isempty(points_lower_than_threshold)
            end_of_AP = NaN;
        else
            end_of_AP = points_lower_than_threshold(1) + max_place;
        end
        
    end
    
    function [baseline_of_AP] = baseline_of_AP_calc(AP)
        averaged_AP = movmean(AP, 5, 'Endpoints', 'discard');
        % Find the peak of AP and the lowest point before the peak
        [~,max_place] = findpeaks(averaged_AP,'MinPeakDistance',50,'MinPeakProminence',0.05);
        if isempty(max_place)
            [~,max_place] = findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.05);
        end
        if max_place<length(averaged_AP)
            [~,min_place] = min(averaged_AP(1:max_place));
        else
            [~,min_place] = min(averaged_AP);
        end
        points = [AP(min_place),AP(min_place+1),AP(min_place+2)];
        baseline_of_AP = mean(points);

    end

    function [start_of_AP] = start_of_AP_calc(AP)
        averaged_AP = movmean(AP, 5, 'Endpoints', 'discard');
        % Find the peak of AP and the lowest point before the peak
        [~,max_place] = findpeaks(averaged_AP,'MinPeakDistance',50,'MinPeakProminence',0.05);
        if isempty(max_place)
            [~,max_place] = findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.05);
        end
        max_place = max(max_place);
        % Normalize AP
        averaged_AP = averaged_AP - min(averaged_AP);
        averaged_AP = averaged_AP/max(averaged_AP);
        if max_place>length(averaged_AP)
            start_of_AP = NaN;
        else
        [~,start_of_AP] = min(abs(averaged_AP(1:max_place) - 0.5));
        % [~,min_place] = min(averaged_AP(1:max_place));
        start_of_AP = start_of_AP + 2;
        end

    end

    function [fig] = plot_examples_simple(data,x,y,AP_start_points,AP_baselines,AP_end_points,APD,alternance)
        figure
        pixel_example = squeeze(data(x,y,:));
        plot(pixel_example)
        hold on
        colors = lines(length(APD));
        %APDs = [];
        for i = 1:length(APD)
            c = colors(i,:);
            baseline_pixel = ones(1,length(pixel_example))*AP_baselines(i);
            plot(1:length(pixel_example),baseline_pixel,'Color',c,'LineWidth', 3)
            %
            hold on
            S2_start_pixel = ones(1,11)*AP_start_points(i);
            plot(S2_start_pixel, 0:0.1:1,'Color',c,'LineWidth', 3)
            S2_end_pixel = ones(1,11)*AP_end_points(i);
            plot(S2_end_pixel, 0:0.1:1,'Color',c,'LineWidth', 3)
            APDval = APD(i)
            % APD = AP_end_points{x,y}(i) - AP_start_points{x,y}(i)
            % APDs = [APDs, APD];

        end
        alternance
        fig=1;
    end

    function [SNR_mask,SNR_matrix] = SNR_mask_calc(data,length_of_data)

        %% BG time points
        figure('Name','fig1')
        % plot(1:4999,squeeze(data(150,150,:))); % find the time points of BG in a good pixel
        plot(1:length_of_data,squeeze(data(150,150,:)));
        title('Choose background noise')
        % In this figure, select a range of values where there should be no signal
        [xi] = getpts;
        BG_start_point = round(xi(1));
        BG_end_point = round(xi(2));
        close fig1

        %% Signal time points
        figure('Name','fig2')
        plot(1:length_of_data,squeeze(data(150,150,:)))
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
        % threshold.
        SNR_mask = zeros(256,256);
        SNR_mask(SNR_matrix>8) = 1;
        

        % % Create a mask with a specific number of pixels
        % SNR_mask = zeros(256,256);
        % N = 30000;
        % 
        % 
        % [~, idx] = sort(SNR_matrix(:), 'descend');
        % 
        % % Select the top N indices
        % topN = idx(1:N);
        % 
        % % Set those locations to 1 in the mask
        % SNR_mask(topN) = 1;

        % Visualize your final mask
        figure
        imagesc(SNR_mask)

    end
    
    function [long_APD_final,short_APD_final,average_differences]=restitution_vals(alternanceMap,AP_storage)
        % Choose pixels with a positive difference
        idx = find(alternanceMap >= (0.2*max(max(alternanceMap))));     % linear indices
        [row, col] = ind2sub(size(alternanceMap), idx);
        odd_avg  = zeros(size(row));
        even_avg = zeros(size(row));

        % For each pixel, get average of long and average of short
        for pixel_high_alternan = 1:length(row)
            APD_values = AP_storage{row(pixel_high_alternan),col(pixel_high_alternan)};
            if isempty(APD_values)
                odd_avg(pixel_high_alternan) = NaN;
                even_avg(pixel_high_alternan) = NaN;
            else

                odd_avg(pixel_high_alternan)  = mean(APD_values(1:2:end));  % odd indices
                if length(APD_values) >= 2

                    even_avg(pixel_high_alternan) = mean(APD_values(2:2:end));  % even indices
                else
                    even_avg(pixel_high_alternan) = NaN;
                end


            end
        end
        %% Get final values for restitution curve
        long_APD = odd_avg;
        short_APD = even_avg;
        % Choose pixels with a negative difference
        idx = find(alternanceMap <= (0.2*min(min(alternanceMap))));     % linear indices
        [row, col] = ind2sub(size(alternanceMap), idx);
        odd_avg  = zeros(size(row));
        even_avg = zeros(size(row));
        % For each pixel, get average of long and average of short
        for pixel_high_alternan = 1:length(row)
            APD_values = AP_storage{row(pixel_high_alternan),col(pixel_high_alternan)};
            if isempty(APD_values)
                odd_avg(pixel_high_alternan) = NaN;
                even_avg(pixel_high_alternan) = NaN;
            else
                odd_avg(pixel_high_alternan)  = mean(APD_values(1:2:end));  % odd indices
                if length(APD_values) >= 2
                    even_avg(pixel_high_alternan) = mean(APD_values(2:2:end));  % even indices
                else
                    even_avg(pixel_high_alternan) = NaN;
                end
            end
        end

        %% Get final values for restitution curve
        long_APD = [long_APD;even_avg];
        short_APD = [short_APD;odd_avg];

        long_APD_final = mean(long_APD);
        short_APD_final = mean(short_APD);
        average_differences = long_APD_final - short_APD_final;
    end

% function [fig] = plot_examples(data,x,y,AP_start_points_1)
    %     figure
    %     pixel_example = squeeze(data(x,y,:));
    %     plot(pixel_example)
    %     hold on
    %     colors = lines(length(AP_baselines{x,y}));
    %     APDs = [];
    %     for i = 1:length(AP_baselines{x,y})
    %         c = colors(i,:);
    %         baseline_pixel = ones(1,length(pixel_example))*AP_baselines{x,y}(i);
    %         plot(1:length(pixel_example),baseline_pixel,'Color',c,'LineWidth', 3)
    % 
    %         hold on
    %         S2_start_pixel = ones(1,11)*AP_start_points{x,y}(i);
    %         plot(S2_start_pixel, 0:0.1:1,'Color',c,'LineWidth', 3)
    %         S2_end_pixel = ones(1,11)*AP_end_points{x,y}(i);
    %         plot(S2_end_pixel, 0:0.1:1,'Color',c,'LineWidth', 3)
    %         APD = AP_end_points{x,y}(i) - AP_start_points{x,y}(i)
    %         APDs = [APDs, APD];
    % 
    %     end
    %     dif_APDs = [];
    %     for num_APDs = 1:2:length(APDs)-1
    %         dif_APD = APDs(num_APDs+1) - APDs(num_APDs);
    %         dif_APDs = [dif_APDs,dif_APD];
    %     end
    %     mean(dif_APDs)
    %     % figure
    %     % plot(pixel_example)
    %     fig=1;
    % end
    % 
    % 
    % 

%                     start_window = AP(min_place:max_place);
%                     % figure
%                     % plot(new_AP)
%                     window_size = 5;
%                     averaged_new_AP = movmean(start_window, window_size, 'Endpoints', 'discard');
%                     averaged_new_AP = [averaged_new_AP(1);averaged_new_AP(1);averaged_new_AP];
%                     % hold on
%                     % plot(averaged_new_AP)
%                     diff_new_AP = diff(averaged_new_AP);
%                     averaged_differences = movmean(diff_new_AP, window_size, 'Endpoints', 'discard');
%                     averaged_differences = [averaged_differences(1);averaged_differences(1);averaged_differences];
%                     averaged_differences = movmean(averaged_differences, window_size, 'Endpoints', 'discard');
%                     averaged_differences = [averaged_differences(1);averaged_differences(1);averaged_differences];
%                     averaged_differences = movmean(averaged_differences, window_size, 'Endpoints', 'discard');
%                     averaged_differences = [averaged_differences(1);averaged_differences(1);averaged_differences];
%                     % figure
%                     % plot(averaged_differences)
% 
%                     [~,start_of_AP] = max(averaged_differences);
%                     start_of_AP = min_place + start_of_AP;
% 
% 
% 
% 
% 
%                     % new_AP = normalize_data(new_AP);
%                     %
%                     %
%                     % window = round((max_place-min_place)/4);
%                     % start_window = AP(min_place+window:max_place-window);
%                     % % average points to find start points
%                     % windowSize = 20;
%                     %
%                     % % Compute sliding averages efficiently
%                     % avgVals_1 = movmean(start_window, windowSize, 'Endpoints', 'discard');
%                     %
%                     % avgVals = diff(avgVals_1);
%                     % % Find index of maximum average
%                     % [~, idxMax] = max(avgVals);
%                     %
%                     % % Convert sliding-window index to original center position
%                     % start_of_AP = idxMax + floor(windowSize/2)+window+min_place;
% 
% 
% 
% 
% 
% 
%         if max(data(pix_x,pix_y,:))==0
% 
%         else
% 
%            pixel_calc = data(pix_x,pix_y,:);
%             pixel_calc = normalize_data(pixel_calc);
%             pixel_calc = squeeze(pixel_calc);
%             [pks, locs] =findpeaks(pixel_calc,'MinPeakDistance',100,'MinPeakProminence',0.05);
%             if length(pks) == 4
%             % figure
%             % plot(pixel_calc)
%             pixel_APDs = [];
%             AP_start_points_1 = [];
%             AP_end_points_1 = [];
%             AP_baselines_1 = [];
%             for k = 1:length(start_points)
%                 AP = pixel_calc(start_points(k):end_points(k));
% 
%                 %%% START POINT
% 
%                 [~,max_place] = max(AP);
%                 if max_place < 10
%                     break
%                 else
%                     if max_place>120
%                         [~,min_place] = min(AP(max_place-120:max_place));
%                         min_place = max_place-121+min_place;
%                     else
%                         [~,min_place] = min(AP(1:max_place));
% 
%                     end
% 
%                     start_window = AP(min_place:max_place);
%                     % figure
%                     % plot(new_AP)
%                     window_size = 5;
%                     averaged_new_AP = movmean(start_window, window_size, 'Endpoints', 'discard');
%                     averaged_new_AP = [averaged_new_AP(1);averaged_new_AP(1);averaged_new_AP];
%                     % hold on
%                     % plot(averaged_new_AP)
%                     diff_new_AP = diff(averaged_new_AP);
%                     averaged_differences = movmean(diff_new_AP, window_size, 'Endpoints', 'discard');
%                     averaged_differences = [averaged_differences(1);averaged_differences(1);averaged_differences];
%                     averaged_differences = movmean(averaged_differences, window_size, 'Endpoints', 'discard');
%                     averaged_differences = [averaged_differences(1);averaged_differences(1);averaged_differences];
%                     averaged_differences = movmean(averaged_differences, window_size, 'Endpoints', 'discard');
%                     averaged_differences = [averaged_differences(1);averaged_differences(1);averaged_differences];
%                     % figure
%                     % plot(averaged_differences)
% 
%                     [~,start_of_AP] = max(averaged_differences);
%                     start_of_AP = min_place + start_of_AP;
% 
% 
% 
% 
% 
%                     % new_AP = normalize_data(new_AP);
%                     %
%                     %
%                     % window = round((max_place-min_place)/4);
%                     % start_window = AP(min_place+window:max_place-window);
%                     % % average points to find start points
%                     % windowSize = 20;
%                     %
%                     % % Compute sliding averages efficiently
%                     % avgVals_1 = movmean(start_window, windowSize, 'Endpoints', 'discard');
%                     %
%                     % avgVals = diff(avgVals_1);
%                     % % Find index of maximum average
%                     % [~, idxMax] = max(avgVals);
%                     %
%                     % % Convert sliding-window index to original center position
%                     % start_of_AP = idxMax + floor(windowSize/2)+window+min_place;
% 
%                     %%% BASELINE
% 
%                     % baseline as a fixed point before the start point
%                     % % find differences in the signal
%                     % differences_baseline = diff(AP);
%                     % % find the index of the greatest differences
%                     % [~, location_for_baseline] = max(differences_baseline);
%                     location_for_baseline = start_of_AP;
%                     % Subtract a fixed time
%                     % if location_for_baseline > 45
%                     %     AP_baseline = AP(location_for_baseline - 45);
%                     % else
%                     %     AP_baseline = min(AP(1:location_for_baseline));
%                     % end
%                     if location_for_baseline > 75
%                         [~, location_baseline] = min(AP(location_for_baseline-75:location_for_baseline));
%                         if location_baseline > 1
%                             baseline_points=[AP(location_for_baseline-75+location_baseline-1),AP(location_for_baseline-75+location_baseline),AP(location_for_baseline-75+location_baseline+1)];
%                             AP_baseline = mean(baseline_points);
%                         else
%                             baseline_points=[AP(location_for_baseline-75),AP(location_for_baseline-75+1),AP(location_for_baseline-75+2)];
%                             AP_baseline = mean(baseline_points);
%                         end
%                     else
%                         [~, location_baseline] = min(AP(1:location_for_baseline));
%                         if location_baseline > 1
%                             baseline_points=[AP(location_baseline-1),AP(location_baseline),AP(location_baseline+1)];
%                             AP_baseline = mean(baseline_points);
%                         else
%                             baseline_points=[AP(location_baseline),AP(location_baseline+1),AP(location_baseline+2)];
%                             AP_baseline = mean(baseline_points);
%                         end
%                     end
% 
% 
% 
% 
%                     % baseline as the min point before it goes up
%                     % dAP = diff(AP);
%                     % % index of the max derivative
%                     % [~, ind_der] = max(dAP);
%                     % % find last derivative == 0 before that point
%                     % %region = dAP(1:ind_der);
%                     % j_Der = find(dAP(1:ind_der) <= 0, 1, 'last');
%                     % AP_baseline = min(AP);
%                     % if ~isempty(j_Der)
%                     %     AP_baseline = AP(j_Der);
%                     % end
% 
% 
% 
%                     % % baseline as an average of the two points chosen manually
%                     % AP_baseline = mean(pixel_calc(baseline_start_points(k):baseline_end_points(k)));
% 
%                     %%% END POINT
% 
%                     AP = max(AP, AP_baseline);
%                     AP = (AP - min(AP)) / (max(AP) - min(AP));
% 
% 
%                     % figure
%                     % plot(AP)
%                     points_lower_than_threshold = find(AP <= AP_level); % find all time points where signal is less than your threshold
%                     if size(points_lower_than_threshold, 1) > 2
%                         notnotch = 0;
%                         % find how far apart these points are
%                         differences = diff(points_lower_than_threshold);
%                         [~, location] = max(differences);
%                         while notnotch == 0
%                             % find the index of the greatest differences
%                             end_of_AP = location + max(differences);
%                             if length(AP)<(end_of_AP+2)
%                                 p=1;
%                                 break
%                             end
%                             if (AP(end_of_AP)>AP(end_of_AP+1))&&...
%                                     (AP(end_of_AP+1)>AP(end_of_AP+2))
% 
% 
% 
%                                 % [~, start_of_AP] = max(diff(AP));
%                                 apd = end_of_AP - start_of_AP;
%                                 notnotch = 1;
%                             else
%                                 location = location+1;
%                             end
%                         end
% 
%                         % pix_x
%                         % pix_y
%                         % if pix_x == 60
%                         %     a=1;
%                         % end
%                         if ~isempty(apd)
%                             if ((minapd < apd) && (apd < maxapd))
%                                 pixel_APDs = [pixel_APDs,apd];
%                                 AP_start_points_1 = [AP_start_points_1,(start_of_AP+start_points(k))];
%                                 AP_end_points_1 = [AP_end_points_1,(start_points(k)+start_of_AP+apd)];
%                                 AP_baselines_1 = [AP_baselines_1,AP_baseline];
%                             end
%                         end
%                     end
% 
%                 end
%                 AP_storage{pix_x,pix_y}=pixel_APDs;
%                 AP_start_points{pix_x,pix_y} = AP_start_points_1;
%                 AP_end_points{pix_x,pix_y} = AP_end_points_1;
%                 AP_baselines{pix_x,pix_y} = AP_baselines_1;
%                 % Option #1 = average odds, average evens. Subtract the average
%                 odds = [];
%                 evens = [];
%                 for odds_even = 1:2:length(pixel_APDs)-1
%                     odds = [odds, pixel_APDs(odds_even)];
%                     evens = [evens, pixel_APDs(odds_even+1)];
%                 end
%                 odd_average = mean(odds);
%                 even_average = mean(evens);
%                 alternan_diff = odd_average-even_average;
%                 alternanceMap(pix_x,pix_y) = alternan_diff;
%                 % if abs(alternan_diff) > 10
%                 %     if counting < 10
%                 %     figure
%                 %     plot(pixel_calc)
%                 %     hold on
%                 %     baseline_pixel = ones(1,4999)*AP_baseline;
%                 %     plot(1:4999,baseline_pixel)
%                 %     hold on
%                 %     S2_start_pixel = ones(1,11)*(start_of_AP+start_points(k));
%                 %     plot(S2_start_pixel, 0:0.1:1)
%                 %     hold on
%                 %     S2_end_pixel = ones(1,11)*(start_points(k)+start_of_AP+apd);
%                 %     plot(S2_end_pixel, 0:0.1:1)
%                 %     counting = counting +1;
%                 %     end
%                 % end
% 
%                 % Option #2 = Do 2-1, 3-2, 4-3, etc. Average the abs value of all.
%                 AP_diff = abs(diff(pixel_APDs));
%                 alternanceMap_2(pix_x,pix_y) = mean(AP_diff);
%             end
%             end
%         end
%     end
% end
% % figure
% % imagesc(alternanceMap)
% % colorbar
% % figure
% % imagesc(alternanceMap_2)
% % colorbar
% 
% %%%%%%%%%%%%%%%%%%%%%
% 
% take away the top and bottom 0.5%
% A is your 256x256 matrix containing numbers and NaNs

% % Extract only the valid (non-NaN) values
% vals = alternanceMap(~isnan(alternanceMap));
% 
% % Number of pixels to threshold (0.5% of all non-NaN values)
% n = numel(vals);
% k = ceil(0.005 * n);   % 0.5%
% 
% % Sort the values
% sortedVals = sort(vals);
% 
% % Determine lower and upper cutoff thresholds
% lowerThresh = sortedVals(k);
% upperThresh = sortedVals(end-k+1);
% 
% % Create a mask of values to NaN (both top and bottom 0.5%)
% mask = (alternanceMap <= lowerThresh) | (alternanceMap >= upperThresh);
% 
% % Apply the mask
% alternanceMap(mask) = NaN;
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% 
% 
% 
% 
% handles.activeCamData.saveData = alternanceMap;
% alternanceMap = handles.activeCamData.saveData;
% figure;
% imagesc(alternanceMap, 'AlphaData', ~isnan(alternanceMap));
% set(gca, 'Color', 'k');
% colormap redblue(256);
% alternance_max = max(max(abs(alternanceMap(:))));
% alternance_min = -alternance_max;
% if alternance_max<=alternance_min
%     a=1
% end
% clim([alternance_min alternance_max]);
% cb = colorbar;
% cb_label = sprintf('Alternance for APD%d (ms)', int8(handles.percentAPD));
% ylabel(cb, cb_label);
% %
% % handles.activeCamData.saveData = alternanceMap_2;
% % alternanceMap_2 = handles.activeCamData.saveData;
% % figure;
% % imagesc(alternanceMap_2, 'AlphaData', ~isnan(alternanceMap_2));
% % set(gca, 'Color', 'k');
% % colormap redblue(256);
% % alternance_max = max(max(abs(alternanceMap_2(:))));
% % alternance_min = -alternance_max;
% % caxis([alternance_min alternance_max]);
% % cb = colorbar;
% % cb_label = sprintf('Alternance for APD%d (ms)', int8(handles.percentAPD));
% % ylabel(cb, cb_label);
% 
% 
% %% get restitution curves for pixels with high alternan rate
% 
% % Choose pixels with a positive difference
% idx = find(alternanceMap >= (0.2*max(max(alternanceMap))));     % linear indices
% [row, col] = ind2sub(size(alternanceMap), idx);
% odd_avg  = zeros(size(row));
% even_avg = zeros(size(row));
% 
% % For each pixel, get average of long and average of short
% for pixel_high_alternan = 1:length(row)
%     APD_values = AP_storage{row(pixel_high_alternan),col(pixel_high_alternan)};
%     if isempty(APD_values)
%         odd_avg(pixel_high_alternan) = NaN;
%         even_avg(pixel_high_alternan) = NaN;
%     else
% 
%         odd_avg(pixel_high_alternan)  = mean(APD_values(1:2:end));  % odd indices
%         if length(APD_values) >= 2
% 
%             even_avg(pixel_high_alternan) = mean(APD_values(2:2:end));  % even indices
%         else
%             even_avg(pixel_high_alternan) = NaN;
%         end
% 
% 
%     end
% end
% 
% %% Get final values for restitution curve
% long_APD = odd_avg;
% short_APD = even_avg;
% 
% % Choose pixels with a negative difference
% idx = find(alternanceMap <= (0.2*min(min(alternanceMap))));     % linear indices
% [row, col] = ind2sub(size(alternanceMap), idx);
% odd_avg  = zeros(size(row));
% even_avg = zeros(size(row));
% 
% 
% 
% 
% 
% % For each pixel, get average of long and average of short
% for pixel_high_alternan = 1:length(row)
%     APD_values = AP_storage{row(pixel_high_alternan),col(pixel_high_alternan)};
%     if isempty(APD_values)
%         odd_avg(pixel_high_alternan) = NaN;
%         even_avg(pixel_high_alternan) = NaN;
%     else
% 
%         odd_avg(pixel_high_alternan)  = mean(APD_values(1:2:end));  % odd indices
%         if length(APD_values) >= 2
% 
%             even_avg(pixel_high_alternan) = mean(APD_values(2:2:end));  % even indices
%         else
%             even_avg(pixel_high_alternan) = NaN;
%         end
% 
% 
%     end
% end
% 
% %% Get final values for restitution curve
% long_APD = [long_APD;even_avg];
% short_APD = [short_APD;odd_avg];
% 
% long_APD_final = mean(long_APD)
% short_APD_final = mean(short_APD)
% average_differences = long_APD_final - short_APD_final
% 
% [fig] = plot_examples(y,x)
% 
% 
% 
% a=1;
% 
% 
% 
% % average all the longs and all the shorts for a final long and short value
% 
% 
% 
% 
%     function [SNR_mask] = SNR_mask_calc(data,length_of_data)
% 
%         %% BG time points
%         figure('Name','fig1')
%         % plot(1:4999,squeeze(data(150,150,:))); % find the time points of BG in a good pixel
%         plot(1:length_of_data,squeeze(data(89,190,:)));
%         title('Choose background noise')
%         % In this figure, select a range of values where there should be no signal
%         [xi] = getpts;
%         BG_start_point = round(xi(1));
%         BG_end_point = round(xi(2));
%         close fig1
% 
%         %% Signal time points
%         figure('Name','fig2')
%         plot(1:length_of_data,squeeze(data(89,190,:)))
%         title('Choose signal')
%         % In this figure, select a range of values where there is signal
%         [xi] = getpts;
%         signal_start_point = round(xi(1));
%         signal_end_point = round(xi(2));
%         close fig2
% 
%         %% Get SNR mask
%         % For each pixel, find the standard deviation of the data when there is no
%         % signal. Then find the signal level (roughly the maximum value minus the
%         % minimum value when you have an AP). Divide this number by the std of the
%         % noise to get a SNR. For each pixel, store this ratio.
% for i = 1: 256
%     for j = 1:256
%         pixel_noise = squeeze(data(i,j,BG_start_point:BG_end_point));
%         dev_noise = std(pixel_noise);
%         pixel_S2 = squeeze(data(i,j,signal_start_point:signal_end_point));
%         MSV = max(pixel_S2)-min(pixel_S2);
%         SNR = MSV/dev_noise;
%         if isnan(SNR)
%             SNR_matrix(i,j) = 0;
%         else
%             SNR_matrix(i,j) = SNR;
%         end
%     end
% end
% 
% % Visualize the SNR distribution
% figure
% imagesc(SNR_matrix)
% colorbar
% % Create a mask with the pixels where SNR is larger than your selected
% % threshold.
% SNR_mask = zeros(256,256);
% N = 1;
% 
% 
% [~, idx] = sort(SNR_matrix(:), 'descend');
% 
% % Select the top N indices
% topN = idx(1:N);
% 
% % Set those locations to 1 in the mask
% SNR_mask(topN) = 1;
% 
% % Visualize your final mask
% figure
% imagesc(SNR_mask)
% 
% end
% 
%     function [fig] = plot_examples(x,y)
%         figure
%         pixel_example = squeeze(data(x,y,:));
%         plot(pixel_example)
%         hold on
%         colors = lines(length(AP_baselines{x,y}));
%         APDs = [];
%         for i = 1:length(AP_baselines{x,y})
%             c = colors(i,:);
%             baseline_pixel = ones(1,length(pixel_example))*AP_baselines{x,y}(i);
%             plot(1:length(pixel_example),baseline_pixel,'Color',c,'LineWidth', 3)
% 
%             hold on
%             S2_start_pixel = ones(1,11)*AP_start_points{x,y}(i);
%             plot(S2_start_pixel, 0:0.1:1,'Color',c,'LineWidth', 3)
%             S2_end_pixel = ones(1,11)*AP_end_points{x,y}(i);
%             plot(S2_end_pixel, 0:0.1:1,'Color',c,'LineWidth', 3)
%             APD = AP_end_points{x,y}(i) - AP_start_points{x,y}(i)
%             APDs = [APDs, APD];
% 
%         end
%         dif_APDs = [];
%         for num_APDs = 1:2:length(APDs)-1
%             dif_APD = APDs(num_APDs+1) - APDs(num_APDs);
%             dif_APDs = [dif_APDs,dif_APD];
%         end
%         mean(dif_APDs)
%         % figure
%         % plot(pixel_example)
%         fig=1;
%     end
% 
% 
% 
% 
% % % get every APD
% % % [apdMap] = apdMap(data, start, endp,minapd, maxapd, percentAPD, area_coords, Fs, cmap, movie_scrn, handles);
% % %% the function apdMap creates a visual representation of
% % 
% % %% Map calculation
% % % for i = i_min : i_max
% % %     for j = j_min : j_max
% % for i = 150 : i_max
% %     for j = 150 : j_max
% % 
% % 
% % 
% %         % find all APDs
% %         % Option #1 = average odds, average evens. Subtract the average
% %         % Option #2 = Do 2-1, 3-2, 4-3, etc. Average the abs value of all.
% % 
% %         % og code - find first two APs. find diferences between them. set
% %         % alternance to that. 
% %         index = find(ap_data(i, j, :) < AP_level);
% % 
% %         if size(index, 1) > 2
% %             spaces = index(2: end) - index(1: end - 1);
% %             peak_index = find((spaces > APD_min_rescaled) & (spaces < APD_max_rescaled), 2); % find first two indices
% % 
% %             if size(peak_index, 1) == 2  
% %                 first_peak_value = spaces(peak_index(1));
% %                 second_peak_value = spaces(peak_index(2));
% % 
% %                 alternance_value = first_peak_value - second_peak_value;
% %                 alternanceMap(i, j) = alternance_value;
% %             end
% %         end
% %     end
% % end
% % 
% % % account for different sampling frequencies
% % unitFix = 1000.0 / Fs;
% % alternanceMap = alternanceMap * unitFix;
% % 
% % handles.activeCamData.saveData = alternanceMap;
% % 
% % %% Plot
% % cla(movie_scrn);
% % 
% % colormap(handles.activeScreen, cmap);
% % imagesc(movie_scrn, alternanceMap, 'AlphaData', ~isnan(alternanceMap));
% % set(movie_scrn,'Color','k')
% % set(movie_scrn,'YDir','reverse');
% % set(movie_scrn,'YTick',[],'XTick',[]);
% % 
% % alternance_max = max(max(abs(alternanceMap(:))));
% % alternance_min = -alternance_max;
% % 
% % caxis(movie_scrn,[alternance_min alternance_max]);
% % 
% % %% Calculating statistics
% % alternance_mean=nanmean(alternanceMap(:));
% % disp(['The average alternance in the region is ' num2str(alternance_mean) ' (ms).'])
% % alternance_std=nanstd(alternanceMap(:));
% % disp(['The standard deviation of alternance in the region is ' num2str(alternance_std) ' (ms).'])
% % alternance_median=nanmedian(alternanceMap(:));
% % disp(['The median alternance in the region is ' num2str(alternance_median) ' (ms).'])
% % 
% % handles.activeCamData.meanresults           = sprintf('Mean: %0.3f (ms)',alternance_mean);
% % handles.activeCamData.medianresults         = sprintf('Median: %0.3f (ms)',alternance_median);
% % handles.activeCamData.SDresults             = sprintf('S.D.: %0.3f (ms)',alternance_std);
% % handles.activeCamData.num_membersresults    = sprintf('');
% % handles.activeCamData.angleresults          = sprintf('');
% % 
% % set(handles.meanresults,'String',handles.activeCamData.meanresults);
% % set(handles.medianresults,'String',handles.activeCamData.medianresults);
% % set(handles.SDresults,'String',handles.activeCamData.SDresults);
% % set(handles.num_members_results,'String',handles.activeCamData.num_membersresults);
% % set(handles.angleresults,'String',handles.activeCamData.angleresults);

end