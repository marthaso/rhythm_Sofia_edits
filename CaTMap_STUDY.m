%% Function : CaTMap_STUDY

% Purpose: 

% Inputs:
% data - data file you are evaluating
% start - start time of the action potential
% endp - end time of the action potential
% minapd - minimum acceptable value of action potential duration
% maxapd - maximum acceptable value of action potential duration
% percentAPD - defines if we are calculating APD80 or APD50
% The other inputs usually won't be changed. They are parameters that describe the data

% Outputs:
% Most important one: APD_average - the average APD for this action
% potential. This is the value that will be stored in the results vector in
% the other function.
% apdMap1 - matrix with all the APDs
% filtered2 - same as apdMap1 but taking away outliers
% AP_start_points - matrix defining start point for EACH PIXEL
% AP_baselines - matrix defining what is the baseline for EACH PIXEL
% AP_end_points - matrix defining end point for EACH PIXEL


function [apdMap1,filtered2,AP_start_points,AP_baselines,AP_end_points,AP_storage, APD_average] = CaTMap_STUDY(data,...
    start, endp,...
    minapd, maxapd,...
    percentAPD,...
    area_coords,...
    Fs, cmap, movie_scrn, handles, SNR_mask,Hz,b,filename_pdf,snr,study)

%% Create initial variables

% initialize your apd map
apdMap1 = nan(size(data, 1), size(data, 2));
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


%% SNR mask
% [SNR_mask, SNR_matrix] = SNR_mask_calc(data,length_of_data);
% If you want to choose just one specific pixel:
% SNR_mask(160,222) = 1;

% Multiply our data times the SNR mask we generated previously to only
% compute APDs in the good pixels.
all_data = data;
data = data.*SNR_mask;

%% Now find the APDs for one pixel

% If you want to see a specific pixel, you can change the "i_min"/"j_min"
for pix_x = i_min : i_max
    for pix_y = j_min : j_max
        % Discard pixels with no signal
        if max(data(pix_x,pix_y,start:endp))==0

        else
            % Check if this pixel has the right number of peaks. Otherwise,
            % discard.
            pixel_calc = data(pix_x,pix_y,start:endp);
            pixel_calc = normalize_data(pixel_calc);
            pixel_calc = squeeze(pixel_calc(25:end-10));
            % I found these values to work in our data, can adjust as
            % needed.
            % [pks, locs] =findpeaks(pixel_calc,'MinPeakDistance',50,'MinPeakProminence',0.3,'MaxPeakWidth',120);
            [pks, locs] =findpeaks(pixel_calc,'MinPeakDistance',50,'MinPeakProminence',0.2,'MaxPeakWidth',120); % file 19
            
           
            % If it has the right number of peaks, continue with the analysis.
            % This part of the code was written for alternan calculations
            % where we have multiple APs. For APD/CaT, we'll usually just
            % have one AP.
            if length(pks) == 1
                % Create vectors where you will store APDs, start points, end
                % points, and baselines for all APs in this pixel.
                

                % Take the start point and end point, your entire AP should be
                % within these points. This code works well for rat data so far
                % but might need to be modified if there are any
                % errors!
                AP = pixel_calc(1:end);

                % Find specific start point of AP. Go to function. If
                % the function determines there is no start point, then
                % store NAN, otherwise, store the start value.
                [start_of_AP] = start_of_AP_calc(AP);
                if ~isnan(start_of_AP)
                    start_of_AP_pixel = start_of_AP;
                else
                    start_of_AP_pixel = NaN;
                end


                % Find baseline (go to function).
                if ~isnan(start_of_AP)
                    [baseline_of_AP] = baseline_of_AP_calc(AP(1:start_of_AP));
                end


                % Find endpoint (go to function). If either the start
                % or the end of the AP is NaN, then APD is NaN.
                % Otherwise, find APD using end - start.
                [end_of_AP] = end_of_AP_calc(AP,baseline_of_AP,AP_level);
                if any(~isnan(end_of_AP)) && any(~isnan(start_of_AP))
                    end_of_AP_pixel = end_of_AP;
                    % Find APD
                    apd = end_of_AP - start_of_AP;
                else
                    apd = NaN;
                    end_of_AP_pixel = NaN;
                end


                % If anything went wrong and the final apd is not just one value, display
                % this message and keep going. Shouldn't happen in many pixels.
                % If apd is just one value, check if it's within our
                % APD range. If it is, store the apd, start, end,
                % baseline, and populate our map. If apd is NaN, store NaN for all these
                % values.
                if ~isscalar(apd)
                    % disp('apd is NOT a scalar');
                else
                    if ((minapd < apd) && (apd < maxapd))
                        AP_storage(pix_x,pix_y) = apd;
                        AP_start_points(pix_x,pix_y) = start_of_AP_pixel;
                        AP_end_points(pix_x,pix_y) = end_of_AP_pixel;
                        AP_baselines(pix_x,pix_y) = baseline_of_AP;
                        apdMap1(pix_x,pix_y)=apd;
                    end
                end
                

            end

        end
    end
end

%% Plot our apd Map
[hdiLow, hdiHigh] = hdi(apdMap1,0.99);
filtered2 = apdMap1;
if ~isempty(hdiHigh) && ~isempty(hdiLow)
    filtered2(filtered2>hdiHigh)=NaN;
    filtered2(filtered2<hdiLow)=NaN;
end


%% Comment out these lines if you don't want to plot the map and examples
if sum(~isnan(filtered2(:))) < 50
    APD_average = NaN;
    close all

else
    plotapd(filtered2,percentAPD,Hz,filename_pdf)
    plot_examples(filtered2,data(:,:,start:endp),AP_start_points,AP_baselines,AP_end_points,AP_storage,filename_pdf, start, endp)
    plot_excluded_examples(study,filtered2,all_data(:,:,start:endp),filename_pdf)



close all
m = nanmean(filtered2(:));
APD_average = m;
s = nanstd(filtered2(:)); 
%results = [Hz, b, m, s];
fs = figure('visible','on');
set(gca,'visible','off')
text(0.05, 0.8, sprintf('Hz = %g, %gbin, SNR = %g, Mean = %g, Std dev = %g', Hz,b,snr, m,s),'FontSize', 14)
print(fs,filename_pdf,'-dpsc','-bestfit','-append')
end


close(gcf)





    function [hdiLow, hdiHigh] = hdi(apdMap1, range)
        % Compute highest density interval
       
        % If you don't specify range, set it to 0.95 
        if nargin < 2
            range = 0.95;
        end

        % Remove NaNs just in case
        apdMap1 = apdMap1(~isnan(apdMap1));

        % Sort samples
        x = sort(apdMap1(:));
        n = numel(x);

        % Number of points in interval
        m = floor(range * n);

        % All candidate interval widths
        widths = x(m+1:end) - x(1:end-m);

        % Find shortest interval
        [~, idx] = min(widths);

        % HDI bounds
        hdiLow  = x(idx);
        hdiHigh = x(idx + m);

    end

    function plotapd(apdMap1,percentAPD,Hz,filename_pdf)
        % Plot the apdMap
        figure
        h = imagesc(apdMap1);
        % Mask NaNs
        set(h, 'AlphaData', ~isnan(apdMap1))
        % Set image characteristics
        colormap(jet)
        set(gca, 'Color', 'w')   % background = white
        colorbar
        axis image
        title(sprintf('CaT%g - %g Hz - bin %g',percentAPD, Hz,b),'FontName','Arial','FontSize',12) %change as needed

        print(gcf,filename_pdf,'-dpsc','-bestfit','-append')
        
    end

    function plot_examples(apdMap1,data,AP_start_points,AP_baselines,AP_end_points,AP_storage,filename_pdf,start, endp) 
        % Find the highest and lowest values and plot those pixels.
        % Flatten matrix
        A = apdMap1(:);

        % Remove NaNs
        validIdx = find(~isnan(A));
        A_valid = A(validIdx);

        % Sort values lowest to highest
        [~, sortIdx] = sort(A_valid);

        % Indices of lowest and highest values. Find where in the vector
        % are the highest/lowest values. Right now we are plotting five of
        % the lowest and five of the highest values. Can change depending
        % on what we want.
        lowIdx  = validIdx(sortIdx(1:5));
        highIdx = validIdx(sortIdx(end-4:end));

        middleIdx = validIdx(sortIdx(round(length(sortIdx)/2:round(length(sortIdx)/2+9))));
        [middle_row, middle_col] = ind2sub(size(apdMap1), middleIdx);

        % Convert linear indices to row/column.
        [rowLow,  colLow]  = ind2sub(size(apdMap1), lowIdx);
        [rowHigh, colHigh] = ind2sub(size(apdMap1), highIdx);

        % Combine lowest and highest coordinates.
        y=[rowLow;rowHigh];
        x=[colLow;colHigh];
        
        figure
        for pix = 1:length(x)
            subplot(2,5,pix);
            pixel_example = squeeze(data(y(pix),x(pix),20:end-10));
            pixel_example_norm = normalize_data(pixel_example');
            plot(pixel_example_norm)
            hold on
            baseline = ones(1,length(pixel_example))*AP_baselines(y(pix),x(pix));
            plot(1:length(pixel_example),baseline,'LineWidth',1)
            start_point = ones(1,11)*AP_start_points(y(pix),x(pix));
            plot(start_point,0:0.1:1,'LineWidth',1)
            end_point = ones(1,11)*AP_end_points(y(pix),x(pix));
            plot(end_point,0:0.1:1,'LineWidth',1)
            title(sprintf('CaT = %d, x = %d, y = %d', AP_storage(y(pix),x(pix)),y(pix),x(pix)),'FontName','Arial','FontSize',5)

        end

        print(gcf,filename_pdf,'-dpsc','-bestfit','-append')
        y = middle_row;
        x = middle_col;
        figure
        for pix = 1:length(x)
            subplot(2,5,pix);
            pixel_example = squeeze(data(y(pix),x(pix),20:end-10));
             pixel_example_norm = normalize_data(pixel_example');
            plot(pixel_example_norm)
            hold on
            baseline = ones(1,length(pixel_example))*AP_baselines(y(pix),x(pix));
            plot(1:length(pixel_example),baseline,'LineWidth',1)
            start_point = ones(1,11)*AP_start_points(y(pix),x(pix));
            plot(start_point,0:0.1:1,'LineWidth',1)
            end_point = ones(1,11)*AP_end_points(y(pix),x(pix));
            plot(end_point,0:0.1:1,'LineWidth',1)
            title(sprintf('CaT = %d, x = %d, y = %d', AP_storage(y(pix),x(pix)),y(pix),x(pix)),'FontName','Arial','FontSize',5)

        end
        print(gcf,filename_pdf,'-dpsc','-bestfit','-append')
        % close(gcf)
    end

    function [end_of_AP] = end_of_AP_calc(AP,AP_baseline,AP_level)
        % % If the pixel is noisy, you can further filter it. For rat data we
        % % shouldn't need to. Comment out the first two lines if using
        % this part of the code. Change AP to averaged_AP.
        % averaged_AP = movmean(AP, 5, 'Endpoints', 'discard');
        % % Find the peak of AP
        % [~,max_place] = findpeaks(averaged_AP,'MinPeakDistance',50,'MinPeakProminence',0.3);
        % max_place = max(max_place);
        % if isempty(max_place)
        %     [~,max_place] = findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.3);
        % end

        % Find the location of the peak of the AP. These values work well
        % for rat data, can adjust as needed.
        % [~,max_place] = findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.3);
        [~,max_place] = findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.2); % file 19
        
        max_place=max(max_place); % choose the max peak if there are several
        % Set Baseline as min
        if ~isempty(max_place)
            % Take your calculated baseline and set any points lower than
            % that value equal to baseline. Similarly, set any points above
            % the peak equal to the peak. These lines help with noisy
            % pixels!
            norm_AP = max(AP,AP_baseline); 
            norm_AP(norm_AP>norm_AP(max_place))=norm_AP(max_place);
            % Now that our minimum is the baseline and the max is our peak,
            % normalize the data.
            norm_AP = (norm_AP - min(norm_AP)) / (max(norm_AP) - min(norm_AP));
            % The end point will be AFTER the peak.
            norm_AP = norm_AP(max_place:length(norm_AP));
            % Find at what point you reach the threshold. Since we
            % normalized, this is just 0.5 (APD50), 0.2 (APD80), etc. 
            points_lower_than_threshold = find(norm_AP <= AP_level);
            % If for whatever reason there are not points below our
            % threshold, set the end point as NaN. If there are points,
            % choose the first one and add it to peak location. This tells
            % you the end point in terms of the whole AP.
            if isempty(points_lower_than_threshold)
                end_of_AP = NaN;
            else
                end_of_AP = points_lower_than_threshold(1) + max_place;
            end
        else % if no peaks are located, there is no end of AP.
            end_of_AP=NaN;
        end

    end

    function [baseline_of_AP] = baseline_of_AP_calc(AP)
        % Find the peak of AP and the lowest point before the peak. If
        % there is noise and there are points lower than the baseline you
        % might need to a) change your start time in the RHYTHM GUI or b) set some point
        % as the AP start (instead of going 1:max_place, you could do
        % max_place-30:max_place. Depends on your data!)
        baseline_of_AP = min(AP);
        % [~,max_place] = findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.3);
        % max_place = max(max_place);
        % 
        % 
        % if max_place-50<1
        %     new_AP = AP(1:max_place);
        %     baseline_of_AP = min(new_AP);
        % else
        %     new_AP = AP(max_place-50:max_place);
        %     baseline_of_AP = min(new_AP);
        % end
            

    end

    function [start_of_AP] = start_of_AP_calc(AP)
        % The rat data has been pretty clean, max derivative is usually a
        % good start point for the AP. If it looks off, you can set a
        % specific range where the start should be, or further filter the
        % pixel.
        % You can plot a specific pixel (x,y) by running these lines:
        % % figure
        % % plot(squeeze(data(x,y,:)))
        % You can plot the derivatives:
        % % hold on
        % % plot(diff(squeeze(data(x,y,:))))
        % This can give you an idea of what the derivatives look like and
        % why it might be choosing the wrong point.
        % [peak, location] =findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.3,'MaxPeakWidth',120);
        [peak, location] =findpeaks(AP,'MinPeakDistance',50,'MinPeakProminence',0.2,'MaxPeakWidth',120); % file 19
        
        
        
        [~,start_of_AP] = max(diff(AP(1:location)));
    end

    function [SNR_mask,SNR_matrix] = SNR_mask_calc(data,length_of_data)

        %% BG time points
        % Plot one good pixel to choose the background noise and actual
        % signal. Right now it just plots pixel 150,150, might need to
        % change depending on the data. Make sure to change the pixel for
        % both figures.
        figure('Name','fig1')
        
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
        % threshold. Here we are saying SNR>2 but can change as needed.
        SNR_mask = zeros(256,256);
        SNR_mask(SNR_matrix>20) = 1;

        % Visualize your final mask
        figure
        imagesc(SNR_mask)

    end

    function plot_single_example (data, y, x, AP_baselines, AP_start_points, AP_end_points, AP_storage )
        figure
        pixel_example = squeeze(data(y,x,:));
        plot(pixel_example)
        hold on
        baseline = ones(1,length(pixel_example))*AP_baselines(y,x);
        plot(1:length(pixel_example),baseline,'LineWidth',1)
        start_point = ones(1,11)*AP_start_points(y,x);
        plot(start_point,0:0.1:1,'LineWidth',1)
        end_point = ones(1,11)*AP_end_points(y,x);
        plot(end_point,0:0.1:1,'LineWidth',1)
        title(sprintf('APD = %d, x = %d, y = %d', AP_storage(y,x),y,x),'FontName','Arial','FontSize',5)
    end

    function plot_excluded_examples(study,apdMap1,data,filename_pdf) 
    % Find pixels where the og mask is 1 but the final APD map is NaN.
        excluded_pixels = zeros(256,256);
    % load og mask
        mask = load(sprintf('%s_Ca.txt',study));
        
        % compare the same pixel in both maps and mark the ones where mask
        % == 1 and apdMap1 == NaN
        for x = 1:256
            for y = 1:256
                if mask(x,y) == 1 && isnan(apdMap1(x,y))
                    excluded_pixels(x,y) = 1;
                end
            end
        end

        figure
        imagesc(excluded_pixels)

        [r, c] = find(excluded_pixels == 1);
        pixels = randperm(length(r),20);
        x = c([pixels]);
        y = r([pixels]);          

        
        figure
        
        for pix = 1:10
            subplot(2,5,pix);
            pixel_example = squeeze(data(y(pix),x(pix),20:end-10));
            pixel_example_norm = normalize_data(pixel_example');
            plot(pixel_example_norm)
            % hold on
            % baseline = ones(1,length(pixel_example))*AP_baselines(y(pix),x(pix));
            % plot(1:length(pixel_example),baseline,'LineWidth',3)
            % start_point = ones(1,11)*AP_start_points(y(pix),x(pix));
            % plot(start_point,0:0.1:1,'LineWidth',3)
            % end_point = ones(1,11)*AP_end_points(y(pix),x(pix));
            % plot(end_point,0:0.1:1,'LineWidth',3)
            title(sprintf('x = %d, y = %d',y(pix),x(pix)),'FontName','Arial','FontSize',5)
            
        end

       print(gcf,filename_pdf,'-dpsc','-bestfit','-append')
        figure
        
        for pix = 11:20
            subplot(2,5,pix-10);
            pixel_example = squeeze(data(y(pix),x(pix),20:end-10));
             pixel_example_norm = normalize_data(pixel_example');
                plot(pixel_example_norm)
            % hold on
            % baseline = ones(1,length(pixel_example))*AP_baselines(y(pix),x(pix));
            % plot(1:length(pixel_example),baseline,'LineWidth',3)
            % start_point = ones(1,11)*AP_start_points(y(pix),x(pix));
            % plot(start_point,0:0.1:1,'LineWidth',3)
            % end_point = ones(1,11)*AP_end_points(y(pix),x(pix));
            % plot(end_point,0:0.1:1,'LineWidth',3)
            title(sprintf('x = %d, y = %d',y(pix),x(pix)),'FontName','Arial','FontSize',5)
            
        end

       print(gcf,filename_pdf,'-dpsc','-bestfit','-append')
        % close(gcf)
    end


end


