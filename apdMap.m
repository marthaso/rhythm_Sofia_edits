% Function name: apdMap

% Function Summary: 

% Inputs: 

% Outputs: 


function [apdMap] = apdMap(data,...
                           start, endp,...
                           minapd, maxapd,...
                           percentAPD,...
                           area_coords,...
                           Fs, cmap, movie_scrn, handles)

%% What do you need to do?
num_heart = 1;
while(1)

%% First off, find the APDs in the data.

%% Load data.
data = uigetfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\data.mat');
data = load(data);
data = struct2cell(data);
data= cell2mat(data);
disp('Loaded data') % Display messages throughout just to know where we are.

%% Get a mask to get rid of background. Can upload one or clean one using the maskfix function.
mask3 = uigetfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\mask3_new.txt');
mask3 = load(mask3);

%% Regional vs total

if area_coords(1)~=0
    % Make a mask of 1s in the selected rectangle. 0 everywhere else.
    mask_ROI = zeros(size(mask3));
    rect = round(area_coords);
    rect = abs(rect);
    mask_ROI(rect(2):rect(2)+rect(4),rect(1):rect(1)+rect(3)) = 1;
    data = data.*mask_ROI;
else
    data = data.*mask3;
end




% Cut the data to just the time you want.
start1 = 1 + round(start * Fs);
endp1 = round(endp * Fs);
ap_data = data(:, :, start1 : endp1);

% Re-normalize your data once you cut it
ap_data = normalize_data(ap_data);

%New APD map - all NAN in the size of your data.
apdMap = nan(size(ap_data, 1), size(ap_data, 2));

% This is the level you want. For example, 80% APD is when the AP level is
% at 0.2 above baseline (data is normalized so max is 1).
AP_level = 1.0 - percentAPD / 100;

% Select the APDs you do want to consider. 

%% Find points of max derivative
% Create a cell for the derivative location for each pixel
all_derivatives = {};
count = 0;

%% Go through each pixel (for time efficiency, doing every 10 pixels). Get locations of all max derivatives. 
for i = 1:10:size(ap_data,1)
    for j = 1:10:size(ap_data,1)
        % Choose one pixel
        pixel = ap_data(i,j,:);
        pixel = squeeze(pixel);
        % Get the derivatives of that signal
        derivatives = diff(pixel);
        % If it's outisde the mask, the derivatives will be zero.
        if mean(derivatives) ~= 0 % if it is not background...
            %og
            %[pks, locs] = findpeaks(derivatives,'MinPeakProminence',0.05,'MinPeakDistance',200); % find all peaks. prominence and distance
            % can and might be changed if peaks are not identified
            % correctly
            % edited at CL 200
             [pks, locs] = findpeaks(derivatives,'MinPeakProminence',0.05,'MinPeakDistance',100);
            count = count + 1;
            all_derivatives(count) = {locs}; % store all the peaks of that pixel
        end
    end
end

%% We only want to keep the pixels that had the right number of peaks. We
% expect most pixels to find the right number.

num_peaks = []; % here we will store how many peaks were found in each pixel
for i = 1 : width(all_derivatives)
    cell_length = size(all_derivatives{i},1); % go through each cell (each pixel) and find # of peaks
    num_peaks = [num_peaks;cell_length]; % store that number in a vector
end

mode_val = mode(num_peaks); % what value is repeated the most? We expect most pixels
% to find the correct number and few outliers.
if mode_val == 0
    num_peaks(num_peaks==0) = NaN;
    mode_val = mode(num_peaks);
end
peaks = []; % Here we will store the peak locations

%% Once again, go through each pixel. If that pixel has the right number of peaks,
% store the peaks locations.
for i = 1 : width(all_derivatives)
    if size(all_derivatives{i},1) == mode_val
        peaks = [peaks,all_derivatives{i}];
    end
end

%% Now, for each peak, find the average value between pixels. They should be pretty close.
final_peaks1 = mean(peaks,2);
final_peaks=[];


%% make sure peaks are within our time frame. We will add/subtract a few ms to 
% encompass all pixels. Here we are using 30 ms both ways.
for i = 1:length(final_peaks1)
    if final_peaks1(i)>60 && final_peaks1(i)< 4900
        final_peaks=[final_peaks,final_peaks1(i)];
    end
end

%% Plot one pixel and the peaks found. Here we are plotting
% pixel 100,100. If it looks bad, you can change pixel.
figure
plot(squeeze(ap_data(100,100,:)))
hold on
plot(final_peaks,ones(length(final_peaks),1)*(mean(ap_data(100,100,:))+std(ap_data(100,100,:))),'o')
% 
% figure
% plot(squeeze(data(50,50,:)))
% hold on
% plot(final_peaks,ones(length(final_peaks),1)*(mean(data(50,50,:))+std(data(50,50,:))),'o')


%% Select any peaks you want to exclude from analysis.
if num_heart == 3
    b=2;
end
[xi,yi] = getpts; % on the figure, click near the peaks you want to exclude. Click enter when done.
% If you don't wish to exclude any peaks just click enter.
minIdx =[];
if ~isempty(xi) % If you did exclude peaks...
    x = round(xi); % getpts doesn't choose pixels so you need to round to whole numbers.
    %% Find the times of the peaks you want to exclude.
    for i = 1:numel(x) % go through each point you clicked
        difference = abs(ones(numel(final_peaks),1)*x(i)-final_peaks'); % subtract the rounded value
        % from all the values in final_peaks.
        [~,minIdx(i)] = min(difference); % the minimum difference is the location where
        %the peak you want to exclude is located. Get indices. Eg. your clicked
        %point is at x=87, the closes peak was at x=93, and that is the first
        %peak found. So minIdx would be 1.
    end

    %% Exclude them.
    counter = 0;
    final_peaks_chosen = [];
    for i = 1:numel(final_peaks) % Go through however many peaks were found
        val = ismember(i,minIdx); % If that index is found in minIdx (val=1) don't do anything
        if val == 0 % If that index is not in minIdx, then add it to the final_peaks_chosen vector
            counter = counter + 1;
            final_peaks_chosen(counter) = final_peaks(i);
        end
    end
else % If you didn't choose any peaks, just take all the previous peaks.
    final_peaks_chosen = [];
    final_peaks_chosen = final_peaks;
end

%% Figure to check it worked correctly. Shouldn't have the excluded peaks. Can uncomment if you want to see it. 
% figure
% plot(squeeze(data(100,100,:)))
% hold on
% plot(final_peaks_chosen,ones(length(final_peaks_chosen),1)*(mean(data(100,100,:))+std(data(100,100,:))),'o')


% Somehow separate the APs
% final_peaks_chosen have the starting point of each AP. A whole AP in
% theory will be from the start of an AP until the start of the next one.
% Either that or the max APD set by the user. Can account for both of these
% situations.
% Create empty vector to store average APDs. So for the first AP, you will
% have the average over many pixels. 
APD_Vector = [];
% The final_peaks vector has an index where each peak is. It is an average
% over pixels so it is not a round number. Here we make it a whole number.
rounded_final_peaks = round(final_peaks_chosen);
% Now we will find the duration for each AP (each peak we found).
apdmap_1 = {};
counterr = 0;
for i = 1:numel(rounded_final_peaks)
    % If there are two APs next to each other, the first one should end
    % when the second one starts. If they aren't next to each other, the AP
    % shouldn't be longer than what we defined at the beginning.
    if i < numel(rounded_final_peaks)
        if rounded_final_peaks(i+1)-rounded_final_peaks(i)> maxapd
            % If there are not next to each other. Set the end of the AP to the
            % start plus whatever the max APD allowable was.
            AP_end = (rounded_final_peaks(i)) + maxapd;
            AP = ap_data(:,:,(rounded_final_peaks(i)-50):AP_end);
        else
            % This first AP will be all the pixels (256x256) of your FOV. Start
            % at the first AP start minus 50 (because it is an average, we
            % don't want to miss the beginning. Think about this to be
            % consistent between pixels/APs). Go until the start of the next
            % AP.
            AP = ap_data(:,:,(rounded_final_peaks(i)-50):rounded_final_peaks(i+1));

        end
    else
        % If you are at the last AP. Set the end of the AP to the
        % start plus whatever the max APD allowable was.
        AP_end = (rounded_final_peaks(i)) + maxapd;
        AP = ap_data(:,:,(rounded_final_peaks(i)-50):AP_end);
    end

    % So now AP is a matrix. It has all of our pixels, it has the time
    % points that include the first AP in totality.
    %AP = normalize_data(AP);
    % Now, for each pixel, find the time points where the "voltage"
    % value is below the level you specified (AP_level)
    AP_indices = [];
    for j = 1:size(ap_data,1) % Go through each pixel
       
        for k = 1:size(ap_data,2)
            % if j > 150 && k > 150
            %     figure
            %     plot(pixel_AP)
            %     a=1;
            % end
            % Normalize the data in this pixel
            if num_heart == 3 && i==numel(rounded_final_peaks)-1 
                b=2;
            end
            pixel_AP = squeeze(AP(j,k,:));
            minval = min(pixel_AP);
            maxval = max(pixel_AP);
            dif = maxval - minval;
            pixel_AP = pixel_AP - minval;
            pixel_AP = pixel_AP./dif;
            % 
            % figure
            % plot(pixel_AP)

            first_values = pixel_AP(1:30);
            %first_values = first_values(first_values>0.5);
            first_values = first_values(first_values>0.9);
            final_values = pixel_AP(250:end);
            %final_values = final_values(final_values>0.5);
            final_values = final_values(final_values>0.9);

            if isempty(first_values) && isempty(final_values)

                % figure
                % plot(pixel_AP)

                % Find location of biggest derivative. acttime will have the
                % LOCATION of max derivative (start for this pixel)
                [~, acttime] = max(diff(pixel_AP));
                % Find all the indices of where your AP level is below the
                % desired value. Remember you normalized.
                index_of_APD = find(pixel_AP<=AP_level);
                % Go through each index until the APD is greater than your min
                % required. (there might be some false positives at the
                % beginning.)
                % first subtract each time point where value is below AP level
                % from activation time. each one of these values are a
                % possibility for an APD.
                possible_APDs = index_of_APD-acttime;
                % of these options, choose the ones that are above our minapd
                % value
                [AP_index] = find(possible_APDs > minapd & possible_APDs < maxapd);
               
                % store each APD (PER PIXEL FOR THE FIRST AP)
                if ~isempty(AP_index)
                    % if there is data in this pixel, choose the first APD that
                    % satisfies all the conditions. For this pixel, this is the
                    % APD for this AP
                    AP_indices = [AP_indices, possible_APDs(AP_index(1))];
                    if possible_APDs(AP_index(1))>235
                        % if counterr<10
                        %     b=2;
                        %     figure
                        %     plot(pixel_AP)
                        %     title(num2str(i))
                        %     counterr= counterr+1;
                        % end
                    end
                else
                    % There might not be a value if there is noise, bg, or all
                    % the APDs values are not within range
                    AP_indices = [AP_indices, NaN];

                end
               
            else
                AP_indices = [AP_indices, NaN];
            end
        end
    end
    
    bgAPD = min(AP_indices);
    AP_indices(AP_indices==bgAPD) = NaN;
    disp(nanmean(AP_indices))
    
    AP_indices = reshape(AP_indices,[256,256])';
    % figure
    % imagesc(AP_indices)
    % colorbar
    % title(num2str(i))
    apdmap_1 (i) = {AP_indices};
    % AP_indices = reshape(AP_indices,1,[]);
    % figure
    % boxplot(AP_indices)
   


  

    % At the end of this round, you should have 256x256 values.

    % Now repeat for the second AP.

    % Once you store the values you should have two values for each pixel.
    % 256x256x2.

    % Continue until you have all the APDs

    % get an average PER PIXEL. 

    % at the end you should have a 256x256x1 with an average APD per pixel.
   
   

end

%% Make average map
B = cat(3,apdmap_1{:});
average_apd = zeros(size(ap_data,1),size(ap_data,2));
average_apd(average_apd==0) = NaN;
for p = 1: size(ap_data,1)
   for q = 1: size(ap_data,1)
       pixel = B(p,q,:);
       pixel_nan = isnan(pixel);
       pixel_sum = sum(pixel_nan);
       if pixel_sum <= numel(final_peaks_chosen)/2
           average_apd(p,q)=nanmean(B(p,q,:),3);
       end
   end
end


%average_apd_map = nanmean(B,3);
figure
imagesc(average_apd)
colorbar
title('Average')
disp(nanmean(nanmean(average_apd)))
average_apd=reshape(average_apd,[],1);
data=[];
g=[];
data=[data;average_apd];
g1 = repmat({'1'},length(average_apd),1);
g=[g;g1];
for i = 1:size(B,3)
    data_1 = B(:,:,i);
    data_1 = reshape(data_1,[],1);
    data = [data;data_1];
    g_1=repmat({num2str(i+1)},length(data_1),1);
    g=[g;g_1];
end




figure
boxplot(data,g)
APD_heart (num_heart) = {average_apd};
num_heart = num_heart + 1;


 choice = menu('Add another heart?','Yes','No');
    if choice==2 || choice==0
        
        break;
    end
end


g=[];
data_2 = [];
C = cat(3,APD_heart{:});
for i = 1:4
    data_3 = C(:,:,i);
    data_3 = reshape(data_3,[],1);
    data_2 = [data_2;data_3];
    g_1=repmat({num2str(i)},65536,1);
    g=[g;g_1];
end
figure
boxplot(data_2,g)

a=1;


%% OLD CODE
% Normalize the AP
% set a baseline and find values above your value. need to make sure there
% is no drift. 
% average the values per pixel.
% plot. 
% the function apdMap creates a visual representation of the action potential duration 
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
% A figure that has a color repersentation for action potential duration
% times
%
% METHOD
% Finds the largest time interval when AP height exceeds AP_level
% Ex.: AP_level = 0.2 when we calculate APD80
%
% AUTHOR: Matt Sulkin (sulkin.matt@gmail.com)
%
% MAINTED BY: Christopher Gloschat - (cgloschat@gmail.com) - [Jan. 2015 - Mar. 2019]
%             Andrey Pikunov - (pikunov@phystech.edu) - [Mar. 2019 - Present]  
%
% MODIFICATION LOG:
% Jan. 26, 2015 - The input cmap was added to input the colormap and code
% was added at the end of the function to set the colormap to the user
% determined values. In this case the most immediate purpose is to
% facilitate inversion of the default colormap.
%
% Mar. 21, 2019 - New method of APD calculation.
%
% Email optocardiography@gmail.com for any questions or concerns.
% Refer to efimovlab.org for more information.
% %% Create initial variables
% start = 1 + round(start * Fs);
% endp = round(endp * Fs);
% ap_data = data(:, :, start : endp);
% ap_data = normalize_data(ap_data);
% 
% apdMap = nan(size(ap_data, 1), size(ap_data, 2));
% 
% APD_min_rescaled = minapd * Fs / 1000;
% APD_max_rescaled = maxapd * Fs / 1000;
% 
% AP_level = 1.0 - percentAPD / 100;
% 
% %area_coords = [105, 101, 57, 58]  ;
% %area_coords = [55, 126, 57, 58]  ;
% %area_coords = [105, 71, 57, 58]  ;
% %area_coords = [55, 96, 57, 58]  ;
% %area_coords = [5, 5, 200, 200]  ;
% %area_coords = [100, 80, 50, 50]  ;
% %area_coords = [150, 68, 35, 35]  ;
% area_coords = [100, 93, 35, 35]  ;
% 
% 
% 
% %area_coords = int8(area_coords);
% area_coords = round(area_coords);
% j_min = 1 + area_coords(1);
% i_min = 1 + area_coords(2);
% j_max = area_coords(1) + area_coords(3);
% i_max = area_coords(2) + area_coords(4);
% 
% %% Map calculation
% for i = i_min : i_max
%     for j = j_min : j_max
%         % In all the time, find which points of that pixel are below your
%         % value. You find indices, not voltage values.
%         index = find(ap_data(i, j, :) <= AP_level);
% 
%         % If you have more than two values...
%         if size(index, 1) > 2
%             % Take an index and substract the previous index. Now you have
%             % a new vector that is the difference between adjacent pixels.
%             apd = max(index(2: end) - index(1: end - 1));
%             % find the point where the difference between continuous
%             % indices is between your range. call that your APD for that
%             % pixel.
%             if ((APD_min_rescaled < apd) && (apd < APD_max_rescaled))
%                 apdMap(i, j) = apd;
%             end
%         end
% 
%     end
% end
% 
% % account for different sampling frequencies
% unitFix = 1000.0 / Fs;
% % Calculate Action Potential Duration
% apdMap = apdMap * unitFix;
% 
% 
% %% Plot APDMap
% handles.activeCamData.saveData = apdMap;
% 
% 
% cla(movie_scrn);
% 
% colormap(handles.activeScreen, cmap);
% imagesc(apdMap,'Parent', movie_scrn, 'AlphaData', ~isnan(apdMap));
% %axis(movie_scrn,'off');
% set(movie_scrn,'Color','k');
% set(movie_scrn,'YDir','reverse');
% set(movie_scrn,'YTick',[],'XTick',[]);
% 
% %Setting up values to use for color axis
% APD_min = prctile(apdMap(isfinite(apdMap)),1);
% APD_max = prctile(apdMap(isfinite(apdMap)),99);
% caxis(movie_scrn,[APD_min APD_max])
% 
% figure;
% %ax1 = axes;
% image(handles.activeCamData.bgRGB);
% %colormap(ax1,'gray');
% %ax2 = axes;
% hold on
% imagesc(apdMap,'AlphaData',~isnan(apdMap));
% colormap('jet');
% colorbar;
% 
% figure
% apdMap_V = load('apdMap_V.mat','apdMap');
% apdMap_V = cell2mat(struct2cell(apdMap_V));
% apdV = apdMap_V(68:103, 150:185);
% apdV_mat = nan(256,256);
% apdV_mat(93:128, 100:135) = apdV;
% %ax1 = axes;
% image(handles.activeCamData.bgRGB);
% %colormap(ax1,'gray');
% %ax2 = axes;
% hold on
% imagesc(apdV_mat,'AlphaData',~isnan(apdV_mat));
% colormap('jet');
% colorbar;
% 
% figure
% diff_time = apdMap - apdV_mat;
% image(handles.activeCamData.bgRGB);
% hold on
% imagesc(diff_time,'AlphaData',~isnan(diff_time));
% colormap('jet');
% colorbar;
% 
% %% Plot Histogram of APDMap
% %figure('Name','Histogram of APD')
% %hist(reshape(apdMap,[],1),floor(APD_max-APD_min))
% %xlim([APD_min APD_max])
% 
% %% Calculating statistics
% apd_mean=nanmean(apdMap(:));
% disp(['The average APD in the region is ' num2str(apd_mean) ' (ms).'])
% apd_std=nanstd(apdMap(:));
% disp(['The standard deviation of APDs in the region is ' num2str(apd_std) ' (ms).'])
% apd_median=nanmedian(apdMap(:));
% disp(['The median APD in the region is ' num2str(apd_median) ' (ms).'])
% 
% handles.activeCamData.meanresults           = sprintf('Mean: %0.3f (ms)',apd_mean);
% handles.activeCamData.medianresults         = sprintf('Median: %0.3f (ms)',apd_median);
% handles.activeCamData.SDresults             = sprintf('S.D.: %0.3f (ms)',apd_std);
% handles.activeCamData.num_membersresults    = sprintf('');
% handles.activeCamData.angleresults          = sprintf('');
% 
% set(handles.meanresults,'String',handles.activeCamData.meanresults);
% set(handles.medianresults,'String',handles.activeCamData.medianresults);
% set(handles.SDresults,'String',handles.activeCamData.SDresults);
% set(handles.num_members_results,'String',handles.activeCamData.num_membersresults);
% set(handles.angleresults,'String',handles.activeCamData.angleresults);
% 
% end
