% Function Name: averaged_amap

% Function Summary: This code will find upstrokes in the signal, allow user
% to choose APs to consider, get max derivatives, and output an average activation
% time map.

% Inputs: Your full data and the sampling frequency.

% Outputs: A matrix the size of your FOV with values starting at 1 showing
% when each pixel depolarizaes wrt the first one. 

function [aMap1] = averaged_amap(data,Fs)

%% Find points of max derivative
% Create a cell for the derivative location for each pixel
all_derivatives = {};
count = 0;

%% Go through each pixel (for time efficiency, doing every 10 pixels). Get locations of all max derivatives. 
for i = 1:10:size(data,1)
    for j = 1:10:size(data,1)
        % Choose one pixel
        pixel = data(i,j,:);
        pixel = squeeze(pixel);
        % Get the derivatives of that signal
        derivatives = diff(pixel);
        % If it's outisde the mask, the derivatives will be zero.
        if mean(derivatives) ~= 0 % if it is not background...
            [pks, locs] = findpeaks(derivatives,'MinPeakProminence',0.01,'MinPeakDistance',200); % find all peaks. prominence and distance
            % can and might be changed if peaks are not identified
            % correctly
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
    if final_peaks1(i)>30 && final_peaks1(i)< 4970
        final_peaks=[final_peaks,final_peaks1(i)];
    end
end

%% Plot one pixel and the peaks found. Here we are plotting
% pixel 100,100. If it looks bad, you can change pixel.
figure
plot(squeeze(data(100,100,:)))
hold on
plot(final_peaks,ones(length(final_peaks),1)*(mean(data(100,100,:))+std(data(100,100,:))),'o')

%% Select any peaks you want to exclude from analysis.
[xi,yi] = getpts; % on the figure, click near the peaks you want to exclude. Click enter when done.
% If you don't wish to exclude any peaks just click enter.
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
    for i = 1:numel(final_peaks) % Go through however many peaks were found
        val = ismember(i,minIdx); % If that index is found in minIdx (val=1) don't do anything
        if val == 0 % If that index is not in minIdx, then add it to the final_peaks_chosen vector
            counter = counter + 1;
            final_peaks_chosen(counter) = final_peaks(i);
        end
    end
else % If you didn't choose any peaks, just take all the previous peaks.
    final_peaks_chosen = final_peaks;
end

%% Figure to check it worked correctly. Shouldn't have the excluded peaks. Can uncomment if you want to see it. 
% figure
% plot(squeeze(data(100,100,:)))
% hold on
% plot(final_peaks_chosen,ones(length(final_peaks_chosen),1)*(mean(data(100,100,:))+std(data(100,100,:))),'o')

%% Compute start and end points for all the APs. Here we are using a rise of 60 ms
stat = final_peaks_chosen - 30;
endp = final_peaks_chosen + 30;
% Create a  cell to store all the aMaps 
amap = {};

%% Create an activation map for each AP
% Go through each peak
for i = 1:length(final_peaks_chosen)
    % Compute Amap for that peak
    [actMap1] = activationmap(stat(i), Fs, endp(i), data);
    % Store the Amap
    amap(i) = {actMap1};
    % Use following lines to plot activation map for each AP if you want
    % figure
    % imagesc(actMap1)
    % title(i)
    % colorbar
end


%% Get the average of all the maps

B = cat(3,amap{:});
% Get an aMap that is the average of all the chosen peaks
data_avg = mean(B,3);
% Can set up values to exclude if there's specific noise in the data
% data_avg(data_avg >= 100) = NaN;
% data_avg(data_avg <= 0.5) = NaN;
% Do an offset so the earliest pixel activates at time=0.
min_val = min(min(data_avg));
offset = ones(size(data_avg,2), size(data_avg,2)).* min_val;
aMap1 = data_avg - offset;
% Can use code to visualize the average activation map
% figure
% imagesc(data_avg)
% title('mean')
% colorbar

%% Save the map for conduction velocity calc
handles.activeCamData.saveData = aMap1;
end

