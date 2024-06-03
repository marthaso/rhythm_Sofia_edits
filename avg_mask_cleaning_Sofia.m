% Get a mask for your data. Get a matrix with average activation times for
% your data.

function [data_avg, mask3] = avg_mask_cleaning_Sofia(Fs, data)

%% Get a mask to get rid of background. Can upload one or clean one using the maskfix function.
mask3 = uigetfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\mask3.txt');
mask3 = load(mask3);
%mask3 = maskfix;
disp('Finished Mask')
toc
tic

%% Multiply your data by the mask to get rid of background.
data = data.*mask3;

%% Find points of max derivative
% Create a cell for the derivative location for each pixel
all_derivatives = {};
count = 0;

% Go through each pixel. Get locations of all max derivatives. 
for i = 1:10:size(data,1)
    for j = 1:10:size(data,1)
        
        % Choose one pixel
        pixel = data(i,j,:);
        pixel = squeeze(pixel);
        % Get the derivatives of that signal
        derivatives = diff(pixel);
        % If it's outisde the mask, the derivatives will be zero.
        if mean(derivatives) ~= 0
            [pks, locs] = findpeaks(derivatives,'MinPeakProminence',0.04,'MinPeakDistance',200);
            count = count + 1;
            all_derivatives(count) = {locs};
        end
    end
end

% We only want to keep the pixels that had the right number of peaks. We
% expect most pixels to find the right number.

num_peaks = [];
for i = 1 : width(all_derivatives)
    cell_length = size(all_derivatives{i},1);
    num_peaks = [num_peaks;cell_length];
end

average = round(mean(num_peaks));
peaks = [];

for i = 1 : width(all_derivatives)
    if size(all_derivatives{i},1) == average
        peaks = [peaks,all_derivatives{i}];
    end
end

final_peaks = mean(peaks,2);

% Can plot one pixel and the peaks found just to check
% figure
% plot(squeeze(data(50,50,:)))
% hold on
% plot(final_peaks,ones(length(final_peaks))*0.5,'o')

disp('Found all max derivatives')
toc
tic
%% Compute start and end points for all the APs. Here we are using a rise of 60 ms

stat = final_peaks - 30;
endp = final_peaks + 30;
amap = {};

%% Create an activation map for each AP

for i = 1:length(final_peaks)
    [actMap1] = activationmap(stat(i), Fs, endp(i), data);
    amap(i) = {actMap1};
    % Use following lines to plot activation map for each AP
    % figure
    % imagesc(actMap1)
    % title(i)
    % colorbar
end


%% Get the average of all the maps

B = cat(3,amap{:});
data_avg = mean(B,3);
data_avg(data_avg >= 100) = NaN;
data_avg(data_avg <= 5) = NaN;
min_val = min(min(data_avg));
offset = ones(size(data_avg,2), size(data_avg,2)).* min_val;
data_avg = data_avg - offset;
% Can use code to visualize the average activation map
% figure
% imagesc(data_avg)
% title('mean')
% colorbar
disp('Made average act map')
toc
tic
end

function [actMap1, mask] = activationmap(stat, Fs, endp, data)

stat=round(stat);
endp=round(endp);
dataCroppedInTime = data(:,:,stat:endp); % truncate data

% Re-normalize data in case of drift
dataCroppedInTime = normalize_data(dataCroppedInTime);

% identify channels that have been zero-ed out due to noise
mask = max(dataCroppedInTime,[],3) > 0;

% Find First Derivative and time of maxium
derivatives = diff(dataCroppedInTime,1,3); % first derivative
[~,max_i] = max(derivatives,[],3); % find location of max derivative

% Create Activation Map
actMap1 = max_i.*mask;
actMap1(actMap1 == 0) = nan;
offset1 = min(min(actMap1));
actMap1 = actMap1 - offset1*ones(size(data,1),size(data,2));
actMap1 = actMap1/Fs*1000; %% time in ms

end

function normData = normalize_data(data)

    if size(data,3) == 1
        min_data = repmat(min(data,[],2),[1 size(data,2)]);
        diff_data = repmat(max(data,[],2)-min(data,[],2),[1 size(data,2)]);
    else
        min_data = repmat(min(data,[],3),[1 1 size(data,3)]);
        diff_data = repmat(max(data,[],3)-min(data,[],3),[1 1 size(data,3)]);
    end

    nonzero = find(diff_data);
    normData = (data-min_data);
    normData(nonzero) = normData(nonzero)./(diff_data(nonzero));

    end