% Function Name: calculate_amap

% Function Summary: This code will find upstrokes in a pixel and output an activation
% time map.

% Inputs: Starting time (stat), end time (endp), sampling frequency (Fs),
% and your data (aka normalized fluorescence values over time for each
% pixel).

% Outputs: A matrix the size of your FOV with values starting at 1 showing
% when each pixel depolarizaes wrt the first one.

function [actMap1] = activationmap(stat, Fs, endp, data)

%% First, make sure your timepoints are integers within your data.
stat=round(stat)+1;
endp=round(endp)-1;

%% Truncate your data in the specified time frame
dataCroppedInTime = data(:,:,stat:endp); 

%% Re-normalize data in case of drift
dataCroppedInTime = normalize_data(dataCroppedInTime);

%% identify channels that have been zero-ed out due to noise
mask = max(dataCroppedInTime,[],3) > 0;

%% Find First Derivative and time of maxium
derivatives = diff(dataCroppedInTime,1,3); % first derivative
[~,max_i] = max(derivatives,[],3); % find location of max derivative

%% Create Activation Map
actMap1 = max_i.*mask;
actMap1(actMap1 == 0) = nan;
offset1 = min(min(actMap1)); % make sure first activation point is at time=1
actMap1 = actMap1 - offset1*ones(size(data,1),size(data,2));
actMap1 = actMap1/Fs*1000; %% time in ms

end
