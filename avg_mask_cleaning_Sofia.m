load('C:\Users\Sofia\Desktop\Optical data - Mice\Heart #2\182024-01-25-143523_N256 (IF1-CAM1)')
load('C:\Users\Sofia\Desktop\Optical data - Mice\Heart #2\mask3.txt')

stat = 2.88;
endp = 2.95;
Fs = frequency;
data = cmosData.*mask3;

[actMap1, mask] = activationmap(stat, Fs, endp, data);
imagesc(actMap1)
colorbar

 function [actMap1, mask] = activationmap(stat, Fs, endp, data)
        
        stat=round(stat*Fs)+1;
        endp=round(endp*Fs)+1;
        %actMap = zeros(size(data,1),size(data,2));
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
%% The function normalizes CMOS data between 0 and 1

% INPUTS
% data = cmos data

% OUTPUT
% normData = normalized data matrix

% METHOD
% Normalize data finds the minimum, maximum, and the difference in
% data values. The normalized data subtracts off the minimum values and
% divides by the difference between the min and max.

% Email optocardiography@gmail.com for any questions or concerns.
% Refer to efimovlab.org for more information.


%% Code
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