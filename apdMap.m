function [apdMap] = apdMap(data,...
                           start, endp,...
                           minapd, maxapd,...
                           percentAPD,...
                           area_coords,...
                           Fs, cmap, movie_scrn, handles)
%% the function apdMap creates a visual representation of the action potential duration 
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
%% METHOD
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

%% Load data.
data = uigetfile('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\data.mat'); %choose your clean data file (change to your directory)
data = load(data);
data = struct2cell(data);
data= cell2mat(data);
disp('Loaded data') % Display messages throughout just to know where we are.

% plot peaks

pixel = data(100,100,:);
pixel = squeeze(pixel); % you will get 1x1x(voltage). You want to get rid of 1x1. You end up just with the voltage data through time
figure
plot(pixel)
[xi] = getpts;


start_points=[];
end_points=[];
for peak = 1:2:length(xi)-1
    start_points = [start_points, xi(peak)];
    end_points = [end_points, xi(peak+1)];
end

count = 1;
mean =[];
std= [];
median = [];
for i = 1:length(start_points)
        [apdMap, APD_fig, apd_mean, apd_std, apd_median] = APDcalc(count, start_points(i), Fs, end_points(i), data, minapd, maxapd, percentAPD, area_coords, handles, movie_scrn, cmap);
        save(['C:\Users\Sofia\Desktop\C25001_OMFigs\try\APD_data_',num2str(i),'.mat'],'apdMap') %change to your directory
        savefig(APD_fig, ['C:\Users\Sofia\Desktop\C25001_OMFigs\try\APD_fig_', num2str(i), '.fig']); % change to your directory
        mean = [mean;apd_mean];
        std = [std;apd_std];
        median = [median;apd_median];
        count = count + 1;
end
mean
std
median

end




function [apdMap, APD_fig, apd_mean, apd_std, apd_median] = APDcalc(count, start, Fs, endp, data, minapd, maxapd, percentAPD, area_coords, handles, movie_scrn, cmap)
%% Create initial variables
%start = 1 + round(start * Fs);
%endp = round(endp * Fs);
ap_data = data(:, :, round(start) : round(endp));
ap_data = normalize_data(ap_data);

apdMap = nan(size(ap_data, 1), size(ap_data, 2));

APD_min_rescaled = minapd * Fs / 1000;
APD_max_rescaled = maxapd * Fs / 1000;

AP_level = 1.0 - percentAPD / 100;

area_coords = int16(area_coords);
j_min = 1 + area_coords(1);
i_min = 1 + area_coords(2);
j_max = area_coords(1) + area_coords(3);
i_max = area_coords(2) + area_coords(4);

%% Map calculation
for i = i_min : i_max
    for j = j_min : j_max
        
        index = find(ap_data(i, j, :) <= AP_level);
        
        if size(index, 1) > 2
            apd = max(index(2: end) - index(1: end - 1));
            if ((APD_min_rescaled < apd) && (apd < APD_max_rescaled))
                apdMap(i, j) = apd;
            end
        end
        
    end
end

% account for different sampling frequencies
unitFix = 1000.0 / Fs;
% Calculate Action Potential Duration
apdMap = apdMap * unitFix;


%% Plot APDMap
handles.activeCamData.saveData = apdMap;

cla(movie_scrn);

colormap(handles.activeScreen, cmap);
imagesc(apdMap,'Parent', movie_scrn, 'AlphaData', ~isnan(apdMap));
%axis(movie_scrn,'off');
set(movie_scrn,'Color','k');
set(movie_scrn,'YDir','reverse');
set(movie_scrn,'YTick',[],'XTick',[]);

%Setting up values to use for color axis
APD_min = prctile(apdMap(isfinite(apdMap)),1);
APD_max = prctile(apdMap(isfinite(apdMap)),99);
%caxis(movie_scrn,[APD_min APD_max])

%% Plot Histogram of APDMap
% figure('Name','Histogram of APD')
% hist(reshape(apdMap,[],1),floor(APD_max-APD_min))
% xlim([APD_min APD_max])

%% Calculating statistics
apd_mean=nanmean(apdMap(:));
disp(['The average APD in the region is ' num2str(apd_mean) ' (ms).'])
apd_std=nanstd(apdMap(:));
disp(['The standard deviation of APDs in the region is ' num2str(apd_std) ' (ms).'])
apd_median=nanmedian(apdMap(:));
disp(['The median APD in the region is ' num2str(apd_median) ' (ms).'])

handles.activeCamData.meanresults           = sprintf('Mean: %0.3f (ms)',apd_mean);
handles.activeCamData.medianresults         = sprintf('Median: %0.3f (ms)',apd_median);
handles.activeCamData.SDresults             = sprintf('S.D.: %0.3f (ms)',apd_std);
handles.activeCamData.num_membersresults    = sprintf('');
handles.activeCamData.angleresults          = sprintf('');

set(handles.meanresults,'String',handles.activeCamData.meanresults);
set(handles.medianresults,'String',handles.activeCamData.medianresults);
set(handles.SDresults,'String',handles.activeCamData.SDresults);
set(handles.num_members_results,'String',handles.activeCamData.num_membersresults);
set(handles.angleresults,'String',handles.activeCamData.angleresults);

APD_fig = figure;
imagesc(apdMap);
title(num2str(count))
colorbar
end