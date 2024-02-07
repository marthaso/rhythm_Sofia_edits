% Load the saved array
loadedArray = load('example8.mat');

% Access the loaded array
loadedData = loadedArray.arrayToSave;

badpixel1 = loadedData(2,2,1:4999);
badpixel2 = loadedData(254, 254, 1:4999);
goodpixel1 = loadedData(120,120,1:4999);
goodpixel2 = loadedData(128,128,1:4999);
goodpixel3 = loadedData(130,130,1:4999);

%% fft of bad pixel
badpixel1_sq = squeeze(badpixel1);
badpixel_fft = fft(badpixel1_sq);

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 4999;             % Length of signal
t = (0:L-1)*T;        % Time vector

plot(Fs/L*(0:L-1),abs(badpixel_fft),"LineWidth",3)
title("Complex Magnitude of fft Spectrum")
xlabel("f (Hz)")
ylabel("|fft(X)|")

%% fft of good signal

goodpixel2_sq = squeeze(goodpixel2);
goodpixel_fft = fft(goodpixel2_sq);

Fs = 1000;            % Sampling frequency                    
T = 1/Fs;             % Sampling period       
L = 4999;             % Length of signal
t = (0:L-1)*T;        % Time vector
figure
plot(Fs/L*(0:L-1),abs(goodpixel_fft),"LineWidth",3)
title("good pixel")
xlabel("f (Hz)")
ylabel("|fft(X)|")

%% peak finder

%bad pixels
vector_badpixel1 = reshape(badpixel1, 1, []);
[pks, locs]= findpeaks(vector_badpixel1);
time = 1:4999;

plot(time,vector_badpixel1,time(locs),pks,'o')
size(pks)

vector_badpixel2 = reshape(badpixel2, 1, []);
[pks, locs]= findpeaks(vector_badpixel2);
time = 1:4999;

figure
plot(time,vector_badpixel2,time(locs),pks,'o')
size(pks)


vector_goodpixel1 = reshape(goodpixel1, 1, []);
[pks, locs]= findpeaks(vector_goodpixel1);
time = 1:4999;

figure
plot(time,vector_goodpixel1,time(locs),pks,'o')
size(pks)

vector_goodpixel2 = reshape(goodpixel2, 1, []);
[pks, locs]= findpeaks(vector_goodpixel2);
time = 1:4999;

figure
plot(time,vector_goodpixel2,time(locs),pks,'o')
size(pks)


vector_goodpixel3 = reshape(goodpixel3, 1, []);
[pks, locs]= findpeaks(vector_goodpixel3);
time = 1:4999;

figure
plot(time,vector_goodpixel3,time(locs),pks,'o')
size(pks)


%% try to make an algorithm


mask = zeros(256,256);

for i = 1:256
    for j = 1:256
        pixel = loadedData(i,j,1:4999);
        vector_pixel = reshape(pixel, 1, []);
        pks = findpeaks(vector_pixel);
        if max(size(pks)) < 170
            mask(i,j) = 1;
        end
    end
end
figure
imagesc(mask)

%% now try to smooth the mask

mask1 = mask;

%%
% Define the threshold for changing pixel values
threshold = 5;

% Iterate over each pixel
for i = 2:size(mask1, 1)-1
    for j = 2:size(mask1, 2)-1
        % Extract the current pixel value
        current_pixel = mask1(i, j);
        
        % Extract the values of the neighboring pixels
        neighbors = mask1(i-1:i+1, j-1:j+1);
        neighbors = neighbors(:);
        
        % Count the number of neighbors with a different value from the current pixel
        different_neighbors = sum(neighbors ~= current_pixel);
        
        % If five or more neighbors are different, change the current pixel value
        if different_neighbors >= threshold
            mask1(i, j) = ~current_pixel; % Toggle the value
        end
    end
end

figure
imagesc(mask1)



