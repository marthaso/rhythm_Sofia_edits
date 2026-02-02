% Clean data ideas


B = load('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\C25004_18_V_bin5.mat');
A = B.cmosData;

%% 


figure
plot(1:2499,squeeze(A(100,50,:)));

[xi] = getpts;


start_points=[];
end_points=[];
for peak = 1:2:length(xi)-1
    start_points = [start_points, xi(peak)];
    end_points = [end_points, xi(peak+1)];
end

% background noise

BG_data = squeeze(A(150,150,start_points(1):end_points(1)));
std_dev_noise = std(BG_data);

%%

% choose the AP range time
figure
plot(1:2499,squeeze(A(100,50,:)));

[xi] = getpts;
start_time = round(xi(1));
end_time = round(xi(2));

% go through each pixel

for i = 1 : 256
    for j = 1 :256
        pixel = squeeze(A(i,j,start_time:end_time));
        
        %pixel = normalize(pixel);
        MSV = max(pixel)-min(pixel);
        SNR = MSV/std_dev_noise;
        SNR_matrix(i,j) = SNR;
    end
end

figure
imagesc(SNR_matrix)
colorbar


%% plot a couple examples

% good

[x,y] = find((2<SNR_matrix)&(SNR_matrix<10));
for i = 300:305
    figure
    plot(1:2499,squeeze(A(x(i),y(i),:)))
end



a=1;
%%
