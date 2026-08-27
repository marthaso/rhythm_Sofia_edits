
%% Function : auto_CaT_STUDY

% Purpose: Upload the data you want to evaluate. Clean it if necessary.
% Find the peaks and define start/end times. Call another function to
% calculate APD of each action potential, and generate two vectors with the
% results.

% Inputs:
% start - start time of the data set 
% endp - end time of the data set
% study - study number
% file_number - file to be analyzed
% filename_pdf - name of the results pdf
% firstAP - first action potential to analyze
% lastAP - last action potential to analyze
% cycle_length - shortest pacing cycle length 

% Outputs:
% APD50_vec/APD80_vec - vectosr containing the action potential durations of the
% specified action potentials. Calculates APD50 and APD80 respectively.


function [APD50_vec, APD80_vec] = auto_CaT_STUDY(start, endp, study, file_number,filename_pdf, AP_num,firstAP,lastAP,cycle_length)
area_coords = [0,0,256,256];
Fs = 1000;
cmap = 1;
movie_scrn = 1;
handles = 1;

%% 1. Calculate APD 50

% These are filtering parameters to clean the data. Usually don't need to
% change them.
bin = [3];
freq = [100];
SNR_value = [5];

% If you wanted to check different values for SNR_value, bin, or freq, you
% would do each for loop multiple times. If you only have one value for
% each of these parameters, the loop will only run once.

for snr = 1:length(SNR_value)
    for b1 = 1: length(bin)
        for f1 = 1: length(freq)
            % see if the .mat file exists
            b = bin(b1);
            f = freq(f1);
            s = SNR_value(snr);
            % If the .mat file does not exist, get the raw data file and
            % convert it.
            if ~isfile(sprintf('%s_%g_Ca_bin%g_%gHz.mat',study,file_number,b,f))
                % Change to the directory where your RAW file is located.
                % and specific the name of the file (CAM2 is calcium and
                % CAM1 is voltage).
                directory = fullfile('C:', 'Users', 'Sofia', 'Downloads', study);
                file_name = sprintf('%g_CAM2.raw',file_number);
                data = CMOSconverter(directory, file_name);
                % Make sure you have .txt mask for this file.
                mask = load(sprintf('%s_Ca.txt',study));
                % bin data
                kernel_name = 'gaussian';
                kernel_size = b;
                [data] = binning(data, mask, kernel_size, kernel_name);
                % filter data
                data = filter_data(data, Fs, 100, 0.5, f);
                data = remove_60hz(data, Fs);
                % % Invert data for Ca. Comment out if it's voltage.
                data=-data+max(data(:))+min(data(:));

                % Plot one pixel as an example.
                data = normalize_data(data);
                figure
                plot(squeeze(data(150,150,:)))
                
                %data = handles.activeCamData.cmosData;
                title = sprintf('%s_%g_Ca_bin%g_%gHz.mat',study,file_number,b,f);
                save(title,'data','-v7.3');

            end

            % Load the .mat file
            data = load(sprintf('%s_%g_Ca_bin%g_%gHz.mat',study,file_number,b,f));
            data = struct2cell(data);

            % Plot one pixel as an example
            data{1,1} = data{1,1}(:,:,1:4999);
            data= cell2mat(data);
            data = normalize_data(data);
            % You might need to change the pixel depending on the dataset.
            pixel_calc = data(150,150,:);
            pixel_calc = normalize_data(pixel_calc);
            pixel_calc = squeeze(pixel_calc);
            
            % Find peaks in the data. Might need to adjust the parameters
            % if the wrong peaks are found.
            [pks, locs] =findpeaks(pixel_calc,'MinPeakDistance',50,'MinPeakProminence',0.3,'MaxPeakWidth',120);
            
            % if you didn't choose a last AP, set the last AP to be the
            % last peak found
            if isempty(lastAP)
                lastAP = length(locs);
            end


            % Find start/end points for each action potential (AP)
            % Plot the action potentials that will be analyzed. This is for
            % visualization purposes. At this point you can go back and
            % edit first/last AP or the parameters if something looks
            % wrong. Then run again.
            figure
            plot(1:4999,pixel_calc,locs(firstAP:lastAP),pks(firstAP:lastAP),'o')
            
            % This line will add the graph to our final pdf so we can see
            % which APs were analyzed.
            print(gcf,filename_pdf,'-dpsc','-bestfit','-append')

            % Now create a vector with the start times and a vector with
            % the end times for each action potential. 

            % If you are starting with the first peak found (firstAP == 1),
            % your first start point will be 1. After that, the start point
            % will be the previous peak found plus a bit of time (to avoid
            % having two peaks in the dataset). Might need to adjust this
            % extra time depending on the data.

            % If you are not starting in the first AP, then your first
            % start time is the previous peak from your first peak, then
            % continue on.

            if firstAP == 1
                start = [1, [locs(1:lastAP-1)+20]'];                
            else
                start = [locs(firstAP-1:lastAP-1)+20'];
            end

            % Similar logic for the end times. The end of one AP will be
            % the next peak minus a bit of time to avoid two peaks. If you
            % are evaluating all APs, then the final end time is just 4999
            % (end of our data file).
            if lastAP == length(locs)
                endp = [[locs(firstAP+1:end)-20]', 4999];
            else
                endp = [locs(firstAP+1:lastAP+1)-20'];

            end

            % Useful for S1s, you can see the time difference between
            % consecutive peaks. Can uncomment if needed.

            % locs_new = locs(2:end);
            % locs_new=[locs_new;5000];
            % APs = locs_new-locs


            % These lines will generate a signal to noise ratio. It is
            % needed for certain data sets. For now, you can just leave
            % unchanged.
            if ~isfile(sprintf('%s_Ca_bin%g_%gHz_SNR%g.mat',study,b,f,s))
                SNR_mask = ones(256,256);
                % SNR_mask = SNR_mask_calculation(data,start(2), endp(2), Fs,s);
                mask_name = sprintf('%s_Ca_bin%g_%gHz_SNR%g.mat',study,b,f,s);
                save(mask_name,'SNR_mask')
            end
            SNR_mask = load(sprintf('%s_Ca_bin%g_%gHz_SNR%g.mat',study,b,f,s));
            SNR_mask = struct2cell(SNR_mask);
            SNR_mask = cell2mat(SNR_mask);
            

            % Create empty vector where we will store each action potential
            % duration.
            APD50_vec = [];

            % this "for loop" will go through each of the action potentials
            % (length(start) is how long the start vector is. if you have 5
            % APs, you will have 5 start times and the loop will run 5
            % times).
            for i = 1 : length(start)

                % Call function to calculate action potential duration. Make sure to change
                % "STUDY". Read description in the other file to know what
                % all these variables mean.
                [apdMap1, filtered2, AP_start_points ,AP_baselines, AP_end_points, AP_storage, APD_average] = CaTMap_STUDY(data, start(i), endp(i),...
                    15, cycle_length*0.9, 50, area_coords, Fs, cmap, movie_scrn, handles, SNR_mask,f,b,filename_pdf,s,study);

                % Once we get a result, store in our results vector.
                APD50_vec = [APD50_vec, APD_average];
                % Display the action potential that was just analyzed. Can
                % be useful to know how far along you are if there are a
                % lot of APS.

                % AP_num = i

                % You can save all the data files if needed. For now, keep
                % commented out.

                % filename_filtered2 = sprintf('APD50_bin%g_%gHz_SNR%g_AP%g_matrix_90_hdi.mat', b, f, s, AP_num);
                % directory = [fullfile('C:', 'Users', 'Sofia', 'Desktop', 'PAH', 'APDMaps', study, num2str(file_number))];
                % filename = append(directory,filename_filtered2);
                % save(filename, 'filtered2');
                % 
                % filename_1 = sprintf('APD50_bin%g_%gHz_SNR%g_AP%g_all.mat', b, f, s, AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'apdMap1');
                % % 
                % filename_1 = sprintf('APD50_bin%g_%gHz_SNR%g_AP%g_AP_start_point.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_start_points');
                % 
                % filename_1 = sprintf('APD50_bin%g_%gHz__AP%g_AP_end_points.mat', b, f,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_end_points');
                % 
                % filename_1 = sprintf('APD50_bin%g_%gHz_SNR%g__AP%g_AP_baselines.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_baselines');
                % 
                % filename_1 = sprintf('APD50_bin%g_%gHz_SNR%g_AP%g_AP_storage.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_storage');
            end

            clear apdMap
        end
    end
end

%% Same as previous loop. Only difference is that we are calculating APD80 instead of APD50.

disp('CaT80')
% APD80_vec = [];
% 
% 
for snr = 1:length(SNR_value)
    for b1 = 1: length(bin)
        for f1 = 1: length(freq)
            % see if the file exists
            b = bin(b1);
            f = freq(f1);
            s = SNR_value(snr);
            data = load(sprintf('%s_%g_Ca_bin%g_%gHz.mat',study,file_number,b,f));
            data = struct2cell(data);
            data{1,1} = data{1,1}(:,:,1:4999);
            % data{1,1} = data{1,1}(:,:,1:1999); % for file 28 
            
            data= cell2mat(data);
            data = normalize_data(data);
            SNR_mask = load(sprintf('%s_Ca_bin%g_%gHz_SNR%g.mat',study,b,f,s));
            SNR_mask = struct2cell(SNR_mask);
            SNR_mask = cell2mat(SNR_mask);

            APD80_vec = [];

            for i = 1: length(start)


                [apdMap1, filtered2, AP_start_points ,AP_baselines, AP_end_points, AP_storage, APD_average] = CaTMap_oct30(data, start(i), endp(i), ...
                    20, cycle_length, 80, area_coords, Fs, cmap, movie_scrn, handles, SNR_mask,f,b,filename_pdf,s,study);

                APD80_vec = [APD80_vec, APD_average];
                AP_num = i

                % filename_filtered2 = sprintf('APD80_bin%g_%gHz_SNR%g_AP%g_matrix_90_hdi.mat', b, f, s, AP_num);
                % directory = [fullfile('C:', 'Users', 'Sofia', 'Desktop', 'PAH', 'APDMaps', study, num2str(file_number))];
                % filename = append(directory,filename_filtered2);
                % save(filename, 'filtered2');
                % 
                % filename_1 = sprintf('APD80_bin%g_%gHz_SNR%g_AP%g_all.mat', b, f, s, AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'apdMap1');
                % 
                % filename_1 = sprintf('APD80_bin%g_%gHz_SNR%g_AP%g_AP_start_point.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_start_points');
                % 
                % filename_1 = sprintf('APD80_bin%g_%gHz__AP%g_AP_end_points.mat', b, f,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_end_points');
                % 
                % filename_1 = sprintf('APD80_bin%g_%gHz_SNR%g__AP%g_AP_baselines.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_baselines');
                % 
                % filename_1 = sprintf('APD80_bin%g_%gHz_SNR%g_AP%g_AP_storage.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_storage');
            end


            clear apdMap
        end
    end
end

end

