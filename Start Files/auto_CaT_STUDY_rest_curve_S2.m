

function [APD50_vec, APD80_vec] = auto_CaT_STUDY_rest_curve_S2(start, endp, study, file_number,filename_pdf, AP_num,firstAP,lastAP)
area_coords = [0,0,256,256];
Fs = 1000;
cmap = 1;
movie_scrn = 1;
handles = 1;

%% APD 50

bin = [3];
freq = [100];
SNR_value = [5];

for snr = 1:length(SNR_value)
    for b1 = 1: length(bin)
        for f1 = 1: length(freq)
            % see if the file exists
            b = bin(b1);
            f = freq(f1);
            s = SNR_value(snr);
            if ~isfile(sprintf('%s_%g_Ca_bin%g_%gHz.mat',study,file_number,b,f))
                % create the file
                directory = fullfile('C:', 'Users', 'Sofia', 'Downloads', study);
                file_name = sprintf('%g_CAM2.raw',file_number);
                data = CMOSconverter(directory, file_name);
                % get mask
                mask = load(sprintf('%s_Ca.txt',study));
                % bin data
                kernel_name = 'gaussian';
                kernel_size = b;
                [data] = binning(data, mask, kernel_size, kernel_name);
                % filter data
                data = filter_data(data, Fs, 100, 0.5, f);
                data = remove_60hz(data, Fs);
                % % Invert data for Ca
                data=-data+max(data(:))+min(data(:));

                data = normalize_data(data);
                figure
                plot(squeeze(data(150,150,:)))
                %data = handles.activeCamData.cmosData;
                title = sprintf('%s_%g_Ca_bin%g_%gHz.mat',study,file_number,b,f);
                save(title,'data','-v7.3');

            end
            data = load(sprintf('%s_%g_Ca_bin%g_%gHz.mat',study,file_number,b,f));
            data = struct2cell(data);
           
            

            data{1,1} = data{1,1}(:,:,1:4999);
            data= cell2mat(data);
            data = normalize_data(data);
            % pixel_calc = data(150,150,:);
            pixel_calc = data(100,100,:); %aug26
            pixel_calc = normalize_data(pixel_calc);
            pixel_calc = squeeze(pixel_calc);
            [pks, locs] =findpeaks(pixel_calc,'MinPeakDistance',50,'MinPeakProminence',0.3,'MaxPeakWidth',110);
            figure
            findpeaks(pixel_calc,'MinPeakDistance',50,'MinPeakProminence',0.3,'MaxPeakWidth',110)

            if isempty(lastAP)
                lastAP = length(locs);
            end


            % Find start/end points
            figure
            plot(1:4999,pixel_calc,locs(AP_num),pks(AP_num),'o')
            print(gcf,filename_pdf,'-dpsc','-bestfit','-append')


            if firstAP == 1
                start = [1, locs(1:lastAP-1)'];                
            else
                start = [locs(firstAP-1:lastAP-1)'];                
            end
            if lastAP == length(locs)
                endp = [locs(firstAP+1:end)', 4999];
            else
                endp = [locs(firstAP+1:lastAP+1)'];
            end


            if ~isfile(sprintf('%s_Ca_bin%g_%gHz_SNR%g.mat',study,b,f,s))
                SNR_mask = SNR_mask_calculation(data,start(1), endp(1), Fs,s);
                mask_name = sprintf('%s_Ca_bin%g_%gHz_SNR%g.mat',study,b,f,s);
                save(mask_name,'SNR_mask')
            end
            SNR_mask = load(sprintf('%s_Ca_bin%g_%gHz_SNR%g.mat',study,b,f,s));
            SNR_mask = struct2cell(SNR_mask);
            SNR_mask = cell2mat(SNR_mask);

            % start = start(AP_num);
            % endp = endp(AP_num);
            start = start(AP_num);
            endp = start + 300;
            figure
            plot(start:endp,pixel_calc(start:endp),locs(AP_num),pks(AP_num),'o')
            print(gcf,filename_pdf,'-dpsc','-bestfit','-append')
            APD50_vec = [];
            APD80_vec = [];
                   

            APD50_vec = [];

            for i = 1 : length(start)
                
                [apdMap1, filtered2, AP_start_points ,AP_baselines, AP_end_points, AP_storage, APD_average] = CaTMap_STUDY_rest_curve_S2(data, start(i), endp(i),...
                    20,165, 50, area_coords, Fs, cmap, movie_scrn, handles, SNR_mask,f,b,filename_pdf,s,study);
               
                APD50_vec = [APD50_vec, APD_average];
                AP_num = i;

                % filename_filtered2 = sprintf('CaT50_bin%g_%gHz_SNR%g_AP%g_matrix_90_hdi.mat', b, f, s, AP_num);
                % directory = [fullfile('C:', 'Users', 'Sofia', 'Desktop', 'PAH', 'CaTMaps', study) filesep];
                % filename = append(directory,filename_filtered2);
                % save(filename, 'filtered2');
                % 
                % filename_1 = sprintf('CaT50_bin%g_%gHz_SNR%g_AP%g_all.mat', b, f, s, AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'apdMap1');
                % 
                % filename_1 = sprintf('CaT50_bin%g_%gHz_SNR%g_AP%g_AP_start_point.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_start_points');
                % 
                % filename_1 = sprintf('CaT50_bin%g_%gHz__AP%g_AP_end_points.mat', b, f,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_end_points');
                % 
                % filename_1 = sprintf('CaT50_bin%g_%gHz_SNR%g__AP%g_AP_baselines.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_baselines');
                % 
                % filename_1 = sprintf('CaT50_bin%g_%gHz_SNR%g_AP%g_AP_storage.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_storage');
            end

            clear apdMap
        end
    end
end

%% APD 80

% disp('CaT80')
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
            data= cell2mat(data);
            data = normalize_data(data);
            SNR_mask = load(sprintf('%s_Ca_bin%g_%gHz_SNR%g.mat',study,b,f,s));
            SNR_mask = struct2cell(SNR_mask);
            SNR_mask = cell2mat(SNR_mask);

            APD80_vec = [];

            for i = 1: length(start)


                [apdMap1, filtered2, AP_start_points ,AP_baselines, AP_end_points, AP_storage, APD_average] = CaTMap_STUDY_rest_curve_S2(data, start(i), endp(i), 20,...
                    210, 80, area_coords, Fs, cmap, movie_scrn, handles, SNR_mask,f,b,filename_pdf,s,study);

                APD80_vec = [APD80_vec, APD_average];
                AP_num = i;

                % filename_filtered2 = sprintf('CaT80_bin%g_%gHz_SNR%g_AP%g_matrix_90_hdi.mat', b, f, s, AP_num);
                % directory = [fullfile('C:', 'Users', 'Sofia', 'Desktop', 'PAH', 'CaTMaps', study) filesep];
                % filename = append(directory,filename_filtered2);
                % save(filename, 'filtered2');
                % 
                % filename_1 = sprintf('CaT80_bin%g_%gHz_SNR%g_AP%g_all.mat', b, f, s, AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'apdMap1');
                % 
                % filename_1 = sprintf('CaT80_bin%g_%gHz_SNR%g_AP%g_AP_start_point.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_start_points');
                % 
                % filename_1 = sprintf('CaT80_bin%g_%gHz__AP%g_AP_end_points.mat', b, f,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_end_points');
                % 
                % filename_1 = sprintf('CaT80_bin%g_%gHz_SNR%g__AP%g_AP_baselines.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_baselines');
                % 
                % filename_1 = sprintf('CaT80_bin%g_%gHz_SNR%g_AP%g_AP_storage.mat', b, f, s,AP_num);
                % filename = append(directory,filename_1);
                % save(filename, 'AP_storage');
            end


            clear apdMap
        end
    end
end

end

