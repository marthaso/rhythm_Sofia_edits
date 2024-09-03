function [X, Y, U, V] = downsample(croppedAmap,xx,yy,Vx,Vy,downsample_factor)

        % How long is your ROI
        length_rect = size(croppedAmap,1);
        % How wide is your ROI
        width_rect = size(croppedAmap,2);
        % Basically, make four matrices the size of your ROI. So in the
        % first spot of each matrix, you have info about that first pixel.
        % You have the pixel's x coord, y coord, Vx and Vy.
        X_plot_rect = reshape(xx,[length_rect,width_rect]);
        Y_plot_rect = reshape(yy,[length_rect,width_rect]);
        Vx_plot_rect = reshape(Vx,[length_rect,width_rect]);
        Vy_plot_rect = reshape(Vy,[length_rect,width_rect]);
        %Create index vectors for rows and columns
        row_indices1 = 1:downsample_factor:length_rect;
        col_indices1 = 1:downsample_factor:width_rect;
        % get the downsampled matrices and convert them to vectors
        downsampled_Vx1 = Vx_plot_rect(row_indices1, col_indices1);
        U = reshape(downsampled_Vx1,[],1);
        downsampled_Vy1 = Vy_plot_rect(row_indices1, col_indices1);
        V = reshape(downsampled_Vy1,[],1);
        downsampled_X_plot = X_plot_rect(row_indices1, col_indices1);
        X = reshape(downsampled_X_plot,[],1);
        downsampled_Y_plot = Y_plot_rect(row_indices1, col_indices1);
        Y = reshape(downsampled_Y_plot,[],1);

    end