%% Manual calc #2
function [vels] = manual_cal_2(activation_times)

num_of_neighbors = 3;
vels = [];
activation_times = activation_times.*0.001;

for i = num_of_neighbors+1:size(activation_times,1)-num_of_neighbors
    for j = num_of_neighbors+1:size(activation_times,2)-num_of_neighbors
        % your pixel is (i,j)
        % get neighbors:
        neighbor_1 = activation_times(i-num_of_neighbors,j);
        neighbor_2 = activation_times(i+num_of_neighbors,j);
        neighbor_3 = activation_times(i, j+num_of_neighbors);
        neighbor_4 = activation_times(i, j-num_of_neighbors);
        % get velocities
        distance = 0.01749 * num_of_neighbors; %cm
        if activation_times(i,j) == neighbor_1
            velocity_1=NaN;
        else
        velocity_1 = abs(distance/(neighbor_1-activation_times(i,j)));
        end
        if activation_times(i,j) == neighbor_2
            velocity_2=NaN;
        else
        velocity_2 = abs(distance/(neighbor_2-activation_times(i,j)));
        end
        if activation_times(i,j) == neighbor_3
            velocity_3=NaN;
        else
        velocity_3 = abs(distance/(neighbor_3-activation_times(i,j)));
        end
        if activation_times(i,j) == neighbor_4
            velocity_4=NaN;
        else
        velocity_4 = abs(distance/(neighbor_4-activation_times(i,j)));
        end
        velocities = [velocity_1,velocity_2,velocity_3,velocity_4];
        final_vel = nanmean(velocities);
        vels(i,j) = final_vel;
    end
end
vels(vels==0) = NaN;

figure
imagesc(vels)
colorbar

average_vel = nanmean(nanmean(vels))

a=1
end


