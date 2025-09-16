% map creater
mask = zeros(256, 256);
y_coords = [67:177];
x_coords = [35:155];

mask(x_coords,y_coords) = 1;

save('C:\Users\Sofia\Desktop\Rhythm (2)\Rhythm\rhythm_try2\new_mask_fig4.txt','mask','-ascii', '-tabs')

