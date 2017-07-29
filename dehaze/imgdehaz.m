
image_hazy = imread('img.png');
gamma = 1;
A = reshape(estimate_airlight(im2double(image_hazy).^(gamma)),1,1,3);

   % Dehaze the image	
[image_dehazed, transmission_refined] = non_local_dehazing(image_hazy, A, gamma );

imshow(image_dehazed);