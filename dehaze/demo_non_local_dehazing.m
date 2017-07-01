% This is a demo script demonstrating the non-local image dehazing algorithm
% described in the paper:
% Non-Local Image Dehazing. Berman, D. and Treibitz, T. and Avidan S., CVPR2016,
% which can be found at:
% www.eng.tau.ac.il/~berman/NonLocalDehazing/NonLocalDehazing_CVPR2016.pdf
% If you use this code, please cite our paper.
% 
% Please read the instructions on README.md in order to use this code.
%
% Author: Dana Berman, 2016. 
% 
% The software code of the Non-Local Image Dehazing algorithm is provided
% under the attached LICENSE.md


% Choose image to use, four example image are supplied with the code in the
% sub-folder "images":
image_name = 'pumpkins'; % 'train'; % 'cityscape'; % 'forest'; % 
image_hazy = imread([image_name,'.png']);

% Load the gamma from the param file. 
% These values were given by Ra'anan Fattal, along with each image:
% http://www.cs.huji.ac.il/~raananf/projects/dehaze_cl/results/
%fid = fopen([image_name,'.txt'],'r');
%[C] = textscan(fid,'%s %f');
%fclose(fid);
gamma = 1;

% Estimate air-light using our method described in:
% Air-light Estimation using Haze-Lines. Berman, D. and Treibitz, T. and 
% Avidan S., ICCP 2017
A = reshape(estimate_airlight(im2double(image_hazy).^(gamma)),1,1,3);

% Dehaze the image	
[image_dehazed, transmission_refined] = non_local_dehazing(image_hazy, A, gamma );

% Display results
figure('Position',[50,50, size(image_hazy,2)*3 , size(image_hazy,1)]);
subplot(1,3,1); imshow(image_hazy);    title('Hazy input')
subplot(1,3,2); imshow(image_dehazed); title('De-hazed output')
subplot(1,3,3); imshow(transmission_refined); colormap('jet'); title('Transmission')
