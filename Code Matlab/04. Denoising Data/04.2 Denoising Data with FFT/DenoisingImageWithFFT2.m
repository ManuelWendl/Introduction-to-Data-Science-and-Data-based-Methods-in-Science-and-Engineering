clc, clear, close all

%% Import Data White Noise
imageimp = imread('Einstein.jpg');
image = double(imageimp);

noise = randn(size(image))*100;

noisyimage = image + noise;

%% Compute Fourier Coefficient matrix

FC = fft2(noisyimage);

%% Filter Noise Frequencies
% only pass low frequencies (middle)

FCshifted = fftshift(FC);
m = round(size(FCshifted,1)/2);
n = round(size(FCshifted,2)/2);

a = 190;
FCclean = zeros(size(FCshifted));
FCclean(m-a:m+a,n-a:n+a)=FCshifted(m-a:m+a,n-a:n+a);


%% Reconstruct denoised image

FCclean = fftshift(FCclean);
Imagedenoised = ifft2(FCclean);

figure('Name','Denoised Picture');
subplot(2,3,1)
imshow(uint8(noisyimage));
title('Noisy image')
subplot(2,3,2)
imshow(uint8(abs(Imagedenoised)));
title('Denoised image')
subplot(2,3,3)
imshow(uint8(image));
title('Original')

FClog = log(abs(fftshift(FC))+1);
subplot(2,3,4)
imshow(mat2gray(FClog));

FCcleanlog = log(abs(fftshift(FCclean))+1);
subplot(2,3,5)
imshow(mat2gray(FCcleanlog));

FCcleanlog = log(abs(fftshift(fft2(image)))+1);
subplot(2,3,6)
imshow(mat2gray(FCcleanlog));

%% Import Data High Frequent Noise

Import = imread('NoisyMoonLanding.png');
Image = double(Import);


%% Compute Fourier Coefficient matrix

FC = fft2(Image);

%% Filter Noise Frequencies

FClog = log(abs(fftshift(FC))+1);
figure('Name','Fourier coefficients')
subplot(2,2,3)
imshow(mat2gray(FClog));

% only pass low frequencies (middle)

FCshifted = fftshift(FC);
m = size(FCshifted,1);
n = size(FCshifted,2);

a = 44;
FCclean = zeros(size(FCshifted));
FCclean(m/2-a:m/2+a,n/2-a:n/2+a)=FCshifted(m/2-a:m/2+a,n/2-a:n/2+a);

FCcleanlog = log(abs(FCclean)+1);
subplot(2,2,4)
imshow(mat2gray(FCcleanlog));


%% Reconstruct denoised image

FCclean = fftshift(FCclean);
Imagedenoised = ifft2(FCclean);

subplot(2,2,1)
imshow(uint8(Image));
title('noisy image')
subplot(2,2,2)
imshow(uint8(abs(Imagedenoised)));
title('denoised image')

%% Filtering Noise special Frequencies

FClog = log(abs(fftshift(FC))+1);
figure('Name','Fourier coefficients')
subplot(1,2,1)
hold on
surf(FClog, 'EdgeColor','interp'),colormap('jet');
surf(ones(size(FClog))*9.9,'FaceColor','b','EdgeColor','b');

a = 44;
b = 110;

s=ones(size(FClog))*11;
surf(n/2-b:n/2+b,m/2-b:m/2+b,s(m/2-b:m/2+b,n/2-b:n/2+b),'FaceColor','k');
legend('Fourier coefficients','Filtering layer 1','Inner filtering layer')

ind = FClog < 9.9;
indinner = FClog < 11;
FCclean2 = FCshifted.*ind;
FCclean2inner = FCshifted.*indinner;
FCclean2(m/2-b:m/2+b,n/2-b:n/2+b)=FCclean2inner(m/2-b:m/2+b,n/2-b:n/2+b);
FCclean2(m/2-a:m/2+a,n/2-a:n/2+a)=FCshifted(m/2-a:m/2+a,n/2-a:n/2+a);

FCclean2log = log(abs(FCclean2)+1);
subplot(1,2,2)
imshow(mat2gray(FCclean2log));


FCclean = fftshift(FCclean2);
Imagedenoised = ifft2(FCclean2);

figure('Name','Denoised Picture');
subplot(1,2,1)
imshow(uint8(Image));
title('noisy image')
subplot(1,2,2)
imshow(uint8(abs(Imagedenoised)));
title('denoised image')
