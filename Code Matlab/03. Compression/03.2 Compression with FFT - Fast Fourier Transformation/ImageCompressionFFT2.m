clc,clear,close all
%% Import Picture
Image = imread('Einstein.png');
Image = double(Image);

%% Plotting Surface
fig1 = figure('Name','Surface Plot');
sp1 = subplot(1,2,1);
colormap(sp1 ,gray)
imshow(uint8(Image))
title('Image')
sp2 = subplot(1,2,2);
surf(Image(:,sort(1:2945, 'descend')),'EdgeColor', 'interp')
colormap(sp2 ,jet)
title('Image as Surface')
colorbar
view([-165,79])

%% Plotting Discrete Fourier Transformation Matrix

F = dftmtx(3668);
figure('Name','DFT Matrix')
hold on
title('DFT Matrix of size 3668 \times 3668')
imshow(real(F))


%% Fourier Transormation
FourImage = fft2(Image);


%% Reducing Data
% Percentages to which we reduce our data
perc = [0.99 0.05 0.01 0.001];

% Fourier Coefficients sorted for finding % of largest
FourOrder = sort(abs(FourImage(:)),'descend');


figure('Name','Reduced Data')
hold on

% Looping percentages
for i = 1:length(perc)
    limit = FourOrder(round(perc(i)*size(FourOrder,1)));
    ind = abs(FourImage)>limit;
    RedFourier = FourImage.*ind;
    RedImage = ifft2(RedFourier);
    subplot(2,4,i)
    imshow(uint8(RedImage));
    title(['Coefficients reduced to ',num2str(perc(i)*100),'%']);
    subplot(2,4,i+4)
    Frame = RedImage(1000:1500,1000:1700);
    imshow(uint8(Frame));
    title(['Coefficients reduced to ',num2str(perc(i)*100),'%']);
end


%% Edge Detecting

modes = [50 500 1000 2000];

figure('Name','Edge Detection')

for i = 1:length(modes)
    limit = FourOrder(modes(i));
    ind = abs(FourImage)<limit;
    RedFourier = FourImage.*ind;
    RedImage = ifft2(RedFourier);
    subplot(2,4,i)
    imshow(uint8(RedImage))
    title(['Coefficients reduced by ',num2str(modes(i)),' frequencies']);
    subplot(2,4,i+4)
    RedFourierlog = log(abs(fftshift(RedFourier))+1);
    imshow(mat2gray(RedFourierlog))
    title(['Coefficients reduced by ',num2str(modes(i)),' frequencies']);
end
