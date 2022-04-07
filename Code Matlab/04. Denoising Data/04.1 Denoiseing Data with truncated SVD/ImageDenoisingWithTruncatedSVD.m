clc, clear, close all
%% Generating Noise innitial plots on low rank system
imageimp = imread('Image.jpg');
image = double(rgb2gray(imageimp));

noise = randn(size(image))*100;

noisyimage = image + noise;

figure('Name','Original image, noise and noisy image')
c = [0 0.4470 0.7410];
subplot(2,3,1)
imshow(uint8(image))
title('Original image')

subplot(2,3,2)
imshow(uint8(noise))
title('noise')

subplot(2,3,3)
imshow(uint8(noisyimage))
title('noisy image')

[Uo,So,Vo] = svd(image,'econ');
[Un,Sn,Vn] = svd(noise,'econ');
[Uni,Sni,Vni] = svd(noisyimage,'econ');

subplot(2,3,4)
semilogy(diag(So),'o-','color',c)
title('Singular values original image')
ylim([1e1, 1e5])
xlim([0, length(diag(So))])

subplot(2,3,5)
semilogy(diag(Sn),'-o','color',c)
title('Singular values noise')
ylim([1e1, 1e5])

subplot(2,3,6)
semilogy(diag(Sni),'o-','color',c)
title('Singular values noisy image')
ylim([1e1, 1e5])


%% Computing optimal truncation

svalues = diag(Sni);
svalueslog = log(svalues);
med = median(svalueslog);
indx = find(svalueslog==med);

%Calculate slope
x1 = round(indx/2);
x2 = round((length(svalues)-indx)/2);
m = (svalueslog(indx-x1)-svalueslog(indx+x2))/(x1+x2);

%Calculate truncation value
tau = exp(m*indx+svalueslog(indx));

figure('Name','Truncation Plot')
semilogy(diag(Sni),'-o','color',c)
hold on
% Noise approx line
x = -indx:size(svalues)-indx-1;
semilogy(1:size(svalues), exp(-m*x+svalueslog(indx)),'r','Linewidth',2)
xlim([1, length(svalues)])
semilogy(indx,svalues(indx),'bo','Linewidth',4)
semilogy(indx-x1,svalues(indx-x1),'co','Linewidth',4)
semilogy(indx+x2,svalues(indx+x2),'mo','Linewidth',4)
title('Singular values and linear noise approximation')
legend('singular values','linear noise approximation','median value \Theta','\theta / 2','\theta + (m-\theta)/2')

%% Truncation 
ind = Sni>tau;
truncatedSni = Sni*ind;

imagedenoised = Uni*truncatedSni*Vni';

figure('Name','Original image, Noise and Noisy image')
subplot(2,2,1)
imshow(uint8(image))
title('Original image')

subplot(2,2,2)
imshow(uint8(noisyimage))
title('Noisy image')

subplot(2,2,3)
imshow(uint8(imagedenoised))
title(['Denoised image with computet trunaction of #sv = ', num2str(rank(truncatedSni))])

%% Optimal truncation

for i = 1:60
    sv = diag(Sni);
    sv(i:end)=0;
    imagedenoised = Uni*diag(sv)*Vni';
    
    error(i) = sum(abs(imagedenoised-image),'all')/(size(image,1)*size(image,2));
end


subplot(2,2,4)
hold on
plot(1:i,error,'o-','color',c)
[minimum, in] = min(error);
plot(in,minimum,'ro','LineWidth',4)
ylim([0 80])
xlabel('Number of kept singular values')
ylabel('Weighted error (denoised-image)/#pixels')
legend('errors','minimum error')
title('Error of truncated image to original')

%% High rank system

imageimp = imread('Einstein.png');
image = double(imageimp);

noise = randn(size(image))*100;

noisyimage = image + noise;

[Uni,Sni,Vni] = svd(noisyimage,'econ');

svalues = diag(Sni);
svalueslog = log(svalues);
med = median(svalueslog);
indx = find(svalueslog==med);

%Calculate slope
x1 = round(indx/2);
x2 = round((length(svalues)-indx)/2);
m = (svalueslog(indx-x1)-svalueslog(indx+x2))/(x1+x2);

%Calculate truncation value
tau = exp(m*indx+svalueslog(indx));

ind = Sni>tau;
truncatedSni = Sni*ind;

imagedenoised = Uni*truncatedSni*Vni';

figure('Name','Original image, Noise and Noisy image')
subplot(2,2,1)
imshow(uint8(image))
title('Original image')

subplot(2,2,2)
imshow(uint8(noisyimage))
title('Noisy image')

subplot(2,2,3)
imshow(uint8(imagedenoised))
title(['Denoised image with computet trunaction of #sv = ', num2str(rank(truncatedSni))])

for i = 1:60
    sv = diag(Sni);
    sv(i:end)=0;
    imagedenoised = Uni*diag(sv)*Vni';
    
    error(i) = sum(abs(imagedenoised-image),'all')/(size(image,1)*size(image,2));
end


subplot(2,2,4)
hold on
plot(1:i,error,'o-','color',c)
[minimum, in] = min(error);
plot(in,minimum,'ro','LineWidth',4)
ylim([0 80])
xlabel('Number of kept singular values')
ylabel('Weighted error (denoised-image)/#pixels')
legend('errors','minimum error')
title('Error of truncated image to original')
