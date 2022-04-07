clc, clear, close all
%% Import Data
[gif,map] = imread('CylinderFlow','gif','Frames','all');

%% Resample

for i = 1:226
    img = double(gif(:,:,:,i));
    img = img(91:180,51:end)/255;
    %img = img + randn(size(img))*60; %gaussian Noise
    img = imnoise(img,'salt & pepper',0.33); %salt and pepper noise 
    X(:,i) = reshape(img,90*430,1);
    gif2(:,:,:,i) = mat2gray(img); 
end

implay(gif2);

%% Denoise

[u, s, v] = svd(X, 'econ');

s = diag(s);

figure('Name','Sing Values')
subplot(1,2,1)
semilogy(s(1:50),'o-')
hold on
title('Singular Values')
xlim([1,50])
subplot(1,2,2)
for i = 1:length(s)
    svar(i) = sum(s(1:i))/sum(s);
end
plot(svar(1:226));
xlim([1,226]);
title('Captured Variance')

figure('Name','Principal Flow Modes')
subplot('Position',[0.34,0.77,0.35,0.2])
imshow(mat2gray(reshape(abs(u(:,1)),90,430))), colormap('jet');
title('Mean Flow')

for i = 2:9
    subplot(5,2,i+1)
    imshow(mat2gray(reshape(abs(u(:,i)),90,430))), colormap('jet');
    title(['POD ', num2str(i)])
end


figure('Name','POD and Timecorrelations')
j = 1;
for i = 1:10
    subplot(10,2,j)
    imshow(mat2gray(reshape(abs(u(:,i)),90,430))), colormap('jet');
    j = j+1;
    subplot(10,2,j)
    plot(v(:,i));
    xlim([1,226])
    j = j+1;
end

%% Reconstruct flow with first few POD Modes

Rec = u(:,1:20)*diag(s(1:20))*v(:,1:20)';

for i = 1:226
    img = reshape(Rec(:,i),90,430);
    gif2(:,:,:,i) = mat2gray(img); 
end
implay(gif2);

%% Clean data 

[Xlr, Xs] = SVT(X);

%%

figure('Name','Low Rank U')
subplot(3,1,1)
hold on
imshow(mat2gray(reshape(abs(X(:,1)),90,430))), colormap('gray');
title('Noisy Data')
subplot(3,1,2)
hold on
imshow(mat2gray(reshape(abs(Xlr(:,1)),90,430)))
title('Low Rank Data')
subplot(3,1,3)
hold on
imshow(mat2gray(reshape(abs(Xs(:,1)),90,430)))
title('Sparse Data')

[u, s, v] = svd(Xlr,'econ');

s = diag(s);

figure('Name','Sing Values')
subplot(1,2,1)
semilogy(s(1:50),'o-')
hold on
title('Singular Values')
xlim([1,50])
subplot(1,2,2)
for i = 1:length(s)
    svar(i) = sum(s(1:i))/sum(s);
end
plot(svar(1:226));
xlim([1,226]);
title('Captured Variance')

figure('Name','Principal Flow Modes')
subplot('Position',[0.34,0.77,0.35,0.2])
imshow(mat2gray(reshape(abs(u(:,1)),90,430))), colormap('jet');
title('Mean flow')

for i = 2:9
    subplot(5,2,i+1)
    imshow(mat2gray(reshape(abs(u(:,i)),90,430))), colormap('jet');
    title(['POD',num2str(i)])
end


figure('Name','POD and Timecorrelations')
j = 1;
for i = 1:10
    subplot(10,2,j)
    imshow(mat2gray(reshape(abs(u(:,i)),90,430))), colormap('jet');
    j = j+1;
    subplot(10,2,j)
    plot(v(:,i));
    j = j+1;
end

Rec = u(:,1:20)*diag(s(1:20))*v(:,1:20)';

for i = 1:226
    img = reshape(Rec(:,i),90,430);
    gif2(:,:,:,i) = mat2gray(img); 
end
implay(gif2);

%% SVT Singular Value Tresholding Algorithm 

function x = shrink(X, a)
    x = sign(X).*max(abs(X)-a,0);
end

function x = soft_thresh(X, a)
    [u, s, v] = svd(X,'econ');
    x = u*shrink(s,a)*v';
end

function [L,S] = SVT(X) 
[n1,n2] = size(X);
mu = n1*n2/(4*sum(abs(X(:)))); 
lambda = 1/sqrt(max(n1,n2)); 
thresh = 1e-6*norm(X,'fro');
L = zeros(size(X));
S = zeros(size(X));
Y = zeros(size(X));
count = 0; 
    while((norm(X-L-S,'fro')>thresh)&&(count<1000))
        L = soft_thresh(X-S+(1/mu)*Y,1/mu);
        S = shrink(X-L+(1/mu)*Y,lambda/mu); 
        Y = Y + mu*(X-L-S);
        count = count + 1;
        if mod(count, 10) == 0
            sprintf('Completed %f %%',count/1000)
        end
    end
end


