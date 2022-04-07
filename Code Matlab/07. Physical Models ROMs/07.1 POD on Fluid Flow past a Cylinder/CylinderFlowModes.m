clc, clear, close all
%% Import Data
[gif,map] = imread('CylinderFlow','gif','Frames','all');

%% Resample

for i = 1:226
    img = double(gif(:,:,:,i));
    img = img(91:180,51:end);
    X(:,i) = reshape(img,90*430,1);
    gif2(:,:,:,i) = mat2gray(img); 
end

implay(gif2);

%% Principal Modes
% PCA of the timeframes

[U,S,V] = svd(X,'econ');
S = diag(S);

figure('Name','Sing Values')
subplot(1,2,1)
semilogy(S(1:50),'o-')
hold on
title('Singular Values')
xlim([1,50])
subplot(1,2,2)
for i = 1:length(S)
    svar(i) = sum(S(1:i))/sum(S);
end
plot(svar(1:226));
xlim([1,226]);
title('Captured Variance')

%% Plot first PODs

figure('Name','Principal Flow Modes')
subplot('Position',[0.34,0.77,0.35,0.2])
imshow(mat2gray(reshape(abs(U(:,1)),90,430))), colormap('jet');
title('Mean Flow (POD 1)')
subplot(5,2,3)
imshow(mat2gray(reshape(abs(U(:,2)),90,430))), colormap('jet');
title('POD 2')
subplot(5,2,4)
imshow(mat2gray(reshape(abs(U(:,3)),90,430))), colormap('jet');
title('POD 3')
subplot(5,2,5)
imshow(mat2gray(reshape(abs(U(:,4)),90,430))), colormap('jet');
title('POD 4')
subplot(5,2,6)
imshow(mat2gray(reshape(abs(U(:,5)),90,430))), colormap('jet');
title('POD 5')
subplot(5,2,7)
imshow(mat2gray(reshape(abs(U(:,6)),90,430))), colormap('jet');
title('POD 6')
subplot(5,2,8)
imshow(mat2gray(reshape(abs(U(:,7)),90,430))), colormap('jet');
title('POD 7')
subplot(5,2,9)
imshow(mat2gray(reshape(abs(U(:,8)),90,430))), colormap('jet');
title('POD 8')
subplot(5,2,10)
imshow(mat2gray(reshape(abs(U(:,9)),90,430))), colormap('jet');
title('POD 9')

%% Time Correlation


figure('Name','Principal Flow Modes')
subplot(9,2,1)
imshow(mat2gray(reshape(abs(U(:,1)),90,430))), colormap('jet');
title('POD Modes')
subplot(9,2,2)
title('Corellation to Time')
hold on
plot(V(:,1));
xlim([1,226])
subplot(9,2,3)
imshow(mat2gray(reshape(abs(U(:,2)),90,430))), colormap('jet');
subplot(9,2,4)
plot(V(:,2));
xlim([1,226])
subplot(9,2,5)
imshow(mat2gray(reshape(abs(U(:,3)),90,430))), colormap('jet');
subplot(9,2,6)
plot(V(:,3));
xlim([1,226])
subplot(9,2,7)
imshow(mat2gray(reshape(abs(U(:,4)),90,430))), colormap('jet');
subplot(9,2,8)
plot(V(:,4));
xlim([1,226])
subplot(9,2,9)
imshow(mat2gray(reshape(abs(U(:,5)),90,430))), colormap('jet');
subplot(9,2,10)
plot(V(:,5));
xlim([1,226])
subplot(9,2,11)
imshow(mat2gray(reshape(abs(U(:,6)),90,430))), colormap('jet');
subplot(9,2,12)
plot(V(:,6));
xlim([1,226])
subplot(9,2,13)
imshow(mat2gray(reshape(abs(U(:,7)),90,430))), colormap('jet');
subplot(9,2,14)
plot(V(:,7));
xlim([1,226])
subplot(9,2,15)
imshow(mat2gray(reshape(abs(U(:,8)),90,430))), colormap('jet');
subplot(9,2,16)
plot(V(:,8));
xlim([1,226])
subplot(9,2,17)
imshow(mat2gray(reshape(abs(U(:,9)),90,430))), colormap('jet');
subplot(9,2,18)
plot(V(:,9));
xlim([1,226])

%% Compressed Sensing with time samples
% Only 22 randomly taken time samples

%R = round(rand(1,30)*226);

% Optimisation
R = round(linspace(1,216,40));
R = R+randi(2,1,40);

Xr = X(:,R);

Xa = zeros(size(X)); Xa(:,R)=X(:,R); 
for i = 1:226
    img = reshape(double(Xa(:,i)),90,430);
    gif2(:,:,:,i) = mat2gray(img); 
end
implay(gif2);

[Ur,Sr,Vr] = svd(Xr,'econ');

figure('Name','Principal Flow Modes')
subplot('Position',[0.34,0.77,0.35,0.2])
imshow(mat2gray(reshape(abs(Ur(:,1)),90,430))), colormap('jet');
title('Mean flow (POD 1)')
subplot(5,2,3)
imshow(mat2gray(reshape(abs(Ur(:,2)),90,430))), colormap('jet');
title('POD 2')
subplot(5,2,4)
imshow(mat2gray(reshape(abs(Ur(:,3)),90,430))), colormap('jet');
title('POD 3')
subplot(5,2,5)
imshow(mat2gray(reshape(abs(Ur(:,4)),90,430))), colormap('jet');
title('POD 4')
subplot(5,2,6)
imshow(mat2gray(reshape(abs(Ur(:,5)),90,430))), colormap('jet');
title('POD 5')
subplot(5,2,7)
imshow(mat2gray(reshape(abs(Ur(:,6)),90,430))), colormap('jet');
title('POD 6')
subplot(5,2,8)
imshow(mat2gray(reshape(abs(Ur(:,7)),90,430))), colormap('jet');
title('POD 7')
subplot(5,2,9)
imshow(mat2gray(reshape(abs(Ur(:,8)),90,430))), colormap('jet');
title('POD 8')
subplot(5,2,10)
imshow(mat2gray(reshape(abs(Ur(:,9)),90,430))), colormap('jet');
title('POD 9')

Phi = conj(dftmtx(226)/226);
PSI = Phi(R,:);

%%

for i=1:30
b = Vr(:,i);       
cvx_begin;
    variable xL1(size(PSI,2)) complex; % sparse vector of coefficients 
    minimize( norm(xL1,1) );
    subject to
        norm(PSI*xL1 - b,2) <= 0.005;
cvx_end;
Vrec(:,i) = real(ifft(xL1));  
end

%% Visualize downsampled PODs

figure('Name','Principal Flow Modes downsampled')
subplot(9,2,1)
imshow(mat2gray(reshape(abs(Ur(:,1)),90,430))), colormap('jet');
title('POD Modes downsampled')
subplot(9,2,2)
title('Corellation to Time downsampled')
hold on
plot(Vr(:,1));
xlim([1,30])
subplot(9,2,3)
imshow(mat2gray(reshape(abs(Ur(:,2)),90,430))), colormap('jet');
subplot(9,2,4)
plot(Vr(:,2));
xlim([1,30])
subplot(9,2,5)
imshow(mat2gray(reshape(abs(Ur(:,3)),90,430))), colormap('jet');
subplot(9,2,6)
plot(Vr(:,3));
xlim([1,30])
subplot(9,2,7)
imshow(mat2gray(reshape(abs(Ur(:,4)),90,430))), colormap('jet');
subplot(9,2,8)
plot(Vr(:,4));
xlim([1,30])
subplot(9,2,9)
imshow(mat2gray(reshape(abs(Ur(:,5)),90,430))), colormap('jet');
subplot(9,2,10)
plot(Vr(:,5));
xlim([1,30])
subplot(9,2,11)
imshow(mat2gray(reshape(abs(Ur(:,6)),90,430))), colormap('jet');
subplot(9,2,12)
plot(Vr(:,6));
xlim([1,30])
subplot(9,2,13)
imshow(mat2gray(reshape(abs(Ur(:,7)),90,430))), colormap('jet');
subplot(9,2,14)
plot(Vr(:,7));
xlim([1,30])
subplot(9,2,15)
imshow(mat2gray(reshape(abs(Ur(:,8)),90,430))), colormap('jet');
subplot(9,2,16)
plot(Vr(:,8));
xlim([1,30])
subplot(9,2,17)
imshow(mat2gray(reshape(abs(Ur(:,9)),90,430))), colormap('jet');
subplot(9,2,18)
plot(Vr(:,9));
xlim([1,30])


figure('Name','Principal Flow Modes downsampled')
subplot(9,2,1)
imshow(mat2gray(reshape(abs(Ur(:,1)),90,430))), colormap('jet');
title('POD Modes downsampled')
subplot(9,2,2)
title('Corellation to Time reconstructed')
hold on
plot(Vrec(:,1));
xlim([1,226])
subplot(9,2,3)
imshow(mat2gray(reshape(abs(Ur(:,2)),90,430))), colormap('jet');
subplot(9,2,4)
plot(Vrec(:,2));
xlim([1,226])
subplot(9,2,5)
imshow(mat2gray(reshape(abs(Ur(:,3)),90,430))), colormap('jet');
subplot(9,2,6)
plot(Vrec(:,3));
xlim([1,226])
subplot(9,2,7)
imshow(mat2gray(reshape(abs(Ur(:,4)),90,430))), colormap('jet');
subplot(9,2,8)
plot(Vrec(:,4));
xlim([1,226])
subplot(9,2,9)
imshow(mat2gray(reshape(abs(Ur(:,5)),90,430))), colormap('jet');
subplot(9,2,10)
plot(Vrec(:,5));
xlim([1,226])
subplot(9,2,11)
imshow(mat2gray(reshape(abs(Ur(:,6)),90,430))), colormap('jet');
subplot(9,2,12)
plot(Vrec(:,6));
xlim([1,226])
subplot(9,2,13)
imshow(mat2gray(reshape(abs(Ur(:,7)),90,430))), colormap('jet');
subplot(9,2,14)
plot(Vrec(:,7));
xlim([1,226])
subplot(9,2,15)
imshow(mat2gray(reshape(abs(Ur(:,8)),90,430))), colormap('jet');
subplot(9,2,16)
plot(Vrec(:,8));
xlim([1,226])
subplot(9,2,17)
imshow(mat2gray(reshape(abs(Ur(:,9)),90,430))), colormap('jet');
subplot(9,2,18)
plot(Vrec(:,9));
xlim([1,226])

%% Reconstruct flow
Vrecf = Vrec(:,1:30);

filter = fft(Vrecf(:,1));
    filter = fftshift(filter);
    filter(1:80)=0;filter(150:end)=0;
    filter = ifftshift(filter);
    Vrecf(:,1) = real(ifft(filter));

for i = 2:30
    filter = fft(Vrecf(:,i));
    filter = fftshift(filter);
    filter(1:85)=0;filter(135:end)=0;
    filter = ifftshift(filter);
    Vrecf(:,i) = real(ifft(filter));
end


figure('Name','Principal Flow Modes downsampled')
subplot(9,2,1)
imshow(mat2gray(reshape(abs(Ur(:,1)),90,430))), colormap('jet');
title('POD Modes downsampled')
subplot(9,2,2)
title('Corellation to Time reconstructed')
hold on
plot(Vrecf(:,1));
xlim([1,226])
subplot(10,2,3)
imshow(mat2gray(reshape(abs(Ur(:,2)),90,430))), colormap('jet');
subplot(10,2,4)
plot(Vrecf(:,2));
xlim([1,226])
subplot(10,2,5)
imshow(mat2gray(reshape(abs(Ur(:,3)),90,430))), colormap('jet');
subplot(10,2,6)
plot(Vrecf(:,3));
xlim([1,226])
subplot(10,2,7)
imshow(mat2gray(reshape(abs(Ur(:,4)),90,430))), colormap('jet');
subplot(10,2,8)
plot(Vrecf(:,4));
xlim([1,226])
subplot(10,2,9)
imshow(mat2gray(reshape(abs(Ur(:,5)),90,430))), colormap('jet');
subplot(10,2,10)
plot(Vrecf(:,5));
xlim([1,226])
subplot(10,2,11)
imshow(mat2gray(reshape(abs(Ur(:,6)),90,430))), colormap('jet');
subplot(10,2,12)
plot(Vrecf(:,6));
xlim([1,226])
subplot(10,2,13)
imshow(mat2gray(reshape(abs(Ur(:,7)),90,430))), colormap('jet');
subplot(10,2,14)
plot(Vrecf(:,7));
xlim([1,226])
subplot(10,2,15)
imshow(mat2gray(reshape(abs(Ur(:,8)),90,430))), colormap('jet');
subplot(10,2,16)
plot(Vrecf(:,8));
xlim([1,226])
subplot(10,2,17)
imshow(mat2gray(reshape(abs(Ur(:,9)),90,430))), colormap('jet');
subplot(10,2,18)
plot(Vrecf(:,9));
xlim([1,226])
subplot(10,2,19)
imshow(mat2gray(reshape(abs(Ur(:,10)),90,430))), colormap('jet');
subplot(10,2,20)
plot(Vrecf(:,10));
xlim([1,226])

Rec = Ur(:,1:15)*Sr(1:15,1:15)*Vrecf(:,1:15)';

meanr = mean(Rec);
meano = mean(X);
factor = meano/meanr;

Rec = Rec*factor;

for i = 1:226
    img = reshape(Rec(:,i),90,430);
    gif2(:,:,:,i) = mat2gray(img); 
end
implay(gif2);