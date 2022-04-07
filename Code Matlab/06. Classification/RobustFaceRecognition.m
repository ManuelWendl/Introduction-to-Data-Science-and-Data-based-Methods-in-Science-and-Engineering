clc, clear, close all
%% Import Data
m = 192;
n = 168;
for i=1:37
   for j=1:60 
        if i<10
            url = ['YaleFaces/yaleB0',num2str(i),'/','Picture_',num2str(j),'.pgm'];
            face = importdata(url);
            allfaces(:,(i-1)*60+j) = reshape(face,m*n,1);
        else
            url = ['YaleFaces/yaleB',num2str(i),'/','Picture_',num2str(j),'.pgm'];
            face = importdata(url);
            allfaces(:,(i-1)*60+j) = reshape(face,m*n,1);
        end
   end
end

%% Split Trainings and Testdata
% Two pictures of every person
count = 1;
for i = 0:36
    ind = randi(37,1,2);
    test(:,[count, count+1]) = double(allfaces(:,i*58+ind));
    allfaces(:,i*58+ind) = [];
    count = count+2;
end

X = double(allfaces);

%% Basis Matrix PSI

o = 15;
p = 13;

for i=1:size(X,2)
    img = reshape(X(:,i),m,n);
    PSI(:,i) = reshape(imresize(img,[o, p], 'lanczos3'),o*p,1);
end


%% Resize testfaces

for i=1:size(test,2)
    img = reshape(test(:,i),m,n);
    b(:,i) = double(reshape(imresize(img,[o, p], 'lanczos3'),o*p,1));
end

%% Recognition comparison L1 and L2

num = 22;

% L2:
xL2 = pinv(PSI)*b(:,num);
% L1:
cvx_begin;
    variable xL1(size(PSI,2)); % sparse vector of coefficients 
    minimize( norm(xL1,1) );
    subject to
        norm(PSI*xL1 - b(:,num),2) <= 0.01;
cvx_end;

figure('Name','Prediction')
subplot(4,2,1)
hold on
imshow(reshape(mat2gray(test(:,num)),m,n));
title('Test Picture')
subplot(4,2,2)
hold on
imshow(reshape(mat2gray(b(:,num)),o,p));
title('Downsampled Picture')
subplot(4,2,3)
hold on
title('x least square')
plot(xL2)
xlim([1,2146])
subplot(4,2,4)
hold on
title('x sparse (L_1 norm)')
plot(xL1)
xlim([1,2146])
subplot(4,2,5)
hold on
title('Reconstruction least square')
imshow(reshape(mat2gray(X*xL2),m,n))
subplot(4,2,6)
hold on
title('Reconstruction L_1')
imshow(reshape(mat2gray(X*xL1),m,n))
subplot(4,2,7)
hold on
title('Error least square')
imshow(reshape(mat2gray(test(:,num)-X*xL2),m,n))
subplot(4,2,8)
hold on
title('Sparse error L_1')
imshow(reshape(mat2gray(test(:,num)-X*xL1),m,n))


%% Identification

num = 35;

cvx_begin;
    variable xL1(size(PSI,2)); % sparse vector of coefficients 
    minimize( norm(xL1,1) );
    subject to
        norm(PSI*xL1 - b(:,num),2) <= 0.01;
cvx_end;

for i = 1:37
    Ident(i) = sum(abs(xL1((i-1)*58+1:i*58)))/sum(abs(xL1));
end

figure('Name','Identifikation')
subplot(2,3,1)
hold on
imshow(reshape(mat2gray(test(:,num)),m,n));
title(['Test Picture of Person: ', num2str(round(num/2))])
subplot(2,3,2)
hold on
imshow(reshape(mat2gray(b(:,num)),o,p));
title('Downsampled Picture')
subplot(2,3,3)
hold on
title('x')
plot(xL1)
subplot(2,3,4)
hold on
title('Reconstruction L_1')
imshow(reshape(mat2gray(X*xL1),m,n))
subplot(2,3,5)
hold on
title('Sparse error')
imshow(reshape(mat2gray(test(:,num)-X*xL1),m,n))
subplot(2,3,6)
hold on
bar(Ident)
[~, ind] = max(Ident);
xlabel(['Recognised Peron: ',num2str(ind)]);

%% Performance L1
% count = 0;
% 
% for i = 1:size(b,2)
%     num = i;
%     cvx_begin;
%         variable xL1(size(PSI,2)); % sparse vector of coefficients 
%         minimize( norm(xL1,1) );
%         subject to
%             norm(PSI*xL1 - b(:,num),2) <= 0.01;
%     cvx_end;
% 
%     for ii = 1:37
%         Ident(ii) = sum(abs(xL1((ii-1)*58+1:ii*58)))/sum(abs(xL1));
%     end
%     [~, ind] = max(Ident);
%     if round(i/2) == ind
%         count = count+1;
%     end
% end
% 
% sprintf('Performance: %d',count/74)

%% Corrupt image Identification

num = 15;

face = reshape(test(:,num),m,n);
glasses = double(rgb2gray(importdata('glasses.jpg')));
glasses = imresize(glasses,[NaN, 168]);
glasses = glasses/255;
glasses(glasses>0.8)=1;

face(30:89,:) = face(30:89,:).*glasses;

f = double(reshape(imresize(face,[o, p],'lanczos3'),o*p,1));

cvx_begin;
    variable xL1(size(PSI,2)); % sparse vector of coefficients 
    minimize(norm(PSI*xL1 - f,2)+40*norm(xL1,1) );
%     subject to
%         norm(PSI*xL1 - f,2) <= .01;
cvx_end;

for i = 1:37
    Ident(i) = sum(abs(xL1((i-1)*58+1:i*58)))/sum(abs(xL1));
end

figure('Name','Identifikation')
subplot(2,3,1)
hold on
imshow(mat2gray(face));
title(['Test Picture of Person: ', num2str(round(num/2))])
subplot(2,3,2)
hold on
imshow(reshape(mat2gray(f),o,p));
title('Downsampled Picture')
subplot(2,3,3)
hold on
title('x')
plot(xL1)
subplot(2,3,4)
hold on
title('Reconstruction L_1')
imshow(reshape(mat2gray(X*xL1),m,n))
subplot(2,3,5)
hold on
title('Sparse error')
imshow(mat2gray(face-reshape(X*xL1,m,n)));
subplot(2,3,6)
hold on
bar(Ident)
[maxi, ind] = max(Ident);
xlabel(['Recognised Peron: ',num2str(ind)]);

%% Self developed iterative solution

xopt = pinv(PSI)*f;
i = 0;
while sum(abs(xopt))> 2
    i=i+1;
    [x_s, ind] = sort(abs(xopt));
    x_s(1:i*2)=0;
    PSI_s = PSI(:,ind);
    PSI_red = PSI_s(:,i*2+1:end);
    xpart = pinv(PSI_red)*f;
    x_s(i*2+1:end) = xpart;
    xopt(ind) = x_s;
    if norm(abs(f-PSI*xopt))/(o*p) > 5
        break;
    end
end

for i = 1:37
    Ident(i) = sum(abs(xopt((i-1)*58+1:i*58)))/sum(abs(xL1));
end

figure('Name','Identification')
subplot(2,3,1)
hold on
imshow(mat2gray(face));
title(['Test Picture of Person: ', num2str(round(num/2))])
subplot(2,3,2)
hold on
imshow(reshape(mat2gray(f),o,p));
title('Downsampled Picture')
subplot(2,3,3)
hold on
title('x (optimised Least Square)')
plot(xopt)
subplot(2,3,4)
hold on
title('Reconstruction')
imshow(reshape(mat2gray(X*xopt),m,n))
subplot(2,3,5)
hold on
title('Error')
imshow(mat2gray(face-reshape(X*xopt,m,n)));
subplot(2,3,6)
hold on
bar(Ident)
[maxi, ind] = max(Ident);
xlabel(['Recognised Peron: ',num2str(ind)]);


%% Visualisatio

figure('Name','Visualisation');
subplot('Position',[0.3 0.05 0.16 0.38])
imshow(reshape(mat2gray(face),m,n));
title('New Testimage')
subplot('Position',[0.1 0.05 0.16 0.38])
imshow(mat2gray(imresize(face,[o, p])));
title('Downsized Image')
subplot('Position',[0.1 0.45 0.025 0.48])
imagesc(mat2gray(reshape(imresize(face,[o, p]),o*p,1))),colormap('gray');
axis('off')
title('Reshaped and downsized')
subplot('Position',[0.2 0.6 0.6 0.48])
imshow(mat2gray(PSI));
title('\Psi')
set(gca,'fontsize', 20);
subplot('Position',[0.9 0.05 0.025 0.9])
imagesc(xL1);
axis('off')
title('Sparse x')
