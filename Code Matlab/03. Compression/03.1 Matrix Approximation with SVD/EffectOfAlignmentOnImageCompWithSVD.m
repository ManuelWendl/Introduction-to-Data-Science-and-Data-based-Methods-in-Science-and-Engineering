clear, clc, close all

M = zeros(201);
M0 = M;
M0(90:110,:) = 1;

figure('Name','Shifted Matrices')
subplot(2,6,1)
hold on
axis off
axis square
imagesc(M0),colormap(gray)

[U,S,V] = svd(M0,'econ');

svalues = diag(S);
subplot(2,6,7)
plot(svalues(1:30),'k-o','LineWidth',2)
    
x = 1:201;
for i = 1:5
    f = round(i*10*sin(2*pi/201*x));
    Mi = M;
    for ii = 1:201
        Mi(90+f(ii):110+f(ii),ii) = i;
    end
    [U,S,V] = svd(Mi,'econ');
    
    subplot(2,6,i+1)
    hold on
    axis off
    axis square
    imagesc(Mi)
    
    svalues = diag(S);
    
    subplot(2,6,i+7)
    plot(svalues(1:30),'k-o','LineWidth',2)
end

alimp = imread('Aligned_Picture.jpg');
unalimp = imread('Unaligned_Picture.jpg');

al = rgb2gray(alimp);
unal = rgb2gray(unalimp);

al = al(1:626,1:500);
unal = unal(200:826,150:650);

figure('Name','Alignment practical')
subplot(2,2,1)
imshow(al)

subplot(2,2,2)
imshow(unal)

al = double(al);
unal = double(unal);

[Ua,Sa,Va] = svd(al,'econ');
[Uu,Su,Vu] = svd(unal,'econ');

svalues_aligned = diag(Sa);
svalues_unaligned = diag(Su);

subplot(2,2,3)
semilogy(svalues_aligned(1:200),'k-o')
hold off

subplot(2,2,4)
semilogy(svalues_unaligned(1:200),'k-o')
hold off



