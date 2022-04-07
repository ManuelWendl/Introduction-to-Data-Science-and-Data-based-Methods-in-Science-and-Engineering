clear, clc, close all

I = imread('BW_Picture.png');

J = rgb2gray(I);

P = double(J);

[U,S,V] = svd(P,'econ');

rank_of_S = [10,30,50,70,120,200,300,408];

figure;
hold on

for i = 1:8
v_1 = ones(1,408);
v_1(rank_of_S(i):408)=0;
D_S = diag(v_1); 
S_1 = D_S * S;

Matrix_Pic_reduced = U*S_1*V';
Pic_reduced = mat2gray(Matrix_Pic_reduced);
subplot(4,2,i)
hold on
title(['Rank: ',num2str(rank_of_S(i))])
imshow(Pic_reduced);
end

figure('Name','Singular Values')
svalues = diag(S);
semilogy(svalues(1:200),'LineWidth',2)
hold on
title('First 200 Singular values')