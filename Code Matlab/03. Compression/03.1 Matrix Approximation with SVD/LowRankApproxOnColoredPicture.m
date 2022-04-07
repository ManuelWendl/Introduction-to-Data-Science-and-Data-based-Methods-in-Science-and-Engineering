clear, clc, close all

I = imread('Color_Picture.png');

I_R = double(I(:,:,1));

I_G = double(I(:,:,2));

I_B = double(I(:,:,3));

[U_R,S_Ra,V_R] = svd(I_R);
[U_G,S_Ga,V_G] = svd(I_G);
[U_B,S_Ba,V_B] = svd(I_B);

rank_of_S = [10,30,50,70,120,200,300,563];

figure;
for i=1:length(rank_of_S)
v_1 = ones(1,563);
v_1(rank_of_S(i):563)=0;
D_S = diag(v_1); 
S_R = D_S * S_Ra;
S_G = D_S * S_Ga;
S_B = D_S * S_Ba;

I_R_red = U_R*S_R*V_R';
I_G_red = U_G*S_G*V_G';
I_B_red = U_B*S_B*V_B';

I_red = I_R_red;
I_red(:,:,2) = I_G_red;
I_red(:,:,3) = I_B_red;

subplot(4,2,i);
hold on
title(['Rank: ',num2str(rank_of_S(i))])
imshow(uint8(I_red))
end

figure;
svaluesR = diag(S_R);
svaluesG = diag(S_G);
svaluesB = diag(S_B);
semilogy(svaluesR(1:200),'r','LineWidth',2)
hold on
semilogy(svaluesG(1:200),'g','LineWidth',2)
semilogy(svaluesB(1:200),'b','LineWidth',2)
title('First 200 singular values')
legend('singular values R','singular values G','singular values B');
hold off
