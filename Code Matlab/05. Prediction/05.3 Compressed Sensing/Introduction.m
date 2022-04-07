clc, clear, close all
%% Nyquist-Shannon

dt = .001;
t = 0:dt:1;

f = sin(2*pi*10*t);

smplrate = 1:20:1001;
samples = f(smplrate);

figure('Name','Nyquist and Shannon')
subplot(2,1,1)
title('Sufficiently sampled signal')
hold on
plot(f)
plot(smplrate,samples,'xr')
xlim([1,1000])
legend('Original and reconstructed signal','Samples')

smplrate = 1:90:1001;
samples = f(smplrate);

subplot(2,1,2)
title('Undersampled Signal')
hold on
plot(f)
plot(smplrate,samples,'xr')
plot(-sin(2*pi*1.11*t))
xlim([1,1000])
legend('Original signal','Samples','Reconstructed Signal')

%% Compressed Sensing
dt = .01;
t = 0:dt:1;

f = sin(2*pi*2*t)+cos(2*pi*5*t);

smplrate = round(rand(20,1)*101);
samples = f(smplrate);

figure('Name','Randomly Sampled')
title('Randomly Sampled')
hold on
plot(f)
plot(smplrate,samples,'xr')
xlim([1,100])
legend('Original signal','Samples')

O = conj(dftmtx(101)./length(t));
S = zeros(length(samples),101);

for i = 1:length(samples)
    S(i,smplrate(i)) = 1;
end


figure('Name','Compressed Sensing')
subplot('Position',[0.05 0.5 0.025 0.4])
imagesc(samples');
axis('off')
title('Sampled Data')
subplot('Position',[0.45 0.6 0.3 0.3])
imagesc(real(O))
axis('off')
title('\Phi')
set(gca,'FontSize',15)
subplot('Position',[0.125 0.8 0.3 0.1])
imagesc(S)
axis('off')
title('R')
set(gca,'FontSize',12)

cvx_begin;
    variable xL1(101) complex; % sparse vector of coefficients 
    minimize( 10*norm(xL1,1) );
    subject to
        S*O*xL1 == samples';
cvx_end;

subplot('Position',[0.8 0.1 0.05 0.8])
imagesc(abs(xL1)), colormap('gray')
axis('off')
title('Sparse L_1 x')
subplot('Position',[0.9 0.1 0.05 0.8])
imagesc(abs(pinv(S*O)*samples'))
axis('off')
title('Not sparse L_2 x')
subplot('Position',[0.3 0.3 0.3 0.1])
imagesc(real(S*O))
axis('off')
title('\Psi')
set(gca,'FontSize',15)
