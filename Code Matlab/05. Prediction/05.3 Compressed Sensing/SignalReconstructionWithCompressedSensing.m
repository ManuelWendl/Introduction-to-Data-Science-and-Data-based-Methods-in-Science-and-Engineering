clc, clear, close all
%% Signal

dt = 0.001;
t = (0:dt:1);

f = sin(2*pi*10*t)+sin(2*pi*70*t) +sin(2*pi*160*t) + cos(2*pi*120*t);

figure('Name','Signal and Power spectrum');
subplot(2,1,1)
plot(t,f,'LineWidth',2);
title('Original signal')

%% Fast Fourier Transformation 

% FC are the Fourier Coeficients
FC = fft(f,length(t));

% Power of different frequencies
P = abs(FC)/length(t);

freq = 1/(dt*length(t))*(1:length(t));

subplot(2,1,2)
hold on
plot(freq,P,'LineWidth',2);
xlim([1, 500]);
xlabel('frequencies in Hz')
ylabel('Power of frequencies')
title('Powerspectrum')

%% Downsample data

s = 55;
red = round(rand(s,1)*1001);
R = f(red);

figure('Name','Downsampled Data')
hold on
plot(t,f,'LineWidth',2)
plot(red/1001,R,'xr','LineWidth',4)
legend('Original signal','Random samples')
title('Measurement samples')


%% Create DFT and PSI determine x

DCT = dftmtx(1001);
PSI = conj(DCT(red,:)./length(t));

cvx_begin;
    variable xL1(1001) complex; % sparse vector of coefficients 
    minimize( 10*norm(xL1,1) );
    subject to
        PSI*xL1 == R';
cvx_end;

xL2 = pinv(PSI)*R';

%% Plot 

frecL1 = ifft(xL1);
frecL2 = ifft(xL2);

Pl1 = abs(xL1)/length(t);
Pl2 = abs(xL2)/length(t);

c = [0 0.4470 0.7410];

figure()
subplot(2,2,1)
title('Compressed Sensing with L1 norm')
hold on
plot(t,f,'LineWidth',2);
plot(t,real(frecL1),'r','LineWidth',2);
plot(red/1001,R,'xk','LineWidth',2)
legend('Original signal','Reconstructed signal','Measurement samples')
subplot(2,2,3)
plot(freq,Pl1,'LineWidth',2);
xlabel('frequencies in Hz')
ylabel('Power of frequencies')
xlim([1, 500])
title('Powerspectrum')

subplot(2,2,2)
title('Least square approximation')
hold on
plot(t,f,'LineWidth',2);
plot(t,real(frecL2),'r','LineWidth',2);
plot(red/1001,R,'xk','LineWidth',2)
legend('Original signal','Reconstructed signal','Measurement samples')
subplot(2,2,4)
plot(freq,Pl2,'LineWidth',2);
xlabel('frequencies in Hz')
ylabel('Power of frequencies')
xlim([1, 500])
title('Powerspectrum')

%% Optimization of sparse solution with least squares

ind = Pl1 > 0.1;

xoptL1 = pinv(PSI*diag(ind))*R';

figure('Name','Optimized L1 with least square')
subplot(2,1,1)
title('Least square optimized sparse L_1 norm solution')
hold on
plot(f,'LineWidth',2)
plot(real(ifft(xoptL1)),'r','LineWidth',2)
xlim([1, 1000])
legend('Original signal','Reconstructed signal')
subplot(2,1,2)
plot(abs(xoptL1)/length(t),'LineWidth',2)
xlabel('frequencies in Hz')
ylabel('Power of frequencies')
xlim([1, 500])
title('Powerspectrum')
