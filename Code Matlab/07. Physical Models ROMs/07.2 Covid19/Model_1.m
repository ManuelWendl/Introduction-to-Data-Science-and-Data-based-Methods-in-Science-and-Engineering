%**************************************************************************
%   MA9803 Modelling and Simulation of ODEs
%   Manuel R. Wendl
%   
%   Covid19 Simulation Model_1
%   ******************
%   Modelled Groups:
%   S:  Susceptibel Population (Unprotected Population)
%   I:  Infected Population (Currently carrying the virus)
%   P:  Protected Population (Recovered from Virus w.o. vaccination)
%   
%   Modelling Parameters:
%   k:  Daily Rate of Infection for one Infected
%   tr: Average recovery time 
%**************************************************************************
clc, clear, close all
%% Definition of Variables
t0 = 0; tn = 365; dt = 1; S0 = 980; I0 = 20; P0 = 0; k = 0.5; tr = 17; N = S0+I0+P0;
%% Symbolic Right Handside Function
syms fs(S,I,P,t) fi(S,I,P,t) fp(S,I,P,t)
fs(S,I,P,t) = -k*I/N*S;
fi(S,I,P,t) = k*I/N*S - 1/tr*I;
fp(S,I,P,t) = 1/tr*I;

%% Computation and Plotting
[t,S,I,P] = ForwardEuler(fs,fi,fp,t0,tn,dt,S0,I0,P0); % Executing ForwardEuler
%%
figure(1)
hold on
area(t,[I',S',P']);
legend('Infected Population','Susceptible Population','Protected Population')
axis([0,200,0,1000]); newcolors = [0.8500 0.3250 0.0980; 0 0.4470 0.7410; 0.4 0.4 0.4]; colororder(newcolors)
xlabel('d')
%% Forward Oiler Function
function [t,S,I,P] = ForwardEuler(fs,fi,fp,t0,tn,dt,S0,I0,P0)
    t = t0:dt:tn;
    S = zeros(size(t)); S(1) = S0;
    I = zeros(size(t)); I(1) = I0;
    P = zeros(size(t)); P(1) = P0;
    for i = 1:length(t)-1
        S(i+1) = S(i) + dt*fs(S(i),I(i),P(i),t(i));
        I(i+1) = I(i) + dt*fi(S(i),I(i),P(i),t(i));
        P(i+1) = P(i) + dt*fp(S(i),I(i),P(i),t(i));
    end
end