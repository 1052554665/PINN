function Y=data(theta,phi,B,K,T,SNR,f)
c=340;
lammda=c/f;
d=lammda/2;
M=size(B,1);
xi=B(:,1);
yi=B(:,2);
zi=B(:,3);
for k1=1:length(theta)
  a=exp(-j*2*pi*f*(xi*cosd(theta(k1))*cosd(phi(k1))+yi*sind(theta(k1))*cosd(phi(k1))+zi*sind(phi(k1)))/c);
  A(:,k1)=a;
end 
% true signal
% random signals
% amp = ones(1,K)';
for i=1:K
if T == 1
    X = exp(1i*2*pi*unifrnd(0,1,K,1));
else
    X(i,:) = (randn(1,T) + 1i * randn(1,T)) / sqrt(2);
%     X = diag(amp) * X1;
end
end
% observed signal
%%
Y = A * X;  %生成麦克风数据
sigma2 = 10^(-SNR/10) * norm(Y,'fro')^2 / (M * T);% 噪声方差
% error
E = sqrt(sigma2)/sqrt(2)*randn(M,T) + 1i*sqrt(sigma2)/sqrt(2)*randn(M,T); %噪声矢量
Y = A * X + E;%麦克风数据
% Meann=0;varn=1;%%噪声均值和方差
% t=[0:1:T-1]/200;
% rvar1=sqrt(varn)*10^(SNR/20);%信号功率
% s=rvar1*exp(j*2*pi*f*t);
% e=sqrt(varn/2)*(randn(M,T)+j*randn(M,T));
% Y=A*s+e;