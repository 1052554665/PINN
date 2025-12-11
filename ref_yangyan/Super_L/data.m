function Y=data(theta,phi,B,K,T,SNR,f)
c=340;
lammda=c/f;
d=lammda/2;
M=size(B,1);
xi1=B(1:M/2,1);
yi1=B(1:M/2,2);
zi1=B(1:M/2,3);
for k1=1:length(theta)
  % a=exp(-j*2*pi*f*(xi*cosd(theta(k1))*cosd(phi(k1))+yi*sind(theta(k1))*cosd(phi(k1))+zi*sind(phi(k1)))/c);
  a1=exp(-j*2*pi*f*(xi1*sind(theta(k1))*cosd(phi(k1))+yi1*cosd(theta(k1))*cosd(phi(k1))+zi1*sind(phi(k1)))/c);
  % a1=exp(-j*2*pi*f*(xi1*sind(theta(k1))+yi1*sind(phi(k1))+zi1*sind(phi(k1)))/c);
  A1(:,k1)=a1;
end 
xi2=B(M/2+1:M,1);
yi2=B(M/2+1:M,2);
zi2=B(M/2+1:M,3);
for k1=1:length(theta)
  % a=exp(-j*2*pi*f*(xi*cosd(theta(k1))*cosd(phi(k1))+yi*sind(theta(k1))*cosd(phi(k1))+zi*sind(phi(k1)))/c);
  a2=exp(-j*2*pi*f*(xi2*sind(theta(k1))*cosd(phi(k1))+yi2*cosd(theta(k1))*cosd(phi(k1))+zi2*sind(phi(k1)))/c);
  A2(:,k1)=a2;
end 

xi=B(:,1);
yi=B(:,2);
zi=B(:,3);
for k1=1:length(theta)
  % a=exp(-j*2*pi*f*(xi*cosd(theta(k1))*cosd(phi(k1))+yi*sind(theta(k1))*cosd(phi(k1))+zi*sind(phi(k1)))/c);
  a=exp(-j*2*pi*f*(xi*sind(theta(k1))*cosd(phi(k1))+yi*cosd(theta(k1))*cosd(phi(k1))+zi*sind(phi(k1)))/c);
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
c0=1;
c1=0.3*exp(j*pi/3);
D=B(1:M/2,1)/d;
for k1=1:length(D)
   for k2=1:length(D)
       if(k1-k2==0)
           C(k1,k2)=c0;
       else
    C(k1,k2)=c1*exp(-j*(abs(D(k1)-D(k2))-1)/8)/abs(D(k1)-D(k2));
       end
end
end
Y1 = C*A1*X;  %生成麦克风数据
Y2=C*A2*X;
% Y=[Y1;Y2];
% sigma2 = 10^(-SNR/10) * norm(Y,'fro')^2 / (M * T);% 噪声方差
% E = sqrt(sigma2)/sqrt(2)*randn(M,T) + 1i*sqrt(sigma2)/sqrt(2)*randn(M,T); %噪声矢量
Y = A * X ;%麦克风数据
Y=awgn(Y,SNR,'measured');
