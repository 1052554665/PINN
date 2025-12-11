clear;
clc;
n1=2;
n2=3;
f=1000;
fs=16000;
c=340;
B=cooprime(n1,n2,f);
M=size(B,1);
theta=[30 40 60];
fe=[60 40 30];
K=length(theta);
T=1000;
SNR=20;
 Y=data(theta,fe,B,K,T,SNR,f);
% x0=30;
% y0=30;
% z0=5; %阵列中心，即参考点位置
% l=30;
% %%麦克风阵列房间设置
% % r=zeros(M,3);
% % r(:,1)=[x0+array_structure1,x0*ones(1,(array_num))];
% % r(:,2)=[y0*ones(1,(array_num)),y0+array_structure1];
% % r(:,3)=z0*ones(M,1);
% r=B;
% s = [x0-l*cosd(fe)*sind(theta) y0-l*cosd(fe)*cosd(theta) z0+l*sind(fe)] ;            % Source position [x y z] (m)声源位置
% C = [23 23 23];                % Room dimensions [x y z] (m)
% beta = 0;                 % Reverberation time (s)混响时间
% n = 4096;                   % Number of samples
% mtype = 'omnidirectional';  % Type of microphone
% order = 3;                 % -1 equals maximum reflection order!最大反射系数-1
% dim = 3;                    % Room dimension房间尺寸
% orientation = 0;            % Microphone orientation (rad)
% hp_filter = 1;              % Enable high-pass filter
% h = rir_generator(c, fs, r, s, C, beta, n, mtype, order, dim, orientation, hp_filter);
% %%
%  t=0:1/fs:(n-1)*1/fs;
% S1=10*cos(2*pi*f*t); %声音信号
% S2=10*sin(2*pi*800*t); 
% S=S1+S2;
% a=zeros(M,n);
% for i=1:M
%     a(i,:)=fftfilt(h(i,:),S); 
%     Y(i,:)=fft(a(i,:),n);
%     i=i+1;
% end
% Y=Y(:,round(f*n/fs)+1);  % 选择600Hz频率点：(n-1)*fs/N=600Hz,n为600Hz对应的采样点,即n=f*N/fs+1
[J,B_vir,vir_B,vir_Y]=difference(B,Y,T,f);
% vir_R=vir_Y*vir_Y';
% [U,D,V] = svd(vir_R);  %求R的奇异值分解
% Un = U(:,K+1:end); %噪声子空间
xi=vir_B(:,1);
yi=vir_B(:,2);
zi=zeros(size(vir_B,1),1);
ang1=20:1:90;  %扫面方位角
ang2=30:1:85;  %扫描俯仰角
for k1=1:length(ang1)
   for k2=1:length(ang2)
    %a1=exp(-j*2*pi*f*(0:size(Rx1,1)-1)'*d*sind(ang1(k1))*cosd(ang2(k2))/c);
    %a2=exp(-j*2*pi*f*(0:size(Rx2,1)-1)'*d*cosd(ang1(k1))*cosd(ang2(k2))/c);
    a=exp(-j*2*pi*f*(xi*sind(ang1(k1))*cosd(ang2(k2))+yi*cosd(ang1(k1))*cosd(ang2(k2))+zi*sind(ang2(k2)))/c);
    %w1=[a1;a2];
%     SP(k2,k1)=abs(1/(a'*vir_R*a));  %波束形成
%       SP(k2,k1)=abs(1/(a'*Un*Un'*a));%   music算法
      A(:,(k1-1)*length(ang2)+k2)=a;
   end
end
% ys=[real(vir_Y);imag(vir_Y)];
% As=[real(A.'),imag(A.')].';
lambda0 = 1e-2; 
learn_Lambda0 = 1; 
Max_iter=5000;
[Weight0,count]=IF_SBL(A,vir_Y,Max_iter);
Weight=reshape(Weight0.x,[length(ang2),length(ang1)]);
% [Weight0,gamma_est0,gamma_used0,count0] = MSBL(A,vir_Y, lambda0, learn_Lambda0);
% Gamma=diag(gamma_est0);
% Weight=reshape(Weight0,[length(ang2),length(ang1)]);
SP=abs(Weight);
figure(1)
plot(B(:,1),B(:,2),'r*');
title('实际麦克风阵列');
figure(2)
plot(B_vir(:,1),B_vir(:,2),'r*');
title('虚拟麦克风阵列');
figure(3)
plot(xi,yi,'r*');
title('连续部分虚拟麦克风阵列');
figure(4)
surf(ang1,ang2,SP);
title('三维声源图');
figure(5)
pcolor(ang1,ang2,SP);
title('二维声源图');