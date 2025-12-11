clear;
clc;
n1=2;
n2=3;
f=3500;
d=0.072;
fs=16000;
c=340;
B=nested(n1,n2,d);
M=size(B,1);
% theta=[-27.5 -21.5 -10.5 0.5 10 15.5];
% fe=[42.5 38.5 30.5 33 31.5 41];
theta=60;
fe=40;
K=length(theta);
T=1000;
SNR=10;
% Y=data(theta,fe,B,K,T,SNR,f);
x0=2;
y0=2;
z0=0; %阵列中心，即参考点位置
%%麦克风阵列房间设置
r=zeros(M,3);
r(:,1)=x0+B(:,1);
r(:,2)=y0+B(:,2);
r(:,3)=z0+B(:,3);
% r=B;
% s = [x0-l*cosd(fe)*sind(theta) y0-l*cosd(fe)*cosd(theta) z0+l*sind(fe)]            % Source position [x y z] (m)声源位置
C = [5 5 5];                % Room dimensions [x y z] (m)
beta = 0;                 % Reverberation time (s)混响时间
n = 4096;                   % Number of samples
mtype = 'omnidirectional';  % Type of microphone
order = 3;                 % -1 equals maximum reflection order!最大反射系数-1
dim = 3;                   % Room dimension房间尺寸
orientation = 0;            % Microphone orientation (rad)
hp_filter = 1;              % Enable high-pass filter
% h = rir_generator(c, fs, r, s, C, beta, n, mtype, order, dim, orientation, hp_filter);
%%
t=0:1/fs:(n-1)*1/fs;
zz=0.5:0.2:5;
for p=1:length(zz)
s=[2 2 zz(p)];
h = rir_generator(c, fs, r, s, C, beta, n, mtype, order, dim, orientation, hp_filter);
S=1000*cos(2*pi*f*t);
% S2=10*sin(2*pi*800*t); 
% S=S1+S2;
a=zeros(M,n);
for i=1:M
    a(i,:)=fftfilt(h(i,:),S); 
    Y(i,:)=fft(a(i,:),n);
    i=i+1;
end
Ys=Y(:,round(f*n/fs)+1);  
% 
[J,B_vir,vir_B,vir_Y]=difference(B,Ys,T,f,d);
vir_R=vir_Y*vir_Y'/T;
 [U,D,V] = svd(vir_R);  %求R的奇异值分解
 Un = U(:,K+1:end); %噪声子空间
xi=x0+vir_B(:,1);
yi=y0+vir_B(:,2);
zi=z0+zeros(size(vir_B,1),1);
% ang1=0:1:80;  %扫面方位角
% ang2=20:1:60;  %扫描俯仰角
% for k1=1:length(ang1)
%    for k2=1:length(ang2)
%     %a1=exp(-j*2*pi*f*(0:size(Rx1,1)-1)'*d*sind(ang1(k1))*cosd(ang2(k2))/c);
%     %a2=exp(-j*2*pi*f*(0:size(Rx2,1)-1)'*d*cosd(ang1(k1))*cosd(ang2(k2))/c);
%     a=exp(-j*2*pi*f*(xi*sind(ang1(k1))*cosd(ang2(k2))+yi*cosd(ang1(k1))*cosd(ang2(k2))+zi*sind(ang2(k2)))/c);
%     %w1=[a1;a2];
% %     SP(k2,k1)=abs(1/(a'*vir_R*a));  %波束形成
%       SP(k2,k1)=abs(1/(a'*Un*Un'*a));%   music算法
%       A(:,(k1-1)*length(ang2)+k2)=a;
%    end
% end
%%
x2 = x0;
y2 = y0;
z2 = z0;
%-------扫描范围------%
step_x = 0.01;  % 步长设置为0.1
step_y = 0.01;
x = (0:step_x:4);  % 扫描范围 
z = zz(p);
y = (0:step_y:4);
for k1=1:length(y)
        for k2=1:length(x)
       Ri = sqrt((x(k2)-xi).^2+(y(k1)-yi).^2+(z-zi).^2);
        Ri2 = sqrt((x(k2)-x2).^2+(y(k1)-y2).^2+(z-z2).^2);
        % 该扫描点到各阵元的聚焦距离矢量   一行一行的扫描，把数值再一行一行的赋给Pcbf
        Rn = Ri-Ri2;
         Rn2=Ri2./Ri;
        % 扫描点到各阵元与参考阵元的程差矢量
        b =Rn2.*exp(-j*2*pi*f*Rn/c); % 声压聚焦方向矢量
        A(:,(k1-1)*length(x)+k2)=b;
%         SP(k1,k2) = abs(b'*Un*Un'*b); %music
          DAS_result(k1,k2) = abs(b'*vir_R*b); % cbf
        end
end
for k1 = 1:length(y);
   pp1(k1) = max(DAS_result(k1,:)); % Pcbf 的第k1行的最大元素的值
end
SP=DAS_result/max(pp1);
SP=SP';
[PSF,intx]=max(SP);
PSF=20*log10(PSF);
PSF=PSF-max(PSF);
indx1=find(PSF>-3,1,"first");
indx2=find(PSF>-3,1,"last");
bandwidth(p)=abs(x(indx1)-x(indx2)) ;       %3dB带宽
[num,loc] = findpeaks(PSF);
[a_num,a_loc] = max(num);                         %在全部峰值里面找出最大的一个a_num，包含其位置a_loc
location_in_x_1 = loc(a_loc);                      %最大的峰值对应的位置
num(a_loc) = min(num);                                   %在找出的全部峰值数组中，将最大的峰值赋值为0
num_del_max = num;
[b_num,b_loc] = max(num);                         %找剩下的峰值中的最大值
DR(p)=a_num-b_num ;                %动态范围
end
%%  
figure(1)
plot(zz,bandwidth);
figure(2)
plot(zz,DR);