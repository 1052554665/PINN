%% 模拟水听器阵列的信号处理和频域常规波束形成
clear
close all
clc
 
% 度与弧度常数
deg2rad = pi/180; % 度转换成弧度
rad2deg = 180/pi; % 弧度转换成度
 
 
%% 水听器阵列参数设置
c = 1500;             % 水体声速，米/秒 m/s
N = 70;               % 阵元数
sqrtN=sqrt(N);        % 导向向量归一化常数
T=1;                 % 数据时长，秒sec
FS=24e2;              % 采样频率，赫兹Hz
LEN=T*FS;             % 数据长度，点数
t=(0:LEN-1)./FS;      % 数据时间轴，秒sec，1 x LEN 
fc=300;               % 阵列中心频率，赫兹Hz
lambda = c/fc;        % 波长，米m
lambda2 = lambda / 2; % 半波长，米m   d<(lambda/2)?
d = 0.27;             % 阵元间距，米m
ra=d*(0:N-1).';       % 阵元坐标，N x 1   
 
%% 目标方位角设置
thetas_deg=90;                 % 目标方位角，度deg
thetas_rad=thetas_deg*deg2rad; % 目标方位角，弧度rad
theta = 0:0.1:180;                  % 任意信号入射角度
 
%% 目标信号生成
f0=300;               % 声源频率，Hz
s=exp(1j*2*pi*f0.*t); % 目标复数信号，1 x LEN  "1j为复数，1为数字一，j为负数"  
as=exp(1j*2*pi*ra*f0*cos(thetas_rad)/c); % 目标导向向量，N x 1，导向向量=exp(-2*pi*f0*τ*1j*)
snr=40;               % 信噪比设置
rx=awgn(real(as*s), snr, 'measured');     % 含加性高斯噪声的水听器输出时间波形实数矩阵，Add white Gaussian noise to signal
x = rx.';                                 % 输出波形矩阵转置，便于二维数组FFT列运算，LEN x N，即接收到的信号
 
%% 频域常规波束形成
tic
xx = (as*s).';
for i = 1:length(theta)
    tao = cosd(theta(i))*ra/c;   %在角度theta(i)下得到的时延
    pt_sig = exp(1j*2*pi*f0*tao)' * xx.';  %波束，1*LEN
    s_sig(i,:) = pt_sig/N;    %size(s_sig) = i*LEN
end
toc
figure(1);
subplot(2,1,1);
plot(theta,s_sig(:,1),'b.-');title("频域常规波束形成");  
subplot(2,1,2);
sig_db = 20*log10(abs(s_sig(:,1).^2));
plot(theta,sig_db,'r.-');title("频域常规波束形成——分贝图");  