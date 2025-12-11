% 模拟和分析一个线性天线阵列在存在干扰信号情况下的波束形成性能
clc
clear

%%设置初始参数
% 定义信号的基本特性、天线阵列的结构、信号的环境条件
v=3e8;  % 光速，单位m/s
f=2.4e9;    % 信号频率，Hz
lamda=v/f;  % 波长
d=0.5*lamda;    % 阵元间距，设置为半波长
w=2*pi*f;   % 角频率
m=10;   % 阵元数
n=1000; % 快拍数
snr=0;  % 信噪比，单位dB
inr=20; % 干扰噪声比
thetas=90/180*pi;   % 期望信号方向，单位rad
thetai=40/180*pi;   % 干扰信号方向

%%生成导向向量
% 计算天线阵列在不同方向上的导向向量，atheta是一个矩阵，每一列对应一个方向的导向向量
theta=0:180;    % 方向角度范围
theta=theta/180*pi; % 转换为弧度
tao=d*cos(theta)/v; % 时间延迟
atheta = zeros(181,10); % 初始化阵列流形向量
for ii=1:181
    for jj=1:10
        atheta(ii,jj)=exp(1i*w*tao(ii)*(jj-1));
    end
end
atheta=atheta.';    % 转置


%期望和入射信号导向向量产生
a_theta_s=zeros(1,10);
a_theta_i=zeros(1,10);
for ii=1:10
    a_theta_s(ii)=exp(1j*w*tao(91)*(ii-1));
    a_theta_i(ii)=exp(1j*w*tao(41)*(ii-1));
end
a_theta_i=a_theta_i.';
a_theta_s=a_theta_s.';
    
%%入射信号及干扰信号的产生、
S=zeros(2,1000);    % 初始化信号矩阵
T=randi([0,1],[1,2*n]); % 生成随机比特
T_IQ=reshape(T,2,n).';  %重塑为IQ信号 
symbol=bi2de(T_IQ,'left-msb'); % 把比特转换为符号
Table=(1/sqrt(2))*[-1-j -1+j 1-j 1+j];  % 映射表
S(1,:)=Table(symbol+1); % 期望信号
T=randi([0,1],[1,2*n]);%2000的行向量
T_IQ=reshape(T,2,n).';%reshape成两个1000的行向量的转置
symbol=bi2de(T_IQ,'left-msb'); %把10变成2
Table=(1/sqrt(2))*[-1-j -1+j 1-j 1+j];  % 映射表
S(2,:)=Table(symbol+1); % 干扰信号
Noise=sqrt(1/2)*(randn(m,n)+j*randn(m,n));  % 高斯噪声
X=a_theta_s*10^(snr/10)*S(1,:)+a_theta_i*10^(inr/10)*S(2,:)+Noise;  % 接收信号
R0=X*X'/n;  % 实际接收信号协方差矩阵
R1=a_theta_s*a_theta_s'+a_theta_i*a_theta_i'*100+eye(m);    % 理论协方差矩阵

% 计算波束形成权重，使用最小均方误差（MMSE）波束形成方法计算权重
w0=inv(R0)*a_theta_s/(a_theta_s'*inv(R0)*a_theta_s);    % 基于R0（实际接收信号协方差矩阵）的波束形成权重
w1=inv(R1)*a_theta_s/(a_theta_s'*inv(R1)*a_theta_s);    % 基于R1（理论协方差矩阵）的波束形成权重


% 绘制波束图
H0=w0'*atheta;  % 基于w0的波束相应
H1=w1'*atheta;  % 基于w1的波束相应

H0=10*log(abs(H0)); % 转换为分贝
H1=10*log(abs(H1)); % 转换为分贝

angles=0:1:180; % 方向角度范围
subplot(2,1,1)
plot(angles,H0);
set(gca,'YLim',[-100,0])
title('H0')
subplot(2,1,2)
plot(angles,H1);
set(gca,'YLim',[-100,0])
title('H1')
