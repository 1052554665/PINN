%{
使用LCMV波束形成器将阵列响应的零点指向干扰源的方向。
该阵列是一个 10 元素均匀线性阵列 （ULA）。
默认情况下，ULA 元件是由相控天线创建的各向同性天线。
设置天线元件的频率范围，使载波频率在工作范围内。载波频率为 1 GHz
%}
fc = 1e9;   % 载波频率 fc=1 GHz
lambda = physconst('LightSpeed')/fc;
array = phased.ULA('NumElements',10,'ElementSpacing',lambda/2);
array.Element.FrequencyRange = [8e8 1.2e9]; % 设置阵元工作频率范围 [0.8, 1.2] GHz，保证 fc 在范围内
% 使用简单的矩形脉冲模拟测试信号
t = linspace(0,0.3,300)';   % 定义时间向量 t（0–0.3 秒，300 个点）
testsig = zeros(size(t));
testsig(201:205) = 1;   % 构造一个 矩形脉冲（在第 201–205 个采样点为 1，其余为 0）
% 假设矩形脉冲从30°方位角和0°仰角入射到ULA上。
% 使用 ULA 系统对象的 collectPlaneWave 函数模拟从入射角接收脉冲波形。
angle_of_arrival = [30;0];
x = collectPlaneWave(array,testsig,angle_of_arrival,fc);
% 信号 x 是一个有十列的矩阵。每列表示其中一个阵元的接收信号。
convbeamformer = phased.PhaseShiftBeamformer('SensorArray',array,...
    'OperatingFrequency',1e9,'Direction',angle_of_arrival,...
    'WeightsOutputPort',true);
% 将复值高斯白噪声添加到信号 x。设置可重现结果的默认随机数流。
rng default
npower = 0.5;
x = x + sqrt(npower/2)*(randn(size(x)) + 1i*randn(size(x)));    % 为了更真实，给接收信号 x 加上复高斯噪声
% 创建一个 10W 干扰源。指定阻塞干扰器的有效辐射功率为 10 W。
% 来自阻塞干扰器的干扰信号从120°方位角和0°仰角入射到ULA上。
% 使用 ULA 系统对象的 collectPlaneWave 函数模拟干扰器信号的接收。
jamsig = sqrt(10)*randn(300,1);
jammer_angle = [120;0];
jamsig = collectPlaneWave(array,jamsig,jammer_angle,fc);
% 添加复值高斯白噪声以模拟与干扰信号不直接相关的噪声贡献。
% 同样，为可重现的结果设置默认随机数流。
% 该噪声功率比干扰器功率低 0 dB。使用传统的波束成形器对信号进行波束成形。
noisePwr = 1e-5;
rng(2008);
noise = sqrt(noisePwr/2)*...
    (randn(size(jamsig)) + 1j*randn(size(jamsig)));
jamsig = jamsig + noise;
rxsig = x + jamsig; % 最终接收信号 = 目标信号 + 干扰信号
[yout, w] = convbeamformer(rxsig);
% 使用相同的 ULA 阵列实现自适应 LCMV 波束形成器。
% 使用无目标数据 jamsig 作为训练数据。
% 输出波束形成信号和波束形成器权重。
steeringvector = phased.SteeringVector('SensorArray',array,...
    'PropagationSpeed',physconst('LightSpeed'));
LCMVbeamformer = phased.LCMVBeamformer('DesiredResponse',1,...
    'TrainingInputPort',true,'WeightsOutputPort',true);
LCMVbeamformer.Constraint = steeringvector(fc,angle_of_arrival);
LCMVbeamformer.DesiredResponse = 1;
[yLCMV,wLCMV] = LCMVbeamformer(rxsig,jamsig);
% 绘制常规波束形成器输出和自适应波束形成器输出
subplot(211)
plot(t,abs(yout))
axis tight
title('Conventional Beamformer')
ylabel('Magnitude')
subplot(212)
plot(t,abs(yLCMV))
axis tight
title('LCMV (Adaptive) Beamformer')
xlabel('Seconds')
ylabel('Magnitude')
% 使用常规和 LCMV 权重，绘制每个波束形成器的响应
subplot(211)
pattern(array,fc,[-180:180],0,'PropagationSpeed',physconst('LightSpeed'),...
    'CoordinateSystem','rectangular','Type','powerdb','Normalize',true,...
    'Weights',w)
title('Array Response with Conventional Beamforming Weights');
subplot(212)
pattern(array,fc,[-180:180],0,'PropagationSpeed',physconst('LightSpeed'),...
    'CoordinateSystem', 'rectangular', 'Type','powerdb','Normalize',true,...
    'Weights',wLCMV)
title('Array Response with LCMV Beamforming Weights');
