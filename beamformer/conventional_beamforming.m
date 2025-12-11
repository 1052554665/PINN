% 创建一个10元素均匀线性阵列（ULA）并进行波束赋形。
% 假设载波频率为1G赫兹，将阵列元素间距设置为载波波长的一半。
fc = 1e9;
lambda = physconst('LightSpeed')/fc;
array = phased.ULA('NumElements',10,'ElementSpacing',lambda/2);
% 默认情况下，ULA元件是由phased.IsotropicAntennaElement系统对象创建的各向同性天线。
% 设置天线元件的频率范围，使载波频率处于工作范围内。
array.Element.FrequencyRange = [8e8 1.2e9];
% 模拟一个测试信号。使用一个简单的矩形脉冲。
t = linspace(0,0.3,300)';
testsig = zeros(size(t));
testsig(201:205) = 1;
% 假设矩形脉冲从方位角30°、仰角0°入射到均匀线阵（ULA）。
% 使用ULA系统对象的collectPlaneWave函数来模拟从指定角度接收脉冲波形。
angle_of_arrival = [30;0];
x = collectPlaneWave(array, testsig, angle_of_arrival, fc);
% 信号|x|是一个具有十列的矩阵。
% 每一列代表阵列中一个阵元接收到的信号。
% 向信号|x|添加复值高斯噪声。
% 重置默认随机数流以获得可重现的结果。
% 绘制均匀线性阵列（ULA）前四个阵元接收脉冲的幅度。
rng default
npower = 0.5;
x = x + sqrt(npower/2)*(randn(size(x)) + 1i*randn(size(x)));
figure;
subplot(221)
plot(t,abs(x(:,1)))
title('Element 1 (magnitude)')
axis tight
ylabel('Magnitude')
subplot(222)
plot(t, abs(x(:,2)))
title('Element 2 (magnitude)')
axis tight
ylabel('Magnitude')
subplot(223)
plot(t,abs(x(:,3)))
title('Element 3 (magnitude)')
axis tight
xlabel('Seconds')
ylabel('Magnitude')
subplot(224)
plot(t,abs(x(:,4)))
title('Element 4 (magnitude)')
axis tight
xlabel('Seconds')
ylabel('Magnitude')
% 构建一个相移波束形成器。
% 将WeightsOutputPort属性设置为true，以输出将波束形成器指向到达角的空间滤波器权重。
beamformer = phased.PhaseShiftBeamformer('SensorArray', array,...
    'OperatingFrequency',1e9,'Direction',angle_of_arrival,...
    'WeightsOutputPort',true);
% 执行相移波束形成器，以计算波束形成器输出并计算所应用的权重。
[y,w] = beamformer(x);
% 绘制输出波形的幅度，并附上无噪声的原始波形以作比较。
figure;
subplot(211)
plot(t,abs(testsig))
axis tight
title('Original Signal')
ylabel('Magnitude')
subplot(212)
plot(t,abs(y))
axis tight
title('Received Signal with Beamforming')
ylabel('Magnitude')
xlabel('Seconds')
% 为了研究波束赋形权重对阵列响应的影响，
% 绘制有无波束赋形权重时的阵列归一化功率响应。
azang = -180:30:180;
figure;
subplot(211)
pattern(array,fc,[-180:180],0,'CoordinateSystem','rectangular',...
    'Type','powerdb','PropagationSpeed',physconst('LightSpeed'))
xticks(azang)
set(gca, 'Fontsize',9)
title('Array Response without Beamforming Weights')
subplot(212)
pattern(array,fc,[-180:180],0,'CoordinateSystem','rectangular',...
    'Type','powerdb','PropagationSpeed',physconst('LightSpeed'),...
    'Weights',w)
xticks(azang)
set(gca,'Fontsize',9)
title('Array Response with Beamforming Weights')
