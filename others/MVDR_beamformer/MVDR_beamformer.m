% 对一个5元均匀线阵（ULA）应用最小方差无失真响应（MVDR）波束形成器。
% 信号的入射方位角为45度，仰角为0度。信号频率为0.01赫兹，载波频率为300兆赫兹。
t = [0:.1:200]';
fr = .01;
xm = sin(2*pi*fr*t);
c = physconst('LightSpeed');
fc = 300e6;
rng('default');
incidentAngle = [45;0];
array = phased.ULA('NumElements',5,'ElementSpacing',0.5);
x = collectPlaneWave(array,xm,incidentAngle,fc,c);
noise = 0.1*(randn(size(x)) + 1j*randn(size(x)));
rx = x + noise;

% 计算波束赋形权重
beamformer = phased.MVDRBeamformer('SensorArray',array,...  % 创建一个具有默认属性值的 MVDR 波束形成器系统对象
    'PropagationSpeed',c,'OperatingFrequency',fc,...
    'Direction',incidentAngle,'WeightsOutputPort',true);
[y,w] = beamformer(rx);  % 返回波束赋形权重W

figure;
plot(t,real(rx(:,3)),'r:',t,real(y))
xlabel('Time')
ylabel('Amplitude')
legend('Original','Beamformed')

figure;
pattern(array,fc,[-180:180],0,'PropagationSpeed',c,...
    'Weights',w,'CoordinateSystem','rectangular',...
    'Type','powerdb');