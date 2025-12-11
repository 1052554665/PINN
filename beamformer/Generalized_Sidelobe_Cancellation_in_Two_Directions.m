%{
为空气中的 11 个元件声学阵列创建 GSC 波束形成器。
线性调频信号以 -50 方位角和 0 仰角入射到阵列上。计算入射波方向和另一个方向的波束信号。比较两个波束成型输出。
信号传播速度为340 m/s，采样率为8 kHz。
创建麦克风和阵列系统对象。阵列元件间距为二分之一波长。将信号频率设置为奈奎斯特频率的二分之一。
%}
c = 340.0;  % 空气中的声速
fs = 8.0e3; % 采样率 8 kHz
fc = fs/2;
lam = c/fc;
transducer = phased.OmnidirectionalMicrophoneElement('FrequencyRange',[20 20000]);
array = phased.ULA('Element',transducer,'NumElements',11,'ElementSpacing',lam/2);
% 模拟带宽为 500 Hz 的线性调频信号
t = 0:1/fs:0.5;
signal = chirp(t,0,0.5,500);
% 创建一个击中阵列的入射波场
collector = phased.WidebandCollector('Sensor',array,'PropagationSpeed',c,...
    'SampleRate',fs,'ModulatedInput',false,'NumSubbands',512);
incidentAngle = [-50;0];
signal = collector(signal.',incidentAngle);
noise = 0.1*randn(size(signal));
recsignal = signal + noise;
% 执行GSC波束成形并绘制波束形成器输出。还绘制到达阵列中间元件的非波束成形信号。
gscbeamformer = phased.GSCBeamformer('SensorArray',array,...
    'PropagationSpeed',c,'SampleRate',fs,'DirectionSource','Input port',...
    'FilterLength',5);  % DirectionSource='Input port' 允许在调用时指定波束方向。
ygsci = gscbeamformer(recsignal,incidentAngle);      % 入射方向 (-50°,0°)，真实信号方向
ygsco = gscbeamformer(recsignal,[20;30]);   % 另一个方向 (20°,30°)
plot(t*1000,recsignal(:,6),t*1000,ygsci,t*1000,ygsco)
xlabel('Time (ms)')
ylabel('Amplitude')
legend('Received signal at element', 'GSC beamformed signal (incident direction)',...
    'GSC beamformed signal (other direction)', 'Location','southeast')
% 放大输出的一小部分。
idx = 1000:1300;
plot(t(idx)*1000,recsignal(idx,6),t(idx)*1000,ygsci(idx),t(idx)*1000,ygsco(idx))
xlabel('Time (ms)')
legend('Received signal at element','GSC beamformed signal (incident direction)',...
    'GSC beamformed signal (other direction)', 'Location','southeast')