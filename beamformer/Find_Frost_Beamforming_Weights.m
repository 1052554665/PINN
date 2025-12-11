%{
求 Frost 波束成形应用于 7 单元声学 ULA 阵列接收信号的波束形成器权重。
输入信号的入射角为方位角-20和仰角30。该信号添加了高斯白噪声。假设空气中的声速为 340 m/s。使用过滤器长度 15。
%}
numelements = 7;
element = phased.OmnidirectionalMicrophoneElement('FrequencyRange', [50,10000]);    % 阵元是全向声学麦克风
array = phased.ULA('Element',element,'NumElements',numelements,'ElementSpacing',0.04);
fs = 8e3;   % 采样率 8 kHz
t = 0:1/fs:0.3;     % 信号时长 0.3 s
x = chirp(t,0,1,500);   % 信号是 啁啾信号，频率从 0 Hz 到 500 Hz
c = 340.0;
collector = phased.WidebandCollector('Sensor',array,...
    'PropagationSpeed',c,'SampleRate',fs,...
    'ModulatedInput',false,'NumSubbands',8192); % NumSubbands=8192 说明使用频率子带方法来模拟宽带信号
incidentAngle = [-20;30];
x = collector(x.',incidentAngle);
noise = 0.2*randn(size(x));
rx = x + noise;
% 创建滤波器长度为 15 的波束形成器。
% 然后，对到达的信号进行波束成形并获得波束形成器权重。
filterlength = 15;  % 滤波器长度为 15（即每个阵元对应一个 15 阶 FIR 滤波器）
beamformer = phased.FrostBeamformer('SensorArray',array,...
    'PropagationSpeed',c,'SampleRate',fs,'WeightsOutputPort',true,...
    'Direction',incidentAngle,'FilterLength',filterlength);
[y,wt] = beamformer(rx);
size(wt)    % wt 大小为 (15 × 7)，即 7 个阵元，每个阵元一个 15 阶 FIR 滤波器
% 将波束成形输出与到达阵列中间元件的信号进行比较。
plot(1000*t,rx(:,4),'r:',1000*t,y)
xlabel('time (msec)')
ylabel('Amplitude')
legend('Middle Element', 'Beamformed')
%{
画出500Hz,1000Hz,2000Hz频率下Frost波束形成器的等效方向图
1.选取频率
2.对每个阵元的 FIR 滤波器权重做 FFT，取该频率点对应的响应值
3.把得到的 7 个复数值拼成 7x1 权重向量，交给 pattern
%} 
% 计算并绘制不同频率下的等效方向图
nfft = 512;
wt_mat = reshape(wt,filterlength,numelements);  % 把 wt 转换成 [滤波器长度 × 阵元数] 的矩阵 wt_mat
% 对每个阵元的 FIR 滤波器做 FFT，得到在频域的响应 H 
% H 大小为 (512 × 7)，每一行对应一个频率点的权重，7 列对应 7 个阵元
H = fft(wt_mat,nfft);   
freqs = (0:nfft-1)/nfft*fs;

% 想看的频率点
freq_list = [500,1000,2000];
figure;
for k = 1:length(freq_list)
    f0 = freq_list(k);
    [~,idx] = min(abs(freqs - f0)); 
    w_equiv = H(idx,:).';   % % H(idx,:) 给出该频率下 7 个阵元的复数权重，构成等效的波束形成权向量

    subplot(1,length(freq_list),k);
    pattern(array,f0,-180:180,0,...
        'PropagationSpeed',c,...
        'CoordinateSystem','rectangular',...
        'Weights',w_equiv);
    title(sprintf('Frost azimuth pattern @ %.0f Hz',f0));
end
