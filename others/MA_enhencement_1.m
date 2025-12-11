%Matlab中搭建麦克风阵列，采用八元均匀线性阵列，阵元间距3cm，声速340m/s。
%选取多组语音进行实验，所有语音均取自清华大学开源语音数据集THCHH30，采样率为16kHz，入射角度为30°。
%使用MATLAB自带的阵列函数产生多通道语音信号。对麦克风接收到的信号进行分帧，分帧长度为25ms，帧移为12.5ms

%%实验一：考察算法对非相干噪声的去噪能力，为每个麦克风叠加白噪声，
%%麦克风之间的噪声不相干，信噪比分别为-5、0、5、10dB

% 麦克风阵列参数
numElements = 8; % 阵元数量
elementSpacing = 0.03; % 阵元间距（3cm）
fs = 16e3; % 采样率（16kHz）
c = 340; % 声速（340m/s）
azimuthAngle = 30; % 入射角度（30°）

%使用 phased.ULA 创建均匀线性阵列。
ula = phased.ULA('NumElements', numElements, 'ElementSpacing', elementSpacing);
ula.Element.FrequencyRange = [0, fs/2]; % 设置频率范围

% 读取语音信号
[voiceSignal, fsVoice] = audioread('D:/datasets/THCHS-30/resource/noise/car/car.wav'); % 替换为语音文件路径
if fsVoice ~= fs
    voiceSignal = resample(voiceSignal, fs, fsVoice); % 重采样到16kHz
end

%产生多通道语音信号。使用 phased.WidebandCollector 将单通道语音信号扩展到多通道信号。
% 信号收集器
collector = phased.WidebandCollector('Sensor', ula, 'PropagationSpeed', c, ...
    'SampleRate', fs, 'NumSubbands', 1000);
% 产生多通道信号
multiChannelSignal = collector(voiceSignal, [azimuthAngle; 0]);

if size(multiChannelSignal, 1) == length(voiceSignal)
    multiChannelSignal = multiChannelSignal'; % 转置
end

% 分帧参数
frameLength = round(0.025 * fs); % 25ms
frameShift = round(0.0125 * fs); % 12.5ms
signalLength = size(multiChannelSignal, 2); % 获取信号长度
numFrames = floor((length(voiceSignal) - frameLength) / frameShift) + 1;

% 初始化分帧后的信号
framedSignal = zeros(numElements, frameLength, numFrames);

%分帧处理。对多通道语音信号进行分帧处理，分帧长度为 25ms，帧移为 12.5ms。
% 分帧处理
for i = 1:numFrames
    startIdx = (i - 1) * frameShift + 1;
    endIdx = startIdx + frameLength - 1;

    % 确保 endIdx 不会超出数组边界
    if endIdx > signalLength
        endIdx = signalLength;
    end

    % 检查索引范围是否正确
    if startIdx > signalLength || endIdx > signalLength
        error('索引超出数组边界：startIdx = %d, endIdx = %d, signalLength = %d', startIdx, endIdx, signalLength);
    end

    framedSignal(:, :, i) = multiChannelSignal(:, startIdx:endIdx);
end

% 添加白噪声。为每个麦克风叠加白噪声，信噪比分别为 -5dB、0dB、5dB 和 10dB。

% 定义信噪比
snrValues = [-5, 0, 5, 10];

% 初始化加噪后的信号
noisySignal = zeros(size(framedSignal));

for snrIdx = 1:length(snrValues)
    snr = snrValues(snrIdx);
    % 计算噪声功率
    signalPower = var(framedSignal(:));
    noisePower = signalPower / 10^(snr / 10);
    
    % 添加白噪声
    for i = 1:numFrames
        noise = sqrt(noisePower) * randn(numElements, frameLength);
        noisySignal(:, :, i) = framedSignal(:, :, i) + noise;
    end
    
    % 合并加噪后的信号
    noisySignalConcat = reshape(noisySignal, numElements, []);
    noisySignalConcat = noisySignalConcat(1, :); % 取第一个阵元的信号

    % 保存加噪后的信号
    noisySignalConcat = noisySignalConcat / max(abs(noisySignalConcat)); % 归一化
    audiowrite(['noisy_signal_snr_' num2str(snr) 'dB.wav'], noisySignalConcat, fs);
end


%去噪处理。使用简单的波束形成器进行去噪处理。
% 波束形成器
beamformer = phased.PhaseShiftBeamformer('SensorArray', ula, 'OperatingFrequency', fs/2, ...
    'DirectionSource', 'Property', 'Direction', [azimuthAngle; 0]);

% 初始化去噪后的信号
denoisedSignal = zeros(size(framedSignal));

for snrIdx = 1:length(snrValues)
    snr = snrValues(snrIdx);
    % 加载加噪后的信号
    noisySignalConcat = audioread(['noisy_signal_snr_' num2str(snr) 'dB.wav']);

    % 重新分帧处理，确保数据维度正确
    % 这里需要将信号重塑为 [numElements, numSamples] 的形式
    noisySignalConcat = reshape(noisySignalConcat, numElements, []);
    
    % 分帧处理
    framedNoisySignal = zeros(numElements, frameLength, numFrames);
    for i = 1:numFrames
        startIdx = (i - 1) * frameShift + 1;
        endIdx = startIdx + frameLength - 1;
        % 确保 endIdx 不超出信号长度
        if endIdx > size(noisySignalConcat, 2)
            endIdx = size(noisySignalConcat, 2);
        end

        % 确保 startIdx 不超过 endIdx
        if startIdx > endIdx
            % 如果 startIdx 已经超过 endIdx，说明没有有效数据，整帧置零
            framedNoisySignal(:, :, i) = 0;
            continue;
        end

        % 计算当前帧的有效长度
        validLength = endIdx - startIdx + 1;
        if validLength < frameLength
            % 处理不完整的帧，补零
            framedNoisySignal(:, 1:validLength, i) = noisySignalConcat(:, startIdx:endIdx);
            framedNoisySignal(:, validLength + 1:frameLength, i) = 0;
        else
            framedNoisySignal(:, :, i) = noisySignalConcat(:, startIdx:endIdx);
        end
    end

    % 波束形成
    for i = 1:numFrames
        % 调用波束形成器
        beamformedOutput = beamformer(framedNoisySignal(:, :, i)');
        % 检查输出维度并进行调整
        if size(beamformedOutput, 1) == frameLength && size(beamformedOutput, 2) == 1
            % 若输出为 400x1，转置为 1x400 并复制到 8 行
            beamformedOutput = repmat(beamformedOutput', numElements, 1);
        end
        denoisedSignal(:, :, i) = beamformedOutput;   
    end
    
    % 合并去噪后的信号
    denoisedSignalConcat = reshape(denoisedSignal, numElements, []);
    denoisedSignalConcat = denoisedSignalConcat(1, :); % 取第一个阵元的信号
    
    % 保存去噪后的信号
    denoisedSignalConcat = denoisedSignalConcat / max(abs(denoisedSignalConcat)); % 归一化
    audiowrite(['denoised_signal_snr_' num2str(snr) 'dB.wav'], denoisedSignalConcat, fs);
end

%语音质量评估。使用分段信噪比（segSNR）和感知评估语音质量（PESQ）评估语音质量。
% 分段信噪比（segSNR）
segSNRValues = zeros(length(snrValues), 2); % 第一列：加噪后，第二列：去噪后

% PESQ 评估
pesqValues = zeros(length(snrValues), 2); % 第一列：加噪后，第二列：去噪后

% 保存原始语音信号为临时文件
tempCleanFile = 'temp_clean_signal.wav';
audiowrite(tempCleanFile, voiceSignal, fs);


for snrIdx = 1:length(snrValues)
    snr = snrValues(snrIdx);
    % 加载加噪和去噪后的信号
    noisySignalConcat = audioread(['noisy_signal_snr_' num2str(snr) 'dB.wav']);
    denoisedSignalConcat = audioread(['denoised_signal_snr_' num2str(snr) 'dB.wav']);

    % 保存加噪和去噪后的信号为临时文件
    tempNoisyFile = ['temp_noisy_signal_snr_' num2str(snr) 'dB.wav'];
    tempDenoisedFile = ['temp_denoised_signal_snr_' num2str(snr) 'dB.wav'];
    audiowrite(tempNoisyFile, noisySignalConcat, fs);
    audiowrite(tempDenoisedFile, denoisedSignalConcat, fs);

    
    % 分段信噪比
    segSNRValues(snrIdx, 1) = segsnr(voiceSignal, noisySignalConcat);
    segSNRValues(snrIdx, 2) = segsnr(voiceSignal, denoisedSignalConcat);

    % 检查PESQ函数是否存在
    if exist('pesq', 'file') == 2
    % PESQ 评估
        pesqNoisy = pesq(tempCleanFile, tempNoisyFile, fs);
        pesqDenoised = pesq(tempCleanFile, tempDenoisedFile, fs);

        % 取 pesq 函数返回值的第一个元素进行赋值
        pesqValues(snrIdx, 1) = pesqNoisy(1);
        pesqValues(snrIdx, 2) = pesqDenoised(1);
    else
        fprintf('pesq 函数未找到，跳过 PESQ 评估。\n');
        pesqValues(snrIdx, :) = NaN;
    end

    % 删除临时文件
    delete(tempNoisyFile);
    delete(tempDenoisedFile);
end

% 删除原始语音临时文件
delete(tempCleanFile);

% 输出结果
disp('分段信噪比（segSNR）：');
disp(segSNRValues);
disp('PESQ 评估结果：');
disp(pesqValues);


% 画出语谱图对比。绘制加噪前后的语谱图。
% 绘制语谱图
figure;
subplot(3, 1, 1);
spectrogram(voiceSignal, 256, 250, 256, fs, 'yaxis');
title('原始语音语谱图');

for snrIdx = 1:length(snrValues)
    snr = snrValues(snrIdx);
    noisySignalConcat = audioread(['noisy_signal_snr_' num2str(snr) 'dB.wav']);
    denoisedSignalConcat = audioread(['denoised_signal_snr_' num2str(snr) 'dB.wav']);
    
    subplot(3, 1, 2);
    spectrogram(noisySignalConcat, 256, 250, 256, fs, 'yaxis');
    title(['加噪语音语谱图（SNR = ' num2str(snr) 'dB）']);
    
    subplot(3, 1, 3);
    spectrogram(denoisedSignalConcat, 256, 250, 256, fs, 'yaxis');
    title(['去噪语音语谱图（SNR = ' num2str(snr) 'dB）']);
    
    pause(2); % 暂停2秒以便观察
end

