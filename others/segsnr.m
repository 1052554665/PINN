function seg_snr = segsnr(clean_speech, noisy_speech, fs, frame_len, overlap)
    % 检查输入参数
    if nargin < 3
        fs = 16000;
    end
    if nargin < 4
        frame_len = 256;
    end
    if nargin < 5
        overlap = 128;
    end

    % 分帧
    frames_clean = enframe(clean_speech, frame_len, overlap);
    frames_noisy = enframe(noisy_speech, frame_len, overlap);
    num_frames = size(frames_clean, 1);

    seg_snr = zeros(num_frames, 1);

    for i = 1:num_frames
        % 计算纯净语音和带噪语音的功率
        clean_power = mean(frames_clean(i, :).^2);
        noise_power = mean((frames_noisy(i, :) - frames_clean(i, :)).^2);

        % 避免除零错误
        if noise_power == 0
            seg_snr(i) = 30; % 假设信噪比为 30dB
        else
            seg_snr(i) = 10 * log10(clean_power / noise_power);
        end
    end

    % 返回平均分段信噪比
    seg_snr = mean(seg_snr);
end

% 辅助函数：分帧
function frames = enframe(signal, frame_len, overlap)
    signal_len = length(signal);
    step = frame_len - overlap;
    num_frames = floor((signal_len - frame_len) / step) + 1;

    frames = zeros(num_frames, frame_len);

    for i = 1:num_frames
        start_idx = (i - 1) * step + 1;
        end_idx = start_idx + frame_len - 1;
        frames(i, :) = signal(start_idx:end_idx);
    end
end