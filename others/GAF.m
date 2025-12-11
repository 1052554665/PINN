function wav_to_image(file_path)
    % 加载.wav文件
    [audioSignal, fs] = audioread('C:\Users\Chen\Desktop\cafe.wav');
    
    % 如果是立体声，只取一个通道
    if size(audioSignal, 2) > 1
        audioSignal = audioSignal(:, 1);
    end
    
    % 限制样本数
    maxSamples = 10000;
    if length(audioSignal) > maxSamples
        audioSignal = audioSignal(1:maxSamples);
    end
    
    % 格拉姆角场变换
    gafImage = gramianAngularField(audioSignal);
    
    % 调整图像大小到224x224
    resizedImage = imresize(gafImage, [224, 224]);
    
    % 将灰度图像扩展为彩色图像
    colorImage = repmat(resizedImage, [1, 1, 3]);
    
    % 显示图像
    figure;
    imshow(colorImage);
    title('Gramian Angular Field Color Image from WAV File');
    axis off;
    
    % 保存图像
    imwrite(colorImage, 'output_image.png');
end

function gafImage = gramianAngularField(timeSeries)
    % 归一化时间序列到 [-1, 1]
    normalizedSeries = 2 * (timeSeries - min(timeSeries)) / (max(timeSeries) - min(timeSeries)) - 1;
    
    % 计算极坐标角度
    theta = acos(normalizedSeries);
    
    % 计算余弦矩阵
    cosMatrix = cos(theta' - theta);
    
    % 将余弦矩阵转换为 [0, 1] 范围
    gafImage = (cosMatrix + 1) / 2;
end
