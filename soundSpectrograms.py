import requests
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt


fileName = "1-103995-A-30.wav"
response = requests.get(f"https://github.com/karolpiczak/ESC-50/raw/master/audio/{fileName}")

with open(fileName, "wb") as f:
    f.write(response.content)


def downSampling(fileName, SamplingRate, showPlot=False):
    fileSamp, fileData = wavfile.read(fileName)
    # normalizing the input to floating point value
    if fileData.dtype is np.float32:
        fileData32 = fileData
    else:
        fileData32 = fileData/np.iinfo(fileData.dtype).max

    fileData = signal.resample(fileData32, (len(fileData32) // fileSamp * SamplingRate))
    if showPlot:
        plt.subplot(2, 1, 1)
        plt.plot(fileData32,label='Original Data')
        plt.legend()
        plt.grid()
        plt.subplot(2, 1, 2)
        plt.plot(fileData,label='Resampled Data')
        plt.legend()
        plt.grid()
        plt.show()
    return fileData


samplingRate = 16000
soundData = downSampling(fileName, samplingRate)
soundData = soundData[:samplingRate]

# defining the window saize and step size
windowSize = 256
stepSize = 128

# finding out numbers of frame

frames = int((len(soundData)-(windowSize-stepSize)) // stepSize)+1
hanningWindow = np.hanning(windowSize)
freqBins = int(windowSize // 2+1)
Spectrogram = np.empty((frames, freqBins))

for frame in range(frames):
    temp = frame*stepSize
    startFrame,endFrame=temp,temp+windowSize
    print(startFrame,endFrame)
    smoothData = soundData[startFrame:endFrame]*hanningWindow
    fftData = np.fft.rfft(smoothData)
    # plt.subplot(2,1,1)
    # plt.plot(smoothData)
    # plt.subplot(2,1,2)
    # plt.plot(np.absolute(fftData))
    # plt.draw()
    # plt.pause(0.001)
    # plt.clf()
    
    Spectrogram[frame-1] = np.absolute(fftData)

height = Spectrogram.T.shape[0]
X = np.arange(samplingRate, step=height + 1)
Y = np.fft.rfftfreq(windowSize, d=1./samplingRate)
print(X,Y)
plt.pcolormesh(X, Y, Spectrogram.T, cmap='viridis', shading='auto')
plt.colorbar()
plt.show()








