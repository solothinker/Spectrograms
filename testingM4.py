from cmsisdsp import arm_float_to_q15
from scipy.io import wavfile
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numpy import pi as PI
from cmsisdsp import arm_cos_f32, arm_float_to_q15
from cmsisdsp import arm_mult_q15
from cmsisdsp import arm_rfft_instance_q15, arm_rfft_init_q15, arm_rfft_q15, arm_cmplx_mag_q15
from cmsisdsp import arm_q15_to_float


fileName = "1-100038-A-14.wav"
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

audio_samples_q15 = arm_float_to_q15(soundData)

# defining the window size and step size
windowSize = 256
stepSize = 128

hanning_window_f32 = np.zeros(windowSize)
for i in range(windowSize):
    hanning_window_f32[i] = 0.5 * (1 - arm_cos_f32(2 * PI * i / windowSize ))

hanning_window_q15 = arm_float_to_q15(hanning_window_f32)

plt.subplot(2,1,1)
plt.plot(audio_samples_q15, label= 'quant sound',color='r')
plt.legend()
plt.grid()
plt.subplot(2,1,2)
plt.plot(soundData, label='float sound')
plt.legend()
plt.grid()
plt.show()

window_1_q15 = audio_samples_q15[:windowSize]
processed_window_1_q15 = arm_mult_q15(window_1_q15, hanning_window_q15)
plt.plot(processed_window_1_q15)
plt.show()

# Initialize the FFT
rfft_instance_q15 = arm_rfft_instance_q15()
status = arm_rfft_init_q15(rfft_instance_q15, windowSize, 0, 1)

# Apply the FFT to the audio
rfft_1_q15 = arm_rfft_q15(rfft_instance_q15, processed_window_1_q15)

# Take the absolute value
fft_bins_1_q15 = arm_cmplx_mag_q15(rfft_1_q15)[:windowSize // 2 + 1]

xf = np.fft.rfftfreq(len(processed_window_1_q15), d=1./samplingRate)
fft_bins_1_q15_scaled = arm_q15_to_float(fft_bins_1_q15) * 512

plt.plot(xf,fft_bins_1_q15_scaled)
plt.show()