#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

sr = 1*10**5
baseline_s = 1
end_s = 1
chirpLen_s = 28
numSamples = chirpLen_s*sr

start_Hz = 203.00725
stop_Hz = .1

start_Hz = .1
stop_Hz = 200


phase_rad = 0


# =============================================================================
# times_s = np.linspace(0, chirpLen_s, numSamples) # Chirp times.
# k = (stop_Hz - start_Hz) / chirpLen_s # Chirp rate.
# sweepFreqs_Hz = (start_Hz + k/2. * times_s) * times_s
# chirp = np.sin(phase_rad + 2 * np.pi * sweepFreqs_Hz)
# =============================================================================


times_s = np.arange(0, chirpLen_s, 1.0/sr)
A = 50 # pA
k = np.power((stop_Hz/start_Hz),1/chirpLen_s)                                 # phase, radiants.                                                 # sampling rate
chirp =  A* np.sin(phase_rad + 2*np.pi*start_Hz*((np.power(k,times_s)-1)/np.log(k)))       # time function, sinusoid.
chirp=np.concatenate([np.zeros(baseline_s*sr),chirp,np.zeros(end_s*sr)])

chirpLen_s = chirpLen_s+baseline_s+end_s
times_s= np.arange(0, chirpLen_s, 1.0/sr)
numSamples = times_s[-1]*sr


bin_window = .5 #s
bin_window_step= bin_window*sr
bin_step = .1 #s
bin_step_step = bin_step*sr
Pxx_dens = list()
freqlimit=np.max([start_Hz,stop_Hz])*2
t_fft = np.arange(bin_window/2,chirpLen_s-bin_window/2,bin_step)
for start_idx, end_idx in zip(np.arange(0,numSamples-bin_window_step,bin_step_step),np.arange(bin_window_step,numSamples,bin_step_step)):
    f, Pxx_den = signal.welch(chirp[int(start_idx):int(end_idx)], sr,nperseg=bin_window_step)
    Pxx_dens.append(Pxx_den[:np.argmax(f>freqlimit)])

#%

f_plot= f[:np.argmax(f>freqlimit)]
fig = plt.figure(figsize = [5,5])
ax_chirp = fig.add_subplot(211)
ax_fft = fig.add_subplot(212)
ax_chirp.plot(times_s,chirp)
fftfig=ax_fft.imshow(np.stack(Pxx_dens).T[::-1,:], extent=[0,3,0,1])
yticks=ax_fft.get_yticks()
yticks_f= f_plot[np.asarray(yticks*(len(f_plot)-1),int)]
ax_fft.set_yticklabels(yticks_f)
xticks=ax_fft.get_xticks()/3
xticks_t= t_fft[np.asarray(xticks*(len(t_fft)-1),int)]
ax_fft.set_xticklabels(xticks_t.round(2))
ax_chirp.set_ylabel('Injected current (pA)')
ax_fft.set_ylabel('Chirp frequency (Hz)')
ax_fft.set_xlabel('Time (s)')
fftfig.set_clim([0, 100])
#ax_fft.set_yscale('log')
#ax_fft.set_ylim([0,200])
#ax_fft.semilogy(f, Pxx_den)

#ax_fft.set_xlim([start_Hz/100,stop_Hz*100])