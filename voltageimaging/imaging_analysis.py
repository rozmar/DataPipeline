import os
os.chdir('/home/rozmar/Scripts/Python/DataPipeline')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import datajoint as dj
dj.conn()
from pipeline import pipeline_tools
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
from voltageimaging import voltage_imaging_utils
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pandas as pd
import time
from plot.plot_imaging import *
import scipy.ndimage as ndimage
font = {'size'   : 16}

matplotlib.rc('font', **font)
from pathlib import Path

homefolder = dj.config['locations.mr_share']

#%% depth histogram
depth = (ephys_patch.Cell()*imaging_gt.CellMovieCorrespondance()).fetch('depth')
fig = plt.figure(figsize = [5,5])
ax_hist = fig.add_subplot(111)
ax_hist.hist(depth,np.arange(10,100,10))
ax_hist.set_xlabel('depth (microns)')
ax_hist.set_ylabel('movie count')

#%% save average images for Carsen
from PIL import Image
subject_ids,sessions,movie_nums, meanimages = (imaging.RegisteredMovie()&'motion_correction_method = "VolPy"').fetch('subject_id','session','movie_number','registered_movie_mean_image')
for subject_id,session,movie_num, meanimage in zip(subject_ids,sessions,movie_nums, meanimages):
    im = Image.fromarray(meanimage)
    im.save("/home/rozmar/Data/Voltage_imaging/mean_images/anm{}_session{}_movie{}.tiff".format(subject_id,session,movie_num))
    break


#%% homefolder = str(Path.home())
AP_tlimits = [-.005,.01]
subject_ids = np.unique(imaging_gt.CellMovieCorrespondance().fetch('subject_id'))
for subject_id in subject_ids:
    key ={'subject_id':subject_id}
    cell_numbers =np.unique((imaging_gt.CellMovieCorrespondance()&key).fetch('cell_number'))
    for cell_number in cell_numbers:
        
        key['cell_number']=cell_number
        #%
        key ={'subject_id':456462, 'cell_number':6}
        #key ={'subject_id':466774, 'cell_number':0}
        key['roi_type'] = "VolPy"
        thresh_all,apmaxtimes_all,baseline_all =(imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()*ephysanal.ActionPotentialDetails()*ephysanal.ActionPotential&key&'ap_real = 1').fetch('ap_threshold','ap_max_time','ap_baseline_value')
        fig = plt.figure(figsize = [15,15])
        ax_hist = fig.add_subplot(311)
        ax_o_ap = fig.add_subplot(313)
        ax_hist.hist(thresh_all,100)
        ax_hist.set_title(key)
        apwavetimes,apwaves,famerates,snratio,apnums,ap_threshold = ((imaging_gt.GroundTruthROI()*imaging.Movie()*imaging_gt.ROIAPWave()*ephysanal.ActionPotentialDetails())&key&'ap_real = 1').fetch('apwave_time','apwave_dff','movie_frame_rate','apwave_snratio','ap_num','ap_threshold')
        thresh_ophys = list()
        for apwavetime, apwave in zip(apwavetimes,apwaves):
            neededidx = (apwavetime < -.005) & (apwavetime > -.015)
            thresh_ophys.append(np.mean(apwave[neededidx]))
            ax_o_ap.plot(apwavetime,apwave-np.mean(apwave[neededidx]))
            #ax_o_ap.plot(apwavetime[neededidx],apwave[neededidx],'ro')
        ax_thresh = fig.add_subplot(312)
        ax_thresh.plot(ap_threshold,thresh_ophys,'ko')
        break
    break
#%%  check ephys-ophys correspondance
def moving_average(a, n=3) : # moving average 
    if n>2:
        begn = int(np.ceil(n/2))
        endn = int(n-begn)-1
        a = np.concatenate([a[begn::-1],a,a[:-endn:-1]])
    ret = np.cumsum(a,axis = 0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
import scipy



tau_1_on_voltron = .64/1000
tau_2_on_voltron = 4.1/1000
tau_1_ratio_on_voltron =  .61
tau_1_off_voltron = .78/1000
tau_2_off_voltron = 3.9/1000
tau_1_ratio_off_voltron = 55
    
    
downsampled_rate = 10000 #Hz
junction_potential = 13.5


key ={'subject_id':456462}
key['movie_number'] = 3 #3


key ={'subject_id':462149}
key['movie_number'] = 0 #3

key = (imaging_gt.CellMovieCorrespondance()&key).fetch(as_dict = True)[0]
key['roi_type'] = "VolPy"
sweep_numbers = key['sweep_numbers']
del key['sweep_numbers']

#%
motion_corr_vectors = np.asarray((imaging.MotionCorrection()*imaging.RegisteredMovie()&key&'motion_correction_method  = "VolPy"'&'motion_corr_description= "rigid motion correction done with VolPy"').fetch1('motion_corr_vectors'))
#% ophys related stuff
session_time, cell_recording_start = (experiment.Session()*ephys_patch.Cell()&key).fetch1('session_time','cell_recording_start')
first_movie_start_time =  np.min(np.asarray(((imaging.Movie()*imaging_gt.GroundTruthROI())&key).fetch('movie_start_time'),float))
first_movie_start_time_real = first_movie_start_time + session_time.total_seconds()
frame_times = (imaging.MovieFrameTimes()&key).fetch1('frame_times') - cell_recording_start.total_seconds() + session_time.total_seconds()
roi_dff,roi_f0,roi_spike_indices,framerate = (imaging.Movie()*imaging.ROI*imaging_gt.GroundTruthROI()&key).fetch1('roi_dff','roi_f0','roi_spike_indices','movie_frame_rate')
roi_spike_indices = roi_spike_indices-1
roi_f = (roi_dff*roi_f0)+roi_f0

xvals = frame_times-frame_times[0]
yvals = roi_f
out = scipy.optimize.curve_fit(lambda t,a,b,c,d,e: a*np.exp(-t/b) + c + d*np.exp(-t/e),  xvals,  yvals,bounds=((0,0,-np.inf,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf)))
f0_fit_f = out[0][0]*np.exp(-xvals/out[0][1])+out[0][2] +out[0][3]*np.exp(-xvals/out[0][4])
dff_fit = (roi_f-f0_fit_f)/f0_fit_f



#%
# calculating F0s
aptimes = (ephysanal.ActionPotential()*imaging_gt.ROIAPWave()*imaging_gt.GroundTruthROI()&key).fetch('ap_max_time')
aptimes = np.array(aptimes ,float)
real_spike_indices = list()
for aptime in aptimes:
    real_spike_indices.append(np.argmax(frame_times>aptime))
real_spike_indices = np.asarray(real_spike_indices)

xvals = frame_times[real_spike_indices]-frame_times[real_spike_indices][0]
yvals = roi_f[real_spike_indices]
out = scipy.optimize.curve_fit(lambda t,a,b,c,d,e: a*np.exp(-t/b)*0 + c + d*np.exp(-t/e),  xvals,  yvals,bounds=((0,0,-np.inf,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf)))
xvals = frame_times-frame_times[real_spike_indices][0]
f0_fit_ap_peak = out[0][0]*np.exp(-xvals/out[0][1])+out[0][2] +out[0][3]*np.exp(-xvals/out[0][4])
dff_fit_ap_peak = (roi_f-f0_fit_ap_peak)/f0_fit_ap_peak

xvals = frame_times[real_spike_indices-1]-frame_times[real_spike_indices-1][0]
yvals = roi_f[real_spike_indices-1]
out = scipy.optimize.curve_fit(lambda t,a,b,c,d,e: a*np.exp(-t/b)*0 + c + d*np.exp(-t/e),  xvals,  yvals,bounds=((0,0,-np.inf,0,0),(np.inf,np.inf,np.inf,np.inf,np.inf)))
xvals = frame_times-frame_times[real_spike_indices-1][0]
f0_fit_ap_thresh = out[0][0]*np.exp(-xvals/out[0][1])+out[0][2] +out[0][3]*np.exp(-xvals/out[0][4])
dff_fit_ap_thresh = (roi_f-f0_fit_ap_thresh)/f0_fit_ap_thresh




# extracting ephys
traces=list()
traces_t=list()
traces_conv = list()
for sweep_number in sweep_numbers:
    #%
    key_sweep = key.copy()
    key_sweep['sweep_number'] = sweep_number

    neutralizationenable,e_sr= (ephys_patch.SweepMetadata()&key_sweep).fetch1('neutralizationenable','sample_rate')
    try:
        uncompensatedRS =  float((ephysanal.SweepSeriesResistance()&key_sweep).fetch1('series_resistance_residual'))
    except:
        uncompensatedRS = 0
    v = (ephys_patch.SweepResponse()&key_sweep).fetch1('response_trace')
    i = (ephys_patch.SweepStimulus()&key_sweep).fetch1('stimulus_trace')
    tau_1_on =.1/1000
    t = np.arange(0,.001,1/e_sr)
    f_on = np.exp(t/tau_1_on) 
    f_on = f_on/np.max(f_on)
    kernel = np.concatenate([f_on,np.zeros(len(t))])[::-1]
    kernel  = kernel /sum(kernel )  
    i_conv = np.convolve(i,kernel,'same')
    v_comp = (v - i*uncompensatedRS*10**6)*1000 - junction_potential
    i = i * 10**12
    
    sweep_start_time  = float((ephys_patch.Sweep()&key_sweep).fetch('sweep_start_time')) 
    trace_t = np.arange(len(v))/e_sr + sweep_start_time
    downsample_factor = int(np.round(e_sr/downsampled_rate))
    #%downsampling
    v_out = moving_average(v_comp, n=downsample_factor)
    v_out = v_out[int(downsample_factor/2)::downsample_factor]
    i_out = moving_average(i, n=downsample_factor)
    i_out = i_out[int(downsample_factor/2)::downsample_factor]
    t_out = moving_average(trace_t, n=downsample_factor)
    t_out = t_out[int(downsample_factor/2)::downsample_factor]
    
    #convolving
    t = np.arange(0,.01,1/downsampled_rate)
    f_on = tau_1_ratio_on_voltron*np.exp(t/tau_1_on_voltron) + (1-tau_1_ratio_on_voltron)*np.exp(-t/tau_2_on_voltron)
    f_off = tau_1_ratio_off_voltron*np.exp(t[::-1]/tau_1_off_voltron) + (1-tau_1_ratio_off_voltron)*np.exp(-t[::-1]/tau_2_off_voltron)
    f_on = f_on/np.max(f_on)
    f_off = f_off/np.max(f_off)
    kernel = np.concatenate([f_on,np.zeros(len(f_off))])[::-1]
    kernel  = kernel /sum(kernel )
    
    trace_conv0 = np.convolve(np.concatenate([v_out[500::-1],v_out,v_out[:-500:-1]]),kernel,mode = 'same') 
    trace_conv0 = trace_conv0[500:500+len(v_out)]
    
    kernel = np.ones(int(np.round(downsampled_rate/framerate)))
    kernel  = kernel /sum(kernel )
    trace_conv = np.convolve(np.concatenate([v_out[500::-1],trace_conv0,v_out[:-500:-1]]),kernel,mode = 'same') 
    trace_conv = trace_conv[500:500+len(v_out)]
# =============================================================================
#     trace_conv = trace_conv0
# =============================================================================
    
    
    
    traces.append(v_out)
    traces_t.append(t_out)
    traces_conv.append(trace_conv)
#%% generate downsampled trace
traces_t_all = np.concatenate(traces_t)
traces_conv_all = np.concatenate(traces_conv)
trace_im = list()
e_o_timediff =list()
for frame_time in frame_times:
    idx = np.argmax(traces_t_all>frame_time)
    e_o_timediff.append(frame_time-traces_t_all[idx])
    trace_im.append(traces_conv_all[idx])  
thrashidx = np.abs(e_o_timediff) > 2*np.median(np.diff(traces_t_all))
trace_im=np.asarray(trace_im)
trace_im[thrashidx]=np.nan
#%% calculate threshold values
thresh_window_size = 1 #frames
thresh_step_back = 1 #frames from AP ak
ap_min_frame_diff = 5 # min frames from previous ap
spikes_needed = np.concatenate([[True],ap_min_frame_diff<np.diff(real_spike_indices)])
thresh_times = list()
thresh_values = list()
thresh_values_dff_filt = list()
thresh_values_dff_filt_ap_peak = list()
thresh_values_dff_filt_ap_thresh = list()
thresh_values_downsampled_ephys = list()
for spike_idx in real_spike_indices[spikes_needed]:
    thresh_times.append(np.mean(frame_times[spike_idx-thresh_step_back-thresh_window_size:spike_idx-thresh_step_back]))
    thresh_values.append(np.mean(roi_f[spike_idx-thresh_step_back-thresh_window_size:spike_idx-thresh_step_back]))
    thresh_values_dff_filt.append(np.mean(dff_fit[spike_idx-thresh_step_back-thresh_window_size:spike_idx-thresh_step_back]))
    thresh_values_dff_filt_ap_peak.append(np.mean(dff_fit_ap_peak[spike_idx-thresh_step_back-thresh_window_size:spike_idx-thresh_step_back]))
    thresh_values_dff_filt_ap_thresh.append(np.mean(dff_fit_ap_thresh[spike_idx-thresh_step_back-thresh_window_size:spike_idx-thresh_step_back]))
    thresh_values_downsampled_ephys.append(np.mean(trace_im[spike_idx-thresh_step_back-thresh_window_size:spike_idx-thresh_step_back]))
#%% calculate optimal F0
sigma = .01
trace_im_filt = ndimage.gaussian_filter(trace_im,sigma*framerate)
roi_f_filt = ndimage.gaussian_filter(roi_f,sigma*framerate)
F0_optimal = roi_f_filt/((trace_im_filt-trace_im_filt[0])*-.0035+1)
fig=plt.figure()
ax = fig.add_subplot(321)
ax.plot(trace_im*-1)
ax.plot(roi_f)
axx=fig.add_subplot(322, sharex=ax)
ax.plot(F0_optimal,'g-')
dff_optimal=(roi_f-F0_optimal)/F0_optimal
dff_optimal_filt = ndimage.gaussian_filter(dff_optimal,sigma*framerate)
axx.plot(trace_im*-.001)
axx.plot(dff_optimal)

axxx=fig.add_subplot(325)
axxx.hist2d(trace_im,dff_optimal_filt,150,[[np.nanmin(trace_im),-40],[np.nanmin(dff_optimal_filt),np.nanmax(dff_optimal_filt)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))

axxxx=fig.add_subplot(324)
F0_optimal_fft = F0_optimal[~np.isnan(F0_optimal)]
fftout = np.abs(np.fft.rfft(F0_optimal_fft))
freqs = np.fft.fftfreq(len(F0_optimal_fft))*framerate
axxxx.plot(freqs[100:round(len(freqs)/2)],fftout[100:round(len(freqs)/2)])
#axxxx.set_yscale('log')
axxxx.set_xlim([0,10])

ax_motion = fig.add_subplot(323, sharex=ax)
ax_motion.plot(motion_corr_vectors)
#%%
sigma = .025
roi_f_filt = ndimage.gaussian_filter(roi_f,sigma*framerate)
roi_dff_filt = ndimage.gaussian_filter(roi_dff,sigma*framerate)
dff_fit_filt = ndimage.gaussian_filter(dff_fit,sigma*framerate)
dff_fit_ap_peak_filt =ndimage.gaussian_filter(dff_fit_ap_peak,sigma*framerate)
dff_fit_ap_thresh_filt =ndimage.gaussian_filter(dff_fit_ap_thresh,sigma*framerate)
trace_im_filt = ndimage.gaussian_filter(trace_im,sigma*framerate)


fig=plt.figure()
ax = fig.add_subplot(721)
ax.plot(frame_times,roi_f)
ax.plot(frame_times[real_spike_indices],roi_f[real_spike_indices],'ro')
#ax.plot(frame_times[real_spike_indices[spikes_needed]],roi_f[real_spike_indices[spikes_needed]],'bo')
ax.plot(thresh_times,thresh_values,'rx')
ax.plot(frame_times,roi_f0,label = 'Volpy F0')
ax.plot(frame_times,f0_fit_f,'g-',label = 'F fitted F0')
ax.plot(frame_times,f0_fit_ap_peak,'k-',label = 'peak fitted F0')
ax.plot(frame_times,f0_fit_ap_thresh,'b-',label = 'threshold fitted F0')
ax.legend()
ax.axes.get_xaxis().set_visible(False)
ax.set_title('original F and F0s')

ax_dff = fig.add_subplot(723, sharex=ax)
ax_dff.plot(frame_times,roi_dff,'y-') #moving_average(roi_dff,4)
#ax_dff.plot(frame_times,roi_dff_filt,'r-')
ax_dff.plot(frame_times[real_spike_indices],roi_dff[real_spike_indices],'ro')
ax_dff.plot(frame_times[real_spike_indices],roi_dff_filt[real_spike_indices],'bo')
ax_dff.axes.get_xaxis().set_visible(False)
ax_dff.set_title('VolPy')
ax_dff.invert_yaxis()

ax_dff_f0fit = fig.add_subplot(725, sharex=ax)
ax_dff_f0fit.plot(frame_times,dff_fit,'g-')
#ax_dff_f0fit.plot(frame_times,dff_fit_filt,'r-')
ax_dff_f0fit.plot(frame_times[real_spike_indices],dff_fit[real_spike_indices],'ro')
ax_dff_f0fit.plot(frame_times[real_spike_indices],dff_fit_filt[real_spike_indices],'bo')
ax_dff_f0fit.plot(thresh_times,thresh_values_dff_filt,'rx')
ax_dff_f0fit.set_title('single exponential fit')
ax_dff_f0fit.axes.get_xaxis().set_visible(False)
ax_dff_f0fit.invert_yaxis()

ax_dff_peakfit = fig.add_subplot(727, sharex=ax)
ax_dff_peakfit.plot(frame_times,dff_fit_ap_peak,'k-')
#ax_dff_peakfit.plot(frame_times,dff_fit_ap_peak_filt,'r-')
ax_dff_peakfit.plot(frame_times[real_spike_indices],dff_fit_ap_peak[real_spike_indices],'ro')
ax_dff_peakfit.plot(frame_times[real_spike_indices],dff_fit_ap_peak_filt[real_spike_indices],'bo')
ax_dff_peakfit.plot(thresh_times,thresh_values_dff_filt_ap_peak,'rx')
ax_dff_peakfit.set_title('single exponential fit on AP peaks')
ax_dff_peakfit.axes.get_xaxis().set_visible(False)
ax_dff_peakfit.invert_yaxis()

ax_dff_threshfit = fig.add_subplot(729, sharex=ax)
ax_dff_threshfit.plot(frame_times,dff_fit_ap_thresh,'k-')
#ax_dff_threshfit.plot(frame_times,dff_fit_ap_thresh_filt,'r-')
ax_dff_threshfit.plot(frame_times[real_spike_indices],dff_fit_ap_thresh[real_spike_indices],'ro')
ax_dff_threshfit.plot(frame_times[real_spike_indices],dff_fit_ap_thresh_filt[real_spike_indices],'bo')
ax_dff_threshfit.plot(thresh_times,thresh_values_dff_filt_ap_thresh,'rx')
ax_dff_threshfit.set_title('single exponential fit on AP thresh')
ax_dff_threshfit.axes.get_xaxis().set_visible(False)
ax_dff_threshfit.invert_yaxis()


ax_ephys = fig.add_subplot(7,2,11, sharex=ax)
for t,v,v_conv in zip(traces_t,traces,traces_conv):
    #ax_ephys.plot(t,v,'k-')
    ax_ephys.plot(t,v_conv,'k-')
ax_ephys.plot(frame_times,trace_im,'g-')
#ax_ephys.plot(frame_times,trace_im_filt,'r-')
ax_ephys.plot(thresh_times,thresh_values_downsampled_ephys,'rx')


ax_filt = fig.add_subplot(722, sharex=ax)
ax_filt.plot(frame_times,roi_f,'g-')
ax_filt.plot(frame_times,roi_f_filt,'k-')


ax_dff_corr = fig.add_subplot(724)
ax_dff_corr.hist2d(trace_im_filt,roi_dff_filt,150,[[np.nanmin(trace_im_filt),-40],[np.nanmin(roi_dff_filt),np.nanmax(roi_dff_filt)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
ax_dff_corr.invert_yaxis()

ax_dff_f0fit_corr = fig.add_subplot(726)
ax_dff_f0fit_corr.hist2d(trace_im_filt,dff_fit_filt,150,[[np.nanmin(trace_im_filt),-40],[np.nanmin(dff_fit_filt),np.nanmax(dff_fit_filt)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
ax_dff_f0fit_corr.invert_yaxis()

ax_dff_peakfit_corr = fig.add_subplot(728)
ax_dff_peakfit_corr.hist2d(trace_im_filt,dff_fit_ap_peak_filt,150,[[np.nanmin(trace_im_filt),-40],[np.nanmin(dff_fit_ap_peak_filt),np.nanmax(dff_fit_ap_peak_filt)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
ax_dff_peakfit_corr.invert_yaxis()

ax_dff_threshfit_corr = fig.add_subplot(7,2,10)
ax_dff_threshfit_corr.hist2d(trace_im_filt,dff_fit_ap_thresh_filt,150,[[np.nanmin(trace_im_filt),-40],[np.nanmin(dff_fit_ap_thresh_filt),np.nanmax(dff_fit_ap_thresh_filt)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
ax_dff_threshfit_corr.invert_yaxis()

#%%
# =============================================================================
#         break
#     break
# #%%
# =============================================================================
thresh_ophys= np.asarray(thresh_ophys)
apmaxtimes_all = np.asarray(apmaxtimes_all,float)
apmaxdiffs = np.concatenate([[10],np.diff(apmaxtimes_all)])
apmaxdiffs[apmaxdiffs>1]=1
needed= apmaxdiffs>.05
plt.plot(apmaxdiffs[needed],thresh_all[needed],'ko',ms=1)
#%%
plt.plot(baseline_all[needed],thresh_all[needed],'ko',ms=1)
# =============================================================================
# vs, stims= (ephys_patch.SweepStimulus()*ephysanal.SweepResponseCorrected()*ephys_patch.Sweep()&key&'sweep_start_time >150' &'sweep_start_time <300').fetch('response_trace_corrected','stimulus_trace')
# v = (ephysanal.SweepResponseCorrected()&key&'sweep_number = 42').fetch1('response_trace_corrected')
# =============================================================================
#%%
thresh_all =(imaging_gt.GroundTruthROI()*imaging_gt.ROIAPWave()*ephysanal.ActionPotentialDetails()&'roi_type = "SpikePursuit_base_subtr_mean"').fetch('ap_threshold')
plt.hist(thresh_all,1000)



#%%
fig=plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.plot(cells_sp['SN'],cells_volpy['SN'],'ko')
ax.set_xlabel('S/N SpikePursuit')
ax.set_ylabel('S/N VolPy')
fig.savefig('./figures/SN_compared.png', bbox_inches = 'tight')
#%% potential roi types
roi_types = imaging.ROIType().fetch()
print('potential roi types: {}'.format(roi_types))

#%% ephys recording histograms

fig=plt.figure()
ax_hist = fig.add_axes([0,0,1,1])
ax_hist.hist(cells['RS'])
ax_hist.set_xlabel('Access resistance (MOhm)')
ax_hist.set_ylabel('# of cells')
fig.savefig('./figures/RS_hist.png'.format(roi_type), bbox_inches = 'tight')

fig=plt.figure()
ax_hist = fig.add_axes([0,0,1,1])
ax_hist.hist(cells['V0'])
ax_hist.set_xlabel('Resting membrane potential during movie (mV)')
ax_hist.set_ylabel('# of cells')
fig.savefig('./figures/V0_hist.png'.format(roi_type), bbox_inches = 'tight')
                   
fig=plt.figure()
ax_hist = fig.add_axes([0,0,1,1])
ax_hist.hist(cells['holding'])
ax_hist.set_xlabel('Injected current (pA)')
ax_hist.set_ylabel('# of cells')
fig.savefig('./figures/holding_hist.png'.format(roi_type), bbox_inches = 'tight')                 
#%% IVs
for cell in cells.iterrows():
    ivnum = 0
    try:
        fig = plot_IV(subject_id = cell[1]['subject_id'], cellnum = cell[1]['cell_number'], ivnum = ivnum,IVsweepstoplot = None)
        fig.savefig('./figures/IV_{}_{}_iv{}.png'.format(cell[1]['subject_id'],cell[1]['cell_number'],ivnum), bbox_inches = 'tight')         
    except:
        print('no such IV')
    #break
    #print(cell)

          
#%% PLOT AP waveforms

session = 1
subject_id = 456462
cell_number = 3
roi_type = 'Spikepursuit'#'Spikepursuit'#'VolPy_denoised'#'SpikePursuit'#'VolPy_dexpF0'#'VolPy'#'SpikePursuit_dexpF0'#'VolPy_dexpF0'#''Spikepursuit'#'VolPy'#

binwidth = 30 #s
firing_rate_window = 1 #s
frbinwidth = .01

AP_tlimits = [-.005,.01] #s

#%
key = {'session':session,'subject_id':subject_id,'cell_number':cell_number,'roi_type':roi_type }
gtdata = pd.DataFrame((imaging_gt.GroundTruthROI()&key))
key_cell = gtdata.groupby(['session', 'subject_id','cell_number','motion_correction_method','roi_type']).size().reset_index(name='Freq')
#%
for key_cell in key_cell.iterrows():key_cell = dict(key_cell[1]); del key_cell['Freq']
plot_AP_waveforms(key_cell,AP_tlimits)
#%%
plot_precision_recall(key_cell,binwidth =  binwidth ,frbinwidth = frbinwidth,firing_rate_window =  firing_rate_window)    
# =============================================================================
# data = plot_ephys_ophys_trace(key_cell,time_to_plot=15,trace_window = 5,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
# data['figure_handle'].savefig('./figures/{}_cell_{}_roi_type_{}_short.png'.format(key_cell['subject_id'],key_cell['cell_number'],key_cell['roi_type']), bbox_inches = 'tight')
# =============================================================================
#%%
data = plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 50,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
data['figure_handle'].savefig('./figures/{}_cell_{}_roi_type_{}_long.png'.format(key_cell['subject_id'],key_cell['cell_number'],key_cell['roi_type']), bbox_inches = 'tight')
#%%
plot_precision_recall(key_cell,binwidth =  binwidth ,frbinwidth = frbinwidth,firing_rate_window =  firing_rate_window)    
#%%
plot_precision_recall(key_cell,binwidth =  30,frbinwidth = 0.001,firing_rate_window = 1)    
#%%
plot_ephys_ophys_trace(key_cell,time_to_plot=250,trace_window = 100,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
#%% plot everything..
binwidth = 30 #s
firing_rate_window = 1 #s
frbinwidth = .01

#cell_index = 15
for cell_index in range(len(cells)):

    cell = cells.iloc[cell_index]
    key_cell = dict(cell)    
    del key_cell['Freq']
    if imaging.Movie&key_cell:#&'movie_frame_rate>800':
        #%
        plot_precision_recall(key_cell,binwidth =  binwidth ,frbinwidth = frbinwidth,firing_rate_window =  firing_rate_window)    
# =============================================================================
#         data = plot_ephys_ophys_trace(key_cell,time_to_plot=None,trace_window = 50,show_stimulus = True,show_e_ap_peaks = True,show_o_ap_peaks = True)
#         data['figure_handle'].savefig('./figures/{}_cell_{}_roi_type_{}_long.png'.format(key_cell['subject_id'],key_cell['cell_number'],key_cell['roi_type']), bbox_inches = 'tight')
#         print(cell)
# =============================================================================
#%%    #%%
data = plot_ephys_ophys_trace(key_cell,time_to_plot=25,trace_window = 1,show_e_ap_peaks = True,show_o_ap_peaks = True)
#%%
session = 1
subject_id = 456462
cell_number = 5
roi_type = 'SpikePursuit_base_subtr_mean'#'Spikepursuit'#'VolPy_denoised'#'SpikePursuit'#'VolPy_dexpF0'#'VolPy'#'SpikePursuit_dexpF0'#'VolPy_dexpF0'#''Spikepursuit'#'VolPy'#
key_cell = {'session':session,'subject_id':subject_id,'cell_number':cell_number,'roi_type':roi_type }

session_time, cell_recording_start = (experiment.Session()*ephys_patch.Cell()&key_cell).fetch1('session_time','cell_recording_start')
first_movie_start_time =  np.min(np.asarray(((imaging.Movie()*imaging_gt.GroundTruthROI())&key_cell).fetch('movie_start_time'),float))
first_movie_start_time_real = first_movie_start_time + session_time.total_seconds()
threshold,apmaxtime = (imaging_gt.ROIAPWave()*ephysanal.ActionPotential()*ephysanal.ActionPotentialDetails()&key_cell&'ap_real=1').fetch('ap_threshold','ap_max_time')
threshold=np.asarray(threshold,float)
apmaxtime=np.asarray(apmaxtime,float)

# =============================================================================
# session_time_to_plot = time_to_plot+first_movie_start_time  # time relative to session start
# cell_time_to_plot= session_time_to_plot + session_time.total_seconds() -cell_recording_start.total_seconds() # time relative to recording start
# =============================================================================

#%
time_to_plot = apmaxtime[np.argmin(threshold)]+cell_recording_start.total_seconds() - first_movie_start_time_real
data = plot_ephys_ophys_trace(key_cell,
                              time_to_plot=time_to_plot,
                              trace_window = .5,
                              show_stimulus = False,
                              show_e_ap_peaks = False,
                              show_o_ap_peaks = False)
    #%%


#%%plot time offset between 1st ROI and real elphys
min_corr_coeff = .1
subject_ids = ephys_patch.Cell().fetch('subject_id')
sessions = ephys_patch.Cell().fetch('session')
cellnums = ephys_patch.Cell().fetch('cell_number')
roi_types = ['Spikepursuit','VolPy']
for roi_type_idx ,roi_type in enumerate(roi_types):
    fig=plt.figure()
    axs_delay_sweep = list()
    axs_delay_coeff = list()
    axs_coeff_sweep = list()
    axs_delay_sweep.append(fig.add_axes([0,-roi_type_idx,.8,.8]))
    axs_coeff_sweep.append(fig.add_axes([1,-roi_type_idx,.8,.8]))
    axs_delay_coeff.append(fig.add_axes([2,-roi_type_idx,.8,.8]))
    for subject_id,session,cellnum in zip(subject_ids,sessions,cellnums):
        key = { 'subject_id': subject_id, 'session':session,'cell_number':cellnum,'roi_type':roi_type}
        roi_numbers,corrcoeffs = (imaging_gt.ROIEphysCorrelation()&key).fetch('roi_number','corr_coeff')
        if len(corrcoeffs)>0:
            if np.max(np.abs(corrcoeffs))>min_corr_coeff:
                roi_number = np.min(roi_numbers)#roi_numbers[np.argmax(np.abs(corrcoeffs))]
                key['roi_number'] = roi_number
                sweep_number,lag,corrcoeffs = (imaging.Movie()*imaging_gt.ROIEphysCorrelation()&key&'movie_frame_rate>100').fetch('sweep_number','time_lag','corr_coeff')
                needed = np.abs(corrcoeffs)>min_corr_coeff
                #print(lag[needed])
                #print(corrcoeffs[needed])
                axs_delay_sweep[-1].plot(lag[needed],'o-')#sweep_number[needed]-sweep_number[needed][0],
                axs_delay_sweep[-1].set_title(roi_type)
                axs_delay_sweep[-1].set_xlabel('sweep number from imaging start')
                axs_delay_sweep[-1].set_ylabel('time offset (ephys-ophys, ms)')
                axs_delay_sweep[-1].set_ylim([-10,0])
                
                axs_delay_coeff[-1].plot(np.abs(corrcoeffs[needed]),lag[needed],'o')
                axs_delay_coeff[-1].set_title(roi_type)
                axs_delay_coeff[-1].set_xlabel('correlation coefficient')
                axs_delay_coeff[-1].set_ylabel('time offset (ephys-ophys, ms)')
                
                axs_coeff_sweep[-1].plot(np.abs(corrcoeffs[needed]),'-o') #sweep_number[needed]-sweep_number[needed][0]
                axs_coeff_sweep[-1].set_title(roi_type)
                axs_coeff_sweep[-1].set_xlabel('sweep number from imaging start')
                axs_coeff_sweep[-1].set_ylabel('correlation coefficient')
#%% photocurrent
window = 3 #seconds
roi_type = 'Spikepursuit'
key = {'roi_type':roi_type}
gtrois = (imaging_gt.GroundTruthROI()&key).fetch('subject_id','session','cell_number','movie_number','motion_correction_method','roi_type','roi_number',as_dict=True) 
for roi in gtrois:
    session_time = (experiment.Session()&roi).fetch('session_time')[0]
    cell_time = (ephys_patch.Cell()&roi).fetch('cell_recording_start')[0]
    movie_start_time = float((imaging.Movie()&roi).fetch1('movie_start_time'))
    movie_start_time = session_time.total_seconds() + movie_start_time - cell_time.total_seconds()
    
    
    
    sweeps = (imaging_gt.ROIEphysCorrelation()&roi).fetch('sweep_number')
    sweep_now = ephys_patch.Sweep()&roi&'sweep_number = '+str(sweeps[0])
    trace,sr = ((ephys_patch.SweepResponse()*ephys_patch.SweepMetadata())&sweep_now).fetch1('response_trace','sample_rate')
    sweep_start_time = float(sweep_now.fetch1('sweep_start_time'))
    trace_time = np.arange(len(trace))/sr+sweep_start_time
    neededidx = (trace_time>movie_start_time-window) & (trace_time<movie_start_time)
    fig=plt.figure()
    ax = fig.add_axes([0,0,.8,.8])
    ax.plot(trace_time[neededidx],trace[neededidx])
    print(roi)
    
    #fig.show()


# =============================================================================
#         print(key_cell)
#         print('waiting')
#         time.sleep(3)
# =============================================================================
#%% subthreshold correlations
convolve_voltron_kinetics = True
tau_1_on = .64/1000
tau_2_on = 4.1/1000
tau_1_ratio_on =  .61
tau_1_off = .78/1000
tau_2_off = 3.9/1000
tau_1_ratio_off = 55

movingaverage_windows = [0,.01,.02,.03,.04,.05,.1]    

session = 1
subject_id = 456462
cell_number = 3
roi_type = 'VolPy_denoised_raw'#'Spikepursuit_dexpF0'#'VolPy_denoised'#'SpikePursuit'#'VolPy_dexpF0'#'VolPy'#'SpikePursuit_dexpF0'#'VolPy_dexpF0'#''Spikepursuit'#'VolPy'#
key = {'session':session,'subject_id':subject_id,'cell_number':cell_number,'roi_type':roi_type }
gtdata = pd.DataFrame((imaging_gt.GroundTruthROI()&key))
key_cell = gtdata.groupby(['session', 'subject_id','cell_number','motion_correction_method','roi_type']).size().reset_index(name='Freq')
for key_cell in key_cell.iterrows():key_cell = dict(key_cell[1]); del key_cell['Freq']

movie_numbers,famerates = ((imaging_gt.GroundTruthROI()*imaging.Movie())&key).fetch('movie_number','movie_frame_rate')
session_time = (experiment.Session()&key_cell).fetch('session_time')[0]
cell_time = (ephys_patch.Cell()&key_cell).fetch('cell_recording_start')[0]
for movie_number in movie_numbers:
    
    frame_rate = ((imaging.Movie())&key_cell & 'movie_number = '+str(movie_number)).fetch('movie_frame_rate')[0]
    #frame_num = ((imaging.Movie())&key_cell & 'movie_number = '+str(movie_number)).fetch('movie_frame_num')[0]
    movie_start_time = float(((imaging.Movie())&key_cell & 'movie_number = '+str(movie_number)).fetch('movie_start_time')[0])
    movie_start_time = session_time.total_seconds() + movie_start_time - cell_time.total_seconds()
    movie_time = (imaging.MovieFrameTimes()&key_cell & 'movie_number = '+str(movie_number)).fetch('frame_times')[0] -cell_time.total_seconds() +session_time.total_seconds()
    movie_end_time = movie_time[-1]

    sweeps_needed = ephys_patch.Sweep()&key_cell&'sweep_start_time < '+str(movie_end_time) & 'sweep_end_time > '+str(movie_start_time)
    sweep_start_ts, sweep_end_ts, traces,sweep_nums, sample_rates= (sweeps_needed*ephys_patch.SweepResponse()*ephys_patch.SweepMetadata()).fetch('sweep_start_time','sweep_end_time','response_trace','sweep_number','sample_rate')
    trace_times = list()
    for sweep_start_t, sweep_end_t, trace,sample_rate in zip(sweep_start_ts, sweep_end_ts, traces,sample_rates):
        trace_times.append(np.arange(float(sweep_start_t), float(sweep_end_t)+1/sample_rate,1/sample_rate))#np.arange(len(trace))/sample_rate+float(sweep_start_t)
    #%
    dff = (imaging.ROI()*imaging_gt.GroundTruthROI()&key_cell&'movie_number = {}'.format(movie_number)).fetch1('roi_dff')
    for trace,tracetime,sweep_number,sample_rate in zip(traces,trace_times,sweep_nums,sample_rates):  
        #%
        
        start_t = tracetime[0]
        start_t = movie_time[np.argmax(movie_time>=start_t)]
        end_t = np.min([tracetime[-1],movie_time[-1]])
        t = np.arange(0,.01,1/sample_rate)
        
        if convolve_voltron_kinetics:
            f_on = tau_1_ratio_on*np.exp(t/tau_1_on) + (1-tau_1_ratio_on)*np.exp(-t/tau_2_on)
            f_off = tau_1_ratio_off*np.exp(t[::-1]/tau_1_off) + (1-tau_1_ratio_off)*np.exp(-t[::-1]/tau_2_off)
            f_on = f_on/np.max(f_on)
            f_off = f_off/np.max(f_off)
            kernel = np.concatenate([f_on,np.zeros(len(f_off))])[::-1]
            kernel  = kernel /sum(kernel )
            trace_conv = np.convolve(trace,kernel,mode = 'same') 
        else:
            trace_conv = trace
        
        kernel = np.ones(int(np.round(sample_rate/frame_rate)))
        kernel  = kernel /sum(kernel )
        trace_conv = np.convolve(trace_conv,kernel,mode = 'same') 
        trace_conv_time   = tracetime#[down_idxes]
        
        f_idx_now = (movie_time>=start_t) & (movie_time<=end_t)
        dff_time_now = movie_time[f_idx_now]
        e_idx_original = list()
        for t in dff_time_now:
            e_idx_original.append(np.argmin(trace_conv_time<t))
        #timelag = imaging_gt.GroundTruthROI()*imaging_gt.ROIEphysCorrelation()&key_cell&'sweep_number = {}'.format(sweep_number)
        
            #%
        e_vals = trace[e_idx_original]
        f_vals = dff[f_idx_now]
        #%
        for movingaverage_window in movingaverage_windows:
            if movingaverage_window >1/frame_rate:
                e_vals_filt = voltage_imaging_utils.moving_average(e_vals,int(np.round(movingaverage_window/(1/frame_rate))))
                f_vals_filt = voltage_imaging_utils.moving_average(f_vals,int(np.round(movingaverage_window/(1/frame_rate))))
                
             #%
            
            fig=plt.figure()
            ax = fig.add_axes([0,0,1.2,1.2])
            if movingaverage_window >1/frame_rate:
                ax.hist2d(e_vals_filt*1000,f_vals_filt,150,[[np.min(e_vals_filt)*1000,-20],[np.min(f_vals_filt),np.max(f_vals_filt)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
            else:
                ax.hist2d(e_vals*1000,f_vals,150,[[np.min(e_vals)*1000,-20],[np.min(f_vals),np.max(f_vals)]],norm=colors.PowerNorm(gamma=0.3),cmap =  plt.get_cmap('jet'))
            ax.set_xlabel('mV')
            ax.set_ylabel('dF/F')
            ax.invert_yaxis()
            if movingaverage_window >1/frame_rate:
                ax.set_title('subject: {} cell: {} movie: {} sweep: {} moving average: {} ms'.format(key_cell['subject_id'],key_cell['cell_number'],movie_number,sweep_number,movingaverage_window*1000))
                fig.savefig('./figures/subthreshold_{}_cell{}_movie{}_sweep{}_{}_averaging_{}ms.png'.format(key_cell['subject_id'],key_cell['cell_number'],movie_number,sweep_number,roi_type,int(np.round(movingaverage_window*1000))), bbox_inches = 'tight')
            else:
                ax.set_title('subject: {} cell: {} movie: {} sweep: {}'.format(key_cell['subject_id'],key_cell['cell_number'],movie_number,sweep_number))
                fig.savefig('./figures/subthreshold_{}_cell{}_movie{}_sweep{}_{}.png'.format(key_cell['subject_id'],key_cell['cell_number'],movie_number,sweep_number,roi_type), bbox_inches = 'tight')
        #%
        #break
    #%%

    
        
        
        




#%% save movie - IR + voltron
from pathlib import Path
import os
import caiman as cm
import numpy as np
#%

#%% patching movie
sample_movie_dir = homefolder+'/Data/Voltage_imaging/sample_movie/'
files = np.sort(os.listdir(sample_movie_dir))
fnames_full = list()
for fname in files:
    fnames_full.append(os.path.join(sample_movie_dir,fname))
#%
m_orig = cm.load(fnames_full[0:10])
#%
minvals = np.percentile(m_orig,5,(1,2))
maxvals = np.percentile(m_orig,95,(1,2)) - minvals 
#%
m_now = (m_orig-minvals[:,np.newaxis,np.newaxis])/maxvals[:,np.newaxis,np.newaxis]
#%
m_now.play(fr=20, magnification=.5,save_movie = True)  # press q to exit
#%%
#%% save movie - motion correction, denoising, etc
key =  {'session': 1,
 'subject_id': 462149,
 'cell_number': 1,
 'motion_correction_method': 'VolPy',
 'roi_type': 'VolPy'}
key = {'session': 1,
 'subject_id': 456462,
 'cell_number': 3,
 'motion_correction_method': 'Matlab',
 'roi_type': 'SpikePursuit'}
movie_nums = (imaging_gt.GroundTruthROI()*imaging.Movie()&key).fetch('movie_number')
movie_name = (imaging_gt.GroundTruthROI()*imaging.Movie()&key &'movie_number = {}'.format(min(movie_nums))).fetch1('movie_name')
session_date = str((experiment.Session()*imaging_gt.GroundTruthROI()*imaging.Movie()&key &'movie_number = {}'.format(min(movie_nums))).fetch1('session_date'))
fnames,dirs = (imaging.MovieFile()*imaging_gt.GroundTruthROI()*imaging.Movie()&key &'movie_number = {}'.format(min(movie_nums))).fetch('movie_file_name','movie_file_directory')
allfnames = list()

for fname,directory in zip(fnames,dirs):
    allfnames.append(os.path.join(homefolder,directory,fname))
    
originaldir = os.path.join(homefolder,directory)
puthere = originaldir.find('Voltage_imaging')+len('Voltage_imaging')
denoiseddir = os.path.join(originaldir[:puthere],'denoised_volpy',originaldir[puthere+1:],movie_name)
volpydir = os.path.join(originaldir[:puthere],'VolPy',originaldir[puthere+1:],movie_name)
files = os.listdir(denoiseddir)
for file in files:
    if movie_name in file and file[-4:] == 'mmap':
        denoised_file = file
    if 'memmap_' in file and file[-4:] == 'mmap':
        motioncorrected_denoised_file = file
files = os.listdir(volpydir)
for file in files:
    if 'memmap_' in file and file[-4:] == 'mmap':
        volpy_file = file        

m_orig = cm.load(allfnames[:5])
m_denoised = cm.load(os.path.join(denoiseddir,denoised_file))
m_mocorr_denoised = cm.load(os.path.join(denoiseddir,motioncorrected_denoised_file))
m_volpy = cm.load(os.path.join(volpydir,volpy_file))
#%%
startframe = 9900
framenum = 200
baseline_window = 200
baselinesubtract = False
#offset = int(np.round(baseline_window/2))

m_orig_now = m_orig[startframe:startframe+framenum,:,:]
m_orig_now = m_orig_now.copy()
#m_orig_now =(m_orig_now - np.mean(m_orig_now , axis=0))
#m_orig_now =np.diff(m_orig_now,axis = 0)
if baselinesubtract :
    m_orig_baseline = voltage_imaging_utils.moving_average(m_orig_now, n=baseline_window)
    m_orig_now = m_orig_now/m_orig_baseline
#%
m_volpy_now = m_volpy[startframe:startframe+framenum,:,:]
m_volpy_now = m_volpy_now.copy()
#m_volpy_now=(m_volpy_now- np.mean(m_volpy_now, axis=0))
#m_volpy_now =np.diff(m_volpy_now,axis = 0)
#m_volpy_now  = (m_volpy_now - np.mean(m_volpy_now, axis=(1,2))[:,np.newaxis,np.newaxis])
if baselinesubtract :
    m_volpy_baseline = voltage_imaging_utils.moving_average(m_volpy_now, n=baseline_window)
    m_volpy_now = m_volpy_now/m_volpy_baseline#[offset:m_volpy_baseline.shape[0]+offset,:,:]
#%%
m_denoised_now = m_denoised[startframe:startframe+framenum,:,:]
m_denoised_now = m_denoised_now.copy()
#m_denoised_now=(m_denoised_now- np.mean(m_denoised_now, axis=0))
#m_denoised_now =np.diff(m_denoised_now,axis = 0)
#m_denoised_now  = (m_denoised_now - np.mean(m_denoised_now, axis=(1,2))[:,np.newaxis,np.newaxis])
if baselinesubtract :
    m_denoised_baseline = voltage_imaging_utils.moving_average(m_denoised_now, n=baseline_window)
    m_denoised_now = m_denoised_now/m_denoised_baseline 
#%
m_mocorr_denoised_now = m_mocorr_denoised[startframe:startframe+framenum,:,:]#.copy()
#m_mocorr_denoised_now=(m_mocorr_denoised_now- np.mean(m_mocorr_denoised_now, axis=0))
#m_mocorr_denoised_now =np.diff(m_mocorr_denoised_now,axis = 0)
if baselinesubtract :
    m_mocorr_denoised_baseline = voltage_imaging_utils.moving_average(m_mocorr_denoised_now, n=baseline_window)
    m_mocorr_denoised_now = m_mocorr_denoised_now/m_mocorr_denoised_baseline 
#%

m_now =  cm.concatenate([m_orig_now,m_volpy_now, m_denoised_now,m_mocorr_denoised_now], axis=1)#
#%%
#%%
m_now =  cm.concatenate([m_orig_now,m_volpy_now], axis=1)#
#%%
m_now.play(fr=900, magnification=2,q_max=99.9, q_min=0.1,save_movie = True)
#m_orig = cm.load(allfnames[0:3])
#m_volpy_now.play(fr=400, magnification=1,q_max=99.5, q_min=0.5,save_movie = False)
#%% Szar van a palacsintaban
subject_ids,movie_names,frame_times,sessions,movie_numbers = (imaging.Movie*imaging.MovieFrameTimes()).fetch('subject_id','movie_name','frame_times','session','movie_number')
for subject_id,movie_name,frame_time,session,movie_number in zip(subject_ids,movie_names,frame_times,sessions,movie_numbers):
    frametimediff = np.diff(frame_time)
    if np.min(frametimediff)<.5*np.median(frametimediff):
        key = {'subject_id':subject_id,'movie_number':movie_number,'session':session}
        fig=plt.figure()
        ax = fig.add_axes([0,0,1,1])
        ax.plot(np.diff(frame_time))
        ax.set_title([subject_id,movie_name])
        
        #(imaging.Movie()&key).delete()
    
    
  #%% comparing denoising to original motion corrected movie with caiman
# =============================================================================
# import caiman as cm
# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io as imio
# basedir ='/groups/svoboda/home/rozsam/Data/'
# 
# #data_dir = '/home/rozmar/Data/Voltage_imaging/VolPy/Voltage_rig_1P/rozsam/20191201/40x_patch1' #data_dir = sys.argv[1]
# out_dir = os.path.join(basedir,'Voltage_imaging/sgpmd-nmf/Voltage_rig_1P/rozsam/20191201/40x_patch1')#out_dir = sys.argv[3]
# data_dir = out_dir
# mov_in  = 'memmap__d1_128_d2_512_d3_1_order_C_frames_80000_.mmap'#mov_in = sys.argv[2]
# denoised = 'denoised.tif'
# trend = 'trend.tif'
# dtrend_nnorm = 'detr_nnorm.tif'
# sn_im = 'Sn_image.tif'
# #%%
# i_sn = imio.imread(os.path.join(out_dir,sn_im))[:,:,0]
# m_orig = cm.load(os.path.join(out_dir,mov_in))
# m_denoised = cm.load(os.path.join(out_dir,denoised)).transpose(2,0,1)
# m_trend = cm.load(os.path.join(out_dir,trend)).transpose(2,0,1)
# #m_denoised_w_trend = m_denoised + m_trend
# #m_dtrend_nnorm = cm.load(os.path.join(out_dir,dtrend_nnorm)).transpose(2,0,1)
# #m_noise_substracted = m_orig[:m_denoised.shape[0]]-(m_dtrend_nnorm-m_denoised)*i_sn
# #%%
# #m_pwrig = cm.load(mc.mmap_file)
# ds_ratio = 0.2
# moviehandle = cm.concatenate([m_orig[:m_denoised.shape[0]].resize(1, 1, ds_ratio),
#                               m_noise_substracted.resize(1, 1, ds_ratio)], axis=2)
# 
# moviehandle.play(fr=60, q_max=99.5, magnification=2)  # press q to exit
# #%%
# # % movie subtracted from the mean
# m_orig2 = (m_orig[:m_denoised.shape[0]] - np.mean(m_orig[:m_denoised.shape[0]], axis=0))
# m_denoised2 = (m_noise_substracted - np.mean(m_noise_substracted, axis=0))
# #%%
# ds_ratio = 0.2
# moviehandle1 = cm.concatenate([m_orig2.resize(1, 1, ds_ratio),
#                                m_denoised2.resize(1, 1, ds_ratio),], axis=2)
# moviehandle1.play(fr=60, q_max=99.5, magnification=2)  
# =============================================================================
