import numpy as np
import matplotlib.pyplot as plt
import datajoint as dj
import pandas as pd
import scipy.signal as signal
import scipy.ndimage as ndimage
from pipeline import pipeline_tools, lab, experiment, behavioranal, ephys_patch#, ephysanal
#dj.conn()
#%%
schema = dj.schema(pipeline_tools.get_schema_name('ephys-anal'),locals())

#%%

@schema
class ActionPotential(dj.Computed):
    definition = """
    -> ephys_patch.Sweep
    ap_num : smallint unsigned # action potential number in sweep
    ---
    ap_max_index=null : int unsigned # index of AP max on sweep
    ap_max_time=null : decimal(8,4) # time of the AP max relative to recording start
    """
    def make(self, key):
        
        #%%
        #key = {'subject_id': 454263, 'session': 1, 'cell_number': 1, 'sweep_number': 62}
        #print(key)
        keynow = key.copy()
        if len(ActionPotential()&keynow) == 0:
            pd_sweep = pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepResponse()&key)*(ephys_patch.SweepStimulus()&key)*(ephys_patch.SweepMetadata()&key))
            if len(pd_sweep)>0:
                trace = pd_sweep['response_trace'].values[0]
                sr = pd_sweep['sample_rate'][0]
                si = 1/sr
                sigma = .00005
                trace_f = ndimage.gaussian_filter(trace,sigma/si)
                d_trace_f = np.diff(trace_f)/si
                peaks = d_trace_f > 40
                peaks = ndimage.morphology.binary_dilation(peaks,np.ones(int(round(.002/si))))
        
                spikemaxidxes = list()
                while np.any(peaks):
                    #%%
                    spikestart = np.argmax(peaks)
                    spikeend = np.argmin(peaks[spikestart:])+spikestart
                    if spikestart == spikeend:
                        if sum(peaks[spikestart:]) == len(peaks[spikestart:]):
                            spikeend = len(trace)
                    try:
                        sipeidx = np.argmax(trace[spikestart:spikeend])+spikestart
                    except:
                        print(key)
                        sipeidx = np.argmax(trace[spikestart:spikeend])+spikestart
                    spikemaxidxes.append(sipeidx)
                    peaks[spikestart:spikeend] = False
                    #%%
                if len(spikemaxidxes)>0:
                    spikemaxtimes = spikemaxidxes/sr + float(pd_sweep['sweep_start_time'].values[0])
                    spikenumbers = np.arange(len(spikemaxidxes))+1
                    keylist = list()
                    for spikenumber,spikemaxidx,spikemaxtime in zip(spikenumbers,spikemaxidxes,spikemaxtimes):
                        keynow =key.copy()
                        keynow['ap_num'] = spikenumber
                        keynow['ap_max_index'] = spikemaxidx
                        keynow['ap_max_time'] = spikemaxtime
                        keylist.append(keynow)
                        #%%
                    self.insert(keylist,skip_duplicates=True)
@schema
class ActionPotentialDetails(dj.Computed):
    definition = """
    -> ActionPotential
    ---
    ap_real : tinyint # 1 if real AP
    ap_threshold : float # mV
    ap_threshold_index : int #
    ap_halfwidth : float # ms
    ap_amplitude : float # mV
    ap_dv_max : float # mV/ms
    ap_dv_max_voltage : float # mV
    ap_dv_min : float # mV/ms
    ap_dv_min_voltage : float # mV    
    """
    def make(self, key):
        #%%
        sigma = .00003 # seconds for filering
        step_time = .0001 # seconds
        threshold_value = 10 # mV/ms
        #%%
# =============================================================================
#         key = {'subject_id': 454263, 'session': 1, 'cell_number': 1, 'sweep_number': 56, 'ap_num': 30}
#         print(key)
# =============================================================================
        keynow = key.copy()
        del keynow['ap_num']
        pd_sweep = pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepResponse()&key)*(ephys_patch.SweepStimulus()&key)*(ephys_patch.SweepMetadata()&key))
        pd_ap = pd.DataFrame(ActionPotential()&keynow)
        #%%
        if len(pd_ap)>0 and len(ActionPotentialDetails()&dict(pd_ap.loc[0])) == 0:
            #%%
            print(key)
            trace = pd_sweep['response_trace'].values[0]
            sr = pd_sweep['sample_rate'][0]
            si = 1/sr
            step_size = int(np.round(step_time/si))
            ms5_step = int(np.round(.005/si))
            trace_f = ndimage.gaussian_filter(trace,sigma/si)
            d_trace_f = np.diff(trace_f)/si
            tracelength = len(trace)
            #%%
            keylist = list()
            for ap_now in pd_ap.iterrows():
                ap_now = dict(ap_now[1])
                ap_max_index = ap_now['ap_max_index']
                dvmax_index = ap_max_index
                while dvmax_index>step_size*2 and trace_f[dvmax_index]>0:
                    dvmax_index -= step_size
                while dvmax_index>step_size*2 and dvmax_index < tracelength-step_size and  np.max(d_trace_f[dvmax_index-step_size:dvmax_index])>np.max(d_trace_f[dvmax_index:dvmax_index+step_size]):
                    dvmax_index -= step_size
                if dvmax_index < tracelength -1:
                    dvmax_index = dvmax_index + np.argmax(d_trace_f[dvmax_index:dvmax_index+step_size])
                else:
                    dvmax_index = tracelength-2
                    
                
                dvmin_index = ap_max_index
                #%
                while dvmin_index < tracelength-step_size and  (trace_f[dvmin_index]>0 or np.min(d_trace_f[np.max([dvmin_index-step_size,0]):dvmin_index])>np.min(d_trace_f[dvmin_index:dvmin_index+step_size])):
                    dvmin_index += step_size
                    #%
                dvmin_index -= step_size
                dvmin_index = dvmin_index + np.argmin(d_trace_f[dvmin_index:dvmin_index+step_size])
                
                thresh_index = dvmax_index
                while thresh_index>step_size*2 and (np.min(d_trace_f[thresh_index-step_size:thresh_index])>threshold_value):
                    thresh_index -= step_size
                thresh_index = thresh_index - np.argmax((d_trace_f[np.max([0,thresh_index-step_size]):thresh_index] < threshold_value)[::-1])
                ap_threshold = trace_f[thresh_index]
                ap_amplitude = trace_f[ap_max_index]-ap_threshold
                hw_step_back = np.argmax(trace_f[ap_max_index:np.max([ap_max_index-ms5_step,0]):-1]<ap_threshold+ap_amplitude/2)
                hw_step_forward = np.argmax(trace_f[ap_max_index:ap_max_index+ms5_step]<ap_threshold+ap_amplitude/2)
                ap_halfwidth = (hw_step_back+hw_step_forward)*si
                
                if ap_amplitude > .01 and ap_halfwidth>.0001:
                    ap_now['ap_real'] = 1
                else:
                    ap_now['ap_real'] = 0
                ap_now['ap_threshold'] = ap_threshold*1000
                ap_now['ap_threshold_index'] = thresh_index
                ap_now['ap_halfwidth'] =  ap_halfwidth*1000
                ap_now['ap_amplitude'] =  ap_amplitude*1000
                ap_now['ap_dv_max'] = d_trace_f[dvmax_index]
                ap_now['ap_dv_max_voltage'] = trace_f[dvmax_index]*1000
                ap_now['ap_dv_min'] =  d_trace_f[dvmin_index]
                ap_now['ap_dv_min_voltage'] = trace_f[dvmin_index]*1000
                del ap_now['ap_max_index']
                del ap_now['ap_max_time']
                keylist.append(ap_now)
                #%%
            self.insert(keylist,skip_duplicates=True)
# =============================================================================
#             fig=plt.figure()
#             ax_v = fig.add_axes([0,0,.8,.8])
#             ax_dv = fig.add_axes([0,-1,.8,.8])
#             ax_v.plot(trace[ap_max_index-step_size*10:ap_max_index+step_size*10],'k-')
#             ax_v.plot(trace_f[ap_max_index-step_size*10:ap_max_index+step_size*10])
#             ax_dv.plot(d_trace_f[ap_max_index-step_size*10:ap_max_index+step_size*10])
#             dvidx_now = dvmax_index - ap_max_index + step_size*10
#             ax_dv.plot(dvidx_now,d_trace_f[dvmax_index],'ro')
#             ax_v.plot(dvidx_now,trace_f[dvmax_index],'ro')
#             dvminidx_now = dvmin_index - ap_max_index + step_size*10
#             ax_dv.plot(dvminidx_now,d_trace_f[dvmin_index],'ro')
#             ax_v.plot(dvminidx_now,trace_f[dvmin_index],'ro')
#             threshidx_now = thresh_index - ap_max_index + step_size*10
#             ax_dv.plot(threshidx_now,d_trace_f[thresh_index],'ro')
#             ax_v.plot(threshidx_now,trace_f[thresh_index],'ro')
#             plt.show()
#             break
# =============================================================================
        #%%
    
@schema
class SquarePulse(dj.Computed):
    definition = """
    -> ephys_patch.Sweep
    square_pulse_num : smallint unsigned # action potential number in sweep
    ---
    square_pulse_start_idx: int unsigned # index of sq pulse start
    square_pulse_end_idx: int unsigned # index of sq pulse end
    square_pulse_start_time: decimal(8,4) # time of the sq pulse start relative to recording start
    square_pulse_length: decimal(8,4) # length of the square pulse in seconds
    square_pulse_amplitude: float #amplitude of square pulse
    """
    def make(self, key):
        
        pd_sweep = pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepStimulus()&key)*(ephys_patch.SweepMetadata()&key))
        if len(pd_sweep)>0:
            
            stim = pd_sweep['stimulus_trace'].values[0]
            sr = pd_sweep['sample_rate'].values[0]
            sweepstart = pd_sweep['sweep_start_time'].values[0]
            dstim = np.diff(stim)
            square_pulse_num = -1
            while sum(dstim!=0)>0:
                square_pulse_num += 1
                stimstart = np.argmax(dstim!=0)
                amplitude = dstim[stimstart]
                dstim[stimstart] = 0
                stimend = np.argmax(dstim!=0)
                dstim[stimend] = 0
                stimstart += 1
                stimend += 1
                key['square_pulse_num'] = square_pulse_num
                key['square_pulse_start_idx'] = stimstart
                key['square_pulse_end_idx'] = stimend
                key['square_pulse_start_time'] = stimstart/sr + float(sweepstart)
                key['square_pulse_length'] = (stimend-stimstart)/sr
                key['square_pulse_amplitude'] = amplitude
                self.insert1(key,skip_duplicates=True)
            
@schema
class SquarePulseSeriesResistance(dj.Computed):
    definition = """
    -> SquarePulse
    ---
    series_resistance_squarepulse: decimal(8,2) # series resistance in MOhms 
    """    
    def make(self, key):
        time_back = .0002
        time_capacitance = .0001
        time_forward = .0002
        df_squarepulse = pd.DataFrame((SquarePulse()&key)*ephys_patch.Sweep()*ephys_patch.SweepResponse()*ephys_patch.SweepMetadata())
        stimamplitude = df_squarepulse['square_pulse_amplitude'].values[0]
        if np.abs(stimamplitude)>=40*10**-12:
            trace = df_squarepulse['response_trace'].values[0]
            start_idx = df_squarepulse['square_pulse_start_idx'][0]
            end_idx = df_squarepulse['square_pulse_end_idx'][0]
            sr = df_squarepulse['sample_rate'][0]
            step_back = int(np.round(time_back*sr))
            step_capacitance = int(np.round(time_capacitance*sr))
            step_forward = int(np.round(time_forward*sr))
            
            
            v0_start = np.mean(trace[start_idx-step_back:start_idx])
            vrs_start = np.mean(trace[start_idx+step_capacitance:start_idx+step_capacitance+step_forward])
            v0_end = np.mean(trace[end_idx-step_back:end_idx])
            vrs_end = np.mean(trace[end_idx+step_capacitance:end_idx+step_capacitance+step_forward])
            
            dv_start = vrs_start-v0_start
            RS_start = dv_start/stimamplitude 
            dv_end = vrs_end-v0_end
            RS_end = dv_end/stimamplitude*-1
            
            RS = np.round(np.mean([RS_start,RS_end])/1000000,2)
            key['series_resistance_squarepulse'] = RS
            self.insert1(key,skip_duplicates=True)

@schema
class SweepSeriesResistance(dj.Computed):
    definition = """
    -> ephys_patch.Sweep
    ---
    series_resistance_residual = null: decimal(8,2) # residual series resistance after bridge balance in MOhms 
    series_resistance_bridged = null: decimal(8,2) # bridged series resistance in MOhms 
    series_resistance = null: decimal(8,2) # total series resistance in MOhms 
    """    
    def make(self, key):
        #%%
        #key = {'subject_id':454263,'session':1,'cell_number':0,'sweep_number':0}
        if len((SquarePulseSeriesResistance()&key).fetch('series_resistance_squarepulse'))>0:
            if (ephys_patch.SweepMetadata()&key).fetch('bridgebalenable')[0] == 1:
                bridgeR = (ephys_patch.SweepMetadata()&key).fetch('bridgebalresist')[0]/10**6
            else:
                bridgeR = 0
            try:
                rs_residual = float(np.mean((SquarePulseSeriesResistance()&key).fetch('series_resistance_squarepulse')))
            except:
                print(key)
                rs_residual = float(np.mean((SquarePulseSeriesResistance()&key).fetch('series_resistance_squarepulse')))
            key['series_resistance_residual'] = rs_residual
            key['series_resistance_bridged'] = bridgeR
            key['series_resistance'] = rs_residual + bridgeR
        self.insert1(key,skip_duplicates=True)
        #%%
        
@schema
class SweepFrameTimes(dj.Computed):
    definition = """
    -> ephys_patch.Sweep
    ---
    frame_idx : longblob # index of positive square pulses
    frame_sweep_time : longblob  # time of exposure relative to sweep start in seconds
    frame_time : longblob  # time of exposure relative to recording start in seconds
    """    
    def make(self, key):
        #%%
        #key = {'subject_id':456462,'session':1,'cell_number':3,'sweep_number':24}
        exposure = (ephys_patch.SweepImagingExposure()&key).fetch('imaging_exposure_trace')
        if len(exposure)>0:
            si = 1/(ephys_patch.SweepMetadata()&key).fetch('sample_rate')[0]
            sweeptime = float((ephys_patch.Sweep()&key).fetch('sweep_start_time')[0])
            exposure = np.diff(exposure[0])
            peaks = signal.find_peaks(exposure)
            peaks_idx = peaks[0]
            key['frame_idx']= peaks_idx 
            key['frame_sweep_time']= peaks_idx*si
            key['frame_time']= peaks_idx*si + sweeptime
            self.insert1(key,skip_duplicates=True)
    
    
    
    
# =============================================================================
#     step = 100
#     key = {
#            'subject_id' : 453476,
#            'session':29,
#            'cell_number':1,
#            'sweep_number':3,
#            'square_pulse':0}
#     sqstart = trace[start_idx-step:start_idx+step]
#     sqend = trace[end_idx-step:end_idx+step]
#     time = np.arange(-step,step)/sr*1000
#     fig=plt.figure()
#     ax_v=fig.add_axes([0,0,.8,.8])
#     ax_v.plot(time,sqstart)    
#     ax_v.plot([time_back/2*-1*1000,(time_forward/2+time_capacitance)*1000],[v0_start,vrs_start],'o')    
#     ax_v.plot(time,sqend)    
#     ax_v.plot([time_back/2*-1*1000,(time_forward/2+time_capacitance)*1000],[v0_end,vrs_end],'o')    
# =============================================================================

