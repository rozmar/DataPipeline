import numpy as np
import matplotlib.pyplot as plt
import datajoint as dj
import pandas as pd
from pipeline import pipeline_tools, lab, experiment, behavioranal, ephys_patch
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
    ap_max_index : int unsigned # index of AP max on sweep
    ap_max_time : decimal(8,4) # time of the AP max relative to recording start
    """
    def make(self, key):

        pd_sweep = pd.DataFrame((ephys_patch.Sweep()&key)*(ephys_patch.SweepResponse()&key)*(ephys_patch.SweepStimulus()&key)*(ephys_patch.SweepMetadata()&key))
        trace = pd_sweep['response_trace'].values[0]
        sr = pd_sweep['sample_rate'][0]
        
        peaks = trace > 0
        spikemaxidxes = list()
        while np.any(peaks):
            spikestart = np.argmax(peaks)
            spikeend = np.argmin(peaks[spikestart:])+spikestart
            sipeidx = np.argmax(trace[spikestart:spikeend])+spikestart
            spikemaxidxes.append(sipeidx)
            peaks[spikestart:spikeend] = False
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
            self.insert(keylist,skip_duplicates=True)
       
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
class SeriesResistance(dj.Computed):
    definition = """
    -> SquarePulse
    ---
    series_resistance: decimal(8,2) # series resistance in MOhms 
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
            key['series_resistance'] = RS
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

