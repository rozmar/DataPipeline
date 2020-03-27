import numpy as np
import datajoint as dj
dj.conn()
from pipeline import lab, experiment, ephys_patch, ephysanal, imaging, imaging_gt
#%
def moving_average(a, n=3) : # moving average 
    if n>2:
        begn = int(np.ceil(n/2))
        endn = int(n-begn)-1
        a = np.concatenate([a[begn::-1],a,a[:-endn:-1]])
    ret = np.cumsum(a,axis = 0, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def get_sweep(key_sweep,junction_potential = 13.5, downsampled_rate = 10000):
    key_sweep['cell_number'] = (imaging_gt.CellMovieCorrespondance()&key_sweep).fetch1('cell_number')
    #%
# =============================================================================
#     key_sweep = {'subject_id': 456462,'session' : 1, 'movie_number' : 0, 'cell_number' : 3,'sweep_number':28} 
#     junction_potential = 13.5 #mV
#     downsampled_rate = 10000 #Hz
# =============================================================================

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
    v_comp = (v - i_conv*uncompensatedRS*10**6)*1000 - junction_potential
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
    
    return v_out, i_out, t_out