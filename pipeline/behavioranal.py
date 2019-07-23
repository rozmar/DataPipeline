import datajoint as dj
import pandas as pd
import numpy as np
import decimal
import pipeline.lab as lab
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
#from . import get_schema_name
# 
schema = dj.schema(get_schema_name('behavior-anal'),locals())

@schema
class TrialReactionTime(dj.Computed):
    definition = """
    -> experiment.BehaviorTrial
    ---
    reaction_time : decimal(8,4) # reaction time in seconds (first lick relative to go cue) [-1 in case of ignore trials]
    first_lick_time : decimal(8,4) # time of the first lick after GO cue from trial start in seconds [-1 in case of ignore trials]
    """
    def make(self, key):
        df_licks=pd.DataFrame((experiment.ActionEvent & key).fetch())
        df_gocue = pd.DataFrame((experiment.TrialEvent() & key).fetch())
        gocue_time = df_gocue['trial_event_time'][df_gocue['trial_event_type'] == 'go']
        lick_times = (df_licks['action_event_time'][df_licks['action_event_time'].values>gocue_time.values] - gocue_time.values).values
        key['reaction_time'] = -1
        key['first_lick_time'] = -1
        if len(lick_times) > 0:
            key['reaction_time'] = float(lick_times[0])  
            key['first_lick_time'] = float(lick_times[0])  + float(gocue_time.values)
        self.insert1(key,skip_duplicates=True)
@schema
class SessionReactionTimeHistogram(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    reaction_time_bins : longblob # reaction time bin edges in seconds (first lick relative to go cue)
    reaction_time_values_all_trials  : longblob # trial numbers for each reaction time bin (ignore trials are not included))
    reaction_time_values_miss_trials  : longblob # trial numbers for each reaction time bin
    reaction_time_values_hit_trials  : longblob # trial numbers for each reaction time bin
    """
    def make(self, key):
        df_behaviortrial = pd.DataFrame(((experiment.BehaviorTrial() & key) * (experiment.SessionTrial() & key)  * (TrialReactionTime() & key)).fetch())
        reaction_times_all = np.array((df_behaviortrial['reaction_time'][df_behaviortrial['outcome']!='ignore']).values, dtype=np.float32)
        reaction_times_hit = np.array((df_behaviortrial['reaction_time'][df_behaviortrial['outcome']=='hit']).values, dtype=np.float32)
        reaction_times_miss = np.array((df_behaviortrial['reaction_time'][df_behaviortrial['outcome']=='miss']).values, dtype=np.float32)        
        vals_all,bins = np.histogram(reaction_times_all,100,(0,1))
        vals_hit,bins = np.histogram(reaction_times_hit,100,(0,1))
        vals_miss,bins = np.histogram(reaction_times_miss,100,(0,1))        
        key['reaction_time_bins'] = bins
        key['reaction_time_values_all_trials'] = vals_all
        key['reaction_time_values_miss_trials'] = vals_miss
        key['reaction_time_values_hit_trials'] = vals_hit
        self.insert1(key,skip_duplicates=True)
@schema
class SessionLickRhythmHistogram(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    lick_rhythm_bins : longblob # lick rhythm time bin edges in seconds (lick time relative to the first lick)
    lick_rhythm_values_all_trials  : longblob # trial numbers for each lick rhythm time bin (ignore trials are not included))
    lick_rhythm_values_miss_trials  : longblob # trial numbers for each lick rhythm time bin
    lick_rhythm_values_hit_trials  : longblob # trial numbers for each lick rhythm time bin
    """
    def make(self, key):
        df_licks=pd.DataFrame((experiment.ActionEvent() & key) * (experiment.BehaviorTrial() & key) * (TrialReactionTime() & key))
        alltrials = df_licks['outcome']!='ignore'
        misstrials = df_licks['outcome']=='miss'
        hittrials = df_licks['outcome']=='hit'
        lick_times_from_first_lick_all = np.array( df_licks['action_event_time'][alltrials] - df_licks['first_lick_time'][alltrials] , dtype=np.float32)
        lick_times_from_first_lick_miss = np.array( df_licks['action_event_time'][misstrials] - df_licks['first_lick_time'][misstrials] , dtype=np.float32)
        lick_times_from_first_lick_hit = np.array( df_licks['action_event_time'][hittrials] - df_licks['first_lick_time'][hittrials] , dtype=np.float32)
        vals_all,bins = np.histogram(lick_times_from_first_lick_all,100,(0,1))
        vals_miss,bins = np.histogram(lick_times_from_first_lick_miss,100,(0,1))
        vals_hit,bins = np.histogram(lick_times_from_first_lick_hit,100,(0,1))
        key['lick_rhythm_bins'] = bins
        key['lick_rhythm_values_all_trials'] = vals_all
        key['lick_rhythm_values_miss_trials'] = vals_miss
        key['lick_rhythm_values_hit_trials'] = vals_hit
        self.insert1(key,skip_duplicates=True)
        
@schema
class SessionTrainingType(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_task_protocol : tinyint # the number of the dominant task protocol in the session
    """
    def make(self, key):
        df_taskdetails = pd.DataFrame(experiment.BehaviorTrial() & key)
        key['session_task_protocol'] = df_taskdetails['task_protocol'].median()
        self.insert1(key,skip_duplicates=True)
        
        
        