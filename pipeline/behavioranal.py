import datajoint as dj
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import curve_fit
#import decimal
import warnings
import pipeline.lab as lab
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
#from . import get_schema_name
# 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
schema = dj.schema(get_schema_name('behavior-anal'),locals())

logistic_regression_trials_back = 30
logistic_regression_first_session = 8
logistic_regression_max_bias = .5 # 0 = no bias, 1 = full bias
#%%
def calculate_average_likelihood(local_income,choice,parameters):
    #%%
# =============================================================================
#         local_income = np.asarray(df_choices['local_fractional_income'][0].tolist())
#         choice = np.asarray(df_choices['choice_local_fractional_income'][0].tolist())
#         mu = df_psycurve_fractional['sigmoid_fit_mu']
#         sigma = df_psycurve_fractional['sigmoid_fit_sigma']
# =============================================================================
    if parameters['fit_type'] == 'sigmoid':
        prediction = norm.cdf(local_income, parameters['mu'], parameters['sigma'])
    else:
        prediction = local_income*parameters['slope']+parameters['c']
        #prediction = np.polyval([parameters['slope'],parameters['c']],local_income)
    if len(prediction) > 0:
        score = np.zeros(len(prediction))
        score[choice==1]=np.log(prediction[choice==1])
        score[choice==0]=np.log(1-prediction[choice==0])
        score = score[~np.isinf(score)&~np.isnan(score)]
        SCORE = np.exp(sum(score)/len(score))
        return SCORE
    

def calculate_average_likelihood_series(local_income,choice,parameters,local_filter=np.ones(10)):
    local_filter = local_filter/sum(local_filter)
    if parameters['fit_type'] == 'sigmoid':
        prediction = norm.cdf(local_income, parameters['mu'], parameters['sigma'])
    else:
        prediction = np.asarray(local_income)*parameters['slope']+parameters['c']
    
    score = np.zeros(len(prediction))
    score[choice==1]=np.log(prediction[choice==1])
    score[choice==0]=np.log(1-prediction[choice==0])
    score_series = np.exp(np.convolve(score,local_filter,mode = 'same'))
    return score_series

def calculate_local_income(df_behaviortrial,filter_now):
    trialnum_differential = df_behaviortrial['trial']
    trialnum_fractional = df_behaviortrial['trial']
    right_choice = (df_behaviortrial['trial_choice'] == 'right').values
    left_choice = (df_behaviortrial['trial_choice'] == 'left').values
    right_reward = ((df_behaviortrial['trial_choice'] == 'right')&(df_behaviortrial['outcome'] == 'hit')).values
    left_reward = ((df_behaviortrial['trial_choice'] == 'left')&(df_behaviortrial['outcome'] == 'hit')).values
    
    right_reward_conv = np.convolve(right_reward , filter_now,mode = 'valid')
    left_reward_conv = np.convolve(left_reward , filter_now,mode = 'valid')
    
    right_choice = right_choice[len(filter_now)-1:]
    left_choice = left_choice[len(filter_now)-1:]
    trialnum_differential = trialnum_differential[len(filter_now)-1:]
    
    choice_num = np.ones(len(left_choice))
    choice_num[:]=np.nan
    choice_num[left_choice] = 0
    choice_num[right_choice] = 1
    
    todel = np.isnan(choice_num)
    right_reward_conv = right_reward_conv[~todel]
    left_reward_conv = left_reward_conv[~todel]
    choice_num = choice_num[~todel]
    trialnum_differential = trialnum_differential[~todel]
   
    local_differential_income = right_reward_conv - left_reward_conv
    choice_local_differential_income = choice_num
    
    local_fractional_income = right_reward_conv/(right_reward_conv+left_reward_conv)
    choice_local_fractional_income = choice_num
    
    todel = np.isnan(local_fractional_income)
    local_fractional_income = local_fractional_income[~todel]
    choice_local_fractional_income = choice_local_fractional_income[~todel]
    trialnum_fractional = trialnum_differential[~todel]
    return local_fractional_income, choice_local_fractional_income, trialnum_fractional, local_differential_income, choice_local_differential_income, trialnum_differential

def bin_psychometric_curve(local_income,choice_num,local_income_binnum):
    #%
    bottoms = np.arange(0,100, 100/local_income_binnum)
    tops = np.arange(100/local_income_binnum,100.005, 100/local_income_binnum)
    tops[tops > 100] = 100
    bottoms[bottoms < 0] = 0
    reward_ratio_mean = list()
    reward_ratio_sd = list()
    choice_ratio_mean = list()
    choice_ratio_sd = list()
    n = list()
    for bottom,top in zip(bottoms,tops):
        minval = np.percentile(local_income,bottom)
        maxval = np.percentile(local_income,top)
        if minval == maxval:
            idx = (local_income== minval)
        else:
            idx = (local_income>= minval) & (local_income < maxval)
        reward_ratio_mean.append(np.mean(local_income[idx]))
        reward_ratio_sd.append(np.std(local_income[idx]))
# =============================================================================
#         choice_ratio_mean.append(np.mean(choice_num[idx]))
#         choice_ratio_sd.append(np.std(choice_num[idx]))
# =============================================================================
        bootstrap = bs.bootstrap(choice_num[idx], stat_func=bs_stats.mean)
        choice_ratio_mean.append(bootstrap.value)
        choice_ratio_sd.append(bootstrap.error_width())
        n.append(np.sum(idx))
        #%
    return reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n

#%%
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
            key['reaction_time'] = float(min(lick_times))  
            key['first_lick_time'] = float(min(lick_times))  + float(gocue_time.values)
        self.insert1(key,skip_duplicates=True)
        
@schema
class TrialLickBoutLenght(dj.Computed):
    definition = """
    -> experiment.BehaviorTrial
    ---
    lick_bout_length : decimal(8,4) # lick bout lenght in seconds
    """
    def make(self, key):
        maxlickinterval = .2
        df_lickrhythm = pd.DataFrame(((experiment.BehaviorTrial()*experiment.ActionEvent()) & key)*TrialReactionTime())
        
        if len(df_lickrhythm )>0 and df_lickrhythm['outcome'][0]== 'hit':
            df_lickrhythm['licktime'] = np.nan
            df_lickrhythm['licktime'] = df_lickrhythm['action_event_time']-df_lickrhythm['first_lick_time']
            df_lickrhythm['lickdirection'] = np.nan
            df_lickrhythm.loc[df_lickrhythm['action_event_type'] == 'left lick','lickdirection'] = 'left'
            df_lickrhythm.loc[df_lickrhythm['action_event_type'] == 'right lick','lickdirection'] = 'right'
            df_lickrhythm['firs_licktime_on_the_other_side'] = np.nan
            df_lickrhythm['lickboutlength'] = np.nan

            firs_lick_on_the_other_side = float(np.min(df_lickrhythm.loc[(df_lickrhythm['lickdirection'] != df_lickrhythm['trial_choice']) & (df_lickrhythm['licktime'] > 0) ,'licktime']))
            if np.isnan(firs_lick_on_the_other_side):
                firs_lick_on_the_other_side = np.inf
            df_lickrhythm['firs_licktime_on_the_other_side'] = firs_lick_on_the_other_side
            lickbouttimes = df_lickrhythm.loc[(df_lickrhythm['lickdirection'] == df_lickrhythm['trial_choice']) & (df_lickrhythm['licktime'] < firs_lick_on_the_other_side) & (df_lickrhythm['licktime'] >= 0),'licktime']
            
            if len(lickbouttimes)>1 and any(lickbouttimes.diff().values>maxlickinterval):
                lickbouttimes  = lickbouttimes[:np.where(lickbouttimes.diff().values>maxlickinterval)[0][0]]
            lickboutlenghtnow = float(np.max(lickbouttimes))
            if np.isnan(lickboutlenghtnow):
                lickboutlenghtnow = 0
            #df_lickrhythm['lickboutlength'] = lickboutlenghtnow 
            #%%
        else:
            lickboutlenghtnow = 0
        key['lick_bout_length'] = lickboutlenghtnow
        self.insert1(key,skip_duplicates=True)
        
        
@schema
class SessionStats(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_trialnum : int #number of trials
    session_blocknum : int #number of blocks
    session_hits : int #number of hits
    session_misses : int #number of misses
    session_ignores : int #number of ignores
    session_autowaters : int #number of autowaters
    session_length : decimal(10, 4) #length of the session in seconds
    """
    def make(self, key):
        keytoadd = key
        keytoadd['session_trialnum'] = len(experiment.SessionTrial()&key)
        keytoadd['session_blocknum'] = len(experiment.SessionBlock()&key)
        keytoadd['session_hits'] = len(experiment.BehaviorTrial()&key&'outcome = "hit"')
        keytoadd['session_misses'] = len(experiment.BehaviorTrial()&key&'outcome = "miss"')
        keytoadd['session_ignores'] = len(experiment.BehaviorTrial()&key&'outcome = "ignore"')
        keytoadd['session_autowaters'] = len(experiment.TrialNote & key &'trial_note_type = "autowater"')
        if keytoadd['session_trialnum'] > 0:
            keytoadd['session_length'] = float(((experiment.SessionTrial() & key).fetch('trial_stop_time')).max())
        else:
            keytoadd['session_length'] = 0
        self.insert1(keytoadd,skip_duplicates=True)
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
        if len(df_licks) > 0: # there might be empty sessions
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
class SessionRuns(dj.Computed):
    definition = """
    # a run is a sequence of trials when the mouse chooses the same option
    -> experiment.Session
    run_num : int # number of choice block
    ---
    run_start : int # first trial #the switch itself
    run_end : int # last trial #one trial before the next choice
    run_choice : varchar(8) # left or right
    run_length : int # number of trials in this run
    run_hits : int # number of hit trials
    run_misses : int # number of miss trials
    run_consecutive_misses: int # number of consecutive misses before switch
    run_ignores : int # number of ignore trials
    """
    def make(self, key):     
        #%%
        #key = {'subject_id':453477,'session':1}
        df_choices = pd.DataFrame(experiment.BehaviorTrial()&key)
        #%%
        if len(df_choices)>0:
            df_choices['run_choice'] = df_choices['trial_choice']
            ignores = np.where(df_choices['run_choice']=='none')[0]
            if len(ignores)>0:
                ignoreblock = np.diff(np.concatenate([[0],ignores]))>1
                ignores = ignores[ignoreblock.argmax():]
                ignoreblock = ignoreblock[ignoreblock.argmax():]
                while any(ignoreblock):
                    df_choices.loc[ignores[ignoreblock],'run_choice'] = df_choices.loc[ignores[ignoreblock]-1,'run_choice'].values
                    ignores = np.where(df_choices['run_choice']=='none')[0]
                    ignoreblock = np.diff(np.concatenate([[0],ignores]))>1
                    try:
                        ignores = ignores[ignoreblock.argmax():]
                        ignoreblock = ignoreblock[ignoreblock.argmax():]
                    except:
                        ignoreblock = []

            df_choices['run_choice_num'] = np.nan
            df_choices.loc[df_choices['run_choice'] == 'left','run_choice_num'] = 0
            df_choices.loc[df_choices['run_choice'] == 'right','run_choice_num'] = 1
            diffchoice = np.abs(np.diff(df_choices['run_choice_num']))
            diffchoice[np.isnan(diffchoice)] = 0
            switches = np.where(diffchoice>0)[0]
            if any(np.where(df_choices['run_choice']=='none')[0]):
                runstart = np.concatenate([[np.max(np.where(df_choices['run_choice']=='none')[0])+1],switches+1])
            else:
                runstart = np.concatenate([[0],switches+1])
            runend = np.concatenate([switches,[len(df_choices)-1]])
            columns = list(key.keys())
            columns.extend(['run_num','run_start','run_end','run_choice','run_length','run_hits','run_misses','run_consecutive_misses','run_ignores'])
            df_key = pd.DataFrame(data = np.zeros((len(runstart),len(columns))),columns = columns)
    
            ## this is where I generate and insert the dataframe
            for keynow in key.keys(): 
                df_key[keynow] = key[keynow]
            for run_num,(run_start,run_end) in enumerate(zip(runstart,runend)):
                df_key.loc[run_num,'run_num'] = run_num + 1 
                df_key.loc[run_num,'run_start'] = run_start +1 
                df_key.loc[run_num,'run_end'] = run_end + 1 
                df_key.loc[run_num,'run_choice'] = df_choices['run_choice'][run_start]
                df_key.loc[run_num,'run_length'] = run_end-run_start+1
                df_key.loc[run_num,'run_hits'] = sum(df_choices['outcome'][run_start:run_end+1]=='hit')
                df_key.loc[run_num,'run_misses'] = sum(df_choices['outcome'][run_start:run_end+1]=='miss')
                #df_key.loc[run_num,'run_consecutive_misses'] = sum(df_choices['outcome'][(df_choices['outcome'][run_start:run_end+1]=='miss').idxmax():run_end+1]=='miss')
                if sum(df_choices['outcome'][run_start:run_end+1]=='miss') == len(df_choices['outcome'][run_start:run_end+1]=='miss'):
                    df_key.loc[run_num,'run_consecutive_misses'] = sum(df_choices['outcome'][run_start:run_end+1]=='miss')
                else:
                    df_key.loc[run_num,'run_consecutive_misses'] = sum(df_choices['outcome'][(df_choices['outcome'][run_start:run_end+1]!='miss')[::-1].idxmax():run_end+1]=='miss')        
                
                df_key.loc[run_num,'run_ignores'] = sum(df_choices['outcome'][run_start:run_end+1]=='ignore')
                #%%
            self.insert(df_key.to_records(index=False))
        #%%
@schema
class SessionTrainingType(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_task_protocol : tinyint # the number of the dominant task protocol in the session
    """
    def make(self, key):
        df_taskdetails = pd.DataFrame(experiment.BehaviorTrial() & key)
        if len(df_taskdetails)>0:  # in some sessions there is no behavior at all..
            key['session_task_protocol'] = df_taskdetails['task_protocol'].median()
            self.insert1(key,skip_duplicates=True)
        
# =============================================================================
# @schema # OBSOLETE
# class SessionRewardRatio(dj.Computed):
#     definition = """
#     -> experiment.Session
#     ---
#     session_reward_ratio : longblob # 
#     session_reward_ratio_left : longblob # 
#     session_reward_ratio_right : longblob # 
#     session_maximal_reward_ratio : longblob # 
#     """    
#     def make(self, key):
#         df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key) * (experiment.SessionBlock() & key))
#         if len(df_behaviortrial)>0:
#             windowsize = 20
#             minperiod = 5
#             df_behaviortrial['reward']=0
#             df_behaviortrial.loc[df_behaviortrial['outcome'] == 'hit' , 'reward'] = 1
#             df_behaviortrial.loc[df_behaviortrial['outcome'] == 'miss' , 'reward'] = 0 
#             moving_all = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()
#             df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'left' , 'reward'] = np.nan
#             moving_right = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()
#             df_behaviortrial.loc[df_behaviortrial['outcome'] == 'hit' , 'reward'] = 1
#             df_behaviortrial.loc[df_behaviortrial['outcome'] == 'miss' , 'reward'] = 0 
#             df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'right' , 'reward'] = np.nan
#             moving_left = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()        
#             key['session_reward_ratio'] = moving_all
#             key['session_reward_ratio_left'] = moving_left
#             key['session_reward_ratio_right'] = moving_right
#             key['session_maximal_reward_ratio'] = np.array(df_behaviortrial['p_reward_left']+df_behaviortrial['p_reward_right'],dtype=np.float32)
#             
#             self.insert1(key,skip_duplicates=True)
# =============================================================================
       
@schema
class SessionBias(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_bias_choice : float # 0 = left , 1 = right
    session_bias_lick : float # 0 = left , 1 = right
    """
    def make(self, key):
        #%%
# =============================================================================
#         wr_name = 'FOR02'
#         session = 8
#         subject_id = (lab.WaterRestriction() & 'water_restriction_number = "'+wr_name+'"').fetch('subject_id')[0]
#         key = {
#                 'subject_id':subject_id,
#                 'session':session
#                 } 
# =============================================================================
        choices = (experiment.BehaviorTrial()&key).fetch('trial_choice')
        if len(choices)>0:
            choice_right = sum(choices=='right')
            choice_left = sum(choices=='left')
            licks = (experiment.ActionEvent()&key).fetch('action_event_type')
            lick_left = sum(licks =='left lick')
            lick_right = sum(licks =='right lick')
            choice_bias = choice_right/(choice_right+choice_left)
            if len(licks)>0:
                lick_bias = lick_right/(lick_right+lick_left)
            else:
                lick_bias = choice_bias
            key['session_bias_choice'] = choice_bias
            key['session_bias_lick'] = lick_bias
            self.insert1(key,skip_duplicates=True)
        #%%
@schema
class BlockRewardRatio(dj.Computed):
    definition = """
    -> experiment.SessionBlock
    ---
    block_reward_ratio : decimal(8,4) # miss = 0, hit = 1
    block_reward_ratio_first_tertile : decimal(8,4) # 
    block_reward_ratio_second_tertile : decimal(8,4) # 
    block_reward_ratio_third_tertile : decimal(8,4) # 
    block_length : smallint #
    block_reward_ratio_differential : decimal(8,4) # Left = 0, right = 1
    block_reward_ratio_first_tertile_differential : decimal(8,4) # 
    block_reward_ratio_second_tertile_differential : decimal(8,4) # 
    block_reward_ratio_third_tertile_differential : decimal(8,4) # 
    """    
    def make(self, key):
        #%%
        #key = {'subject_id' : 453477, 'session' : 21, 'block':3}
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
        df_behaviortrial['reward']=0
        df_behaviortrial.loc[df_behaviortrial['outcome'] == 'hit' , 'reward'] = 1
        df_behaviortrial.loc[df_behaviortrial['outcome'] == 'miss' , 'reward'] = 0     
        df_behaviortrial['reward_L']=0
        df_behaviortrial['reward_R']=0
        df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'left') & (df_behaviortrial['outcome'] == 'hit') ,'reward_L']=1
        df_behaviortrial.loc[(df_behaviortrial['trial_choice'] == 'right') & (df_behaviortrial['outcome'] == 'hit') ,'reward_R']=1
        trialnum = len(df_behaviortrial)
        key['block_reward_ratio'] = -1
        key['block_reward_ratio_first_tertile'] = -1
        key['block_reward_ratio_second_tertile'] = -1
        key['block_reward_ratio_third_tertile'] = -1
        key['block_reward_ratio_differential'] = -1
        key['block_reward_ratio_first_tertile_differential'] = -1
        key['block_reward_ratio_second_tertile_differential'] = -1
        key['block_reward_ratio_third_tertile_differential'] = -1
        key['block_length'] = trialnum
        if trialnum >10:
            tertilelength = int(np.floor(trialnum /3))
            
            block_reward_ratio = df_behaviortrial.reward.mean()
            block_reward_ratio_first_tertile = df_behaviortrial.reward[:tertilelength].mean()
            block_reward_ratio_second_tertile = df_behaviortrial.reward[-tertilelength:].mean()
            block_reward_ratio_third_tertile = df_behaviortrial.reward[tertilelength:2*tertilelength].mean()
            
            block_reward_ratio_differential = df_behaviortrial.reward_R.sum()/df_behaviortrial.reward.sum()
            if np.isnan(block_reward_ratio_differential):
                block_reward_ratio_differential = -1
            block_reward_ratio_first_tertile_differential = df_behaviortrial.reward_R[:tertilelength].sum()/df_behaviortrial.reward[:tertilelength].sum()
            if np.isnan(block_reward_ratio_first_tertile_differential):
                block_reward_ratio_first_tertile_differential = -1
            block_reward_ratio_second_tertile_differential = df_behaviortrial.reward_R[-tertilelength:].sum()/df_behaviortrial.reward[-tertilelength:].sum()
            if np.isnan(block_reward_ratio_second_tertile_differential):
                block_reward_ratio_second_tertile_differential = -1
            block_reward_ratio_third_tertile_differential = df_behaviortrial.reward_R[tertilelength:2*tertilelength].sum()/df_behaviortrial.reward[tertilelength:2*tertilelength].sum()
            if np.isnan(block_reward_ratio_third_tertile_differential):
                block_reward_ratio_third_tertile_differential = -1
            
            key['block_reward_ratio'] = block_reward_ratio
            key['block_reward_ratio_first_tertile'] = block_reward_ratio_first_tertile
            key['block_reward_ratio_second_tertile'] = block_reward_ratio_second_tertile
            key['block_reward_ratio_third_tertile'] = block_reward_ratio_third_tertile
            key['block_reward_ratio_differential'] = block_reward_ratio_differential
            key['block_reward_ratio_first_tertile_differential'] = block_reward_ratio_first_tertile_differential
            key['block_reward_ratio_second_tertile_differential'] = block_reward_ratio_second_tertile_differential
            key['block_reward_ratio_third_tertile_differential'] = block_reward_ratio_third_tertile_differential
            #%%
        #print(key)
        self.insert1(key,skip_duplicates=True)

@schema
class BlockChoiceRatio(dj.Computed):
    definition = """ # value between 0 and 1 for left and 1 right choices, averaged over the whole block or a fraction of the block
    -> experiment.SessionBlock
    ---
    block_choice_ratio : decimal(8,4) # 0 = left, 1 = right
    block_choice_ratio_first_tertile : decimal(8,4) # 
    block_choice_ratio_second_tertile : decimal(8,4) # 
    block_choice_ratio_third_tertile : decimal(8,4) # 
    """    
    def make(self, key):
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
        df_behaviortrial['reward']=0
        df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'right' , 'reward'] = 1
        df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'left' , 'reward'] = 0        
        trialnum = len(df_behaviortrial)
        key['block_choice_ratio'] = -1
        key['block_choice_ratio_first_tertile'] = -1
        key['block_choice_ratio_second_tertile'] = -1
        key['block_choice_ratio_third_tertile'] = -1
        if trialnum >10:
            tertilelength = int(np.floor(trialnum /3))
            block_choice_ratio = df_behaviortrial.reward.mean()
            block_choice_ratio_first_tertile = df_behaviortrial.reward[:tertilelength].mean()
            block_choice_ratio_second_tertile = df_behaviortrial.reward[-tertilelength:].mean()
            block_choice_ratio_third_tertile = df_behaviortrial.reward[tertilelength:2*tertilelength].mean()
            key['block_choice_ratio'] = block_choice_ratio
            key['block_choice_ratio_first_tertile'] = block_choice_ratio_first_tertile
            key['block_choice_ratio_second_tertile'] = block_choice_ratio_second_tertile
            key['block_choice_ratio_third_tertile'] = block_choice_ratio_third_tertile
        self.insert1(key,skip_duplicates=True)
     
@schema
class BlockAutoWaterCount(dj.Computed):
    definition = """
    -> experiment.SessionBlock
    ---
    block_autowater_count : smallint # number of autowater trials in block
    """
    def make(self, key):
        df_autowater = pd.DataFrame(experiment.TrialNote()*experiment.SessionBlock() & key)
        if len(df_autowater) == 0:
            block_autowater_count = 0
        else:
            block_autowater_count =(df_autowater['trial_note_type']=='autowater').sum()
        key['block_autowater_count'] = block_autowater_count
        self.insert1(key,skip_duplicates=True)

@schema
class SessionBlockSwitchChoices(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    block_length_prev : longblob
    block_length_next : longblob
    choices_matrix : longblob #
    p_r_prev : longblob #
    p_l_prev : longblob #
    p_l_next : longblob # 
    p_r_next : longblob #
    p_l_change : longblob # 
    p_r_change : longblob #
    
    """    
    def make(self, key):
        minblocklength = 20
        prevblocklength = 30
        nextblocklength = 50
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key) * (experiment.SessionBlock() & key) * (BlockRewardRatio()&key))
        if len(df_behaviortrial)>0:
            df_behaviortrial['trial_choice_plot'] = np.nan
            df_behaviortrial.loc[df_behaviortrial['trial_choice']=='left','trial_choice_plot']=0
            df_behaviortrial.loc[df_behaviortrial['trial_choice']=='right','trial_choice_plot']=1
            blockchanges=np.where(np.diff(df_behaviortrial['block']))[0]
            p_change_L = list()
            p_change_R = list()
            p_L_prev = list()
            p_R_prev = list()
            p_L_next = list()
            p_R_next = list()
            choices_matrix = list()
            block_length_prev = list()
            block_length_next = list()
            for idx in blockchanges:
                prev_blocknum = df_behaviortrial['block_length'][idx]
                next_blocknum = df_behaviortrial['block_length'][idx+1]
                prev_block_p_L = df_behaviortrial['p_reward_left'][idx]
                next_block_p_L = df_behaviortrial['p_reward_left'][idx+1]
                prev_block_p_R = df_behaviortrial['p_reward_right'][idx]
                next_block_p_R = df_behaviortrial['p_reward_right'][idx+1]
                if prev_blocknum > minblocklength and next_blocknum > minblocklength:
                    block_length_prev.append(prev_blocknum)
                    block_length_next.append(next_blocknum)
                    p_L_prev.append(float(prev_block_p_L))
                    p_R_prev.append(float(prev_block_p_R))
                    p_L_next.append(float(next_block_p_L))
                    p_R_next.append(float(next_block_p_R))
                    p_change_L.append(float((next_block_p_L-prev_block_p_L)))
                    p_change_R.append(float(next_block_p_R-prev_block_p_R))
                    choices = np.array(df_behaviortrial['trial_choice_plot'][max([np.max([idx-prevblocklength+1,idx-prev_blocknum+1]),0]):idx+np.min([nextblocklength+1,next_blocknum+1])],dtype=np.float32)
                    if next_blocknum < nextblocklength:
                        ending = np.ones(nextblocklength-next_blocknum)*np.nan
                        choices = np.concatenate([choices,ending])
                    if prev_blocknum < prevblocklength:
                        preceding = np.ones(prevblocklength-prev_blocknum)*np.nan
                        choices = np.concatenate([preceding,choices])
                    choices_matrix.append(choices)
            choices_matrix = np.asmatrix(choices_matrix) 
            key['block_length_prev'] = block_length_prev
            key['block_length_next'] = block_length_next
            key['p_l_prev'] = p_L_prev
            key['p_r_prev'] = p_R_prev
            key['p_l_next'] = p_L_next
            key['p_r_next'] = p_R_next
            key['p_l_change'] = p_change_L
            key['p_r_change'] = p_change_R
            key['choices_matrix'] = choices_matrix
            self.insert1(key,skip_duplicates=True)
    
    
    
@schema
class SessionFittedChoiceCoefficients(dj.Computed):    
    definition = """
    -> experiment.Session
    ---
    coefficients_rewards : longblob
    coefficients_choices  : longblob
    score :  decimal(8,4)
    """    
    def make(self, key):
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
        if len(df_behaviortrial)>0:
            trials_back = logistic_regression_trials_back
            idx = np.argsort(df_behaviortrial['trial'])
            choices = df_behaviortrial['trial_choice'][idx].values
            choices_digitized = np.zeros(len(choices))
            choices_digitized[choices=='right']=1
            choices_digitized[choices=='left']=-1
            outcomes = df_behaviortrial['outcome'][idx].values
            rewards_digitized = choices_digitized.copy()
            rewards_digitized[outcomes=='miss']=0
            label = list()
            data = list()
            for trial in range(trials_back,len(rewards_digitized)):
                if choices_digitized[trial] != 0:
                    label.append(choices_digitized[trial])
                    data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
            label = np.array(label)
            if len(data)>0:
                data = np.matrix(data)
                x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                logisticRegr = LogisticRegression(solver = 'lbfgs')
                logisticRegr.fit(x_train, y_train)
                #predictions = logisticRegr.predict(x_test)
                score = logisticRegr.score(x_test, y_test)        
                coefficients = logisticRegr.coef_
                coefficients = coefficients[0]
                coeff_rewards = coefficients[trials_back-1::-1]
                coeff_choices = coefficients[-1:trials_back-1:-1]
                key['coefficients_rewards'] = coeff_rewards
                key['coefficients_choices'] = coeff_choices
                key['score'] = score
                self.insert1(key,skip_duplicates=True)
                
                
@schema
class SubjectFittedChoiceCoefficientsRNRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    coefficients_nonrewards_subject  : longblob
    coefficients_choices_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        trials_back = logistic_regression_trials_back 
        first_session = logistic_regression_first_session
        max_bias = logistic_regression_max_bias
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            #%%
            df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
            df_bias['biasval'] = np.abs(df_bias['session_bias_choice']*2 -1)
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial() & key & 'session =' +str(session)))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)   
            #%%
            #df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        outcomes = df_behaviortrial['outcome'].values[idx]
                        rewards_digitized = choices_digitized.copy()
                        rewards_digitized[outcomes=='miss']=0
                        non_rewards_digitized = choices_digitized.copy()
                        non_rewards_digitized[outcomes=='hit']=0
                        for trial in range(trials_back,len(rewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial],non_rewards_digitized[trial-trials_back:trial]]))
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_rewards = coefficients[trials_back-1::-1]
                    coeff_choices = coefficients[-trials_back-1:trials_back-1:-1]
                    coeff_nonrewards = coefficients[-1:-trials_back-1:-1]
                    key['coefficients_rewards_subject'] = coeff_rewards
                    key['coefficients_choices_subject'] = coeff_choices
                    key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)    
                    print(wrnumber + ' coefficients fitted for reward+choice')
                else:
                    print('not enough data for' + wrnumber)

@schema
class SubjectFittedChoiceCoefficientsRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    coefficients_choices_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        trials_back = logistic_regression_trials_back 
        first_session = logistic_regression_first_session
        max_bias = logistic_regression_max_bias
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
            df_bias['biasval'] = np.abs(df_bias['session_bias_choice']*2 -1)
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial() & key & 'session =' +str(session)))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)
            #df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        outcomes = df_behaviortrial['outcome'].values[idx]
                        rewards_digitized = choices_digitized.copy()
                        rewards_digitized[outcomes=='miss']=0
                        for trial in range(trials_back,len(rewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_rewards = coefficients[trials_back-1::-1]
                    coeff_choices = coefficients[-1:trials_back-1:-1]
                    key['coefficients_rewards_subject'] = coeff_rewards
                    key['coefficients_choices_subject'] = coeff_choices
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)    
                    print(wrnumber + ' coefficients fitted for reward+choice')
                else:
                    print('not enough data for' + wrnumber)
                    
                    
@schema
class SubjectFittedChoiceCoefficientsNRC(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_nonrewards_subject : longblob
    coefficients_choices_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        trials_back = logistic_regression_trials_back 
        first_session = logistic_regression_first_session
        max_bias = logistic_regression_max_bias
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
            df_bias['biasval'] = np.abs(df_bias['session_bias_choice']*2 -1)
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial() & key & 'session =' +str(session)))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)
            #df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        outcomes = df_behaviortrial['outcome'].values[idx]
                        nonrewards_digitized = choices_digitized.copy()
                        nonrewards_digitized[outcomes=='hit']=0
                        for trial in range(trials_back,len(nonrewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(np.concatenate([nonrewards_digitized[trial-trials_back:trial],choices_digitized[trial-trials_back:trial]]))
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_nonrewards = coefficients[trials_back-1::-1]
                    coeff_choices = coefficients[-1:trials_back-1:-1]
                    key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    key['coefficients_choices_subject'] = coeff_choices
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)    
                    print(wrnumber + ' coefficients fitted for reward+choice')
                else:
                    print('not enough data for' + wrnumber)
                    
@schema
class SubjectFittedChoiceCoefficientsRNR(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    coefficients_nonrewards_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        trials_back = logistic_regression_trials_back 
        first_session = logistic_regression_first_session
        max_bias = logistic_regression_max_bias
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
            df_bias['biasval'] = np.abs(df_bias['session_bias_choice']*2 -1)
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial() & key & 'session =' +str(session)))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)
            #df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        outcomes = df_behaviortrial['outcome'].values[idx]
                        rewards_digitized = choices_digitized.copy()
                        rewards_digitized[outcomes=='miss']=0
                        non_rewards_digitized = choices_digitized.copy()
                        non_rewards_digitized[outcomes=='hit']=0
                        for trial in range(trials_back,len(rewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(np.concatenate([rewards_digitized[trial-trials_back:trial],non_rewards_digitized[trial-trials_back:trial]]))
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_rewards = coefficients[trials_back-1::-1]
                    coeff_nonrewards = coefficients[-1:trials_back-1:-1]
                    key['coefficients_rewards_subject'] = coeff_rewards
                    key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)    
                    print(wrnumber + ' coefficients fitted for reward+choice')
                else:
                    print('not enough data for' + wrnumber)
                    

@schema
class SubjectFittedChoiceCoefficientsOnlyRewards(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        #print(key)
        trials_back = logistic_regression_trials_back 
        first_session = logistic_regression_first_session
        max_bias = logistic_regression_max_bias
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
            df_bias['biasval'] = np.abs(df_bias['session_bias_choice']*2 -1)
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial() & key & 'session =' +str(session)))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)
            #df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        outcomes = df_behaviortrial['outcome'].values[idx]
                        rewards_digitized = choices_digitized.copy()
                        rewards_digitized[outcomes=='miss']=0
                        for trial in range(trials_back,len(rewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(rewards_digitized[trial-trials_back:trial])
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_rewards = coefficients[::-1]
                    key['coefficients_rewards_subject'] = coeff_rewards
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)
                    print(wrnumber + ' coefficients fitted for only reward')
                else:
                    print('not enough data for' + wrnumber)
            else:
                print('not enough data for ' + wrnumber)
                
@schema
class SubjectFittedChoiceCoefficientsOnlyUnRewardeds(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_nonrewards_subject : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        #print(key)
        trials_back = logistic_regression_trials_back 
        first_session = logistic_regression_first_session
        max_bias = logistic_regression_max_bias
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
            df_bias['biasval'] = np.abs(df_bias['session_bias_choice']*2 -1)
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial() & key & 'session =' +str(session)))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)
            #df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        outcomes = df_behaviortrial['outcome'].values[idx]
                        unrewards_digitized = choices_digitized.copy()
                        unrewards_digitized[outcomes=='hit']=0
                        for trial in range(trials_back,len(unrewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(unrewards_digitized[trial-trials_back:trial])
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_nonrewards = coefficients[::-1]
                    key['coefficients_nonrewards_subject'] = coeff_nonrewards
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)
                    print(wrnumber + ' coefficients fitted for only reward')
                else:
                    print('not enough data for' + wrnumber)
            else:
                print('not enough data for ' + wrnumber)                
# =============================================================================
#         else:
#             print('no WR number for this guy')
# =============================================================================

@schema
class SubjectFittedChoiceCoefficientsOnlyChoices(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_choices_subject : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        #print(key)
        trials_back = logistic_regression_trials_back 
        first_session = logistic_regression_first_session
        max_bias = logistic_regression_max_bias
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0:
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            df_bias = pd.DataFrame(SessionBias()& key & 'session >' +str(first_session-1))
            df_bias['biasval'] = np.abs(df_bias['session_bias_choice']*2 -1)
            sessionsneeded = (df_bias.loc[df_bias['biasval']<=max_bias,'session']).values
            df_behaviortrial_all = None
            for session in sessionsneeded:
                df_behaviortrial_now = pd.DataFrame((experiment.BehaviorTrial() & key & 'session =' +str(session)))
                if type(df_behaviortrial_all) != pd.DataFrame:
                    df_behaviortrial_all  = df_behaviortrial_now
                else:
                    df_behaviortrial_all = df_behaviortrial_all.append(df_behaviortrial_now)
            #df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))
            if len(df_behaviortrial_all)>0:
                sessions = np.unique(df_behaviortrial_all['session'])
                for session in sessions:
                    if session >= first_session:
                        df_behaviortrial=df_behaviortrial_all[df_behaviortrial_all['session']==session]
                        idx = np.argsort(df_behaviortrial['trial'])
                        choices = df_behaviortrial['trial_choice'].values[idx]
                        choices_digitized = np.zeros(len(choices))
                        choices_digitized[choices=='right']=1
                        choices_digitized[choices=='left']=-1
                        for trial in range(trials_back,len(choices_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(choices_digitized[trial-trials_back:trial])
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_rewards = coefficients[::-1]
                    key['coefficients_choices_subject'] = coeff_rewards
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)
                    print(wrnumber + ' coefficients fitted for only reward')
                else:
                    print('not enough data for' + wrnumber)
            else:
                print('not enough data for ' + wrnumber)                
# =============================================================================
#         else:
#             print('no WR number for this guy')
# =============================================================================            
            

@schema
class SubjectFittedChoiceCoefficientsVSTime(dj.Computed):    ## TODO: BIAS correction
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        #print(key)
        
        timesteps_back = 90
        first_session = logistic_regression_first_session
        label = list()
        data = list()
        if len((lab.WaterRestriction()&key).fetch('water_restriction_number'))>0 and len(experiment.TrialEvent()&key)>0:
            #%%
            wrnumber = (lab.WaterRestriction()&key).fetch('water_restriction_number')[0]
            
            #%%
            if len((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)))>0:
                sessions = np.unique((experiment.BehaviorTrial() & key & 'session >' +str(first_session-1)).fetch('session'))
                for session in sessions:
                    if session >= first_session:
                        #%%
                        df_behaviortrial = pd.DataFrame((experiment.SessionTrial()*(experiment.TrialEvent()&'trial_event_type = "go"')*experiment.BehaviorTrial())&key&'session = '+str(session))
                        df_behaviortrial['choicetime'] = np.asarray(df_behaviortrial['trial_start_time'] + df_behaviortrial['trial_event_time'],dtype = float)
                        mintval = np.floor(df_behaviortrial['choicetime'].min())
                        maxtval = np.round(df_behaviortrial['choicetime'].max())+1
                        df_behaviortrial['choicetime'] = df_behaviortrial['choicetime'] - mintval
                        maxtval -= mintval
                        choices_digitized = np.zeros(int(maxtval))
                        choices_digitized[np.asarray(np.floor(df_behaviortrial.loc[df_behaviortrial['trial_choice']=='left','choicetime'].values),dtype = int)] = -1
                        choices_digitized[np.asarray(np.floor(df_behaviortrial.loc[df_behaviortrial['trial_choice']=='right','choicetime'].values),dtype = int)] = 1
                        rewards_digitized = choices_digitized.copy()
                        rewards_digitized[np.asarray(np.floor(df_behaviortrial.loc[df_behaviortrial['outcome']=='miss','choicetime'].values),dtype = int)]=0

                        for trial in range(timesteps_back,len(rewards_digitized)):
                            if choices_digitized[trial] != 0:
                                label.append(choices_digitized[trial])
                                data.append(rewards_digitized[trial-timesteps_back:trial])
                label = np.array(label)
                data = np.matrix(data)
                if len(data) > 1:
                    x_train, x_test, y_train, y_test = train_test_split(data, label, test_size=0.15, random_state=0)
                    logisticRegr = LogisticRegression(solver = 'lbfgs')
                    logisticRegr.fit(x_train, y_train)
                    #predictions = logisticRegr.predict(x_test)
                    score = logisticRegr.score(x_test, y_test)        
                    coefficients = logisticRegr.coef_
                    coefficients = coefficients[0]
                    coeff_rewards = coefficients[::-1]
                    key['coefficients_rewards_subject'] = coeff_rewards
                    key['score_subject'] = score
                    self.insert1(key,skip_duplicates=True)
                    print(wrnumber + ' coefficients fitted versus time')
                else:
                    print('not enough data for' + wrnumber)
            else:
                print('not enough data for ' + wrnumber)
# =============================================================================
#         else:
#             print('no WR number for this guy')
# =============================================================================
    

            
@schema
class SessionPsychometricDataBoxCar(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    local_fractional_income : longblob
    choice_local_fractional_income : longblob
    trialnum_local_fractional_income : longblob
    local_differential_income : longblob
    choice_local_differential_income : longblob
    trialnum_local_differential_income : longblob
    local_filter : longblob
    """  
    def make(self,key):
        warnings.filterwarnings("ignore", category=RuntimeWarning)        
        local_filter = np.ones(10)
        local_filter = local_filter/sum(local_filter)
        df_behaviortrial = pd.DataFrame(experiment.BehaviorTrial()&key)
        if len(df_behaviortrial)>1:
            local_fractional_income, choice_local_fractional_income, trialnum_fractional, local_differential_income, choice_local_differential_income, trialnum_differential  = calculate_local_income(df_behaviortrial,local_filter)
            key['local_fractional_income'] = local_fractional_income
            key['choice_local_fractional_income'] = choice_local_fractional_income
            key['trialnum_local_fractional_income'] = trialnum_fractional
            key['local_differential_income'] = local_differential_income
            key['choice_local_differential_income']= choice_local_differential_income
            key['trialnum_local_differential_income'] = trialnum_differential
            key['local_filter'] = local_filter
            self.insert1(key,skip_duplicates=True)

@schema
class SessionPsychometricDataFitted(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    local_fractional_income : longblob
    choice_local_fractional_income : longblob
    trialnum_local_fractional_income : longblob
    local_differential_income : longblob
    choice_local_differential_income : longblob
    trialnum_local_differential_income : longblob
    local_filter : longblob
    """  
    def make(self,key):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        df_coeff = pd.DataFrame(SubjectFittedChoiceCoefficientsOnlyRewards()&key)
        if len(df_coeff)>0:
            local_filter = df_coeff['coefficients_rewards_subject'][0]#.mean()
            local_filter = local_filter/sum(local_filter)
            df_behaviortrial = pd.DataFrame(experiment.BehaviorTrial()&key)
            if len(df_behaviortrial)>1:
                local_fractional_income, choice_local_fractional_income, trialnum_fractional, local_differential_income, choice_local_differential_income, trialnum_differential  = calculate_local_income(df_behaviortrial,local_filter)
                key['local_fractional_income'] = local_fractional_income
                key['choice_local_fractional_income'] = choice_local_fractional_income
                key['trialnum_local_fractional_income'] = trialnum_fractional
                key['local_differential_income'] = local_differential_income
                key['choice_local_differential_income']= choice_local_differential_income
                key['trialnum_local_differential_income'] = trialnum_differential
                key['local_filter'] = local_filter
                self.insert1(key,skip_duplicates=True)

@schema    
class SubjectPsychometricCurveBoxCarFractional(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    sigmoid_fit_mu : double
    sigmoid_fit_sigma : double 
    linear_fit_slope : double
    linear_fit_c : double
    trial_num : longblob
    local_filter : longblob
    """     
    def make(self,key):  
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataBoxCar()&key & 'session > '+str(minsession-1))
        if len(df_psychcurve )>0:
            
            reward_ratio_combined = np.concatenate(df_psychcurve['local_fractional_income'].values)
            choice_num = np.concatenate(df_psychcurve['choice_local_fractional_income'].values)
            mu,sigma = curve_fit(norm.cdf, reward_ratio_combined, choice_num,p0=[0,1])[0]
            out = curve_fit(lambda t,a,b: a*t+b,  reward_ratio_combined,  choice_num)
            slope = out[0][0]
            c = out[0][1]
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_combined,choice_num,reward_ratio_binnum)
            
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
            key['sigmoid_fit_mu'] = mu
            key['sigmoid_fit_sigma'] = sigma     
            key['linear_fit_slope'] = slope
            key['linear_fit_c'] = c
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter']
            self.insert1(key,skip_duplicates=True)

@schema    
class SubjectPsychometricCurveBoxCarDifferential(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    sigmoid_fit_mu : double
    sigmoid_fit_sigma : double 
    linear_fit_slope : double
    linear_fit_c : double
    trial_num : longblob
    local_filter : longblob
    """     
    def make(self,key): 
        #%%
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataBoxCar()&key & 'session > '+str(minsession-1))
        if len(df_psychcurve )>0:
            
            reward_ratio_combined = np.concatenate(df_psychcurve['local_differential_income'].values)
            choice_num = np.concatenate(df_psychcurve['choice_local_differential_income'].values)
            mu,sigma = curve_fit(norm.cdf, reward_ratio_combined, choice_num,p0=[0,1])[0]
            out = curve_fit(lambda t,a,b: a*t+b,  reward_ratio_combined,  choice_num)
            slope = out[0][0]
            c = out[0][1]
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_combined,choice_num,reward_ratio_binnum)
            
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
            key['sigmoid_fit_mu'] = mu
            key['sigmoid_fit_sigma'] = sigma      
            key['linear_fit_slope'] = slope
            key['linear_fit_c'] = c
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter']
            #%%
            self.insert1(key,skip_duplicates=True)

@schema    
class SubjectPsychometricCurveFittedFractional(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    sigmoid_fit_mu : double
    sigmoid_fit_sigma : double 
    linear_fit_slope : double
    linear_fit_c : double
    trial_num : longblob
    local_filter : longblob
    """     
    def make(self,key):  
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataFitted()&key & 'session > '+str(minsession-1))
        if len(df_psychcurve )>0:
            
            reward_ratio_combined = np.concatenate(df_psychcurve['local_fractional_income'].values)
            choice_num = np.concatenate(df_psychcurve['choice_local_fractional_income'].values)
            mu,sigma = curve_fit(norm.cdf, reward_ratio_combined, choice_num,p0=[0,1])[0]
            out = curve_fit(lambda t,a,b: a*t+b,  reward_ratio_combined,  choice_num)
            slope = out[0][0]
            c = out[0][1]
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_combined,choice_num,reward_ratio_binnum)
            
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
            key['sigmoid_fit_mu'] = mu
            key['sigmoid_fit_sigma'] = sigma
            key['linear_fit_slope'] = slope
            key['linear_fit_c'] = c
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter']
            self.insert1(key,skip_duplicates=True)
    
@schema    
class SubjectPsychometricCurveFittedDifferential(dj.Computed):
    definition = """
    -> lab.Subject
    ---
    reward_ratio_mean  : longblob
    reward_ratio_sd  : longblob
    choice_ratio_mean  : longblob
    choice_ratio_sd  : longblob
    sigmoid_fit_mu : double
    sigmoid_fit_sigma : double
    linear_fit_slope : double
    linear_fit_c : double
    trial_num : longblob
    local_filter : longblob
    """     
    def make(self,key):  
        minsession = 8
        reward_ratio_binnum = 10
        df_psychcurve = pd.DataFrame(SessionPsychometricDataFitted()&key & 'session > '+str(minsession-1))
        if len(df_psychcurve )>0:
            #%%
            reward_ratio_combined = np.concatenate(df_psychcurve['local_differential_income'].values)
            choice_num = np.concatenate(df_psychcurve['choice_local_differential_income'].values)   

            mu,sigma = curve_fit(norm.cdf, reward_ratio_combined, choice_num,p0=[0,1])[0]
            out = curve_fit(lambda t,a,b: a*t+b,  reward_ratio_combined,  choice_num)
            slope = out[0][0]
            c = out[0][1]
# =============================================================================
#             x = np.arange(-3,3,.1)
#             y = norm.cdf(x, mu1, sigma1)
#             plt.plot(x,y)
#             
# =============================================================================
            reward_ratio_mean, reward_ratio_sd, choice_ratio_mean, choice_ratio_sd, n = bin_psychometric_curve(reward_ratio_combined,choice_num,reward_ratio_binnum)
            
            key['reward_ratio_mean'] = reward_ratio_mean
            key['reward_ratio_sd'] = reward_ratio_sd
            key['choice_ratio_mean'] = choice_ratio_mean
            key['choice_ratio_sd'] = choice_ratio_sd
            key['sigmoid_fit_mu'] = mu
            key['sigmoid_fit_sigma'] = sigma
            key['linear_fit_slope'] = slope
            key['linear_fit_c'] = c
            key['trial_num'] = n
            key['local_filter'] = df_psychcurve['local_filter']
            self.insert1(key,skip_duplicates=True)    
            
@schema    
class SessionPerformance(dj.Computed):     
    definition = """
    -> experiment.Session
    ---
    performance_boxcar_fractional  : double
    performance_boxcar_differential  : double
    performance_fitted_fractional  : double
    performance_fitted_differential  : double
    """       
    def make(self,key):  
        #%%
# =============================================================================
#         #key = {'subject_id':453475,'session':21}
# =============================================================================
        df_choices = pd.DataFrame(SessionPsychometricDataBoxCar()&key)
        #%%
        if len(df_choices)>0:
            #%%
            df_psycurve_fractional = pd.DataFrame(SubjectPsychometricCurveBoxCarFractional()&key)
            df_psycurve_differential = pd.DataFrame(SubjectPsychometricCurveBoxCarDifferential()&key)
            #%
            local_income = np.asarray(df_choices['local_fractional_income'][0].tolist())
            choice = np.asarray(df_choices['choice_local_fractional_income'][0].tolist())
            mu = df_psycurve_fractional['sigmoid_fit_mu'][0]
            sigma = df_psycurve_fractional['sigmoid_fit_sigma'][0]
            slope = df_psycurve_fractional['linear_fit_slope'][0]
            c = df_psycurve_fractional['linear_fit_c'][0]
            parameters = dict()
            parameters = {'fit_type':'linear','slope':slope,'c':c}
            key['performance_boxcar_fractional'] =  calculate_average_likelihood(local_income,choice,parameters)
                
            local_income = np.asarray(df_choices['local_differential_income'][0].tolist())
            choice = np.asarray(df_choices['choice_local_differential_income'][0].tolist())
            mu = df_psycurve_differential['sigmoid_fit_mu'][0]
            sigma = df_psycurve_differential['sigmoid_fit_sigma'][0]
            slope = df_psycurve_differential['linear_fit_slope'][0]
            c = df_psycurve_differential['linear_fit_c'][0]
            parameters = {'fit_type':'sigmoid','mu':mu,'sigma':sigma}
            key['performance_boxcar_differential'] =  calculate_average_likelihood(local_income,choice,parameters)
            
            
            #%%
            df_choices = pd.DataFrame(SessionPsychometricDataFitted()&key)
            df_psycurve_fractional = pd.DataFrame(SubjectPsychometricCurveFittedFractional()&key)
            df_psycurve_differential = pd.DataFrame(SubjectPsychometricCurveFittedDifferential()&key)
            
            local_income = np.asarray(df_choices['local_fractional_income'][0].tolist())
            choice = np.asarray(df_choices['choice_local_fractional_income'][0].tolist())
            mu = df_psycurve_fractional['sigmoid_fit_mu'][0]
            sigma = df_psycurve_fractional['sigmoid_fit_sigma'][0]
            slope = df_psycurve_fractional['linear_fit_slope'][0]
            c = df_psycurve_fractional['linear_fit_c'][0]
            parameters = {'fit_type':'linear','slope':slope,'c':c}
            key['performance_fitted_fractional'] =  calculate_average_likelihood(local_income,choice,parameters)
            #%%
            local_income = np.asarray(df_choices['local_differential_income'][0].tolist())
            choice = np.asarray(df_choices['choice_local_differential_income'][0].tolist())
            mu = df_psycurve_differential['sigmoid_fit_mu'][0]
            sigma = df_psycurve_differential['sigmoid_fit_sigma'][0]
            slope = df_psycurve_differential['linear_fit_slope'][0]
            c = df_psycurve_differential['linear_fit_c'][0]
            parameters = {'fit_type':'sigmoid','mu':mu,'sigma':sigma}
            key['performance_fitted_differential'] =  calculate_average_likelihood(local_income,choice,parameters)
            if key['performance_fitted_differential']!= None and key['performance_fitted_fractional']!= None and key['performance_boxcar_fractional']!= None and key['performance_boxcar_differential']!= None :
                self.insert1(key,skip_duplicates=True)
            else:
                print('couldn''t calculate performance for the following:')
                print(key)
    
