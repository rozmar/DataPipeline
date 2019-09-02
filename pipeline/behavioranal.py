import datajoint as dj
import pandas as pd
import numpy as np
import decimal
import pipeline.lab as lab
import pipeline.experiment as experiment
from pipeline.pipeline_tools import get_schema_name
#from . import get_schema_name
# 

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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
            key['reaction_time'] = float(min(lick_times))  
            key['first_lick_time'] = float(min(lick_times))  + float(gocue_time.values)
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
        
@schema
class SessionRewardRatio(dj.Computed):
    definition = """
    -> experiment.Session
    ---
    session_reward_ratio : longblob # 
    session_reward_ratio_left : longblob # 
    session_reward_ratio_right : longblob # 
    session_maximal_reward_ratio : longblob # 
    """    
    def make(self, key):
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key) * (experiment.SessionBlock() & key))
        if len(df_behaviortrial)>0:
            windowsize = 20
            minperiod = 5
            df_behaviortrial['reward']=0
            df_behaviortrial.loc[df_behaviortrial['outcome'] == 'hit' , 'reward'] = 1
            df_behaviortrial.loc[df_behaviortrial['outcome'] == 'miss' , 'reward'] = 0 
            moving_all = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()
            df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'left' , 'reward'] = np.nan
            moving_right = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()
            df_behaviortrial.loc[df_behaviortrial['outcome'] == 'hit' , 'reward'] = 1
            df_behaviortrial.loc[df_behaviortrial['outcome'] == 'miss' , 'reward'] = 0 
            df_behaviortrial.loc[df_behaviortrial['trial_choice'] == 'right' , 'reward'] = np.nan
            moving_left = df_behaviortrial.reward.rolling(window = windowsize,center = True, min_periods=minperiod).mean()        
            key['session_reward_ratio'] = moving_all
            key['session_reward_ratio_left'] = moving_left
            key['session_reward_ratio_right'] = moving_right
            key['session_maximal_reward_ratio'] = np.array(df_behaviortrial['p_reward_left']+df_behaviortrial['p_reward_right'],dtype=np.float32)
            
            self.insert1(key,skip_duplicates=True)
       
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
    """    
    def make(self, key):
        df_behaviortrial = pd.DataFrame((experiment.BehaviorTrial() & key))
        df_behaviortrial['reward']=0
        df_behaviortrial.loc[df_behaviortrial['outcome'] == 'hit' , 'reward'] = 1
        df_behaviortrial.loc[df_behaviortrial['outcome'] == 'miss' , 'reward'] = 0        
        trialnum = len(df_behaviortrial)
        key['block_reward_ratio'] = -1
        key['block_reward_ratio_first_tertile'] = -1
        key['block_reward_ratio_second_tertile'] = -1
        key['block_reward_ratio_third_tertile'] = -1
        key['block_length'] = trialnum
        if trialnum >10:
            tertilelength = int(np.floor(trialnum /3))
            block_reward_ratio = df_behaviortrial.reward.mean()
            block_reward_ratio_first_tertile = df_behaviortrial.reward[:tertilelength].mean()
            block_reward_ratio_second_tertile = df_behaviortrial.reward[-tertilelength:].mean()
            block_reward_ratio_third_tertile = df_behaviortrial.reward[tertilelength:2*tertilelength].mean()
            key['block_reward_ratio'] = block_reward_ratio
            key['block_reward_ratio_first_tertile'] = block_reward_ratio_first_tertile
            key['block_reward_ratio_second_tertile'] = block_reward_ratio_second_tertile
            key['block_reward_ratio_third_tertile'] = block_reward_ratio_third_tertile
        self.insert1(key,skip_duplicates=True)

@schema
class BlockChoiceRatio(dj.Computed):
    definition = """
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
            trials_back = 15
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
class SubjectFittedChoiceCoefficients(dj.Computed):    
    definition = """
    -> lab.Subject
    ---
    coefficients_rewards_subject : longblob
    coefficients_choices_subject  : longblob
    score_subject :  decimal(8,4)
    """    
    def make(self, key):
        print(key)
        trials_back = 15
        first_session = 8
        label = list()
        data = list()
        df_behaviortrial_all = pd.DataFrame((experiment.BehaviorTrial() & key))
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
    
    
    
    
    
    
    
    
    
        