import datajoint as dj
import pipeline.lab as lab#, ccf
import pipeline.experiment as experiment
import pipeline.ephys_patch as ephys_patch
import pipeline.ephysanal as ephysanal
from pipeline.pipeline_tools import get_schema_name
schema = dj.schema(get_schema_name('imaging'),locals())


@schema
class Movie(dj.Imported):
    definition = """
    -> experiment.Session
    movie_number                : smallint
    ---
    movie_name                  : varchar(200)          # movie name
    movie_x_size                : double                # (pixels)
    movie_y_size                : double                # (pixels)
    movie_frame_rate            : double                # (Hz)             
    movie_frame_num             : int                   #number of frames        
    movie_start_time            : decimal(10, 4)        # (s) from session start
    movie_pixel_size            : decimal(5,2)          # in microns
    """ 
    
@schema
class MovieFrameTimes(dj.Imported):
    definition = """
    -> Movie
    ---
    frame_times                : longblob              # timing of each frame relative to Session start
    """
    
@schema
class MovieFile(dj.Imported): #MovieFile
    definition = """
    -> Movie 
    movie_file_number         : smallint
    ---
    movie_file_repository     : varchar(200)          # name of the repository where the data are
    movie_file_directory      : varchar(200)          # location of the files  
    movie_file_name           : varchar(100)          # file name
    movie_file_start_frame    : int                   # first frame of this file that belongs to the movie
    movie_file_end_frame      : int                   # last frame of this file that belongs to the movie
    """

@schema
class MotionCorrectionMethod(dj.Lookup): 
    definition = """
    #
    motion_correction_method  :  varchar(30)
    """
    contents = zip(['Matlab','VolPy','Suite2P'])

@schema
class RegisteredMovie(dj.Imported): #MovieFile
    definition = """
    -> Movie
    -> MotionCorrectionMethod
    ---
    registered_movie_mean_image : longblob
    """

@schema
class MotionCorrection(dj.Imported): 
    definition = """
    -> RegisteredMovie
    ---
    motion_corr_vectors         : longblob              # registration vectors   #motion_corr_parameters      : longblob              # probably a dict?  ##motion_corr_metrics         : longblob              # ??
    """


@schema
class ROIType(dj.Lookup): 
    definition = """
    #
    roi_type  :  varchar(30)
    """
    contents = zip(['SpikePursuit','Suite2P'])

@schema
class ROI(dj.Imported): 
# ROI (Region of interest - e.g. cells)
    definition = """
    -> RegisteredMovie
    roi_number                      : int           # roi number (restarts for every session)
    ---
    -> ROIType    
    roi_f                           : longblob      # raw stuff
    roi_dff                         : longblob      #spikepursuit
    roi_f0                          : longblob      #spikepursuit
    roi_spike_indexes               : longblob      #spikepursuit 
    """
    
# =============================================================================
# @schema
# class TextInList(dj.Imported):
#     definition = """
#     index                     : int
#     ---
#     textlist                : longblob              # file names as a list
#     """
# =============================================================================
