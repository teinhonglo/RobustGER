from scipy.io import wavfile
from scipy.signal import correlate, fftconvolve
from scipy.interpolate import interp1d

import librosa
import librosa.display

import os
import numpy as np
import json
import soundfile
from tqdm import tqdm
'''
import argparse
parser = argparse.ArgumentParser()
args = parser.parse_args()
'''
def merge_dict(first_dict, second_dict):
    third_dict = {**first_dict, **second_dict}
    return third_dict

def get_stats(numeric_list, prefix=""):
    # number, mean, standard deviation (std), median, mean absolute deviation
    stats_np = np.array(numeric_list)
    number = len(stats_np)
    
    if number == 0:
        summ = 0.
        mean = 0.
        std = 0.
        median = 0.
        mad = 0.
        maximum = 0.
        minimum = 0.
    else:
        summ = np.sum(stats_np)
        mean = np.mean(stats_np)
        std = np.std(stats_np)
        median = np.median(stats_np)
        mad = np.sum(np.absolute(stats_np - mean)) / number
        maximum = np.max(stats_np)
        minimum = np.min(stats_np)
    
    stats_dict = {  prefix + "number": number, 
                    prefix + "mean": mean, 
                    prefix + "std": std, 
                    prefix + "median": median, 
                    prefix + "mad": mad, 
                    prefix + "summ": summ,
                    prefix + "max": maximum,
                    prefix + "min": minimum
                 }
    return stats_dict
    
    
class AudioModel(object):
    def __init__(self, sample_rate):
        self.sample_rate = sample_rate
    
    def get_f0(self, speech):
        #frame_length=800, win_length=400, hop_length=160, center=False, 
        f0_org_list, voiced_flag, voiced_probs = librosa.pyin(speech, sr=self.sample_rate,
                                             frame_length=800, hop_length=160, center=True, 
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))
        f0_list = np.nan_to_num(f0_org_list)
        f0_stats = get_stats(f0_list, prefix="f0_")
        f0_stats["f0_list"] = f0_list.tolist()
        f0_stats["f0_voiced_probs"] = voiced_probs.tolist()
        # removed unvoiced frames
        f0_nz_list = f0_list[np.nonzero(f0_list)]
        f0_nz_stats = get_stats(f0_nz_list, prefix="f0_nz_")
       
        # mean-var norm 
        #f0_mvn_list = self.__mvn(f0_nz_list)
        #f0_mvn_stats = get_stats(f0_mvn_list, prefix="f0_mvn_")
        
        # min-max norm
        #f0_mmn_list = self.__mmn(f0_nz_list)
        #f0_mmn_stats = get_stats(f0_mmn_list, prefix="f0_mmn_")
        
        # log norm
        #f0_lgn_list = np.log(f0_nz_list)
        #f0_lgn_stats = get_stats(f0_lgn_list, prefix="f0_lgn_")
        
        f0_stats = merge_dict(f0_stats, f0_nz_stats)
        #f0_stats = merge_dict(f0_stats, f0_mvn_stats)
        #f0_stats = merge_dict(f0_stats, f0_mmn_stats)
        #f0_stats = merge_dict(f0_stats, f0_lgn_stats)
        
        return [f0_list, f0_stats]
    
    def get_energy(self, speech):
        rms = librosa.feature.rms(y=speech, frame_length=800, hop_length=160, center=True)
        rms_list = rms.reshape(rms.shape[1],)

        # Replace zeros with a small positive value to avoid issues with log(0)
        rms_list = np.where(rms_list == 0, np.finfo(float).eps, rms_list)
    
        rms_stats = get_stats(rms_list, prefix="energy_")
        rms_stats["energy_rms_list"] = rms_list.tolist()
        
        # mean-var norm
        rms_mvn_list = self.__mvn(rms_list)
        rms_mvn_stats = get_stats(rms_mvn_list, prefix="rms_mvn_")
        
        # min-max norm
        rms_mmn_list = self.__mmn(rms_list)
        rms_mmn_stats = get_stats(rms_list, prefix="rms_mmn_")
        
        # log norm
        rms_lgn_list = np.log(rms_list)
        rms_lgn_stats = get_stats(rms_lgn_list, prefix="rms_lgn_")
        
        rms_stats = merge_dict(rms_stats, rms_mvn_stats)
        rms_stats = merge_dict(rms_stats, rms_mmn_stats)
        rms_stats = merge_dict(rms_stats, rms_lgn_stats)
        
        return [rms_list, rms_stats]

    # mean-var
    def __mvn(self, np_list):
        mean = np.mean(np_list)
        var = np.var(np_list)
        std = np.maximum(np.sqrt(var), 1.0e-20)
        new_np_list = (np_list - mean) / std
        
        return new_np_list
    
    # mean-max
    def __mmn(self, np_list):
        max_v = np.max(np_list)
        min_v = np.min(np_list)
        new_np_list = (np_list - min_v) / (max_v - min_v)
        
        return new_np_list
    
if __name__ == "__main__":
    import soundfile
    wav_path = "/share/corpus/speechocean762/WAVE/SPEAKER2236/022360265.WAV"
    speech, rate = soundfile.read(wav_path)
    #speech, rate = librosa.load(librosa.ex('trumpet'))
    
    audio_model = AudioModel(rate)
    # f0
    f0_list, f0_info = audio_model.get_f0(speech)
    # energy 
    rms_list, rms_info = audio_model.get_energy(speech)

    print(f"f0_info {f0_info}")
    print(f"rms_info {rms_info}")
