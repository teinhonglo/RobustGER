import os
import json
import soundfile
from tqdm import tqdm
from whisperx_models import SpeechModel
from audio_models import AudioModel
from nlp_models import NlpModel
import numpy as np
import sys
import wave
from whisper.normalizers import EnglishTextNormalizer
import string
import jiwer
import torch
import argparse
import re

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir",
                    default="/share/nas165/teinhonglo/AcousticModel/2020AESRC/s5/data/cv_56",
                    type=str)

parser.add_argument("--model_tag",
                    default="large-v2",
                    type=str)

parser.add_argument("--model_name",
                    default="large-v2",
                    type=str)
                    
parser.add_argument("--sample_rate",
                    default=16000,
                    type=int)

parser.add_argument("--device",
                    default="cuda",
                    type=str)

parser.add_argument("--language",
                    default="en",
                    type=str) 

parser.add_argument("--json_name", default="train.tsv")

parser.add_argument("--condition_on_previous_text", action="store_true", help="if True, provide the previous output of the model as a prompt for the next window; disabling may make the text inconsistent across windows, but the model becomes less prone to getting stuck in a failure loop")
parser.add_argument("--suppress_numeric_tokens", action="store_true")
parser.add_argument("--suppress_punc_tokens", action="store_true")
parser.add_argument("--stt_only", action="store_true")
parser.add_argument("--use_prompt", action="store_true")
args = parser.parse_args()

data_dir = args.data_dir
model_name = args.model_name
model_tag = args.model_tag
sample_rate = args.sample_rate
language = args.language
condition_on_previous_text = args.condition_on_previous_text
device = args.device
stt_only = args.stt_only
use_prompt = args.use_prompt
json_name = args.json_name

output_dir = os.path.join(data_dir, model_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

basename = " ".join(json_name.split(".")[0:-1])
output_json_path = f"{output_dir}/{basename}_raw.json"
tag = args.model_tag
wavscp_dict = {}
text_dict = {}
utt_list = []

speech_model = SpeechModel(
                    tag=model_tag, 
                    device=device, 
                    language=language, 
                    condition_on_previous_text=condition_on_previous_text, 
                    args=args
                )
audio_model = AudioModel(sample_rate)

normalizer = EnglishTextNormalizer()
nlp_model = NlpModel()

with open(os.path.join(data_dir, json_name), "r") as fn:
    data_info = json.load(fn)

utt_list = data_info["id"]

import pprint
pp = pprint.PrettyPrinter(indent=4)

if os.path.isfile(output_json_path):
    with open(output_json_path, "r") as fn:
        all_info = json.load(fn)
else:
    all_info = {}

for i, uttid in tqdm(enumerate(utt_list)):
    if uttid in all_info:
        print(f"Feature extraction has already been completed for {uttid}")
        continue
    
    wav_path = data_info["audio"][i]
    text_prompt = data_info["text"][i]
    # Confirm the sampling rate is equal to that of the training corpus.
    # If not, you need to resample the audio data before inputting to speech2text
    speech, rate = soundfile.read(wav_path)
    assert rate == sample_rate
    total_duration = speech.shape[0] / rate

    # audio feature
    if not stt_only:
        try:
            _, f0_info = audio_model.get_f0(speech)
            _, energy_info = audio_model.get_energy(speech)
        except Exception as e:
            print(f"Raw audio extraction went errors for {uttid} {wav_path} {e}")
            continue

    # fluency feature and confidence feature
    try:
        if use_prompt:
            text_result, ctm_results = speech_model.recog(wav_path, text_prompt=text_prompt)
        else:
            text_result, ctm_results = speech_model.recog(wav_path)
    except Exception as e:
        print("No audio are reconized or aligned for", uttid, wav_path, ":", e)
        text_result = ["", ""]
        ctm_results = ["", ""]
        continue
    
    text, text_norm = text_result
    word_ctm_info, phn_ctm_info = ctm_results
    
    sil_feats_info, response_duration = speech_model.sil_feats(word_ctm_info, total_duration)
    word_feats_info, response_duration = speech_model.word_feats(word_ctm_info, total_duration)
    phone_feats_info, response_duration = speech_model.phone_feats(phn_ctm_info, total_duration)
    vp_feats_info = nlp_model.vocab_profile_feats(text_norm)

    all_info[uttid] = { "stt": text, "prompt": text_prompt,
                        "wav_path": wav_path, 
                        "word_ctm": word_ctm_info, "ctm": phn_ctm_info, 
                        "feats": {  **f0_info, **energy_info, 
                                **sil_feats_info, **word_feats_info,
                                **phone_feats_info, **vp_feats_info,
                                "total_duration": total_duration,
                                "response_duration": response_duration
                                }
                        }
    
    if i % 1000 == 0:
        print(all_info[uttid])

with open(output_json_path, "w") as fn:
    json.dump(all_info, fn, indent=4, ensure_ascii=False, cls=NpEncoder)

