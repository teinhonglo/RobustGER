import whisper
import re
import sys
sys.path.append(".")

import os, random, copy
import numpy as np
import torch
import pandas as pd
import torchaudio
from tqdm.notebook import tqdm
import collections, json
import editdistance
from whisper.normalizers import EnglishTextNormalizer
from argparse import ArgumentParser
from num2words import num2words
sys.path.append('./local/')
from my_jiwer import wer_embdiff
import fasttext
from huggingface_hub import hf_hub_download
from pathlib import Path
from typing import Optional
from sentencepiece import SentencePieceProcessor, SentencePieceTrainer
from sentence_transformers import SentenceTransformer
from argparse import ArgumentParser
from evaluate import load
from lit_gpt.tokenizer import Tokenizer
from tqdm import tqdm
import time
import argparse


print(whisper.__path__)

eval_wer = load("wer")
normalizer = EnglishTextNormalizer()

checkpoint_dir = Path('checkpoints/meta-llama/Llama-2-7b-hf')
tokenizer = Tokenizer(checkpoint_dir)

sbert_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


def calculate_wer(all_hypo, all_refer):
    return eval_wer.compute(predictions=all_hypo, references=all_refer)

def word_emb_diff(reference, hypothesis):
    output, edit_ops = wer_embdiff(reference, hypothesis)
    ref_words, hypo_words = output.references[0], output.hypotheses[0]

    emb_diffs = []
    for op in edit_ops:
        if op.tag == 'replace':
            ref_word, hypo_word = ref_words[op.src_pos], hypo_words[op.dest_pos]
        elif op.tag == 'delete':
            ref_word, hypo_word = ref_words[op.src_pos], None
        elif op.tag == 'insert':
            ref_word, hypo_word = None, hypo_words[op.dest_pos]
        else:
            continue

        ref_emb = torch.from_numpy(sbert_model.encode([ref_word])[0]) if ref_word else torch.zeros([384])
        hypo_emb = torch.from_numpy(sbert_model.encode([hypo_word])[0]) if hypo_word else torch.zeros([384])

        emb_diff = ref_emb - hypo_emb
        emb_diffs.append(emb_diff)

        # print('word', hypo_emb.mean(), ref_emb.mean(), emb_diff.mean())

    if len(emb_diffs) == 0:
        return torch.zeros([384])
    else:
        return torch.stack(emb_diffs, dim=0).mean(dim=0)

def sent_emb_diff(reference, hypothesis):
    embeddings = sbert_model.encode([reference, hypothesis])
    ref_emb, hypo_emb = torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[1])
    emb_diff = ref_emb - hypo_emb
    # print('sentence', hypo_emb.mean(), ref_emb.mean(), emb_diff.mean())

    return emb_diff

def generate_prompt(input1, input2):
    return (
        f"Below is the best-hypotheses transcribed from speech recognition system. Please try to revise it using the words which are only included into other-hypothesis, and write the response for the true transcription.\n\n### Best-hypothesis:\n{input1}\n\n### Other-hypothesis:\n{input2}\n\n### Response:\n"
    )

def atomic_torch_save(obj, path: str):
    tmp_path = path + ".tmp"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)

def atomic_json_save(obj, path: str):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


# -----------------------------
# Whisper padding-free features
# -----------------------------
# Whisper uses 16kHz audio, hop_length=160 (10ms) -> max 3000 mel frames for 30s.
# Encoder time steps are approximately ceil(mel_frames/2) with max 1500.


def compute_whisper_mel_enc_len_from_n16(n_samples_16k: int) -> tuple[int, int]:
    """Compute (mel_len, enc_len) for Whisper given raw 16k-sample length.

    - mel_len is capped at 3000 (30s)
    - enc_len is capped at 1500
    """
    hop = 160
    mel_len = (int(n_samples_16k) + hop - 1) // hop  # ceil
    mel_len = min(mel_len, 3000)
    enc_len = (mel_len + 1) // 2  # ceil(mel/2)
    enc_len = min(enc_len, 1500)
    enc_len = max(enc_len, 1)
    return int(mel_len), int(enc_len)


def slice_whisper_time_axis(feat: torch.Tensor, enc_len: int) -> torch.Tensor:
    """Slice Whisper encoder features to `enc_len` time steps.

    Supports:
      - [T, D]
      - [1, T, D]
    """
    if feat is None:
        return feat
    if not torch.is_tensor(feat):
        return feat
    if feat.dim() == 3 and feat.size(0) == 1:
        return feat[:, :enc_len, :].contiguous()
    if feat.dim() == 2:
        return feat[:enc_len, :].contiguous()
    return feat

parser = argparse.ArgumentParser()
parser.add_argument("--noisy_wavscp", default="dump/raw/ihm_train_sp/wav.scp", type=str)
parser.add_argument("--clean_wavscp", default="dump/raw/ihm_train_sp/wav.scp", type=str)
parser.add_argument("--text_path", default="dump/raw/ihm_train_sp/text", type=str)
parser.add_argument("--output_path", default="data/dump/ihm_train_sp.pt", type=str)
parser.add_argument("--save_every", type=int, default=200)
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model('large-v2').to(DEVICE)
sbert_model = sbert_model.to(DEVICE)

#f_noisy_wav = open(args.noisy_wavscp, 'r')
#f_clean_wav = open(args.clean_wavscp, 'r')
#f_text = open(args.text_path, 'r')

uttid_list = []
noisy_wav_dict = {}
clean_wav_dict = {}
text_dict = {}

with open(args.noisy_wavscp, 'r') as fn:
    for line in fn.readlines():
        info = line.split()
        noisy_wav_dict[info[0]] = info[1]
        uttid_list.append(info[0])

with open(args.clean_wavscp, 'r') as fn:
    for line in fn.readlines():
        info = line.split()
        clean_wav_dict[info[0]] = info[1]

with open(args.text_path, 'r') as fn:
    for line in fn.readlines():
        info = line.split()
        text_dict[info[0]] = " ".join(info[1:])

pt_file = []
all_hypo, all_refer = [], []

state_path = args.output_path + ".state.json"
chunk_id = 0
start_index = 0

meta = {
    "noisy_wavscp": args.noisy_wavscp,
    "clean_wavscp": args.clean_wavscp,
    "text_path": args.text_path,
    "num_utts": len(uttid_list),
}

if args.resume and os.path.exists(state_path):
    with open(state_path, "r") as f:
        st = json.load(f)

    old_meta = st.get("meta", {})
    if old_meta.get("noisy_wavscp") == meta["noisy_wavscp"] and old_meta.get("num_utts") == meta["num_utts"]:
        start_index = int(st.get("next_index", 0))
        chunk_id = int(st.get("chunk_id", 0))
    else:
        print(f"[WARN] state meta mismatch, restart from 0")

start_index = max(0, min(start_index, len(uttid_list)))
options = whisper.DecodingOptions(language='en', beam_size=50)

with torch.inference_mode():
    for idx in tqdm(range(start_index, len(uttid_list)), total=len(uttid_list) - start_index):
        utt_id = uttid_list[idx]
        audio_path = noisy_wav_dict[utt_id]
        clean_utt_id = utt_id.replace("noise-", "")
        print(utt_id, clean_utt_id)
        clean_audio_path = clean_wav_dict[clean_utt_id]
        
        try:
            gt = text_dict[clean_utt_id]

            # -----------------------------
            # No padding on stored features
            # -----------------------------
            # We still run Whisper with 30s pad/trim, but slice encoder features back to the
            # *effective* length computed from raw waveform length.
            audio_raw = whisper.load_audio(audio_path)
            audio_mel_len, audio_enc_len = compute_whisper_mel_enc_len_from_n16(len(audio_raw))
            audio = whisper.pad_or_trim(audio_raw)  # Whisper expects 30s for fixed-shape mel
            mel = whisper.log_mel_spectrogram(audio).to(model.device)

            texts, confidences, audio_features = whisper.decode_score(model, mel, options)
            audio_features = audio_features.to("cpu").detach()
            audio_features = slice_whisper_time_axis(audio_features, audio_enc_len)

            # clean audio feats
            clean_raw = whisper.load_audio(clean_audio_path)
            clean_mel_len, clean_enc_len = compute_whisper_mel_enc_len_from_n16(len(clean_raw))
            clean_audio = whisper.pad_or_trim(clean_raw)
            clean_mel = whisper.log_mel_spectrogram(clean_audio).to(model.device)
            clean_audio_features = model.encoder(clean_mel.unsqueeze(0))[0].to("cpu").detach()
            clean_audio_features = slice_whisper_time_axis(clean_audio_features, clean_enc_len)

            input, score = [], []
            for text, confidence in zip(texts, confidences):
                if len(input) < 5 and len(text) > 0 and text not in input:
                    input.append(text)
                    score.append(confidence)
            
            if len(input) < 5:
                options = whisper.DecodingOptions(language='en', temperature=1.2)
                for _ in range(5 - len(input)):
                    result = whisper.decode(model, mel, options)
                    text, condidence = result.text, result.avg_logprob
                    if text in input:
                        continue
                    inserted = False
                    for i in range(len(input)):
                        if condidence > score[i]:
                            input.insert(i, text)
                            score.insert(i, condidence)
                            inserted = True
                            break
                    if not inserted:
                        input.append(text)
                        score.append(condidence)

            if len(input) < 5:
                num_to_add = 5 - len(input)
                for _ in range(num_to_add):
                    rand_id = random.randint(0, len(input) - 1)
                    rep_input, rep_score = copy.deepcopy(input[rand_id]), copy.deepcopy(score[rand_id])
                    input.insert(rand_id + 1, rep_input)
                    score.insert(rand_id + 1, rep_score)

            for i in range(len(input)):
                try:
                    text = normalizer(input[i])
                    text = re.sub(r"[-+]?\d*\.?\d+|\d+%?", lambda m: num2words(m.group()), text).replace('%', ' percent')
                except Exception:
                    text = normalizer(input[i])
                    print(f'input exception: {text}')
                input[i] = text if len(text) > 0 else '<UNK>'
                
            try:
                output = normalizer(gt)
                output = re.sub(r"[-+]?\d*\.?\d+|\d+%?", lambda m: num2words(m.group()), output).replace('%', ' percent')
            except Exception:
                output = normalizer(gt)
                print(f'output exception: {output}')
            output = output if len(output) > 0 else '<UNK>'

            cur_wer = calculate_wer([input[0]], [output])

            # calculate emb diff
            we_diffs, se_diffs = [], []
            for i in range(5):
                for j in range(i + 1, 5):
                    we_diffs.append(word_emb_diff(input[i], input[j]))
                    se_diffs.append(sent_emb_diff(input[i], input[j]))

            we_diff = torch.stack(we_diffs, dim=0)      # [10, 384]
            se_diff = torch.stack(se_diffs, dim=0)      # [10, 384]
            emb_diff = torch.cat([we_diff, se_diff], dim=0)     # [20, 384]

            # generate ids
            input1 = input[0] + '.'
            input2 = '. '.join(input[1:]) + '.'

            full_prompt = generate_prompt(input1, input2)
            full_prompt_and_response = full_prompt + output
            encoded_full_prompt = tokenizer.encode(full_prompt, max_length=1024)
            encoded_full_prompt_and_response = tokenizer.encode(full_prompt_and_response, eos=True, max_length=1024)
            

            labels = encoded_full_prompt_and_response.clone()
            labels[: len(encoded_full_prompt)] = -1


            data = {
                "id": utt_id,
                "input_ids": encoded_full_prompt_and_response,
                "input_ids_no_response": encoded_full_prompt,
                "labels": labels,
                "input": input,
                "ground_truth": output,
                "am_score": score,
                "emb_diff": emb_diff,
                "audio_features": audio_features,
                "clean_audio_features": clean_audio_features,
                # length metadata (optional but useful)
                "audio_mel_len": int(audio_mel_len),
                "audio_enc_len": int(audio_enc_len),
                "clean_audio_mel_len": int(clean_mel_len),
                "clean_audio_enc_len": int(clean_enc_len),
            }
            # calculate wer
            idx += 1
            print(f'utterance {idx}: wer = {cur_wer}, confidence = {score[0]}')
            all_hypo.append(input[0])
            all_refer.append(output)
            pt_file.append(data)
            del mel, clean_mel, audio, clean_audio
        except:
            idx += 1
            data = {"id": utt_id, "input_ids": None, "input_ids_no_response": None, "labels": None,
                    "input": None, 'ground_truth': None, "am_score": None, 'emb_diff': None, 'audio_features': None, 
                    'clean_audio_features': None}
            print(f"utterance {utt_id}: Something went wrong.")
            pt_file.append(data)


        if args.save_every and args.save_every > 0 and len(pt_file) >= args.save_every:
            chunk_path = f"{args.output_path}.chunk{chunk_id:06d}.pt"
            atomic_torch_save(pt_file, chunk_path)
            pt_file.clear()
            chunk_id += 1

            state = {"next_index": idx + 1, "chunk_id": chunk_id, "meta": meta}
            atomic_json_save(state, state_path)


if args.save_every and args.save_every > 0:
    if len(pt_file) > 0:
        chunk_path = f"{args.output_path}.chunk{chunk_id:06d}.pt"
        atomic_torch_save(pt_file, chunk_path)
        pt_file.clear()
        chunk_id += 1

    state = {"next_index": len(uttid_list), "chunk_id": chunk_id, "meta": meta}
    atomic_json_save(state, state_path)
else:
    torch.save(pt_file, args.output_path)

if len(all_hypo) > 0 and len(all_refer) > 0:
    all_wer = calculate_wer(all_hypo, all_refer)
    print(f'all wer = {all_wer}')

