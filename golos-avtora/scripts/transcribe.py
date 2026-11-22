#!/usr/bin/env python3
"""Расшифровка устной речи с разметкой пауз (для анализа устное → письменное).
python3 transcribe.py <аудио/видео> <папка_модели_gigaam> > out.txt
Паузы: (.) 0.3–0.7 c, (..) 0.7–1.5 c, (...) >1.5 c. Каждый отрезок между длинными паузами — отдельная строка.
"""
import sys, subprocess, re, os, tempfile
import numpy as np, sherpa_onnx

src, model_dir = sys.argv[1], sys.argv[2]
tmp = tempfile.mktemp(suffix=".wav")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", "16000", tmp], check=True)
st_log = subprocess.run(["ffmpeg", "-i", tmp, "-af", "astats", "-f", "null", "-"], capture_output=True, text=True).stderr
floor = float((re.findall(r"Noise floor dB: (-?[\d.]+)", st_log) or ["-45"])[-1])
thr = min(floor + 10, -20)  # порог тишины адаптивный: шумовой пол + 10 дБ
log = subprocess.run(["ffmpeg", "-i", tmp, "-af", f"silencedetect=noise={thr}dB:d=0.3", "-f", "null", "-"],
                     capture_output=True, text=True).stderr
starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", log)]
ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
import soundfile as sf
audio, sr = sf.read(tmp, dtype="float32")
dur = len(audio) / sr
# речь = промежутки между тишинами
cuts, pos = [], 0.0
for s, e in zip(starts, ends):
    if s > pos:
        cuts.append((pos, s, e - s))
    pos = e
if pos < dur:
    cuts.append((pos, dur, 0))
rec = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
    model=os.path.join(model_dir, "model.int8.onnx"), tokens=os.path.join(model_dir, "tokens.txt"), num_threads=4)
out = []
for a, b, pause in cuts:
    # длинные куски режем по 25 c
    t = a
    while t < b:
        seg = audio[int(t * sr):int(min(b, t + 25) * sr)]
        t += 25
        if len(seg) < sr * 0.2:
            continue
        st = rec.create_stream(); st.accept_waveform(sr, seg); rec.decode_stream(st)
        txt = st.result.text.strip()
        if txt:
            out.append(txt)
    mark = "" if pause < 0.3 else "(.)" if pause < 0.7 else "(..)" if pause < 1.5 else "(...)"
    if out and mark:
        out[-1] += " " + mark
line, lines = [], []
for chunk in out:
    line.append(chunk)
    if chunk.endswith("(..)") or chunk.endswith("(...)"):
        lines.append(" ".join(line)); line = []
if line:
    lines.append(" ".join(line))
print(f"# {os.path.basename(src)} — {dur:.0f} c")
print("\n".join(lines))
