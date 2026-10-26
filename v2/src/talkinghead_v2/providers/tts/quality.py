"""Local ASR-based intelligibility evidence, never a substitute for human ratings."""

import argparse
import json
import os
from pathlib import Path
import re
import unicodedata

from talkinghead_v2.assets.artifacts import sha256_file


def words(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", "".join(c if c.isalnum() else " " for c in normalized)).strip().split()


def word_errors(reference: str, hypothesis: str) -> dict:
    expected, actual = words(reference), words(hypothesis)
    # Each cell is (distance, substitutions, deletions, insertions). Prefer a
    # diagonal alignment, then deletion, then insertion on equal-cost ties.
    previous = [(i, 0, 0, i) for i in range(len(actual)+1)]
    for row, token in enumerate(expected, 1):
        current = [(row, 0, row, 0)]
        for col, candidate in enumerate(actual, 1):
            diagonal = previous[col-1]
            substitution = token != candidate
            choices = [(diagonal[0]+substitution, diagonal[1]+substitution, diagonal[2], diagonal[3]),
                       (previous[col][0]+1, previous[col][1], previous[col][2]+1, previous[col][3]),
                       (current[col-1][0]+1, current[col-1][1], current[col-1][2], current[col-1][3]+1)]
            current.append(min(choices, key=lambda x:x[0]))
        previous = current
    distance, substitutions, deletions, insertions = previous[-1]
    return {"reference_words": len(expected), "substitutions": substitutions, "deletions": deletions,
            "insertions": insertions, "errors": distance, "wer": distance/len(expected) if expected else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--texts", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    for entry in manifest["files"]:
        path = args.evaluator / entry["path"]
        if path.stat().st_size != entry["size"] or sha256_file(path) != entry["sha256"]:
            raise ValueError("evaluator identity mismatch")
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import librosa
    import soundfile as sf
    import torch
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    processor = WhisperProcessor.from_pretrained(str(args.evaluator), local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(str(args.evaluator), local_files_only=True,
        use_safetensors=True, trust_remote_code=False, dtype=torch.float32).to("cpu").eval()
    texts = json.loads(args.texts.read_text())
    rows = []
    for provider, root, prefix in (("official_bf16", args.reference, "reference"), ("native_f16_streaming", args.native, "native")):
        for index, request in enumerate(texts):
            path = root / f"{prefix}-{index:03}.wav"
            audio, rate = sf.read(path, dtype="float32")
            audio = librosa.resample(audio, orig_sr=rate, target_sr=16000)
            features = processor(audio, sampling_rate=16000, return_tensors="pt",
                                 return_attention_mask=True, truncation=False, padding="longest")
            with torch.inference_mode():
                tokens = model.generate(input_features=features.input_features, attention_mask=features.attention_mask,
                                        return_timestamps=True, do_sample=False)
            transcript = processor.batch_decode(tokens, skip_special_tokens=True)[0]
            rows.append({"provider": provider, "id": request["id"], "wave_sha256": sha256_file(path),
                         "transcript": transcript, **word_errors(request["text"], transcript)})
    aggregates = {}
    for provider in ("official_bf16", "native_f16_streaming"):
        selected = [row for row in rows if row["provider"] == provider]
        total, errors = sum(r["reference_words"] for r in selected), sum(r["errors"] for r in selected)
        aggregates[provider] = {"reference_words": total, "errors": errors, "wer": errors/total}
    report = {"schema_version": 1, "evidence_category": "local_observation", "evaluator": manifest,
              "device": "cpu", "dtype": "float32", "torch_threads": 2, "requests": rows, "aggregates": aggregates,
              "human_listening": "pending", "human_cloning_qualified": False,
              "normalization": "NFKC, casefold, non-alphanumeric to spaces; no number/abbreviation or accent equivalence",
              "limitations": ["ASR errors and legitimate verbalizations count in strict WER; inspect rows",
                              "Five synthetic-voice text cases; no independent human or population-quality conclusion"]}
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(aggregates))


if __name__ == "__main__":
    main()
