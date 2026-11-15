# Fine-tuning spec — `notebooks/finetune.ipynb` (M3)

Goal: fine-tune the REAL-TSE WeSep BSRNN (causal checkpoint `spk_emb_causal_100`) to
**remove** the target speaker instead of extracting them.

## Notebook cells

1. **Setup** — mount Google Drive; set `HUSH_DATA=/content/drive/MyDrive/HUSH-data`;
   clone this repo and `wesep-real-tse`; `pip install -r requirements-colab.txt`
   and WeSpeaker at the commit the wesep-real-tse README pins. Check the GPU (`nvidia-smi`).
2. **Load model** — build BSRNN from `checkpoints/pretrained/spk_emb_causal_100/config.yaml`,
   load `avg_model.pt`. Freeze the speaker encoder. Register a forward hook on the
   mask-estimation module so the mask `M` is available.
3. **Data** — `MixDataset(target="mikolaj", exclude_sessions=[<test session>])`,
   DataLoader batch 8, 3 s items, `num_workers=2`.
4. **Loss** — with the mask `M` and STFTs of mix X, target S, others R:
   ```
   Y_rest = iSTFT((1 - M) * STFT(R))       # what happens to everyone else
   Y_tgt  = iSTFT((1 - M) * STFT(S))       # what's left of the target
   loss = -SI_SDR(Y_rest, R) + beta * energy(Y_tgt) / (energy(S) + eps)     beta = 3
   ```
   Items with no target: second term = 0. Items with target only: first term = 0.
   If the mask can't be hooked, fall back to: removed = mix - model_output,
   loss = -SI_SDR(removed, R) (pure domain adaptation, no asymmetry).
5. **Train** — Adam lr 1e-4, mixed precision (`torch.autocast`), grad clip 5.
   Time the first 50 steps and print the projected time for 3,000 steps.
   Every 500 steps: save `checkpoints/ours/step_<n>.pt` to Drive and append
   `step,loss,val_si_sdr` to `checkpoints/ours/log.csv` (val = 20 fixed items, excluded session).
   Must resume from the latest checkpoint if Colab disconnects.
6. **Evaluate** — run `score.py --ckpt checkpoints/ours/step_<best>.pt --name finetuned`.
7. **Optional** — repeat with beta in {1, 10} for a small ablation table.

Expected compute: Colab T4 or better; BSRNN (6 recurrent blocks) won't train on a laptop CPU.
