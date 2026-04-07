"""
TFT Attention & Interpretation Extraction

TFT'nin interpret_output() ciktisini cikarir:
1. Variable Selection Network weights — hangi feature ne kadar onemli
2. Attention weights — hangi gecmis zaman adimi tahmini etkiliyor
3. Entity bazli (rota x kabin) disaggregation

Cikti: reports/tft_interpretation.json
"""
import os
import json
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore")

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_DIR, "data", "models")
REPORTS_DIR = os.path.join(PROJECT_DIR, "reports")

CHECKPOINT = os.path.join(MODELS_DIR, "tft_full_checkpoint.ckpt")
DATASET_PT = os.path.join(MODELS_DIR, "tft_training_dataset.pt")


def main():
    print("=" * 60)
    print("TFT Attention & Interpretation Extraction")
    print("=" * 60)

    # 1. Load model & dataset
    print("\n[1/4] Loading TFT model and dataset...")
    from pytorch_forecasting import TemporalFusionTransformer

    # torchmetrics _apply CUDA hatasi bypass — monkey patch
    import torchmetrics
    _orig_apply = torchmetrics.Metric._apply
    def _safe_apply(self, fn, *args, **kwargs):
        try:
            return _orig_apply(self, fn, *args, **kwargs)
        except (AssertionError, RuntimeError):
            # CUDA init hatasi — CPU'da kaldir
            return self
    torchmetrics.Metric._apply = _safe_apply

    model = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT, map_location="cpu")
    model.eval()
    model.freeze()

    # Monkey patch'i geri al
    torchmetrics.Metric._apply = _orig_apply
    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    dataset = torch.load(DATASET_PT, weights_only=False)
    print(f"  Dataset loaded: {len(dataset)} samples")

    # 2. Create dataloader (small batch for interpretation)
    print("\n[2/4] Running interpretation predictions...")
    dataloader = dataset.to_dataloader(train=False, batch_size=64, num_workers=0)

    # Collect interpretation across batches
    all_attention = []
    all_static_vars = []
    all_encoder_vars = []
    all_decoder_vars = []
    entity_ids = []

    batch_count = 0
    max_batches = 50  # enough for representative sample

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break

            try:
                out = model(x)
                interp = model.interpret_output(out, reduction="none")

                # Attention weights: (batch, n_heads, decoder_length, encoder_length)
                if "attention" in interp:
                    att = interp["attention"]
                    if att.ndim == 4:
                        att = att.mean(dim=1)  # average over heads
                    all_attention.append(att.cpu().numpy())

                # Variable importance
                if "static_variables" in interp:
                    all_static_vars.append(interp["static_variables"].cpu().numpy())
                if "encoder_variables" in interp:
                    all_encoder_vars.append(interp["encoder_variables"].cpu().numpy())
                if "decoder_variables" in interp:
                    all_decoder_vars.append(interp["decoder_variables"].cpu().numpy())

                batch_count += 1
                if batch_count % 10 == 0:
                    print(f"  Processed {batch_count}/{max_batches} batches...")

            except Exception as e:
                print(f"  Batch {batch_idx} error: {e}")
                continue

    print(f"  Completed: {batch_count} batches processed")

    # 3. Aggregate results
    print("\n[3/4] Aggregating interpretation results...")

    result = {
        "_meta": {
            "description": "TFT Attention & Variable Importance — interpret_output() sonuclari",
            "source": "tft_full_checkpoint.ckpt + tft_training_dataset.pt",
            "n_batches": batch_count,
        }
    }

    # Variable importance — feature names from model
    # Static variables
    if all_static_vars:
        static = np.concatenate(all_static_vars, axis=0)
        static_mean = np.mean(static, axis=0)
        static_names = model.hparams.get("static_categoricals", []) + model.hparams.get("static_reals", [])
        if not static_names:
            static_names = [f"static_{i}" for i in range(len(static_mean))]
        static_importance = sorted(
            [{"feature": n, "importance": round(float(v), 4)} for n, v in zip(static_names, static_mean)],
            key=lambda x: x["importance"], reverse=True
        )
        result["static_variable_importance"] = static_importance
        print(f"  Static variables: {len(static_importance)}")
        for item in static_importance[:5]:
            print(f"    {item['feature']}: {item['importance']:.4f}")

    # Encoder variables (time-varying known + unknown in encoder)
    if all_encoder_vars:
        enc = np.concatenate(all_encoder_vars, axis=0)
        enc_mean = np.mean(enc, axis=0)
        enc_names = (model.hparams.get("time_varying_categoricals_encoder", []) +
                     model.hparams.get("time_varying_reals_encoder", []))
        if not enc_names:
            enc_names = [f"encoder_{i}" for i in range(len(enc_mean))]
        enc_importance = sorted(
            [{"feature": n, "importance": round(float(v), 4)} for n, v in zip(enc_names, enc_mean)],
            key=lambda x: x["importance"], reverse=True
        )
        result["encoder_variable_importance"] = enc_importance
        print(f"  Encoder variables: {len(enc_importance)}")
        for item in enc_importance[:5]:
            print(f"    {item['feature']}: {item['importance']:.4f}")

    # Decoder variables
    if all_decoder_vars:
        dec = np.concatenate(all_decoder_vars, axis=0)
        dec_mean = np.mean(dec, axis=0)
        dec_names = (model.hparams.get("time_varying_categoricals_decoder", []) +
                     model.hparams.get("time_varying_reals_decoder", []))
        if not dec_names:
            dec_names = [f"decoder_{i}" for i in range(len(dec_mean))]
        dec_importance = sorted(
            [{"feature": n, "importance": round(float(v), 4)} for n, v in zip(dec_names, dec_mean)],
            key=lambda x: x["importance"], reverse=True
        )
        result["decoder_variable_importance"] = dec_importance
        print(f"  Decoder variables: {len(dec_importance)}")
        for item in dec_importance[:5]:
            print(f"    {item['feature']}: {item['importance']:.4f}")

    # Attention pattern — average across all samples
    if all_attention:
        att_all = np.concatenate(all_attention, axis=0)
        # Average attention pattern: (decoder_length, encoder_length)
        att_mean = np.mean(att_all, axis=0)
        # For each decoder step, which encoder steps matter most?
        # Summarize: average over decoder steps → encoder step importance
        encoder_step_importance = np.mean(att_mean, axis=0)
        if encoder_step_importance.ndim == 0:
            encoder_step_importance = np.array([float(encoder_step_importance)])
        encoder_step_importance = encoder_step_importance.flatten().tolist()
        result["attention"] = {
            "description": "Ortalama attention: her encoder (gecmis) adiminin decoder (gelecek) uzerindeki etkisi",
            "encoder_step_importance": [round(v, 4) for v in encoder_step_importance],
            "n_encoder_steps": len(encoder_step_importance),
            "n_decoder_steps": att_mean.shape[0] if att_mean.ndim == 2 else None,
            "pattern_summary": {
                "recent_30d_weight": round(float(np.sum(encoder_step_importance[-30:])) / max(np.sum(encoder_step_importance), 1e-8), 4),
                "mid_30d_weight": round(float(np.sum(encoder_step_importance[-60:-30])) / max(np.sum(encoder_step_importance), 1e-8), 4) if len(encoder_step_importance) > 60 else None,
                "early_weight": round(float(np.sum(encoder_step_importance[:-60])) / max(np.sum(encoder_step_importance), 1e-8), 4) if len(encoder_step_importance) > 60 else None,
            }
        }
        print(f"  Attention: {att_mean.shape}")
        ps = result["attention"]["pattern_summary"]
        print(f"    Recent 30d weight: {ps['recent_30d_weight']:.1%}")
        if ps.get("mid_30d_weight"):
            print(f"    Mid 30d weight: {ps['mid_30d_weight']:.1%}")
        if ps.get("early_weight"):
            print(f"    Early weight: {ps['early_weight']:.1%}")

    # 4. Save
    print("\n[4/4] Saving results...")
    out_path = os.path.join(REPORTS_DIR, "tft_interpretation.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

    # Summary for API
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "static_variable_importance" in result:
        print("\nTop Static Features (rota/kabin ozellikleri):")
        for item in result["static_variable_importance"][:5]:
            print(f"  {item['feature']:30s} {item['importance']:.4f}")
    if "encoder_variable_importance" in result:
        print("\nTop Encoder Features (zaman serisi girdileri):")
        for item in result["encoder_variable_importance"][:5]:
            print(f"  {item['feature']:30s} {item['importance']:.4f}")
    if "attention" in result:
        ps = result["attention"]["pattern_summary"]
        print(f"\nAttention Pattern:")
        print(f"  Son 30 gun etkisi: {ps['recent_30d_weight']:.1%}")


if __name__ == "__main__":
    main()
