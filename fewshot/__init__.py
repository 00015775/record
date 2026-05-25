"""Few-shot isolated sign recognition for UzSL.

Pipeline:
    train.py        — episodic ProtoNet fine-tuning of the ST-GCN encoder
    prototypes.py   — build the global prototype database (mean embedding per sign)
    infer.py        — classify .npz files
    live.py         — webcam inference with smoothing + confidence gate
    add_sign.py     — extend the DB with new signs (no retraining)
"""
