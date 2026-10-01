# RunPod support request — draft for the account owner

**Subject:** Storage recovery request for terminated H100 pod `0tbm05ey3w1xpc` — EEG32 model checkpoint

Hello RunPod Support,

I am trying to recover the trained model checkpoint from my former H100 pod:

- Pod ID: `0tbm05ey3w1xpc`
- Pod name: `b2q-eeg32-h100-fast`
- Region: `US-NE-1`
- Last observed with the checkpoint: August 30, 2026
- File: `/workspace/results/eeg32/best.ckpt`
- Recorded file size: `7,482,838,260` bytes

The pod now returns 404. My account shows zero pods and zero network volumes, and I have no local checkpoint copy. This used pod-local persistent storage, not a detachable network volume. I understand that data export before termination is recommended and that recovery may be unavailable.

Could you investigate whether the original host disk, any retained storage, snapshot or backup still contains this file? If any source remains, please preserve it while you investigate. Please confirm whether recovery is possible and any charges before starting a paid service or provisioning compute.

Recovery priority:

1. `/workspace/results/eeg32/best.ckpt` — the essential trained model.
2. Other files under `/workspace/results/eeg32`, especially checkpoints, Lightning CSV logs, saved predictions and run configuration.
3. Matching preprocessing/cache and subject-index mapping under `/workspace/b2q_cache`, plus any native-preprocessed EEG32 batch used by that run. These help reproduce evaluation, but the checkpoint is the first priority.

If recovery is possible, a secure file export/download is sufficient; I do not need GPU training restarted. Please provide the recovered file's byte size and SHA256 if available. If recovery is impossible, please confirm whether the underlying storage was permanently deleted and whether any backup or retention path exists.

I can provide the complete historical recovery findings and account details through the authenticated support ticket. I am not including credentials in this draft.

Thank you.

---

This draft has **not been sent**. Historical pod/storage facts above come from the user's supplied findings. The original Windows evidence documents are not mounted in this cloud workspace, so this draft does not claim to reproduce their complete contents.
