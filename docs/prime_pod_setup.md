# Prime Intellect — Pod Setup & Smoke Test

End-to-end runbook for provisioning a GPU pod on Prime Intellect, getting this repo onto it, running the smoke test, and pulling logs back. Captures what was verified working on 2026-05-05 (DataCrunch A100 80GB, image `ubuntu_22_cuda_12`).

---

## 1. One-time prerequisites (laptop)

### 1.1 Install the Prime CLI

```bash
uv tool install prime
prime login                              # opens browser, authenticate
prime config view                        # confirm API key + team are set
```

### 1.2 Have an SSH keypair

```bash
ls -la ~/.ssh/id_ed25519*                # should show id_ed25519 and id_ed25519.pub
# If missing:
ssh-keygen -t ed25519 -C "your@email"    # accept defaults
```

### 1.3 Register the public key in two places

**A. Local Prime CLI config** (used when the CLI SSHes for you):

```bash
prime config set-ssh-key-path ~/.ssh/id_ed25519
prime config view | grep "SSH Key Path"
```

**B. Prime account UI** (this is what gets *injected into pods at boot*):

1. [https://app.primeintellect.ai](https://app.primeintellect.ai) → account menu → **Settings → SSH Keys → Add SSH Key**
2. Paste the contents of `~/.ssh/id_ed25519.pub` (whole single line). Quick copy:
  ```bash
   pbcopy < ~/.ssh/id_ed25519.pub
  ```
3. Verify the fingerprint Prime displays matches:
  ```bash
   ssh-keygen -lf ~/.ssh/id_ed25519.pub
  ```

> Both A and B must be set. A only configures the CLI; B is what makes pods actually accept your key. Without B, you get `Permission denied (publickey)` no matter what provider you pick.

---

## 2. Provisioning a pod

### 2.1 Find available GPU

```bash
prime availability gpu-types                                          # see all types
COLUMNS=240 prime availability list --gpu-type A100_80GB --gpu-count 1   # filter

# find specific gpu
prime availability list --gpu-type A100_80GB --gpu-count 1
```

For Qwen3-8B inference (this project), any of:

- **RTX4090 24GB** (~$0.71/hr) — fits FP16 8B with KV-cache room
- **A100 40GB** (~$1.99/hr) — comfortable, Lambda is most reliable
- **L40 / L40S 48GB** (~$0.82–$1.00/hr) — 48 GB headroom, often available
- **A100 80GB** (~$1.79/hr) — overkill but very reliable

### 2.2 Create the pod (non-interactive, AI-friendly)

```bash
prime pods create --plain -y \
  --id <id-from-availability-list> \
  --name qwen8b \
  --image ubuntu_22_cuda_12 \
  --disk-size 200
```

Flags:

- `--plain` — terse output (skip TUI rendering)
- `-y` — skip confirmation prompts
- `--image ubuntu_22_cuda_12` — universally supported across providers; lets `uv` install everything fresh. Other images (e.g. `cuda_12_6_pytorch_2_7`) aren't supported on every provider.
- `--disk-size 200` — sufficient for HF cache + dep wheels; smaller disk also installs faster than the 960 GB default.

### 2.3 Wait for SSH endpoint

```bash
prime pods status <pod-id>
```

Look for `Status: ACTIVE`, `Installation Status: FINISHED`, and **non-`N/A` IP and SSH** fields. Timing is provider-dependent:

- DataCrunch / Lambda / MassedCompute: ~1–3 min
- RunPod: usually 2–10 min, sometimes never (port-forward lag)
- Crusoe: 5–15 min

If still `IP: N/A` after 10 min, terminate and try a different `--id`.

### 2.4 Test SSH

```bash
ssh -o StrictHostKeyChecking=accept-new \
    root@<pod-ip> \
    'whoami; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'
```

Expected: prints `root` and your GPU. If `Permission denied (publickey)`, the public key was missing from your Prime account at pod-create time → recreate after fixing step 1.3 B.

> Username is provider-dependent: `root` on DataCrunch/Crusoe, `ubuntu` on MassedCompute/RunPod. Use whatever `prime pods status` shows in the SSH field.

---

## 3. Bootstrap the pod (one-time per pod)

### 3.1 Install uv on the pod

```bash
ssh root@<pod-ip> 'curl -LsSf https://astral.sh/uv/install.sh | sh > /dev/null && ~/.local/bin/uv --version'
```

### 3.2 Get the repo onto the pod

**Option A — rsync from laptop** (fastest, doesn't require committing):

```bash
rsync -avz \
  --exclude='.venv/' --exclude='__pycache__/' \
  --exclude='results/' --exclude='outputs/' \
  --exclude='.git/' --exclude='.DS_Store' \
  --exclude='.tmp.driveupload/' --exclude='docs/' --exclude='*.log' \
  ./ root@<pod-ip>:nlp-final-project/janys-setup-intellect/
```

**Option B — git clone** (cleaner; requires the branch pushed to GitHub):

```bash
ssh root@<pod-ip> '
  git clone -b janys-setup-intellect \
    https://<USER>:<PAT>@github.com/janys0v0/nlp-final-project.git \
    ~/nlp-final-project/janys-setup-intellect
'
```

### 3.3 Install dependencies

```bash
ssh root@<pod-ip> '
  export PATH="$HOME/.local/bin:$PATH"
  cd nlp-final-project/janys-setup-intellect
  uv sync --extra gpu
'
```

- `--extra gpu` adds Linux-only `bitsandbytes` for 4/8-bit quantization.
- First sync takes ~3–5 min (downloads torch ~700 MB + transformers + datasets + verifiers).
- Subsequent `uv sync` is near-instant (cached in `~/.cache/uv/`).

### 3.4 (Optional) Hugging Face auth

Only needed for gated models or to avoid HF rate limits:

```bash
ssh root@<pod-ip> '
  export PATH="$HOME/.local/bin:$PATH"
  cd nlp-final-project/janys-setup-intellect
  uv run huggingface-cli login                   # paste token interactively
'
```

---

## 4. Run the smoke test

```bash
ssh -t root@<pod-ip> '
  export PATH="$HOME/.local/bin:$PATH"
  cd nlp-final-project/janys-setup-intellect
  mkdir -p results/smoke
  uv run python scripts/smoke_generate.py \
    --max-new-tokens 64 \
    --log-file results/smoke/run-test.log
'
```

What the script does:

- Loads `Qwen/Qwen3-0.6B` (~1.5 GB download on first run, cached after)
- Auto-detects device (CUDA on the pod), uses bf16
- Runs three short prompts, prints throughput
- Tees stdout to the `--log-file` path

Expected throughput on A100: ~20–30 tok/s for the 0.6B model.

---

## 5. Pull results back

```bash
mkdir -p outputs/smoke
rsync -avz \
  root@<pod-ip>:nlp-final-project/janys-setup-intellect/results/smoke/ \
  outputs/smoke/
```

The pod's `results/` is gitignored on the pod; your local `outputs/` is gitignored on the laptop. Logs flow back without polluting the repo.

---

## 6. Tear down (important — billing continues until terminated)

```bash
prime pods terminate -y <pod-id>
prime pods list                          # confirm it's gone
```

A100 80GB at ~$1.79/hr × 1 hour idle = $1.79 wasted. Always terminate when done.

---

## 7. Helper scripts in this repo

- `bin/ssh_pod.sh <pod-id>` — wrapper around `prime pods ssh` (interactive shell on a pod by ID)
- `bin/smoke_on_pod.sh <ssh-dest> [-- args]` — laptop-driven runner: pulls the latest, runs the smoke test, rsyncs the log back. Assumes the repo is already cloned on the pod and `uv` is installed.

---

## 8. Provider notes (empirical, 2026-05-05)


| Provider          | Pros                                                      | Cons                                                                                         |
| ----------------- | --------------------------------------------------------- | -------------------------------------------------------------------------------------------- |
| **DataCrunch**    | SSH endpoint published in ~1 min; root user; no surprises | Limited GPU types vs. RunPod                                                                 |
| **Lambda Labs**   | Most reliable provisioning; key injection always works    | A100 40GB at $1.99/hr is pricey                                                              |
| **MassedCompute** | Cheap (L40 $0.86, A6000 $0.54)                            | SSH key injection failed on `ubuntu` user — recheck after registering public key in Prime UI |
| **RunPod**        | Good selection of consumer GPUs (RTX 4090 $0.71)          | Port-forward proxy can take 10+ min to publish IP/SSH; sometimes never                       |
| **Crusoe**        | Cheap H100s                                               | `cuda_12_6_pytorch_2_7` image install can take 15+ min; got stuck on first attempt           |


If a pod is stuck at `IP: N/A` past 10 min: terminate and try a different `--id` (different provider). Recreating with the same image often works on the next provider.

---

## 9. Common failure modes


| Symptom                                   | Cause                                                            | Fix                                                                                   |
| ----------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| `IP: N/A` for 10+ min                     | Provider→Prime endpoint propagation lag                          | Terminate, try a different `--id`                                                     |
| `Permission denied (publickey)`           | Public key not registered in Prime account UI at pod-create time | Add key (step 1.3 B), terminate, recreate (key injection is at boot, not retroactive) |
| `Provider X is not supported for image Y` | Provider doesn't carry that image                                | Use `ubuntu_22_cuda_12` (universal)                                                   |
| `uv sync` fails on bitsandbytes           | You're on macOS or skipped `--extra gpu`                         | On pods always pass `--extra gpu`; locally bitsandbytes is excluded automatically     |
| HF rate-limited / unauthenticated warning | No HF token                                                      | Run `huggingface-cli login` once on the pod                                           |
| SSH drops kill running script             | Long script running in foreground                                | Wrap in `tmux new -s exp '<cmd>'` so it survives disconnects                          |


---

## 10. Cost discipline checklist

- Always `prime pods terminate -y <id>` when done.
- Smoke-test with `--max-problems 1` / `--max-new-tokens 64` before launching real runs.
- Use `tmux` for any run >5 minutes so SSH drops don't waste compute.
- Pull `results/` back via rsync before terminating — pod disk is ephemeral.
- Track pod IDs + costs in `results/INDEX.md` so the team knows what's been spent.

