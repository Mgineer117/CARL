# CARL: Contraction-Aware Reinforcement Learning

CARL simultaneously learns a tracking policy and a control contraction metric (CCM). By embedding the learned metric directly into the reward formulation, the policy is optimized to minimize the cumulative tracking error within the CCM-induced space, thereby providing formal certificates for the resulting optimal policy.

## Authors

* **Minjae Cho** — *The Grainger College of Engineering, University of Illinois Urbana-Champaign*
* **Hiroyasu Tsukamoto** — *The Grainger College of Engineering, University of Illinois Urbana-Champaign*
* **Huy T. Tran** — *The Grainger College of Engineering, University of Illinois Urbana-Champaign*

Correspondence: Minjae Cho.

## Installation

```bash
git clone https://github.com/Mgineer117/CARL/
cd CARL

conda create -n carl python=3.12
conda activate carl
pip install -r requirements.txt
```

## Training

```bash
python main.py --task car --algo-name carl          # unknown dynamics (learned f and B) — default
python main.py --task car --algo-name carl-exact    # known dynamics (analytic f and B)
```

`--algo-name` is `<base>[-approx|-exact]`, where the base is one of:

| base | description |
|---|---|
| `carl` | this work: a learned contraction metric supplying PPO's reward |
| `ppo` | the same actor and critic on the environment's own reward — the ablation that isolates what the metric contributes |

**Learned dynamics are the default.** A control-affine model is fitted from sampled transitions
first, and every contraction condition is imposed on that learned model. This is the
unknown-dynamics setting and the one worth reporting; `-exact` opts back into the analytic oracle,
and `-approx` names the default explicitly. Note that the simulator always integrates the *true*
dynamics — the learned model reaches only the contraction conditions, so it is the method that is
denied the model, not the world.

Tasks: `car`, `pvtol`, `quadrotor`, `neurallander`. Per-task step budgets live in
`config/<task>.json`; any argument not given on the command line is filled in from there.

## Evaluation

Checkpoints are written into the run's log directory and carry the actor, the critic and the
learned metric, so a run is reproducible from its own logs:

```bash
python evaluate.py --task car --algo-name carl \
    --checkpoint log/train_log/<group>/<run>/best_model.pth
```

`evaluate.py` derives its defaults from the *training* parser, so the architecture it rebuilds
always matches what training produced; adding or renaming a training flag cannot leave it silently
stale. Pass `--actor-dim`, `--metric-dim`, … only when the checkpoint used non-default widths.

The headline metric is **mAUC**: the area under the normalised tracking-error curve
`‖x(t) − x*(t)‖ / ‖x(0) − x*(0)‖`. It begins at 1 by construction, so it measures how quickly error
is contracted independently of how large the initial error happened to be. Episodes that terminate
early are penalised by the fraction of the horizon they completed.

## Repository layout

```
config/       per-task step budgets (JSON)
envs/         the four benchmark systems
  env_base.py   BaseEnv: observation/action contract, Euler integration, reference generation,
                and the offline datasets used by the contraction and dynamics losses
  xyD/ xyzD/    concrete systems, grouped by how many position dimensions they track
policy/
  contraction.py  the CCM conditions, their diagnostics, and the MetricLearner mixin
  autograd.py     batched Jacobian and directional-derivative primitives
  carl.py         CARL = PPO + the metric-shaped reward
  ppo.py          PPO (also the RL half of CARL)
  layers/         actors, the metric network, the learned dynamics model
trainer/      training loops: online (CARL, PPO), offline (the dynamics fit), evaluation
utils/        argument parsing, the algorithm registry, the rollout sampler
```

## The observation contract

Every environment emits one flat vector, parsed by `BasePolicy.trim_state`:

```
[  x_t  |  x*[t : t+H]  |  u*[t]  ]
```

`H = ceil(span / stride)` is the preview horizon, set by `--ref-span` (how far ahead the window
reaches) and `--ref-stride` (waypoint spacing); CARL and PPO default to the whole episode at
stride 2. A window running past the end of the episode holds its final point, so the observation
size is constant. Only the *current* reference control is carried — no network encodes a window of
them, and `u*[t]` is needed solely as the feedforward term below.

The window is fed to the GRU **in reverse**, so the nearest waypoint is the most recent input: read
out through its final hidden state, the encoder would otherwise forget precisely the part that
matters. The stride then trades resolution on the path ahead for reach, and keeps the Jacobian
behind the contraction gain shallow.

The action is the **full control** — the plant applies `clip(u)` and does not re-add `u*`, because
the actor already has. The residual form

```
u = u*[t] + W₂([x′, z]) · tanh( W₁([x′, z]) · e )
```

makes the reference an equilibrium of the closed loop by construction, since the feedback vanishes
with `e`. Position enters only as a displacement — the reference seen from the vehicle in `z`, the
vehicle seen from the reference in `x′` — so no layer receives an absolute location.

## Hardware

CUDA, Apple-Silicon **MPS**, and CPU are all supported; the device is auto-detected and `--device`
overrides it. MPS runs everything this repository needs — including the reverse-over-reverse pass
behind the contraction loss — and agrees with the CPU to float32 precision.
Two MPS gaps are handled internally: `linalg.eigvalsh` is unimplemented there, so the eigenvalue
diagnostics run on the CPU, and `linalg.svd`, used only to build the annihilator of a *learned* `B`,
falls back to the CPU automatically.

## Logging

Weights & Biases, TensorBoard and CSV are written together, into the run's log directory alongside
its checkpoints. Every training key is `<Algorithm>/<group>/<name>`, with five groups and nothing
logged twice:

| group | contents | example |
|---|---|---|
| `loss/` | quantities being minimised | `CARL/loss/actor`, `CARL/loss/critic`, `CARL/loss/total` |
| `metric/` | the contraction conditions | `CARL/metric/cu`, `metric/c1`, `metric/c2`, `metric/overshoot`, `metric/contraction` |
| `analytics/` | diagnostics that are read but not optimised | `CARL/analytics/kl_divergence`, `analytics/entropy`, `analytics/clip_fraction`, `analytics/reward` |
| `grad/` | gradient norms after clipping | `CARL/grad/actor`, `grad/critic`, `grad/metric` |
| `lr/` | current learning rates | `CARL/lr/rl` (one rate for actor and critic), `lr/metric` |

CARL is exactly PPO plus the `metric/` group, so the two are directly comparable. Diagnostics are
deliberately *not* filed under `loss/`: the KL, the entropy and the clip fraction are quantities to
inspect, not quantities being minimised.

Evaluation writes `eval/<name>` with a matching `eval/<name>_ci95`: `mauc`, `m2auc`, `return`,
`control_effort`, `inference_time`, `episode_len`, `overshoot`, `contraction_rate`.

The schema is pinned — group names, snake_case leaves, no duplicated quantity, and a
per-algorithm key budget — so logging cannot quietly creep back up.

## Robot demonstration

The hardware experiment is available [here](https://www.youtube.com/watch?v=gnrHGQBFitA).
