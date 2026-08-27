# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $14,552  
**Projected savings:** $12,581  (**46%**)

## Savings by lever

| Lever | Savings (USD) |
|---|---|
| Inference (cascade/cache/batch) | $1,212 |
| Purchasing (spot/reserved) | $10,114 |
| Right-size util-lies | $655 |
| Kill idle GPUs | $600 |

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cleanest region: europe-north1
- Cheapest-electricity region: us-east-wa
- Balanced cost/carbon region: us-east-wa

## Unit economics and scope

Inference unit cost falls from **$6.488/1M-token** to
**$1.126/1M-token** (82.6% lower).
The monthly baseline combines the same 30-day inference trace with the workload
GPU bill. Savings buckets are additive and use one shared baseline; no lever is
compounded or counted twice.

## GPU-Util lie: diagnosis and financial meaning

| GPU | Current SKU | GPU-Util | MFU | MBU | Candidate SKU | Gross monthly delta |
|---|---|---:|---:|---:|---|---:|
| gpu-h100-4 | H100 | 98.2% | 19.4% | 20.7% | A100 | $511 |
| gpu-a10g-1 | A10G | 96.9% | 26.8% | 30.2% | L4 | $144 |

`GPU-Util` reports that kernels were active during the sampling window; it does
not show how close useful FLOPs came to hardware peak. A GPU can therefore read
98% busy while warps stall on HBM/I/O, launch many small kernels, synchronize,
or execute poorly batched work. Here `gpu-h100-4` is 98% active but delivers only
about 20% MFU, so NimbusAI pays H100 rates without receiving H100 throughput.
The telemetry is a triage signal, not proof of one root cause: profile kernels,
memory stalls, batch size and input pipeline before moving SKUs. The table shows
**gross** price deltas; validate throughput/SLO parity in a canary before booking
the $655/month right-size saving.

## Recommended action order

1. **P0 - eliminate idle leakage and enforce ownership:** automate shutdown of
   idle instances (up to $600/month) and alert owners. Keep the
   current 92% tag coverage above the 80% chargeback gate.
2. **P1 - apply inference routing/cache/batch guardrails:** this saves
   $1,212/month and 82.6% per token in the measured trace. Roll out
   cascade quality checks first; use batch only for latency-tolerant traffic.
3. **P1 - execute the purchasing plan:** it is the largest lever at
   $10,114/month. Use checkpointed spot for finite jobs and
   3-year reservations only for continuously observed production services;
   revalidate demand before signing commitments.
4. **P2 - right-size utilization lies:** benchmark the two candidates and accept
   only changes that preserve throughput, memory capacity and latency SLOs.

## Extension evidence

### Extension 1 - interruption- and duration-aware purchasing

The policy now uses GPU-specific interruption rates (H100 3%, A100 5%, A10G
10%, L4 8%) and chooses a 1-year versus 3-year term from the planning horizon.
Against the original fixed-5%/always-3-year policy, measured monthly cost changes
from **$15,627 to $15,553**,
an additional **$74/month** saving. The difference is
small because the lower H100 rework offsets higher A10G/L4 interruption risk;
the main value is avoiding a false uniform-risk assumption and mismatched terms.

### Extension 3 - prompt-cache break-even

Team/project/model is used as a conservative proxy for a reusable prefix because
the synthetic CSV has no prefix identifier. It yields 150.0
observed reads per proxy prefix. Break-even is **1.39
reads** for the small-tier write profile and **2.22
reads** for the large tier; both clear the threshold. Production metering should
replace this proxy with cache-key-level reads and include storage/expiry charges.

### Extension 4 - reasoning budget

Reasoning is **8.4% of requests** but
**16.5% of optimized inference cost** and
**94.0% of modeled energy**. Its 80x multiplier
acts on each query and the trace also has a 6x output-token tax, explaining the
disproportion. The requested 10% cap is already met, so it truthfully reroutes
0 requests and saves $0 and 0 Wh. A stricter 5%
sensitivity reroutes 81 requests, saving
**$0.97/day** and
**597 Wh/day**. Recommended rule: enable
reasoning only for complex tasks that fail the small-model confidence/quality
threshold, with a 10% hard budget and human-approved exceptions.

### Extension 5 - carbon-aware scheduling

The 1,789.0 kWh interruptible fleet has these placement
outcomes (electricity component only):

| Criterion | Region | Electricity cost | Carbon |
|---|---|---:|---:|
| Current | us-east-1 | $214.68 | 679.8 kgCO2e |
| Cleanest | europe-north1 | $161.01 | 53.7 kgCO2e |
| Cheapest electricity | us-east-wa | $98.39 | 161.0 kgCO2e |
| Balanced normalized score | us-east-wa | $98.39 | 161.0 kgCO2e |

Moving delay-tolerant jobs from `us-east-1` to `europe-north1` saves
**626.1 kgCO2e (92.1%)**
and $53.67
of modeled electricity. `us-east-wa` is the cost-first choice; `europe-north1`
is the carbon-first choice. Electricity is only part of the cloud GPU price, so
confirm regional GPU rates, capacity, data residency, transfer cost and latency.
Cleanest capacity can be farthest from users; schedule only interruptible batch
work there, not latency-critical chat.

## Decision controls

- Track `$ / 1M-token`, quality pass rate, p95 latency, MFU/MBU and idle hours
  together; a cheaper request that fails quality or SLOs is not a saving.
- Review spot interruption observations weekly and reservation coverage monthly.
- Keep showback now; enable chargeback only while required-tag coverage remains
  at or above 80%, with untagged spend assigned to an explicit remediation pool.

_Figures are June-2026 as-of snapshots; re-baseline before acting._