"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing, m4_allocation

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    r4 = m4_allocation.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "cleanest_region": r3["carbon_aware"]["cleanest_region"],
        "cheapest_region": r3["carbon_aware"]["cheapest_region"],
        "balanced_region": r3["carbon_aware"]["balanced_region"],
    }

    lie_rows = []
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        monthly_delta = max(
            0.0,
            num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"]),
        ) * 24 * DAYS
        lie_rows.append(
            f"| {lie['gpu_id']} | {cur} | {lie['gpu_util_pct']:.1f}% | "
            f"{lie['mfu']:.1%} | {lie['mbu']:.1%} | {tgt} | ${monthly_delta:,.0f} |"
        )

    reasoning = r2["reasoning"]
    reasoning_5pct = r2["reasoning_5pct"]
    carbon = r3["carbon_aware"]
    current_region = carbon["regions"]["us-east-1"]
    clean_region = carbon["cleanest_region"]
    clean_values = carbon["regions"][clean_region]
    cheapest_region = carbon["cheapest_region"]
    cheapest_values = carbon["regions"][cheapest_region]
    cache_small = r2["cache_analysis"]["small"]
    cache_large = r2["cache_analysis"]["large"]
    additional = f"""
## Unit economics and scope

Inference unit cost falls from **${r2['baseline_per_m']:.3f}/1M-token** to
**${r2['optimized_per_m']:.3f}/1M-token** ({r2['savings_pct']:.1f}% lower).
The monthly baseline combines the same 30-day inference trace with the workload
GPU bill. Savings buckets are additive and use one shared baseline; no lever is
compounded or counted twice.

## GPU-Util lie: diagnosis and financial meaning

| GPU | Current SKU | GPU-Util | MFU | MBU | Candidate SKU | Gross monthly delta |
|---|---|---:|---:|---:|---|---:|
{chr(10).join(lie_rows)}

`GPU-Util` reports that kernels were active during the sampling window; it does
not show how close useful FLOPs came to hardware peak. A GPU can therefore read
98% busy while warps stall on HBM/I/O, launch many small kernels, synchronize,
or execute poorly batched work. Here `gpu-h100-4` is 98% active but delivers only
about 20% MFU, so NimbusAI pays H100 rates without receiving H100 throughput.
The telemetry is a triage signal, not proof of one root cause: profile kernels,
memory stalls, batch size and input pipeline before moving SKUs. The table shows
**gross** price deltas; validate throughput/SLO parity in a canary before booking
the ${rightsize_savings:,.0f}/month right-size saving.

## Recommended action order

1. **P0 - eliminate idle leakage and enforce ownership:** automate shutdown of
   idle instances (up to ${idle_savings:,.0f}/month) and alert owners. Keep the
   current {r4['tag_coverage']:.0%} tag coverage above the 80% chargeback gate.
2. **P1 - apply inference routing/cache/batch guardrails:** this saves
   ${infer_savings:,.0f}/month and 82.6% per token in the measured trace. Roll out
   cascade quality checks first; use batch only for latency-tolerant traffic.
3. **P1 - execute the purchasing plan:** it is the largest lever at
   ${purchasing_savings:,.0f}/month. Use checkpointed spot for finite jobs and
   3-year reservations only for continuously observed production services;
   revalidate demand before signing commitments.
4. **P2 - right-size utilization lies:** benchmark the two candidates and accept
   only changes that preserve throughput, memory capacity and latency SLOs.

## Extension evidence

### Extension 1 - interruption- and duration-aware purchasing

The policy now uses GPU-specific interruption rates (H100 3%, A100 5%, A10G
10%, L4 8%) and chooses a 1-year versus 3-year term from the planning horizon.
Against the original fixed-5%/always-3-year policy, measured monthly cost changes
from **${r3['legacy_optimized_monthly']:,.0f} to ${r3['optimized_monthly']:,.0f}**,
an additional **${r3['policy_delta_usd']:,.0f}/month** saving. The difference is
small because the lower H100 rework offsets higher A10G/L4 interruption risk;
the main value is avoiding a false uniform-risk assumption and mismatched terms.

### Extension 3 - prompt-cache break-even

Team/project/model is used as a conservative proxy for a reusable prefix because
the synthetic CSV has no prefix identifier. It yields {cache_small['avg_cache_reads']:.1f}
observed reads per proxy prefix. Break-even is **{cache_small['break_even_reads']:.2f}
reads** for the small-tier write profile and **{cache_large['break_even_reads']:.2f}
reads** for the large tier; both clear the threshold. Production metering should
replace this proxy with cache-key-level reads and include storage/expiry charges.

### Extension 4 - reasoning budget

Reasoning is **{reasoning['request_share_pct']:.1f}% of requests** but
**{reasoning['cost_share_pct']:.1f}% of optimized inference cost** and
**{reasoning['energy_share_pct']:.1f}% of modeled energy**. Its 80x multiplier
acts on each query and the trace also has a 6x output-token tax, explaining the
disproportion. The requested 10% cap is already met, so it truthfully reroutes
{reasoning['requests_to_reroute']} requests and saves $0 and 0 Wh. A stricter 5%
sensitivity reroutes {reasoning_5pct['requests_to_reroute']} requests, saving
**${reasoning_5pct['cap_savings_daily']:.2f}/day** and
**{reasoning_5pct['cap_wh_saved_daily']:.0f} Wh/day**. Recommended rule: enable
reasoning only for complex tasks that fail the small-model confidence/quality
threshold, with a 10% hard budget and human-approved exceptions.

### Extension 5 - carbon-aware scheduling

The {carbon['interruptible_kwh']:,.1f} kWh interruptible fleet has these placement
outcomes (electricity component only):

| Criterion | Region | Electricity cost | Carbon |
|---|---|---:|---:|
| Current | us-east-1 | ${current_region['electricity_cost_usd']:,.2f} | {current_region['carbon_g']/1000:,.1f} kgCO2e |
| Cleanest | {clean_region} | ${clean_values['electricity_cost_usd']:,.2f} | {clean_values['carbon_g']/1000:,.1f} kgCO2e |
| Cheapest electricity | {cheapest_region} | ${cheapest_values['electricity_cost_usd']:,.2f} | {cheapest_values['carbon_g']/1000:,.1f} kgCO2e |
| Balanced normalized score | {carbon['balanced_region']} | ${carbon['regions'][carbon['balanced_region']]['electricity_cost_usd']:,.2f} | {carbon['regions'][carbon['balanced_region']]['carbon_g']/1000:,.1f} kgCO2e |

Moving delay-tolerant jobs from `us-east-1` to `{clean_region}` saves
**{carbon['carbon_saved_g']/1000:,.1f} kgCO2e ({carbon['carbon_saved_pct']:.1f}%)**
and ${current_region['electricity_cost_usd'] - clean_values['electricity_cost_usd']:,.2f}
of modeled electricity. `{cheapest_region}` is the cost-first choice; `{clean_region}`
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
"""

    md = report.build_report(
        baseline,
        optimized,
        levers,
        sustainability=sust,
        additional_markdown=additional,
    )
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(
        levers,
        os.path.join(ROOT, "outputs", "savings.png"),
        baseline_usd=baseline,
        optimized_usd=optimized,
    )

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1)}


if __name__ == "__main__":
    run()
