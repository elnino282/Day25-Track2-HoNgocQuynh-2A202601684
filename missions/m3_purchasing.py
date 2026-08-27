"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = legacy_optimized_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # A 30-day production inference trace represents a recurring service;
        # use a 3-year planning horizon for it. Finite train/dev jobs retain
        # their stated duration and are never forced into a long commitment.
        observed_days = num(j["days"])
        planning_days = 1095 if j["kind"] == "infer" and not interruptible and observed_days >= 30 else observed_days
        plan = pricing.recommend_tier(
            hpd,
            interruptible,
            gpu_type=gtype,
            job_days=planning_days,
            return_details=True,
        )
        tier = plan["tier"]
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(
                gpu_hours,
                num(c["spot_hr"]),
                od,
                interrupt_rate=plan["interruption_rate"],
            )
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            price_key = "reserved_3yr_hr" if plan["reserved_term"] == "3yr" else "reserved_1yr_hr"
            opt_cost = gpu_hours * num(c[price_key])
        else:
            opt_cost = on_demand_cost

        # Original policy used one 5% interruption rate and always the 3-year
        # reservation price. Keep it as a measured before/after comparator.
        legacy_tier = pricing.recommend_tier(hpd, interruptible)
        if legacy_tier == "spot":
            legacy_cost = pricing.spot_checkpoint_cost(
                gpu_hours, num(c["spot_hr"]), od, interrupt_rate=0.05
            )["spot_cost"]
        elif legacy_tier == "reserved":
            legacy_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            legacy_cost = on_demand_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        legacy_optimized_monthly += legacy_cost
        recs.append({
            "job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
            "reserved_term": plan["reserved_term"],
            "interruption_rate": plan["interruption_rate"],
            "reason": plan["reason"],
            "on_demand": round(on_demand_cost), "optimized": round(opt_cost),
        })

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0

    # Extension 5: carbon-aware placement of checkpointable jobs. Workload
    # energy is invariant across regions; grid price/intensity changes outcomes.
    interruptible_kwh = 0.0
    for j in jobs:
        if not bool(int(num(j["interruptible"]))):
            continue
        c = cat[j["gpu_type"]]
        gpu_hours = num(j["hours_per_day"]) * num(j["days"]) * int(num(j["num_gpus"]))
        interruptible_kwh += gpu_hours * num(c["watts"]) / 1000.0
    region_comparison = {}
    for region in sustainability.REGION_CARBON:
        region_comparison[region] = {
            "electricity_cost_usd": interruptible_kwh * sustainability.REGION_PRICE_KWH[region],
            "carbon_g": interruptible_kwh * sustainability.REGION_CARBON[region],
        }
    cheapest_region = min(region_comparison, key=lambda r: region_comparison[r]["electricity_cost_usd"])
    cleanest_region = min(region_comparison, key=lambda r: region_comparison[r]["carbon_g"])
    max_cost = max(v["electricity_cost_usd"] for v in region_comparison.values()) or 1.0
    max_carbon = max(v["carbon_g"] for v in region_comparison.values()) or 1.0
    balanced_region = min(
        region_comparison,
        key=lambda r: (
            region_comparison[r]["electricity_cost_usd"] / max_cost
            + region_comparison[r]["carbon_g"] / max_carbon
        ),
    )
    carbon_saved_g = (
        region_comparison["us-east-1"]["carbon_g"]
        - region_comparison[cleanest_region]["carbon_g"]
    )
    carbon_saved_pct = (
        carbon_saved_g / region_comparison["us-east-1"]["carbon_g"] * 100
        if region_comparison["us-east-1"]["carbon_g"] else 0.0
    )

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'term':7}{'risk':>7}{'on-demand':>13}{'optimized':>13}")
        for r in recs:
            od_label = f"${r['on_demand']:,}"
            opt_label = f"${r['optimized']:,}"
            print(
                f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}"
                f"{(r['reserved_term'] or '-'):7}{r['interruption_rate']:>6.0%} "
                f"{od_label:>12}{opt_label:>13}"
            )
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        delta = legacy_optimized_monthly - optimized_monthly
        print(
            f"Extension 1 policy: legacy ${legacy_optimized_monthly:,.0f} -> "
            f"risk/duration-aware ${optimized_monthly:,.0f} ({delta:+,.0f} additional savings)"
        )
        print("\nExtension 5 - interruptible fleet by region:")
        print(f"  energy moved: {interruptible_kwh:,.1f} kWh")
        print(f"  {'region':18}{'$/kWh':>8}{'gCO2/kWh':>12}{'electricity':>14}{'carbon kg':>12}")
        for region, values in region_comparison.items():
            electricity_label = f"${values['electricity_cost_usd']:,.2f}"
            print(
                f"  {region:18}{sustainability.REGION_PRICE_KWH[region]:>8.3f}"
                f"{sustainability.REGION_CARBON[region]:>12,.0f}"
                f"{electricity_label:>14}{values['carbon_g']/1000:>12,.1f}"
            )
        print(
            f"  cheapest={cheapest_region}; cleanest={cleanest_region}; balanced={balanced_region}; "
            f"move us-east-1 -> {cleanest_region} saves {carbon_saved_g/1000:,.1f} kgCO2e "
            f"({carbon_saved_pct:.1f}%)"
        )
        print("  latency caveat: route only delay-tolerant jobs after data-residency and transfer checks")

    return {
        "recommendations": recs,
        "on_demand_monthly": round(on_demand_monthly),
        "optimized_monthly": round(optimized_monthly),
        "legacy_optimized_monthly": round(legacy_optimized_monthly),
        "policy_delta_usd": round(legacy_optimized_monthly - optimized_monthly),
        "savings_pct": round(savings_pct, 1),
        "carbon_aware": {
            "interruptible_kwh": interruptible_kwh,
            "regions": region_comparison,
            "cheapest_region": cheapest_region,
            "cleanest_region": cleanest_region,
            "balanced_region": balanced_region,
            "carbon_saved_g": carbon_saved_g,
            "carbon_saved_pct": carbon_saved_pct,
        },
    }


if __name__ == "__main__":
    run()
