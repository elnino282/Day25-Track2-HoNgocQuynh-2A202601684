"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}


def _optimized_row_cost(r: dict, output_tokens: int | None = None) -> float:
    """Price one row under the cascade/cache/batch policy."""
    inp = int(num(r["input_tokens"]))
    out = int(num(r["output_tokens"])) if output_tokens is None else output_tokens
    pin, pout = MODEL_PRICES[r["route_tier"]]
    return pricing.request_cost(
        inp,
        out,
        pin,
        pout,
        cached_in=int(num(r["cached_input_tokens"])),
        batch=bool(int(num(r["is_batch"]))),
    )


def reasoning_budget_analysis(rows: list[dict], target_share: float = 0.10) -> dict:
    """Quantify reasoning's request, dollar and energy shares.

    For cap scenarios, an excess reasoning request is conservatively converted
    to its non-reasoning equivalent: the synthetic generator's 6x output-token
    tax is removed and the 80x reasoning energy multiplier is removed.
    """
    if not 0 <= target_share <= 1:
        raise ValueError("target_share must be in [0, 1]")
    reasoning_rows = [r for r in rows if bool(int(num(r["is_reasoning"])))]
    normal_rows = [r for r in rows if not bool(int(num(r["is_reasoning"])))]

    def totals(group: list[dict], reasoning: bool) -> tuple[float, float]:
        cost = sum(_optimized_row_cost(r) for r in group)
        wh = sum(
            sustainability.wh_per_query(
                int(num(r["input_tokens"])) + int(num(r["output_tokens"])),
                is_reasoning=reasoning,
            )
            for r in group
        )
        return cost, wh

    reasoning_cost, reasoning_wh = totals(reasoning_rows, True)
    normal_cost, normal_wh = totals(normal_rows, False)
    allowed = int(len(rows) * target_share)
    excess = max(0, len(reasoning_rows) - allowed)

    # Route the most expensive excess requests to the normal path first.
    conversions = []
    for r in reasoning_rows:
        normal_out = max(1, int(num(r["output_tokens"])) // 6)
        before_cost = _optimized_row_cost(r)
        after_cost = _optimized_row_cost(r, output_tokens=normal_out)
        before_wh = sustainability.wh_per_query(
            int(num(r["input_tokens"])) + int(num(r["output_tokens"])), True
        )
        after_wh = sustainability.wh_per_query(
            int(num(r["input_tokens"])) + normal_out, False
        )
        conversions.append((before_cost - after_cost, before_wh - after_wh))
    conversions.sort(reverse=True)

    total_cost = reasoning_cost + normal_cost
    total_wh = reasoning_wh + normal_wh
    return {
        "request_share_pct": len(reasoning_rows) / len(rows) * 100 if rows else 0.0,
        "cost_share_pct": reasoning_cost / total_cost * 100 if total_cost else 0.0,
        "energy_share_pct": reasoning_wh / total_wh * 100 if total_wh else 0.0,
        "reasoning_cost_daily": reasoning_cost,
        "normal_cost_daily": normal_cost,
        "reasoning_wh_daily": reasoning_wh,
        "normal_wh_daily": normal_wh,
        "target_share_pct": target_share * 100,
        "requests_to_reroute": excess,
        "cap_savings_daily": sum(x[0] for x in conversions[:excess]),
        "cap_wh_saved_daily": sum(x[1] for x in conversions[:excess]),
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        opt_cost += _optimized_row_cost(r)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    reasoning = reasoning_budget_analysis(rows, target_share=0.10)
    reasoning_5pct = reasoning_budget_analysis(rows, target_share=0.05)
    cache_profiles = {
        "small": {"write_cost_multiple": 1.25},
        "large": {"write_cost_multiple": 2.00},
    }
    cacheable = [r for r in rows if int(num(r["cached_input_tokens"])) > 0]
    distinct_prefixes = {
        (r.get("team", ""), r.get("project", ""), r.get("route_tier", ""))
        for r in cacheable
    }
    avg_cache_reads = len(cacheable) / len(distinct_prefixes) if distinct_prefixes else 0.0
    cache_analysis = {}
    for tier, profile in cache_profiles.items():
        write_mult = profile["write_cost_multiple"]
        cache_analysis[tier] = {
            "avg_cache_reads": avg_cache_reads,
            "break_even_reads": pricing.cache_break_even_reads(write_mult),
            "worth_it": pricing.cache_is_worth_it(avg_cache_reads, write_mult),
        }

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")
        print("\nExtension 3 - prompt-cache break-even:")
        for tier, result in cache_analysis.items():
            print(
                f"  {tier:5}: observed {result['avg_cache_reads']:.1f} reads/prefix vs "
                f"{result['break_even_reads']:.2f} break-even -> worth it? {result['worth_it']}"
            )
        print("\nExtension 4 - reasoning budget:")
        print(
            f"  {reasoning['request_share_pct']:.1f}% of requests -> "
            f"{reasoning['cost_share_pct']:.1f}% of optimized cost and "
            f"{reasoning['energy_share_pct']:.1f}% of energy"
        )
        print(
            f"  cap at 10%: reroute {reasoning['requests_to_reroute']} requests, "
            f"save ${reasoning['cap_savings_daily']:.2f}/day and "
            f"{reasoning['cap_wh_saved_daily']:.0f} Wh/day"
        )
        print(
            f"  sensitivity cap at 5%: reroute {reasoning_5pct['requests_to_reroute']} requests, "
            f"save ${reasoning_5pct['cap_savings_daily']:.2f}/day and "
            f"{reasoning_5pct['cap_wh_saved_daily']:.0f} Wh/day"
        )

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_analysis": cache_analysis,
        "reasoning": reasoning,
        "reasoning_5pct": reasoning_5pct,
    }


if __name__ == "__main__":
    run()
