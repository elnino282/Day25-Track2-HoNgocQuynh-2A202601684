"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations


# Illustrative per-hour interruption probabilities. Keep these as replaceable
# data so a FinOps team can substitute its provider's observed interruption feed.
SPOT_INTERRUPTION_RATES = {
    "H100": 0.03,
    "H200": 0.04,
    "A100": 0.05,
    "A10G": 0.10,
    "L4": 0.08,
    "B200": 0.06,
    "MI300X": 0.05,
}


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(
    hours_per_day: float,
    interruptible: bool,
    reserved_discount: float = 0.45,
    gpu_type: str | None = None,
    job_days: float | None = None,
    return_details: bool = False,
) -> str | dict:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    The default return remains a backward-compatible tier string. Optional GPU
    type and duration inputs add risk/term evidence; ``return_details=True``
    exposes those assumptions for audit and reporting.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    interruption_rate = SPOT_INTERRUPTION_RATES.get(gpu_type or "", 0.05)
    # Expected rework of 0.5 h/interruption plus 3% checkpoint overhead.
    spot_overhead = 0.03 + interruption_rate * 0.5
    if interruptible and hours_per_day < 24 and spot_overhead <= 0.15:
        tier = "spot"
        term = None
        reason = (
            f"checkpointable; {interruption_rate:.0%}/h interruption assumption "
            f"adds about {spot_overhead:.1%} compute overhead"
        )
    elif duty >= be:
        tier = "reserved"
        term = "3yr" if job_days is not None and job_days >= 730 else "1yr"
        reason = (
            f"{duty:.0%} duty cycle clears the {be:.0%} break-even; "
            f"{term} matches the stated planning horizon"
        )
    else:
        tier = "on_demand"
        term = None
        reason = f"{duty:.0%} duty cycle is below the {be:.0%} commitment break-even"

    if return_details:
        return {
            "tier": tier,
            "reserved_term": term,
            "interruption_rate": interruption_rate,
            "spot_overhead_frac": spot_overhead,
            "reason": reason,
        }
    return tier


def cache_break_even_reads(write_cost_per_m: float, read_discount: float = 0.10) -> float:
    """Reads needed for saved input charges to repay one cache write.

    ``write_cost_per_m`` is relative to the normal input price: 1.25 means a
    cache write costs 1.25 times a normal input read.
    """
    if write_cost_per_m < 0:
        raise ValueError("write_cost_per_m must be non-negative")
    if not 0 <= read_discount < 1:
        raise ValueError("read_discount must be in [0, 1)")
    return write_cost_per_m / (1.0 - read_discount)


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float,
    read_discount: float = 0.10,
) -> bool:
    """Return whether expected prefix reuse clears cache-write break-even."""
    return avg_cache_reads >= cache_break_even_reads(write_cost_per_m, read_discount)


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
