"""Student-authored tests for the Your Turn extensions.

The instructor tests remain untouched; these checks make the added policy and
measurement behavior reproducible for review.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data import generate
from finops import pricing
from missions import m2_inference_levers, m3_purchasing
from missions._common import load_csv


def test_gpu_specific_spot_policy_exposes_risk_assumption():
    plan = pricing.recommend_tier(
        10, True, gpu_type="H100", job_days=14, return_details=True
    )
    assert plan["tier"] == "spot"
    assert plan["interruption_rate"] == 0.03
    assert "checkpointable" in plan["reason"]


def test_reservation_duration_follows_planning_horizon():
    one_year = pricing.recommend_tier(
        24, False, gpu_type="A100", job_days=365, return_details=True
    )
    three_year = pricing.recommend_tier(
        24, False, gpu_type="A100", job_days=1095, return_details=True
    )
    assert one_year["tier"] == three_year["tier"] == "reserved"
    assert one_year["reserved_term"] == "1yr"
    assert three_year["reserved_term"] == "3yr"


def test_cache_break_even_and_decision():
    assert abs(pricing.cache_break_even_reads(1.25, 0.10) - 1.25 / 0.90) < 1e-9
    assert pricing.cache_is_worth_it(2.0, 1.25, 0.10) is True
    assert pricing.cache_is_worth_it(1.0, 1.25, 0.10) is False


def test_reasoning_and_carbon_extensions_are_measurable():
    generate.main()
    reasoning = m2_inference_levers.reasoning_budget_analysis(
        load_csv("token_usage.csv"), target_share=0.05
    )
    purchasing = m3_purchasing.run(verbose=False)

    assert reasoning["requests_to_reroute"] > 0
    assert reasoning["cap_savings_daily"] > 0
    assert reasoning["cap_wh_saved_daily"] > 0
    assert purchasing["policy_delta_usd"] != 0
    assert purchasing["carbon_aware"]["cleanest_region"] == "europe-north1"
    assert purchasing["carbon_aware"]["carbon_saved_pct"] > 90
