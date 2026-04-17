"""
scenario_planner.py
Climate Scenario Simulator for CropAI Platform — SKB_P3

Simulates how genotypes perform under 54+ future climate scenarios.
Applies physics-guided transformations to biological features
based on climate stress (temperature, rainfall, soil pH changes).
"""

import numpy as np
import pandas as pd


def simulate_scenario(df: pd.DataFrame, scenario: dict, model) -> pd.DataFrame:
    """
    Applies a climate scenario to the dataset and predicts crop performance.

    Physics-guided transformations:
      - Higher temp      → more abscisic acid (stress hormone)
      - Lower rainfall   → more proline (protective metabolite)
      - Severe drought   → upregulates drought gene expression
      - Heat stress      → suppresses growth gene expression

    Args:
        df       : Original omics dataframe
        scenario : Dict with temp_increase, rain_factor, soil_ph keys
        model    : Trained CropResilienceModel instance

    Returns:
        pd.DataFrame: df with predicted_yield and predicted_drought_tolerance columns
    """
    df_sim = df.copy()

    temp_inc    = scenario["temp_increase"]
    rain_factor = scenario["rain_factor"]
    new_ph      = scenario["soil_ph"]

    # ── Apply climate changes ────────────────────────────────────────────
    df_sim["temperature_c"] = df_sim["temperature_c"] + temp_inc
    df_sim["rainfall_mm"]   = df_sim["rainfall_mm"]   * rain_factor
    df_sim["soil_ph"]       = new_ph

    # ── Physics-guided biological stress responses ───────────────────────
    # Rule 1: Higher temperature → more abscisic acid (ABA) stress hormone
    df_sim["abscisic_acid"] += temp_inc * 0.2

    # Rule 2: Lower rainfall → higher proline (drought protectant)
    if rain_factor < 1.0:
        df_sim["proline_level"] += (1.0 - rain_factor) * 5.0

    # Rule 3: Severe drought → drought-response genes activated
    if rain_factor < 0.7:
        df_sim["gene_expr_drought"] += 1.5

    # Rule 4: High heat stress → growth genes suppressed
    if temp_inc >= 3:
        df_sim["gene_expr_growth"] -= 1.0

    # Rule 5: Acidic soil → reduced nitrogen uptake effect
    if new_ph < 6.0:
        df_sim["soil_nitrogen"] *= 0.85

    # ── Predict under new conditions ─────────────────────────────────────
    yield_pred, drought_pred = model.predict(df_sim)

    df_sim["predicted_yield"]             = np.clip(yield_pred,   0.5, 10.0)
    df_sim["predicted_drought_tolerance"] = np.clip(drought_pred, 0.0, 10.0)

    return df_sim


def rank_genotypes(df_sim: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Ranks genotypes by combined resilience score.

    Resilience Score = (Yield × 0.6) + (Drought Tolerance × 0.4)
    Weights favor yield slightly since food production is primary goal.

    Args:
        df_sim : Simulated dataframe with predicted columns
        top_n  : Number of top genotypes to return

    Returns:
        pd.DataFrame: Top N genotypes ranked by resilience score
    """
    df_out = df_sim.copy()
    df_out["resilience_score"] = (
        df_out["predicted_yield"]             * 0.6 +
        df_out["predicted_drought_tolerance"] * 0.4
    )
    result = (
        df_out[["sample_id", "predicted_yield", "predicted_drought_tolerance", "resilience_score"]]
        .sort_values("resilience_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    result.index = result.index + 1  # 1-based ranking
    return result


def generate_blueprint(scenario: dict, top_genotypes: pd.DataFrame) -> dict:
    """
    Generates a region-specific breeding strategy and soil fixation blueprint.

    Args:
        scenario      : Climate scenario dict
        top_genotypes : Ranked genotypes dataframe from rank_genotypes()

    Returns:
        dict: Breeding blueprint with strategies, parent recommendations, soil plan
    """
    stress = scenario["stress_label"]

    # ── Breeding strategies by stress level ──────────────────────────────
    strategies = {
        "Severe": [
            "Prioritize extreme drought-tolerant lines (drought score > 8)",
            "Focus on SNP markers linked to heat shock proteins (HSP)",
            "Apply deficit irrigation protocols during vegetative stage",
            "Select parents with high proline accumulation capacity",
            "Consider intercropping with nitrogen-fixing legumes",
            "Implement deep-root selection criteria for water access",
        ],
        "Moderate": [
            "Balance yield potential with moderate stress tolerance",
            "Screen for stay-green trait markers to maintain photosynthesis",
            "Implement precision irrigation scheduling (sensor-based)",
            "Prefer deep-rooted parent lines for better water uptake",
            "Use soil amendments to maintain optimal pH 6.5–7.0",
            "Target genotypes with stable performance across environments",
        ],
        "Mild": [
            "Maximize yield potential — stress risk is manageable",
            "Select high-yielding parents with broad environmental adaptability",
            "Standard fertilization protocol is applicable",
            "Include disease resistance markers as secondary selection criterion",
            "Monitor for emerging pest pressures in warmer conditions",
            "Diversify genetic background to hedge against future stress",
        ],
    }

    # ── Soil fixation recommendations ────────────────────────────────────
    soil_plan = {
        "target_ph"          : f"{scenario['soil_ph']:.1f}",
        "nitrogen_management": (
            "High N input (120 kg/ha) — leaching risk from excess rain"
            if scenario["rain_factor"] > 1.2
            else "Moderate N (80 kg/ha) with slow-release polymer-coated urea"
        ),
        "water_strategy"     : (
            "Drip irrigation + plastic mulching to conserve moisture"
            if scenario["rain_factor"] < 0.7
            else "Furrow irrigation with sub-surface drainage channels"
        ),
        "organic_matter"     : (
            "Add 3–5 t/ha compost to improve water retention"
            if scenario["rain_factor"] < 1.0
            else "Standard 1–2 t/ha organic matter supplementation"
        ),
        "ph_correction"      : (
            "Apply agricultural lime (2 t/ha) to raise pH"
            if scenario["soil_ph"] < 6.0
            else "Apply sulfur (0.5 t/ha) if pH > 7.5"
            if scenario["soil_ph"] > 7.5
            else "No pH correction needed — optimal range"
        ),
    }

    top_parents = top_genotypes["sample_id"].tolist()[:3]

    blueprint = {
        "scenario_name"      : scenario["scenario_name"],
        "scenario_stress"    : stress,
        "temp_increase"      : scenario["temp_increase"],
        "rain_factor"        : scenario["rain_factor"],
        "recommended_parents": top_parents,
        "breeding_strategies": strategies.get(stress, strategies["Mild"]),
        "soil_fixation"      : soil_plan,
        "expected_yield_range": (
            f"{top_genotypes['predicted_yield'].min():.2f} – "
            f"{top_genotypes['predicted_yield'].max():.2f} t/ha"
        ),
        "avg_resilience_score": round(top_genotypes["resilience_score"].mean(), 3),
    }

    return blueprint


def format_blueprint_text(blueprint: dict) -> str:
    """Converts blueprint dict to a readable text report for download."""
    sf = blueprint["soil_fixation"]
    strategies_text = "\n".join(f"  • {s}" for s in blueprint["breeding_strategies"])

    return f"""
╔══════════════════════════════════════════════════════════════╗
   CROP BREEDING BLUEPRINT — {blueprint['scenario_name']}
╚══════════════════════════════════════════════════════════════╝

CLIMATE SCENARIO
  Stress Level      : {blueprint['scenario_stress']}
  Temperature Rise  : +{blueprint['temp_increase']}°C
  Rainfall Factor   : ×{blueprint['rain_factor']}

RECOMMENDED PARENT LINES
  {' | '.join(blueprint['recommended_parents'])}

EXPECTED YIELD RANGE
  {blueprint['expected_yield_range']}

BREEDING STRATEGIES
{strategies_text}

SOIL FIXATION PLAN
  Target pH          : {sf['target_ph']}
  Nitrogen           : {sf['nitrogen_management']}
  Water Management   : {sf['water_strategy']}
  Organic Matter     : {sf['organic_matter']}
  pH Correction      : {sf['ph_correction']}

Avg Resilience Score  : {blueprint['avg_resilience_score']}
Generated by CropAI Platform — SKB_P3
""".strip()
