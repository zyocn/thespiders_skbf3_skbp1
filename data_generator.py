"""
data_generator.py
Simulates Multi-Omics + Climate data for the CropAI platform.
In a real project, replace this with actual genomic/lab datasets.
"""

import numpy as np
import pandas as pd


def generate_omics_data(n_samples: int = 300, n_snps: int = 50, seed: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic multi-omics + enviromics dataset.

    Layers included:
    - Genomics     : SNP markers (0, 1, 2 allele dosage)
    - Transcriptomics : Gene expression levels
    - Metabolomics : Stress-related metabolite concentrations
    - Enviromics   : Climate and soil variables

    Returns:
        pd.DataFrame: Combined feature matrix with yield and drought targets
    """
    np.random.seed(seed)

    # ── 1. GENOMICS: SNP markers ──────────────────────────────────────────
    snp_cols = [f"SNP_{i}" for i in range(n_snps)]
    genomics = pd.DataFrame(
        np.random.choice([0, 1, 2], size=(n_samples, n_snps), p=[0.5, 0.3, 0.2]),
        columns=snp_cols
    )

    # ── 2. TRANSCRIPTOMICS: Gene expression ──────────────────────────────
    transcriptomics = pd.DataFrame({
        "gene_expr_drought" : np.random.normal(5.0, 1.5, n_samples),
        "gene_expr_heat"    : np.random.normal(4.0, 1.2, n_samples),
        "gene_expr_growth"  : np.random.normal(6.0, 2.0, n_samples),
    })

    # ── 3. METABOLOMICS: Key metabolites ─────────────────────────────────
    metabolomics = pd.DataFrame({
        "proline_level"  : np.random.normal(10.0, 3.0, n_samples),   # drought marker
        "chlorophyll"    : np.random.normal(35.0, 8.0, n_samples),   # photosynthesis
        "abscisic_acid"  : np.random.normal(2.5,  0.8, n_samples),   # stress hormone
    })

    # ── 4. ENVIROMICS: Climate + soil ────────────────────────────────────
    enviromics = pd.DataFrame({
        "temperature_c"  : np.random.uniform(20, 45, n_samples),
        "rainfall_mm"    : np.random.uniform(100, 800, n_samples),
        "soil_ph"        : np.random.uniform(5.5, 8.0, n_samples),
        "soil_nitrogen"  : np.random.uniform(10, 80, n_samples),
        "humidity_pct"   : np.random.uniform(20, 90, n_samples),
    })

    # ── 5. TARGET: Yield (tons/ha) ────────────────────────────────────────
    yield_score = (
        3.0
        + genomics[snp_cols[:10]].mean(axis=1) * 0.30
        + transcriptomics["gene_expr_growth"]   * 0.40
        + metabolomics["chlorophyll"]            * 0.05
        - metabolomics["abscisic_acid"]          * 0.30
        - (enviromics["temperature_c"] - 28).abs() * 0.08
        + enviromics["rainfall_mm"]              * 0.003
        + np.random.normal(0, 0.3, n_samples)
    ).clip(0.5, 8.0)

    # ── 6. TARGET: Drought Tolerance (0–10) ──────────────────────────────
    drought_score = (
        metabolomics["proline_level"]            * 0.30
        + transcriptomics["gene_expr_drought"]   * 0.50
        + genomics[snp_cols[0:5]].sum(axis=1)   * 0.10
        + np.random.normal(0, 0.5, n_samples)
    ).clip(0, 10)

    # ── COMBINE ───────────────────────────────────────────────────────────
    df = pd.concat([genomics, transcriptomics, metabolomics, enviromics], axis=1)
    df["yield_tons_ha"]      = yield_score
    df["drought_tolerance"]  = drought_score
    df["sample_id"]          = [f"Genotype_{i:03d}" for i in range(n_samples)]

    print(f"[data_generator] Generated {n_samples} samples | "
          f"Features: {df.shape[1]-2} | "
          f"Avg Yield: {yield_score.mean():.2f} t/ha")
    return df


def get_climate_scenarios() -> pd.DataFrame:
    """
    Generates 54 diverse future climate scenarios by combining:
    - Temperature increases  : 0 to +5°C
    - Rainfall factors       : 0.5× to 1.5×
    - Soil pH levels         : 5.5, 6.5, 7.5

    Returns:
        pd.DataFrame: 54 named climate scenarios
    """
    scenarios = []
    for temp_increase in [0, 1, 2, 3, 4, 5]:
        for rain_factor in [0.5, 0.7, 1.0, 1.3, 1.5]:
            for soil_ph in [5.5, 6.5, 7.5]:
                stress = (
                    "Severe"   if temp_increase >= 4 or rain_factor <= 0.5
                    else "Moderate" if temp_increase >= 2
                    else "Mild"
                )
                scenarios.append({
                    "scenario_name" : f"T+{temp_increase}°C | Rain×{rain_factor} | pH{soil_ph}",
                    "temp_increase" : temp_increase,
                    "rain_factor"   : rain_factor,
                    "soil_ph"       : soil_ph,
                    "stress_label"  : stress,
                })

    df = pd.DataFrame(scenarios[:54])
    print(f"[data_generator] Created {len(df)} climate scenarios")
    return df
