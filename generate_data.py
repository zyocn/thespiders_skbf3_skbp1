import pandas as pd
import numpy as np
import os

def create_synthetic_crop_data(num_samples=2000, output_file='crop_data.csv'):
    print(f"Generating synthetic multi-omics and enviromics dataset ({num_samples} samples)...")
    np.random.seed(42)

    # 1. Generate Genomics Data (e.g., 20 Gene Markers / SNPs)
    # Values between 0 and 2 representing allele frequency or expression level
    gene_columns = [f"Gene_Marker_{i}" for i in range(1, 21)]
    genomics_data = np.random.randint(0, 3, size=(num_samples, len(gene_columns)))

    # 2. Generate Environmental Data (5 climatic variables)
    env_columns = ["Avg_Temp_C", "Annual_Rainfall_mm", "Soil_Nitrogen_ppm", "Sunlight_Hours", "Soil_pH"]
    
    # Realistic ranges for the environment
    temp = np.random.normal(22, 5, num_samples) # Mean 22C, std 5
    rainfall = np.random.normal(800, 200, num_samples) # Mean 800mm, std 200
    nitrogen = np.random.normal(50, 10, num_samples)
    sunlight = np.random.normal(2000, 300, num_samples)
    ph = np.random.normal(6.5, 0.5, num_samples)
    
    env_data = np.column_stack((temp, rainfall, nitrogen, sunlight, ph))

    # 3. Simulate The Target: Crop Yield (bushels/acre)
    # We create biological rules: 
    # Gene 1 and 5 boost yield. Gene 10 reduces yield if Temp is high (Heat stress vulnerability).
    # Good rainfall and nitrogen boost yield.
    
    base_yield = 100 
    
    # Genetic effects
    genetic_effect = (genomics_data[:, 0] * 5) + (genomics_data[:, 4] * 3) - (genomics_data[:, 15] * 2)
    
    # Environmental effects
    env_effect = ((rainfall - 800) * 0.05) + ((nitrogen - 50) * 0.5)
    
    # Complex Gene-Environment Interaction (Gene 9 makes the crop heat tolerant)
    heat_stress = np.where(temp > 28, -15, 0)
    heat_tolerance_gene = genomics_data[:, 8]
    interaction_effect = heat_stress * (1 - (heat_tolerance_gene * 0.5)) # Gene reduces heat stress penalty
    
    # Add random noise (unmeasured factors)
    noise = np.random.normal(0, 5, num_samples)
    
    # Calculate final yield
    crop_yield = base_yield + genetic_effect + env_effect + interaction_effect + noise
    
    # Calculate Drought Tolerance Score (0-100)
    # High rainfall history makes it less drought tolerant, Gene 2 makes it more tolerant
    drought_tolerance = 50 - ((rainfall - 800) * 0.02) + (genomics_data[:, 1] * 10) + np.random.normal(0, 5, num_samples)
    drought_tolerance = np.clip(drought_tolerance, 0, 100)

    # 4. Combine into a Pandas DataFrame
    df_genes = pd.DataFrame(genomics_data, columns=gene_columns)
    df_env = pd.DataFrame(env_data, columns=env_columns)
    
    df = pd.concat([df_genes, df_env], axis=1)
    df['Crop_Yield_bu_ac'] = crop_yield
    df['Drought_Tolerance_Score'] = drought_tolerance
    
    # 5. Save to CSV
    df.to_csv(output_file, index=False)
    print(f"✅ Successfully created '{output_file}'!")
    print(df.head())

if __name__ == "__main__":
    create_synthetic_crop_data()
