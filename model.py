import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiOmicsEncoder(nn.Module):
    """
    Core Objective 1: Multi Omics Big Data Integration
    Encodes various high-dimensional omics layers (Genomics, Transcriptomics, Proteomics, etc.)
    into a unified latent representation.
    """
    def __init__(self, input_dims, hidden_dim=256):
        super().__init__()
        # input_dims is a dict: {'genomics': 10000, 'transcriptomics': 5000, ...}
        self.encoders = nn.ModuleDict()
        for omics_type, dim in input_dims.items():
            self.encoders[omics_type] = nn.Sequential(
                nn.Linear(dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            )
            
    def forward(self, omics_data):
        # omics_data is a dict of tensors
        encoded_features = []
        for omics_type, data in omics_data.items():
            encoded = self.encoders[omics_type](data)
            encoded_features.append(encoded)
            
        # Concatenate encoded omics data along the feature dimension
        return torch.cat(encoded_features, dim=1)


class EnviromicsEncoder(nn.Module):
    """
    Processes multi-environment climatic variables (temperature, precipitation, soil data).
    """
    def __init__(self, input_dim, hidden_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, env_data):
        return self.encoder(env_data)


class PhysicsGuidedAttention(nn.Module):
    """
    Challenge Requirement: Physics-guided machine learning and Model Interpretability.
    This attention mechanism mimics the weighting of biological pathways. In a fully 
    realized model, this could be constrained by a Gene Regulatory Network (GRN) graph matrix.
    """
    def __init__(self, feature_dim):
        super().__init__()
        self.attention_weights = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, feature_dim),
            nn.Sigmoid() # Scale features between 0 and 1 representing biological activation
        )
        
    def forward(self, x):
        weights = self.attention_weights(x)
        return x * weights # Element-wise attention


class CropPerformancePredictor(nn.Module):
    """
    Core Objective 2: Predictive Trait Discovery
    Deep neural genomic prediction model accurately forecasting combined effects 
    of traits on crop output under diverse climates.
    """
    def __init__(self, omics_input_dims, env_input_dim, num_traits=2):
        super().__init__()
        omics_hidden_dim = 256
        env_hidden_dim = 128
        num_omics = len(omics_input_dims)
        
        self.omics_encoder = MultiOmicsEncoder(omics_input_dims, hidden_dim=omics_hidden_dim)
        self.env_encoder = EnviromicsEncoder(env_input_dim, hidden_dim=env_hidden_dim)
        
        # Fusion dimensionality
        fused_dim = (omics_hidden_dim * num_omics) + env_hidden_dim
        
        self.physics_attention = PhysicsGuidedAttention(fused_dim)
        
        self.predictor = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_traits) # Outputs: [Yield, Drought Tolerance Score]
        )
        
    def forward(self, omics_data, env_data):
        # 1. Encode
        omics_features = self.omics_encoder(omics_data)
        env_features = self.env_encoder(env_data)
        
        # 2. Fusion of Biology and Environment
        fused_features = torch.cat([omics_features, env_features], dim=1)
        
        # 3. Interpretability / Biological routing
        attended_features = self.physics_attention(fused_features)
        
        # 4. Predict
        predictions = self.predictor(attended_features)
        return predictions


class GenerativeScenarioPlanner(nn.Module):
    """
    Core Objective 3 & 4: Generative Scenario Planning & Parental Selection
    A Conditional Variational Autoencoder (cVAE) to simulate 'What-If' futures.
    It takes baseline Genotype and conditions on an extreme Climate (Env) to simulate traits.
    """
    def __init__(self, genotype_dim, env_dim, latent_dim=64):
        super().__init__()
        
        # Encoder: (Genotype + Environment) -> Latent Space
        self.encoder = nn.Sequential(
            nn.Linear(genotype_dim + env_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim * 2) # Outputs mu and log_var
        )
        
        # Decoder: (Latent Space + Environment) -> Simulated Phenotype (e.g. Yield)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + env_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 2) # Simulated [Yield, Drought Tolerance]
        )
        
    def encode(self, genotype, env):
        x = torch.cat([genotype, env], dim=1)
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        return mu, logvar
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z, env):
        x = torch.cat([z, env], dim=1)
        return self.decoder(x)
        
    def forward(self, genotype, env):
        mu, logvar = self.encode(genotype, env)
        z = self.reparameterize(mu, logvar)
        simulated_phenotype = self.decode(z, env)
        return simulated_phenotype, mu, logvar


if __name__ == "__main__":
    print("Initializing Multi-Omics AI Platform...")
    
    # --- Mock Data Dimensions ---
    # In reality, this would be SNPs (e.g., 50k), gene expressions, etc.
    omics_dims = {'genomics': 1000, 'transcriptomics': 500} 
    env_dim = 50 # e.g., 50 climatic variables across growth stages
    
    # --- Initialize Models ---
    predictive_model = CropPerformancePredictor(omics_dims, env_dim, num_traits=2)
    generative_model = GenerativeScenarioPlanner(genotype_dim=1000, env_dim=env_dim)
    
    # --- Simulate Data (Batch size of 32) ---
    batch_size = 32
    dummy_genomics = torch.randn(batch_size, 1000)
    dummy_transcriptomics = torch.randn(batch_size, 500)
    dummy_baseline_env = torch.randn(batch_size, 50) # Normal climate
    
    omics_data = {
        'genomics': dummy_genomics,
        'transcriptomics': dummy_transcriptomics
    }
    
    # --- 1. Predictive Trait Discovery ---
    print("\n[Phase 1] Running Predictive Trait Discovery...")
    predictions = predictive_model(omics_data, dummy_baseline_env)
    print(f"-> Predicted Traits Shape (Yield, Drought Tolerance): {predictions.shape}")
    
    # --- 2. Generative Scenario Planning ---
    print("\n[Phase 2] Simulating 'What-If' Extreme Climate Scenarios...")
    # Simulate a severe heatwave (shifting environmental variables drastically)
    extreme_env = dummy_baseline_env + torch.randn(batch_size, 50) * 3.0 
    
    simulated_outcome, mu, logvar = generative_model(dummy_genomics, extreme_env)
    print(f"-> Simulated Phenotype under Extreme Climate Shape: {simulated_outcome.shape}")
    print("\nPlatform architecture is ready for dataset integration and training!")
