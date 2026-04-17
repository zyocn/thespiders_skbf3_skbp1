import torch
import torch.nn as nn
import torch.optim as optim
from model import CropPerformancePredictor, GenerativeScenarioPlanner

def train_predictive_model():
    print("--- Training Predictive Trait Discovery Model ---")
    omics_dims = {'genomics': 1000, 'transcriptomics': 500}
    env_dim = 50
    model = CropPerformancePredictor(omics_dims, env_dim, num_traits=2)
    
    # Loss & Optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    epochs = 5
    batch_size = 16
    
    model.train()
    for epoch in range(epochs):
        # 1. Fetch data (mocked here, replace with your DataLoader)
        dummy_gen = torch.randn(batch_size, 1000)
        dummy_trans = torch.randn(batch_size, 500)
        omics_data = {'genomics': dummy_gen, 'transcriptomics': dummy_trans}
        env_data = torch.randn(batch_size, 50)
        
        # Ground truth targets
        target_traits = torch.randn(batch_size, 2)
        
        # 2. Forward pass
        optimizer.zero_grad()
        predictions = model(omics_data, env_data)
        
        # 3. Calculate Loss
        loss = criterion(predictions, target_traits)
        
        # 4. Backward pass & optimize
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
        
    print("Predictive Model Training Complete!\n")


def loss_function_cvae(recon_x, x, mu, logvar):
    """ Custom loss for the Generative Model (ELBO) """
    BCE = nn.functional.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_generative_model():
    print("--- Training Generative 'What-If' Planner ---")
    genotype_dim = 1000
    env_dim = 50
    model = GenerativeScenarioPlanner(genotype_dim, env_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 5
    batch_size = 16
    
    model.train()
    for epoch in range(epochs):
        dummy_gen = torch.randn(batch_size, genotype_dim)
        dummy_env = torch.randn(batch_size, env_dim)
        
        # Ground truth phenotypic traits under that environment
        target_traits = torch.randn(batch_size, 2) 
        
        optimizer.zero_grad()
        # The VAE aims to reconstruct the traits based on (genotype + env)
        simulated_phenotype, mu, logvar = model(dummy_gen, dummy_env)
        
        loss = loss_function_cvae(simulated_phenotype, target_traits, mu, logvar)
        loss.backward()
        optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], VAE Loss: {loss.item():.4f}")
        
    print("Generative Model Training Complete!")

if __name__ == "__main__":
    train_predictive_model()
    train_generative_model()
