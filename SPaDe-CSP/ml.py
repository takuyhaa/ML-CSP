from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
import pickle
import pandas as pd
# import torch
# from torch import nn, optim


def calculate_MACCS_keys(mol):
    fp = MACCSkeys.GenMACCSKeys(mol)
    bits = list(fp)
    return pd.Series(bits[1:])


def calc_sg_proba_trained_model(smiles):
    trained_model = pickle.load(open('models/Trained_LightGBM_sg_over100.pkl', 'rb'))
    mol = Chem.MolFromSmiles(smiles)
    feature = calculate_MACCS_keys(mol)
    probabilities = trained_model.predict_proba(pd.DataFrame(feature).transpose())
    return probabilities[0]


def calc_z_proba_trained_model(smiles):
    trained_model = pickle.load(open('models/Trained_LightGBM_Z_over100.pkl', 'rb'))
    mol = Chem.MolFromSmiles(smiles)
    feature = calculate_MACCS_keys(mol)
    probabilities = trained_model.predict_proba(pd.DataFrame(feature).transpose())
    return probabilities[0]


def calc_density_trained_model(smiles):
    trained_model = pickle.load(open('models/Trained_LightGBM_density_over100.pkl', 'rb'))
    mol = Chem.MolFromSmiles(smiles)
    feature = calculate_MACCS_keys(mol)
    density = trained_model.predict(pd.DataFrame(feature).transpose())
    return density[0]


def calculate_molecular_weight(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        print('Invalid SMILES')
        return None
    molecular_weight = Descriptors.MolWt(molecule)
    return molecular_weight


def get_valid_classes(probs, classes, threshold=0.01):
    valid_classes = [classes[j] for j in range(len(probs)) if probs[j] >= threshold]
    return valid_classes


### Following VAE is currently under development
# class CVAE(nn.Module):
#     def __init__(self, input_dim, latent_dim, condition_dim):
#         super(CVAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim + condition_dim, 64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.ReLU(),
#         )
#         self.z_mean = nn.Linear(32, latent_dim)
#         self.z_log_var = nn.Linear(32, latent_dim)
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim + condition_dim, 32),
#             nn.ReLU(),
#             nn.Linear(32, 64),
#             nn.ReLU(),
#             nn.Linear(64, input_dim),
#         )
#
#     def reparameterize(self, z_mean, z_log_var):
#         std = torch.exp(0.5 * z_log_var)
#         eps = torch.randn_like(std)
#         return z_mean + eps * std
#
#     def forward(self, x, c):
#         # Encoder
#         xc = torch.cat([x, c], dim=1)  # 入力と条件を結合
#         h = self.encoder(xc)
#         z_mean = self.z_mean(h)
#         z_log_var = self.z_log_var(h)
#         z = self.reparameterize(z_mean, z_log_var)
#         # Decoder
#         zc = torch.cat([z, c], dim=1)  # 潜在変数と条件を結合
#         reconstructed = self.decoder(zc)
#         return reconstructed, z_mean, z_log_var
#
#
# # 損失関数
# def cvae_loss(reconstructed, x, z_mean, z_log_var):
#     reconstruction_loss = nn.MSELoss(reduction='sum')(reconstructed, x)
#     kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
#     return reconstruction_loss + kl_loss
#
#
# def load_model(input_dim=6, latent_dim=3, condition_dim=3):
#     model = CVAE(input_dim, latent_dim, condition_dim)
#     model.load_state_dict(torch.load("models/cvae_model_v020.pth", weights_only=True))
#     model.eval()
#     return model
