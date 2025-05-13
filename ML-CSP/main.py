import pandas as pd
import numpy as np
from ase import Atoms
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from ase.constraints import FixSymmetry
from ase.io import write, read
from tqdm import tqdm
import time, os, glob, warnings, sys, random
import re
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.cif import CifWriter
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import autode
import concurrent.futures
from pymatgen.core import Structure
import yaml
import pickle
from multiprocessing import get_context
from ase.neighborlist import NeighborList
from pyxtal import pyxtal
from pyxtal.lattice import Lattice
from utility import find_wyckoff_pos
import math
from scipy.optimize import NonlinearConstraint
import ml
import traceback

warnings.simplefilter('ignore')

# Get the path to the config.yaml file from command-line arguments
import sys

if len(sys.argv) > 1:
    config_path = sys.argv[1]

# Load the config.yaml file
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Access the parameters
smiles = config['smiles']
num_conformers = config['num_conformers']
num_structures = config['num_structures']
root_folder = config['root_folder']
ml_mode = config['ml_mode']
sg_mode = config['sg_mode']
Z_mode = config['Z_mode']
density_mode = config['density_mode']
model_name = config['model_name']
list_numIons = config['list_numIons']
conformer_mode = config['conformer_mode']
try:
    lattice_mode = config['lattice_mode']
except:
    lattice_mode = None

# Preset variables
# sg99 = sorted([2, 4, 5, 9, 14, 15, 19, 33, 61, 1, 29, 7, 13, 18, 20, 43, 56, 60, 76, 78, 88, 92, 96, 144, 145, 148, 169, 170])
# sg99 = sorted([2, 4, 5, 9, 14, 15, 19, 33, 61, 1, 29, 7, 13, 18, 20, 43, 56, 60, 76, 78, 86, 88, 92, 96, 144, 145, 169, 170])
sg99 = sorted([14, 2, 19, 4, 61, 15, 33, 29, 5, 9, 1, 60, 7, 18, 56, 43, 88, 13, 145, 92, 144, 76, 96, 169, 78, 170, 20, 86, 41, 45, 114, 146])
# sg99_ml_rev = sorted([1, 2, 4, 5, 7, 9, 13, 14, 18, 19, 29, 33, 76, 78, 144, 145])
# sg99_ml_large = sorted([1, 2, 4, 5, 7, 9, 13, 14, 15, 18, 19, 20, 29, 33, 56, 60, 61, 76, 78, 92, 96, 144, 145, 148, 169, 170])
sg95 = sorted([2, 4, 5, 9, 14, 15, 19, 33, 61, 1, 29])
n_workers = 24
volume_factor = 1.1
ml_sg_proba = ml.calc_sg_proba_trained_model(smiles)
ml_z_proba = ml.calc_z_proba_trained_model(smiles)
ml_density = ml.calc_density_trained_model(smiles)
mw = ml.calculate_molecular_weight(smiles)
if ml_mode == 'proba_threshold_random':
    if sg_mode == 'ml':
        # sg99 = ml.get_valid_classes(ml_sg_proba, sg99, threshold=1e-10)
        sg99 = ml.get_valid_classes(ml_sg_proba, sg99, threshold=1e-2)
        ml_sg_proba = np.ones(len(sg99))/len(sg99)
    if Z_mode == 'ml':
        list_numIons = ml.get_valid_classes(ml_z_proba, list_numIons, threshold=0.01)
        ml_z_proba = np.ones(len(list_numIons))/len(list_numIons)
else:
    if sg_mode == 'ml':
        sg99 = ml.get_valid_classes(ml_sg_proba, sg99, threshold=0)
    if Z_mode == 'ml':
        list_numIons = ml.get_valid_classes(ml_z_proba, list_numIons, threshold=0)

print(f'Molecular Weight: {mw:.3f} g/mol')
print(f'Predicted density: {ml_density:.3f} g/cm3')
print(f'SG candidates: {sg99}')
print(f'Z candidates: {list_numIons}')
print(f'Lattice mode: {lattice_mode}')
if lattice_mode == 'VAE':
    import torch
    loaded_cvae = ml.load_model()

##################################################
##### Generate molecules class ###################
##################################################
def generate_conformers(smiles, num_conformers):
    conformers = []
    mol = autode.Molecule(smiles=smiles)
    mol.populate_conformers(n_confs=num_conformers)
    n = len(mol.conformers)
    print(f"{n} conformers found (n_confs:{num_conformers})")

    for i, ade_mol in enumerate(mol.conformers):
        ele = [atom.atomic_symbol for atom in ade_mol.atoms]
        pos = np.array([atom.coord for atom in ade_mol.atoms])
        atoms = Atoms(ele, pos)
        conformers.append(atoms)

    return conformers


def write_conformers_to_xyz(conformers, xyz_folder, prefix):
    for i, atoms in enumerate(conformers):
        xyz_file = f"{xyz_folder}{prefix}_conformer_{i + 1}.xyz"
        write(xyz_file, atoms)


def count_atoms_in_xyz(file_path):
    with open(file_path, 'r') as file:
        num_atoms = int(file.readline().strip())
    return num_atoms


##################################################
##### Generate crystals class #####
##################################################

def choice_sg(sg_mode):
    if sg_mode == 'all':
        sg = int(np.random.randint(1, 231, 1))
    elif sg_mode == 'random': #'sg99':
        sg = np.random.choice(sg99)
    elif sg_mode == 'sg95':
        sg = np.random.choice(sg95)
    elif sg_mode == 'ml':
        # sg = np.random.choice(sg99_ml_rev, p=ml_sg_proba)
        sg = np.random.choice(sg99, p=ml_sg_proba)
    elif type(sg_mode) is int:
        sg = sg_mode

    return sg


def choice_mol(mol_files):
    return str(np.random.choice(mol_files))


def choice_numIons(candidates, Z_mode=None, sg=None):
    if Z_mode == 'ml':
        numIons = np.random.choice(candidates, p=ml_z_proba)
    elif Z_mode == 'random':
        numIons = np.random.choice(candidates)
    elif Z_mode == 'sg_dependent':
        if sg in [1]:
            numIons = 1
        elif sg in [2,4,7]:
            numIons = 2
        elif sg in [144, 145]:
            numIons = 3
        elif sg in [14,19,33,29,9,5,18,13,76,78]:
            numIons = 4
        elif sg in [169,170]:
            numIons = 6
        elif sg in [61,15,60,56,92,96,20,86,41,45,114]:
            numIons = 8
        elif sg in [146]:
            numIons = 9
        elif sg in [43,88]:
            numIons = 16
    return numIons


####### ML lattice generation (2024/10/29 inserted) #######
def generate_lattice_constants(sg):
    length_range = (3, 50)
    angle_range = (60, 120)

    a = round(random.uniform(*length_range), 3)
    b = round(random.uniform(*length_range), 3)
    c = round(random.uniform(*length_range), 3)
    alpha = round(random.uniform(*angle_range), 3)
    beta = round(random.uniform(*angle_range), 3)
    gamma = round(random.uniform(*angle_range), 3)
    if 3 <= sg <= 15:
        alpha, gamma = 90, 90
    elif 16 <= sg <= 74:
        alpha, beta, gamma = 90, 90, 90
    elif 75 <= sg <= 142:
        b = a
        alpha, beta, gamma = 90, 90, 90
    elif 143 <= sg <= 167:
        # b, c = a, a
        # beta, gamma = alpha, alpha
        b = a
        alpha, beta, gamma = 90, 90, 120
    elif 168 <= sg <= 194:
        b = a
        alpha, beta, gamma = 90, 90, 120
    elif 195 <= sg <= 230:
        b, c = a, a
        alpha, beta, gamma = 90, 90, 90
    return a, b, c, alpha, beta, gamma


def generate_lattice_constants_vae(sg):
    with torch.no_grad():
        latent_sample = torch.randn(1, 3)
        condition_sample = torch.tensor([[sg, ml_density, mw]])
        generated_constants = loaded_cvae.decoder(torch.cat([latent_sample, condition_sample], dim=1))

    a = round(generated_constants[0][0].item(), 3)
    b = round(generated_constants[0][1].item(), 3)
    c = round(generated_constants[0][2].item(), 3)
    alpha = round(generated_constants[0][3].item(), 3)
    beta = round(generated_constants[0][4].item(), 3)
    gamma = round(generated_constants[0][5].item(), 3)
    if 3 <= sg <= 15:
        alpha, gamma = 90, 90
    elif 16 <= sg <= 74:
        alpha, beta, gamma = 90, 90, 90
    elif 75 <= sg <= 142:
        b = a
        alpha, beta, gamma = 90, 90, 90
    elif 143 <= sg <= 167:
        # b, c = a, a
        # beta, gamma = alpha, alpha
        b = a
        alpha, beta, gamma = 90, 90, 120
    elif 168 <= sg <= 194:
        b = a
        alpha, beta, gamma = 90, 90, 120
    elif 195 <= sg <= 230:
        b, c = a, a
        alpha, beta, gamma = 90, 90, 90
    return a, b, c, alpha, beta, gamma


def calculate_volume(a, b, c, alpha, beta, gamma):
    # Degree to Radian
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    # Calculate volume
    volume = (
            a * b * c *
            np.sqrt(
                1 - np.cos(alpha_r) ** 2 - np.cos(beta_r) ** 2 - np.cos(gamma_r) ** 2
                + 2 * np.cos(alpha_r) * np.cos(beta_r) * np.cos(gamma_r)
            )
    )
    return volume


def check_density(volume, Z):
    # 計算された密度をチェック
    avogadro_number = 6.022e23  # アボガドロ数 (/mol)
    calculated_density = (Z * mw) / (avogadro_number * volume * 1e-24)  # 密度の計算 (g/cm^3)
    return np.isclose(calculated_density, ml_density, atol=0.5)  # ある程度の許容誤差で一致を確認


def choice_lattice(list_numIons, sg_mode, lattice_mode=None):
    sg = choice_sg(sg_mode)
    Z = choice_numIons(list_numIons, Z_mode, sg)
    found = False
    # count = 0
    while not found:
        # count+=1
        if lattice_mode == 'VAE':
            a, b, c, alpha, beta, gamma = generate_lattice_constants_vae(sg)
        else:
            a, b, c, alpha, beta, gamma = generate_lattice_constants(sg)
        volume = calculate_volume(a, b, c, alpha, beta, gamma)
        if check_density(volume, Z):
            found = True
            # print(count, found, a, b, c, alpha, beta, gamma)
    return sg, Z, a, b, c, alpha, beta, gamma


def gen_one_structure_from_lattice(xyz_file, params):
    sg = params[0]
    num_mol_primitive_cell = params[1]
    while True:
        wyck_pos = find_wyckoff_pos(sg)
        system = sg2system(sg)
        crystal = pyxtal(molecular=True)
        lattice = Lattice.from_para(params[2], params[3], params[4], params[5], params[6], params[7], ltype=system)
        # print(lattice)
        crystal.from_random(
            dim=3,
            group=sg,
            species=[xyz_file],
            numIons=[num_mol_primitive_cell],
            lattice=lattice,
            sites=[{wyck_pos: [round(random.uniform(0,1),3), round(random.uniform(0,1),3), round(random.uniform(0,1),3)]}],
            max_count=1,
        )
        # print(params)
        # 条件を確認
        if crystal.get_neighboring_molecules()[0][0] > 0.6:
            break  # 条件が満たされたらループを抜ける

    return crystal.to_ase(resort=False), sg, num_mol_primitive_cell

#####################################



def gen_one_structure_from_one_xyz(xyz_file, num_mol_primitive_cell, sg):
    while True:
        crystal = pyxtal(molecular=True)
        crystal.from_random(
            dim=3,
            group=sg,
            species=[xyz_file],
            numIons=[num_mol_primitive_cell],
            factor=volume_factor
        )

        # 条件を確認
        if crystal.get_neighboring_molecules()[0][0] > 0.6:
            break  # 条件が満たされたらループを抜ける

    return crystal.to_ase(resort=False), sg, num_mol_primitive_cell


def gen_one_structure_from_multi_xyz(xyz_file, num_mol_primitive_cell, sg):
    crystal = pyxtal(molecular=True)
    crystal.from_random(
        dim=3,
        group=sg,
        species=xyz_file,
        numIons=num_mol_primitive_cell
    )
    return crystal.to_ase(resort=False), sg, num_mol_primitive_cell


def check_distances(atoms, cutoff):
    nl = NeighborList([cutoff / 2] * len(atoms), self_interaction=False, bothways=True)
    nl.update(atoms)

    for i in range(len(atoms)):
        indices, offsets = nl.get_neighbors(i)
        for j, offset in zip(indices, offsets):
            if i < j:
                distance = np.linalg.norm(atoms.positions[i] - (atoms.positions[j] + np.dot(offset, atoms.get_cell())))
                if distance < cutoff:
                    return False
    return True


def count_asym_unit(ase_atoms):
    adaptor = AseAtomsAdaptor()
    pmg_structure = adaptor.get_structure(ase_atoms)
    sga = SpacegroupAnalyzer(pmg_structure)
    symmetrized_structure = sga.get_symmetrized_structure()
    return len(symmetrized_structure.equivalent_sites)


def generate_crystal_structures(mol_files, num_structures, sg_mode='all', atoms_dict=None, conformer_mode='search'):
    num_atoms = count_atoms_in_xyz(mol_files[0])
    if atoms_dict is None:
        structures = {}
    else:
        structures = atoms_dict
    if conformer_mode == 'search':
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            while len(structures) < num_structures:
                if density_mode == 'random':
                    futures = [
                        executor.submit(
                            gen_one_structure_from_one_xyz,
                            choice_mol(mol_files),
                            choice_numIons(list_numIons, Z_mode),
                            choice_sg(sg_mode)
                        )
                        for _ in range(num_structures - len(structures))
                    ]
                elif density_mode == 'ml':
                    futures = [
                        executor.submit(
                            gen_one_structure_from_lattice,
                            choice_mol(mol_files),
                            choice_lattice(list_numIons, sg_mode)
                        )
                        for _ in range(num_structures - len(structures))
                    ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        crystal, sg, num_mol_primitive_cell = future.result()
                        # if check_distances(crystal, 0.85) and count_asym_unit(crystal) == num_atoms:
                        if count_asym_unit(crystal) == num_atoms:
                            id = len(structures)
                            structures[f'ID_{id}'] = crystal
                            print(f'ID_{id} (SG: {sg}, numIons: {num_mol_primitive_cell}, n_atoms: {num_atoms}, n_asym_unit: {count_asym_unit(crystal)}) generated')
                    except:
                        continue
                    # except Exception as e:
                    #     print(f'Error generating structure: {e}')
    elif conformer_mode == 'predefined':
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            while len(structures) < num_structures:
                futures = [
                    executor.submit(
                        gen_one_structure_from_multi_xyz,
                        mol_files,
                        list_numIons,
                        choice_sg(sg_mode)
                    )
                    for _ in range(num_structures - len(structures))
                ]

                for future in concurrent.futures.as_completed(futures):
                    try:
                        crystal, sg, num_mol_primitive_cell = future.result()
                        if check_distances(crystal, 0.9):
                            id = len(structures)
                            structures[f'ID_{id}'] = crystal
                            print(f'ID_{id} (SG: {sg}, numIons: {num_mol_primitive_cell}) generated')
                    except:
                        continue

    return structures


def get_prev_atoms_dict(cif_path):
    from pymatgen.io.cif import CifParser
    cifparser = CifParser(cif_path)
    structures = cifparser.get_structures()
    structures_keys = list(cifparser.as_dict().keys())
    atoms_list = [AseAtomsAdaptor.get_atoms(structure) for structure in structures]
    atoms_dict = {}

    for i in range(len(atoms_list)):
        key = structures_keys[i]
        atoms = atoms_list[i]
        atoms_dict[f'{key}'] = atoms

    return atoms_dict


def write_pkl(atoms_dict, output_file):
    with open(output_file, 'wb') as f:
        pickle.dump(atoms_dict, f)


def read_pkl(input_file):
    with open(input_file, 'rb') as f:
        return pickle.load(f)

# For ML
def get_valid_lattices(sg, mw, Z):
    result = []

    interval = 0.5
    for a in tqdm(np.arange(2, 51, interval)):
        for b in np.arange(2, 51, interval):
            ab = a * b
            # a*bが0の場合はスキップ（実際には0にはなりませんが、安全のため）
            if ab == 0:
                continue
            # cの最小値と最大値を計算
            min_c = max(math.ceil(501 / ab), 2)
            max_c = min(math.floor(999 / ab), 50)
            # 有効なcの範囲が存在するか確認
            if min_c > max_c:
                continue
            for c in np.arange(int(min_c), int(max_c) + 1, interval):
                density = calculate_crystal_density(mw, a , b, c, Z)
                if 1.51 < density < 1.53:
                    result.append((a,b,c))
    return result



##################################################
##### Optimize crystals class #####
##################################################
def set_calculator(model_name):
    if model_name == 'CHGNet':
        from chgnet.model import CHGNet
        from chgnet.model.dynamics import CHGNetCalculator
        import torch
        model = CHGNet().load()
        weights = torch.load('../../chgnet/result/FT_PFPv5_rslt_10000_240321_AdamW_lr0.01/epoch19_e0_f0_sNA_mNA.pth.tar')
        model.load_state_dict(state_dict=weights['model']['state_dict'])
        calculator = CHGNetCalculator(model=model, use_device='cuda:0')
    elif model_name == 'PFP':
        from pfp_api_client.pfp.calculators.ase_calculator import ASECalculator
        from pfp_api_client.pfp.estimator import Estimator, EstimatorCalcMode
        model = Estimator(calc_mode=EstimatorCalcMode.CRYSTAL_U0_PLUS_D3, model_version="v5.0.0")
        calculator = ASECalculator(model)
    return calculator


def optimize_one_structure(atoms_in, calculator, logfile=None, trajectory=None):
    atoms = atoms_in.copy()
    atoms.calc = calculator
    atoms.set_constraint([FixSymmetry(atoms)])
    ecf = FrechetCellFilter(atoms)
    opt = LBFGS(ecf, logfile=logfile, trajectory=trajectory)
    opt.run(fmax=0.2, steps=2000)
    label = False
    try:
        opt.run(fmax=0.05, steps=2000)
        label = opt.converged()
    except Exception as e:
        traceback.print_exc()
    finally:
        atoms.constraints = None
        # print(atoms.cell.cellpar())
    return atoms, label


def optimize_structure(key, atoms, model_name):
    calculator = set_calculator(model_name)
    opt_atoms, label = optimize_one_structure(atoms, calculator)
    struct = AseAtomsAdaptor.get_structure(opt_atoms)
    struct = struct.as_dict()
    struct['properties']['energy'] = opt_atoms.get_potential_energy()
    return key, opt_atoms.todict(), struct, label


def replace_data_block(cif_text, new_data_block):
    pattern = r'(?<=\n)data_.*?(?=\n)'
    replaced_text = re.sub(pattern, new_data_block, cif_text, count=1)
    return replaced_text


def write_cif(atoms_dict, output_file):
    with open(output_file, 'w') as f:
        for key, atoms in atoms_dict.items():
            struct = AseAtomsAdaptor.get_structure(atoms)
            try:
                cif = CifWriter(struct, symprec=0.1)
            except:
                continue
            new_data_block = f'data_{key}'
            modified_cif = replace_data_block(str(cif), new_data_block)
            f.write(modified_cif)


##################################################
# Bayesian optimization for structure generation & optimization
##################################################

def sg2system(sg):
    if 1 <= sg <= 2:
        system = 'triclinic'
    elif 3 <= sg <= 15:
        system = 'monoclinic'
    elif 16 <= sg <= 74:
        system = 'orthorhombic'
    elif 75 <= sg <= 142:
        system = 'tetragonal'
    elif 143 <= sg <= 167:
        system = 'trigonal'
    elif 168 <= sg <= 194:
        system = 'hexagonal'
    elif 195 <= sg <= 230:
        system = 'cubic'
    return system


def adjust_lattice(sg, lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma):
    if 3 <= sg <= 15:
        lat_alpha, lat_gamma = 90, 90
    elif 16 <= sg <= 74:
        lat_alpha, lat_beta, lat_gamma = 90, 90, 90
    elif 75 <= sg <= 142:
        lat_b = lat_a
        lat_alpha, lat_beta, lat_gamma = 90, 90, 90
    elif 143 <= sg <= 167:
        lat_b, lat_c = lat_a, lat_a
        lat_beta, lat_gamma = lat_alpha, lat_alpha
    elif 168 <= sg <= 194:
        lat_b = lat_a
        lat_alpha, lat_beta, lat_gamma = 90, 90, 120
    elif 195 <= sg <= 230:
        lat_b, lat_c = lat_a, lat_a
        lat_alpha, lat_beta, lat_gamma = 90, 90, 90
    return lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma


def gen_one_structure_for_bayes(xyz_file, num_mol_primitive_cell, sg, volume):
    crystal = pyxtal(molecular=True)
    lattice = Lattice(sg2system(sg), volume=volume)
    crystal.from_random(
        dim=3,
        group=sg,
        species=[xyz_file],
        numIons=[num_mol_primitive_cell],
        factor=volume_factor,
        lattice=lattice
    )
    return crystal.to_ase(resort=False), sg, num_mol_primitive_cell


def gen_bayes_structures(mol_files, sg, volume):
    attempts = 0
    max_attempts = 30  # 最大試行回数を設定

    while attempts < max_attempts:
        try:
            crystal = pyxtal(molecular=True)
            # crystal.lattice_attempts = 5
            # crystal.coord_attempts = 5
            lattice = Lattice(sg2system(sg), volume=volume)
            crystal.from_random(
                dim=3,
                group=sg,
                species=[choice_mol(mol_files)],
                numIons=[choice_numIons(list_numIons)],
                factor=volume_factor,
                lattice=lattice
            )
            if crystal.valid:
                if check_distances(crystal.to_ase(), 0.9):
                    print(f"Valid crystal generated after {attempts + 1} attempts.")
                    return crystal

        except Exception as e:
            print(f"Error occurred: {str(e)}. Retrying...")
        return crystal


def calculate_unit_cell_volume(a, b, c, alpha, beta, gamma):
    # 角度をラジアンに変換
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = math.radians(gamma)

    # 体積計算の公式
    volume = a * b * c * math.sqrt(
        1 - math.cos(alpha_rad) ** 2 - math.cos(beta_rad) ** 2 - math.cos(gamma_rad) ** 2
        + 2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)
    )
    return volume


def gen_single_structure(mol_files, sg_idx, numIons, lat_a, lat_b, lat_c,
                         lat_alpha, lat_beta, lat_gamma, wyck_site_x, wyck_site_y, wyck_site_z,
                         phi_1, phi_2, phi_3):

    # Adjust parameters for crystal structure generation
    sg = sg99[int(sg_idx)]
    # numIons = int(numIons)
    phi_1 = int(phi_1) * 30
    phi_2 = int(phi_2) * 30
    phi_3 = int(phi_3) * 30
    wyck_site_x = round(wyck_site_x * 10) / 10
    wyck_site_y = round(wyck_site_y * 10) / 10
    wyck_site_z = round(wyck_site_z * 10) / 10
    lat_a = round(lat_a) #round(lat_a * 10) / 10
    lat_b = round(lat_b) #round(lat_b * 10) / 10
    lat_c = round(lat_c) #round(lat_c * 10) / 10
    lat_alpha = round(lat_alpha) * 5
    lat_beta = round(lat_beta) * 5
    lat_gamma = round(lat_gamma) * 5

    lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma = adjust_lattice(sg, lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma)
    wyck_pos = find_wyckoff_pos(sg)
    sites = {wyck_pos: [wyck_site_x, wyck_site_y, wyck_site_z]}
    mol = choice_mol(mol_files)
    list_angle = [phi_1, phi_2, phi_3]

    crystal = pyxtal(molecular=True)
    lattice = Lattice.from_para(lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma, ltype=sg2system(sg))
    crystal.from_random(
        dim=3,
        group=sg,
        species=[mol],
        numIons=[numIons],
        lattice=lattice,
        sites=[sites]
    )
    crystal.mol_sites[0].update_orientation(list_angle)
    return crystal


def gen_bayes_structures2(mol_files, sg_idx, numIons, lat_a, lat_b, lat_c,
                          lat_alpha, lat_beta, lat_gamma, wyck_site_x, wyck_site_y, wyck_site_z,
                          phi_1, phi_2, phi_3, num_structures=1, max_workers=24):
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(gen_single_structure, mol_files, sg_idx, numIons, lat_a, lat_b, lat_c,
                                   lat_alpha, lat_beta, lat_gamma, wyck_site_x, wyck_site_y, wyck_site_z,
                                   phi_1, phi_2, phi_3) for _ in range(num_structures)]
        results = []
        for future in concurrent.futures.as_completed(futures):
            crystal = future.result()
            if crystal is not None:
                results.append(crystal)

    return results[0]


def bayes_get_opt_structures(
        mol_files,
        num_structures,
        model_name,
        init_cif_path,
        init_pkl_path,
        opt_cif_path,
        opt_pkl_path,
        opt_result_path,
):

    opt_atoms_dict = {}
    init_atoms_dict = {}
    results = {}

    def objective_energy(lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma, wyck_site_x, wyck_site_y, wyck_site_z, phi_1, phi_2, phi_3):
        # phi_1, phi_2, phi_3 = X[:,0], X[:,1], X[:,2]
        sg_idx = 10
        numIons = 4
        # lat_a = 3.7818
        # lat_b = 7.8320
        # lat_c = 26.8720
        # lat_alpha, lat_beta, lat_gamma = 90, 90, 90
        # wyck_site_x = 0.5049
        # wyck_site_y = 0.49
        # wyck_site_z = 0.6251
        # l1 = Lattice.from_para(3.7818, 7.8320, 26.8720, 90.0000, 90.0000, 90.0000, ltype='orthorhombic')
        # phi_1, phi_2, phi_3 = int(phi_1) * 30, int(phi_2) * 30, int(phi_3) * 30

        try:
            crystal = gen_single_structure(mol_files, sg_idx, numIons, lat_a, lat_b, lat_c,
                                           lat_alpha, lat_beta, lat_gamma, wyck_site_x, wyck_site_y, wyck_site_z,
                                           phi_1, phi_2, phi_3)
            # crystal = pyxtal(molecular=True)
            # crystal.from_random(
            #     dim=3,
            #     group=19,  # fix space group
            #     numIons=[4],
            #     species=['../ACNAQU01/xyz/O=C1C(=O)c2cccc3cccc1c23_conformer_1.xyz'],  # fix molecular conformation
            #     lattice=l1,  # fix lattice
            #     sites=[{'4a': [wyck_site_x, wyck_site_y, wyck_site_z]}],  # fix wyckoff position (0<=x<1 each)
            # )
            # list_angle = [phi_1, phi_2, phi_3]
            # crystal.mol_sites[0].update_orientation(list_angle)

            min_distance = crystal.get_neighboring_molecules()[0][0]
            # if min_distance > 0.6 and min_distance < 1:
            density = crystal.to_pymatgen().density
            print(f'{density:.3f} (g/cm^3)')
            if density > 1.0 and min_distance > 0.6:
                key = f'ID_{len(opt_atoms_dict)}'
                print(key)
                init_atoms = crystal.to_ase()
                init_atoms_dict[key] = init_atoms
                key, opt_atoms, struct_dict, label = optimize_structure(key, init_atoms, model_name)

                opt_atoms = Atoms.fromdict(opt_atoms)
                opt_atoms.calc = set_calculator(model_name)
                energy_per_atom = opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms()
                struct = Structure.from_dict(struct_dict)
                results[key] = {
                    'density': float(struct.density),
                    'energy_per_atom': energy_per_atom,
                    'sg_symbol': struct.get_space_group_info()[0],
                    'sg_number': struct.get_space_group_info()[1],
                    'opt': label
                }
                opt_atoms.calc = None
                opt_atoms_dict[key] = opt_atoms

                return -energy_per_atom  # target
            else:
                return 0
        except:
            return 0

    def objective_density(lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma, wyck_site_x, wyck_site_y, wyck_site_z, phi_1, phi_2, phi_3):
        sg_idx = 10
        numIons = 4
        if lat_a * lat_b * lat_c > 1000: # Constraint on cell volume
            return -10
        else:
            try:
                crystal = gen_single_structure(mol_files, sg_idx, numIons, lat_a, lat_b, lat_c,
                                               lat_alpha, lat_beta, lat_gamma, wyck_site_x, wyck_site_y, wyck_site_z,
                                               phi_1, phi_2, phi_3)

                min_distance = crystal.get_neighboring_molecules()[0][0]
                density = crystal.to_pymatgen().density
                if min_distance > 0.6:
                    key = f'ID_{len(opt_atoms_dict)}'
                    print(key)
                    init_atoms = crystal.to_ase()
                    init_atoms_dict[key] = init_atoms
                    key, opt_atoms, struct_dict, label = optimize_structure(key, init_atoms, model_name)

                    opt_atoms = Atoms.fromdict(opt_atoms)
                    opt_atoms.calc = set_calculator(model_name)
                    energy_per_atom = opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms()
                    struct = Structure.from_dict(struct_dict)
                    results[key] = {
                        'density': float(struct.density),
                        'energy_per_atom': energy_per_atom,
                        'sg_symbol': struct.get_space_group_info()[0],
                        'sg_number': struct.get_space_group_info()[1],
                        'opt': label
                    }
                    opt_atoms.calc = None
                    opt_atoms_dict[key] = opt_atoms

                    return density  # target
                else:
                    return 0
            except:
                return 0


    def constraint_function(lat_a, lat_b, lat_c, lat_alpha, lat_beta, lat_gamma, wyck_site_x, wyck_site_y, wyck_site_z, phi_1, phi_2, phi_3):
        const1 = lat_a + lat_b + lat_c
        const2 = lat_alpha + lat_beta + lat_gamma
        return np.array([const1, const2])

    constraint_lower = np.array([10, 100])
    constraint_upper = np.array([50, 300])
    constraint = NonlinearConstraint(constraint_function, constraint_lower, constraint_upper)

    # For bayes_opt package----------------------------
    from bayes_opt import BayesianOptimization
    #
    # small_val = 1e-10
    # pbounds = {
    #     'sg_idx': (0, len(sg99) - small_val),
    #     'numIons': (1, 5 - small_val),
    #     'lat_a': (2, 50),
    #     'lat_b':(2, 50),
    #     'lat_c': (2, 50),
    #     'lat_alpha': (70, 120),
    #     'lat_beta': (70, 120),
    #     'lat_gamma': (70, 120),
    #     'wyck_site_x': (0, 1),
    #     'wyck_site_y': (0, 1),
    #     'wyck_site_z': (0, 1),
    #     'phi_1': (-6, 7 - small_val),
    #     'phi_2': (-6, 7 - small_val),
    #     'phi_3': (-6, 7 - small_val),
    # }
    small_val = 1e-6
    pbounds = {
        'lat_a': (2, 50),
        'lat_b': (2, 50),
        'lat_c': (2, 50),
        'lat_alpha': (18, 18.1), #(12, 24),
        'lat_beta': (18, 18.1), #(12, 24),
        'lat_gamma': (18, 18.1), #(12, 24),
        'wyck_site_x': (0, 1),
        'wyck_site_y': (0, 1),
        'wyck_site_z': (0, 1),
        'phi_1': (-6, 7 - small_val),
        'phi_2': (-6, 7 - small_val),
        'phi_3': (-6, 7 - small_val)
    }
    #
    optimizer = BayesianOptimization(
        f=objective_density,
        # constraint=constraint,
        pbounds=pbounds,
        random_state=1,
        # allow_duplicate_points=True
    )
    # Generate initial structures
    n_initial_points = 100
    # for i in range(n_initial_points):
    #     optimizer.probe(
    #         params={param: np.random.uniform(bounds[0], bounds[1])
    #                 for param, bounds in pbounds.items()},
    #         lazy=True,
    #     )
    optimizer.maximize(init_points=n_initial_points, n_iter=1000)
    # optimizer.maximize(init_points=0, n_iter=num_structures - 10)
    # -------------------------------------------------------------------

    # Here, I add GPyOpt-------------------------------------------------
    # import GPyOpt
    # from GPyOpt.methods import BayesianOptimization

    # 探索空間の定義
    # pbounds = [{'name': 'sg_idx', 'type': 'categorical', 'domain': tuple(np.arange(0, len(sg99), 1))},
    #            {'name': 'numIons', 'type': 'discrete', 'domain': (1, 2, 3, 4)},
    #            {'name': 'lat_a', 'type': 'discrete', 'domain': tuple(np.arange(2, 50, 0.1))},
    #            {'name': 'lat_b', 'type': 'discrete', 'domain': tuple(np.arange(2, 50, 0.1))},
    #            {'name': 'lat_c', 'type': 'discrete', 'domain': tuple(np.arange(2, 50, 0.1))},
    #            {'name': 'lat_alpha', 'type': 'discrete', 'domain': tuple(np.arange(60, 120.1, 1))},
    #            {'name': 'lat_beta', 'type': 'discrete', 'domain': tuple(np.arange(60, 120.1, 1))},
    #            {'name': 'lat_gamma', 'type': 'discrete', 'domain': tuple(np.arange(60, 120.1, 1))},
    #            {'name': 'wyck_site_x', 'type': 'discrete', 'domain': tuple(np.arange(0, 1, 0.1))},
    #            {'name': 'wyck_site_y', 'type': 'discrete', 'domain': tuple(np.arange(0, 1, 0.1))},
    #            {'name': 'wyck_site_z', 'type': 'discrete', 'domain': tuple(np.arange(0, 1, 0.1))},
    #            {'name': 'phi_1', 'type': 'discrete', 'domain': tuple(np.arange(-180, 180, 30))},
    #            {'name': 'phi_2', 'type': 'discrete', 'domain': tuple(np.arange(-180, 180, 30))},
    #            {'name': 'phi_3', 'type': 'discrete', 'domain': tuple(np.arange(-180, 180, 30))}]
    # pbounds = [{'name': 'phi_1', 'type': 'discrete', 'domain': tuple(np.arange(-180, 180, 30))},
    #            {'name': 'phi_2', 'type': 'discrete', 'domain': tuple(np.arange(-180, 180, 30))},
    #            {'name': 'phi_3', 'type': 'discrete', 'domain': tuple(np.arange(-180, 180, 30))}]

    # constrains = [{'name': 'constr_1', 'constraint': 'x[:,2]+x[:,3]+x[:,4]-100'},
    #               {'name': 'constr_2', 'constraint': '-(x[:,2]+x[:,3]+x[:,4])+20'},
    #               {'name': 'constr_3', 'constraint': 'x[:,5]+x[:,6]+x[:,7]-300'},
    #               {'name': 'constr_4', 'constraint': '-(x[:,5]+x[:,6]+x[:,7])-250'}]

    # バッチベイズ最適化の設定
    # optimizer = BayesianOptimization(f=objective_function,
    #                                  domain=pbounds,
    #                                  model_type='GP',
    #                                  acquisition_type='EI',
    #                                  normalize_Y=True,
    #                                  initial_design_numdata=20,
    #                                  batch_size=5,
    #                                  num_cores=4,
    #                                  maximize=False)
    #
    # # 最適化の実行
    # max_iter = 10
    # optimizer.run_optimization(max_iter, verbosity=True)
    # --------------------------------------------------------------------

    # Save locally optimized structures
    write_pkl(init_atoms_dict, init_pkl_path)
    write_cif(init_atoms_dict, init_cif_path)
    write_pkl(opt_atoms_dict, opt_pkl_path)
    write_cif(opt_atoms_dict, opt_cif_path)
    df_results = pd.DataFrame(results).T
    df_results.to_csv(opt_result_path)


##################################################
# Press the green button in the gutter to run the script.
##################################################
if __name__ == '__main__':
    start = time.time()

    init_cif_path = root_folder + 'init_structures.cif'
    init_pkl_path = root_folder + 'init_structures.pkl'
    opt_cif_path = root_folder + 'opt_structures.cif'
    opt_pkl_path = root_folder + 'opt_structures.pkl'
    opt_result_path = root_folder + 'opt_results.csv'
    xyz_folder = root_folder + 'xyz/'

    run_mode = None #'Bayes'
    if run_mode == 'Bayes':
        if not os.path.exists(xyz_folder):
            if conformer_mode == 'search':
                generated_conformers = generate_conformers(smiles, num_conformers)
                os.makedirs(xyz_folder, exist_ok=True)
                write_conformers_to_xyz(generated_conformers, xyz_folder, smiles)
            mol_files = glob.glob(xyz_folder + '*.xyz')
        else:
            mol_files = glob.glob(xyz_folder + '*.xyz')
        bayes_get_opt_structures(
            mol_files,
            num_structures,
            model_name,
            init_cif_path,
            init_pkl_path,
            opt_cif_path,
            opt_pkl_path,
            opt_result_path,
        )
        elapsed = time.time() - start
        print(f'{num_structures} have been optimized. (Total: {elapsed:.1f} sec)')
        sys.exit()

    # Add new structures
    if os.path.exists(init_pkl_path):
        # Load initial structures
        init_atoms_dict = read_pkl(init_pkl_path)
        generated_structures = init_atoms_dict

        if len(generated_structures) != num_structures:
            # Generate crystal structures
            mol_files = glob.glob(xyz_folder + '*.xyz')
            generated_structures = generate_crystal_structures(mol_files, num_structures, sg_mode=sg_mode,
                                                               atoms_dict=init_atoms_dict,
                                                               conformer_mode=conformer_mode)

            # Save initial structures
            write_pkl(generated_structures, init_pkl_path)
            write_cif(generated_structures, init_cif_path)
            elapsed = time.time() - start
            print(f'{num_structures} have been generated. ({elapsed:.1f} sec)')

    # First structure generation & optimization
    if not os.path.exists(init_pkl_path):

        mol_files = glob.glob(xyz_folder + '*.xyz')
        # Generate conformers
        if len(mol_files) == 0 and conformer_mode == 'search':
            generated_conformers = generate_conformers(smiles, num_conformers)
            os.makedirs(xyz_folder, exist_ok=True)
            write_conformers_to_xyz(generated_conformers, xyz_folder, smiles)
            mol_files = glob.glob(xyz_folder + '*.xyz')

        # Generate crystal structures
        generated_structures = generate_crystal_structures(mol_files, num_structures, sg_mode=sg_mode,
                                                           conformer_mode=conformer_mode)

        # Save initial structures
        write_pkl(generated_structures, init_pkl_path)
        write_cif(generated_structures, init_cif_path)
        elapsed = time.time() - start
        print(f'{num_structures} have been generated. ({elapsed:.1f} sec)')

        # Optimize structures
        opt_atoms_dict, results = {}, {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(optimize_structure, key, atoms, model_name) for key, atoms in
                       generated_structures.items()]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                key, opt_atoms, struct_dict, label = future.result()
                opt_atoms = Atoms.fromdict(opt_atoms)
                opt_atoms.calc = set_calculator(model_name)
                struct = Structure.from_dict(struct_dict)
                results[key] = {
                    'density': float(struct.density),
                    'energy_per_atom': opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms(),
                    'sg_symbol': struct.get_space_group_info()[0],
                    'sg_number': struct.get_space_group_info()[1],
                    'opt': label
                }
                opt_atoms.calc = None
                opt_atoms_dict[key] = opt_atoms

        # Save locally optimized structures
        write_pkl(opt_atoms_dict, opt_pkl_path)
        write_cif(opt_atoms_dict, opt_cif_path)
        df_results = pd.DataFrame(results).T
        df_results.to_csv(opt_result_path)
        elapsed = time.time() - start
        print(f'{num_structures} have been optimized. (Total: {elapsed:.1f} sec)')


    # First optimization after structure generation
    elif os.path.exists(init_pkl_path) and not os.path.exists(opt_pkl_path):
        # Load initial structures
        init_atoms_dict = read_pkl(init_pkl_path)
        generated_structures = init_atoms_dict

        # Optimize structures
        opt_atoms_dict, results = {}, {}
        ctx = get_context('spawn')
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
            futures = [executor.submit(optimize_structure, key, atoms, model_name) for key, atoms in
                       generated_structures.items()]

            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                key, opt_atoms, struct_dict, label = future.result()
                opt_atoms = Atoms.fromdict(opt_atoms)
                opt_atoms.calc = set_calculator(model_name)
                struct = Structure.from_dict(struct_dict)
                results[key] = {
                    'density': float(struct.density),
                    'energy_per_atom': opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms(),
                    'sg_symbol': struct.get_space_group_info()[0],
                    'sg_number': struct.get_space_group_info()[1],
                    'opt': label
                }
                opt_atoms.calc = None
                opt_atoms_dict[key] = opt_atoms

        # Save locally optimized structures
        write_pkl(opt_atoms_dict, opt_pkl_path)
        write_cif(opt_atoms_dict, opt_cif_path)
        df_results = pd.DataFrame(results).T
        df_results.to_csv(opt_result_path)
        elapsed = time.time() - start
        print(f'{num_structures} have been optimized. (Total: {elapsed:.1f} sec)')

    # Additional run
    elif os.path.exists(init_pkl_path) and os.path.exists(opt_pkl_path):

        # Load initial & optimized structures
        init_atoms_dict = read_pkl(init_pkl_path)
        opt_atoms_dict = read_pkl(opt_pkl_path)

        # Additional structure generation & optimization
        if len(init_atoms_dict) == len(opt_atoms_dict):
            add_num_structures = num_structures - len(init_atoms_dict)
            n_already = len(init_atoms_dict)

            # Generate crystal structures
            mol_files = glob.glob(xyz_folder + '*.xyz')
            generated_structures = generate_crystal_structures(mol_files, num_structures, sg_mode=sg_mode,
                                                               atoms_dict=init_atoms_dict,
                                                               conformer_mode=conformer_mode)

            # Save additional initial structures
            write_pkl(generated_structures, init_pkl_path)
            write_cif(generated_structures, init_cif_path)
            elapsed = time.time() - start
            print(f'{n_already} already. New {add_num_structures} have been generated. ({elapsed:.1f} sec)')

            # Pick up added structures
            from itertools import islice

            add_generated_structures = dict(
                islice(generated_structures.items(), len(generated_structures) - add_num_structures,
                       len(generated_structures)))

            # Optimize structures
            results = pd.read_csv(opt_result_path, index_col=0).T.to_dict()
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = [executor.submit(optimize_structure, key, atoms, model_name) for key, atoms in
                           add_generated_structures.items()]

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    key, opt_atoms, struct_dict, label = future.result()
                    opt_atoms = Atoms.fromdict(opt_atoms)
                    opt_atoms.calc = set_calculator(model_name)
                    struct = Structure.from_dict(struct_dict)
                    results[key] = {
                        'density': float(struct.density),
                        'energy_per_atom': opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms(),
                        'sg_symbol': struct.get_space_group_info()[0],
                        'sg_number': struct.get_space_group_info()[1],
                        'opt': label
                    }
                    opt_atoms.calc = None
                    opt_atoms_dict[key] = opt_atoms

            # Save locally optimized structures
            write_pkl(opt_atoms_dict, opt_pkl_path)
            write_cif(opt_atoms_dict, opt_cif_path)
            df_results = pd.DataFrame(results).T
            df_results.to_csv(opt_result_path)
            elapsed = time.time() - start
            print(f'{n_already} already. New {add_num_structures} have been optimized. (Total: {elapsed:.1f} sec)')


        # Additional optimization after structure generation
        else:
            # Pick up added structures
            from itertools import islice

            generated_structures = init_atoms_dict
            n_already = len(opt_atoms_dict)
            add_num_structures = len(init_atoms_dict) - n_already
            add_generated_structures = dict(
                islice(generated_structures.items(), len(generated_structures) - add_num_structures,
                       len(generated_structures)))

            # Optimize structures
            results = pd.read_csv(opt_result_path, index_col=0).T.to_dict()
            ctx = get_context('spawn')
            with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
                futures = [executor.submit(optimize_structure, key, atoms, model_name) for key, atoms in
                           add_generated_structures.items()]

                for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                    key, opt_atoms, struct_dict, label = future.result()
                    opt_atoms = Atoms.fromdict(opt_atoms)
                    opt_atoms.calc = set_calculator(model_name)
                    struct = Structure.from_dict(struct_dict)
                    results[key] = {
                        'density': float(struct.density),
                        'energy_per_atom': opt_atoms.get_potential_energy() / opt_atoms.get_global_number_of_atoms(),
                        'sg_symbol': struct.get_space_group_info()[0],
                        'sg_number': struct.get_space_group_info()[1],
                        'opt': label
                    }
                    opt_atoms.calc = None
                    opt_atoms_dict[key] = opt_atoms

            # Save locally optimized structures
            write_pkl(opt_atoms_dict, opt_pkl_path)
            write_cif(opt_atoms_dict, opt_cif_path)
            df_results = pd.DataFrame(results).T
            df_results.to_csv(opt_result_path)
            elapsed = time.time() - start
            print(f'{n_already} already. New {add_num_structures} have been optimized. (Total: {elapsed:.1f} sec)')
