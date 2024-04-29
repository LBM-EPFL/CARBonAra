import numpy as np
import torch as pt


# standard elements (sorted by aboundance) (32)
std_elements = np.array([
    'C', 'O', 'N', 'S', 'P', 'Se', 'Mg', 'Cl', 'Zn', 'Fe', 'Ca', 'Na',
    'F', 'Mn', 'I', 'K', 'Br', 'Cu', 'Cd', 'Ni', 'Co', 'Sr', 'Hg', 'W',
    'As', 'B', 'Mo', 'Ba', 'Pt'
])

# standard residue names: AA/RNA/DNA (sorted by aboundance) (29)
std_resnames = np.array([
    'LEU', 'GLU', 'ARG', 'LYS', 'VAL', 'ILE', 'PHE', 'ASP', 'TYR',
    'ALA', 'THR', 'SER', 'GLN', 'ASN', 'PRO', 'GLY', 'HIS', 'TRP',
    'MET', 'CYS', 'G', 'A', 'C', 'U', 'DG', 'DA', 'DT', 'DC'
])

# standard atom names contained in standard residues (sorted by aboundance) (63)
std_names = np.array([
    'CA', 'N', 'C', 'O', 'CB', 'CG', 'CD2', 'CD1', 'CG1', 'CG2', 'CD',
    'OE1', 'OE2', 'OG', 'OG1', 'OD1', 'OD2', 'CE', 'NZ', 'NE', 'CZ',
    'NH2', 'NH1', 'ND2', 'CE2', 'CE1', 'NE2', 'OH', 'ND1', 'SD', 'SG',
    'NE1', 'CE3', 'CZ3', 'CZ2', 'CH2', 'P', "C3'", "C4'", "O3'", "C5'",
    "O5'", "O4'", "C1'", "C2'", "O2'", 'OP1', 'OP2', 'N9', 'N2', 'O6',
    'N7', 'C8', 'N1', 'N3', 'C2', 'C4', 'C6', 'C5', 'N6', 'N4', 'O2',
    'O4'
])

# backbone
std_backbone = np.array([
    'CA', 'N', 'C', 'O'
    # "P", "OP1", "OP2", "O5'", "C5'", "C4'", "O4'",
    # "C3'", "O3'", "C2'", "O2'", "C1'",
])

# amino-acids
std_aminoacids = np.array([
    'LEU', 'GLU', 'ARG', 'LYS', 'VAL', 'ILE', 'PHE', 'ASP', 'TYR',
    'ALA', 'THR', 'SER', 'GLN', 'ASN', 'PRO', 'GLY', 'HIS', 'TRP',
    'MET', 'CYS',
])

# resname categories
categ_to_resnames = {
    "protein": ['GLU', 'LEU', 'ALA', 'ASP', 'SER', 'VAL', 'GLY', 'THR', 'ARG',
                'PHE', 'TYR', 'ILE', 'PRO', 'ASN', 'LYS', 'GLN', 'HIS', 'TRP',
                'MET', 'CYS'],
    "rna": ['A', 'U', 'G', 'C'],
    "dna": ['DA', 'DT', 'DG', 'DC'],
    "ion": ['MG', 'ZN', 'CL', 'CA', 'NA', 'MN', 'K', 'IOD', 'CD', 'CU', 'FE', 'NI',
            'SR', 'BR', 'CO', 'HG'],
    "ligand": ['SO4', 'NAG', 'PO4', 'EDO', 'ACT', 'MAN', 'HEM', 'FMT', 'BMA',
               'ADP', 'FAD', 'NAD', 'NO3', 'GLC', 'ATP', 'NAP', 'BGC', 'GDP',
               'FUC', 'FES', 'FMN', 'GAL', 'GTP', 'PLP', 'MLI', 'ANP', 'H4B',
               'AMP', 'NDP', 'SAH', 'OXY'],
    "lipid": ['PLM', 'CLR', 'CDL', 'RET'],
}
resname_to_categ = {rn:c for c in categ_to_resnames for rn in categ_to_resnames[c]}

# prepare back mapping
elements_enum = np.concatenate([std_elements, [b'X']])
names_enum = np.concatenate([std_names, [b'UNK']])
resnames_enum = np.concatenate([std_resnames, [b'UNX']])

# prepare config summary
config_encoding = {'std_elements': std_elements, 'std_resnames': std_resnames, 'std_names': std_names}


def onehot(x, v):
    m = (x.reshape(-1,1) == np.array(v).reshape(1,-1))
    return np.concatenate([m, ~np.any(m, axis=1).reshape(-1,1)], axis=1)


def encode_structure(structure, device=pt.device("cpu")):
    # coordinates
    if isinstance(structure['xyz'], pt.Tensor):
        X = structure['xyz'].to(device)
    else:
        X = pt.from_numpy(structure['xyz'].astype(np.float32)).to(device)

    # atom to residues mapping
    if isinstance(structure['resid'], pt.Tensor):
        resids = structure['resid'].to(device)
    else:
        resids = pt.from_numpy(structure['resid']).to(device)
    Mr = (resids.unsqueeze(1) == pt.unique(resids).unsqueeze(0)).float()

    # atom to chain mapping
    if isinstance(structure['cid'], pt.Tensor):
        cids = structure['cid'].to(device)
    else:
        cids = pt.from_numpy(structure['cid']).to(device)
    Mc = (cids.unsqueeze(1) == pt.unique(cids).unsqueeze(0)).float()

    return X, Mr, Mc


def encode_features(structure, device=pt.device("cpu")):
    # charge features
    qe = pt.from_numpy(onehot(structure['element'], std_elements).astype(np.float32)).to(device)
    qr = pt.from_numpy(onehot(structure['resname'], std_resnames).astype(np.float32)).to(device)
    qn = pt.from_numpy(onehot(structure['name'], std_names).astype(np.float32)).to(device)

    return qe, qr, qn


def extract_topology(X, num_nn):
    # compute displacement vectors
    R = X.unsqueeze(0) - X.unsqueeze(1)
    # compute distance matrix
    D = pt.norm(R, dim=2)
    # mask distances
    D = D + 2.0*pt.max(D)*(D < 1e-2).float()
    # normalize displacement vectors
    R = R / D.unsqueeze(2)

    # find nearest neighbors
    knn = min(num_nn, D.shape[0])
    D_topk, ids_topk = pt.topk(D, knn, dim=1, largest=False)
    R_topk = pt.gather(R, 1, ids_topk.unsqueeze(2).repeat((1,1,X.shape[1])))

    return ids_topk, D_topk, R_topk, D, R


def structure_to_data(structure, device=pt.device("cpu")):
    # encode structure and features
    X, M = encode_structure(structure, device=device)
    q = pt.cat(encode_features(structure, device=device), dim=1)

    # extract topology
    ids_topk, D_topk, R_topk, D, R = extract_topology(X, 64)

    return X, ids_topk, q, M
