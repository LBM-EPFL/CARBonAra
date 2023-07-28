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

std_element_radii = {
    'H': 0.58, 'HE': 0.441, 'LI': 1.89, 'BE': 1.26, 'B': 1.08, 'C': 0.73, 'N': 0.75,
    'O': 0.73, 'F': 0.71, 'NE': 0.459, 'NA': 0.75, 'MG': 1.53, 'AL': 1.62, 'SI': 1.35,
    'P': 1.06, 'S': 1.02, 'CL': 0.99, 'AR': 0.792, 'K': 2.52, 'CA': 1.5, 'SC': 1.89,
    'TI': 1.8, 'V': 1.71, 'CR': 1.2, 'MN': 1.62, 'FE': 1.55, 'CO': 1.53, 'NI': 1.44,
    'CU': 1.44, 'ZN': 1.35, 'GA': 1.62, 'GE': 1.35, 'AS': 1.17, 'SE': 1.08, 'BR': 1.14,
    'KR': 0.9, 'RB': 2.7, 'SR': 2.25, 'Y': 2.07, 'ZR': 1.98, 'NB': 1.89, 'MO': 1.8,
    'TC': 1.8, 'RU': 1.71, 'RH': 1.62, 'PD': 1.62, 'AG': 1.62, 'CD': 1.53, 'IN': 1.8,
    'SN': 1.53, 'SB': 1.35, 'TE': 1.26, 'I': 1.33, 'XE': 1.08, 'CS': 2.97, 'BA': 2.52,
    'LA': 2.43, 'CE': 2.43, 'PR': 2.43, 'ND': 2.34, 'PM': 2.34, 'SM': 2.34, 'EU': 2.34,
    'GD': 2.25, 'TB': 2.25, 'DY': 2.25, 'HO': 2.25, 'ER': 2.25, 'TM': 2.16, 'YB': 2.16,
    'LU': 2.07, 'HF': 1.98, 'TA': 1.89, 'W': 1.8, 'RE': 1.8, 'OS': 1.71, 'IR': 1.71,
    'PT': 1.62, 'AU': 1.44, 'HG': 1.62, 'TL': 1.89, 'PB': 2, 'BI': 2, 'PO': 2, 'AT': 2,
    'RN': 2, 'FR': 2, 'RA': 2, 'AC': 2, 'TH': 2, 'PA': 2, 'U': 2, 'NP': 2, 'PU': 2,
    'AM': 2, 'CM': 2, 'BK': 2, 'CF': 2, 'ES': 2, 'FM': 2, 'MD': 2, 'NO': 2, 'LR': 2,
    'RF': 2, 'DB': 2, 'SG': 2, 'BH': 2, 'HS': 2, 'MT': 2, 'DS': 2, 'RG': 2, 'CN': 2,
    'UUT': 2, 'UUQ': 2, 'UUP': 2, 'UUH': 2, 'UUS': 2, 'UUO': 2, 'X': 2
}

# prepare back mapping
elements_enum = np.concatenate([std_elements, [b'X']])
names_enum = np.concatenate([std_names, [b'UNK']])
resnames_enum = np.concatenate([std_resnames, [b'UNX']])

# prepare config summary
config_encoding = {'std_elements': std_elements, 'std_resnames': std_resnames, 'std_names': std_names}


def onehot(x, v):
    m = (x.reshape(-1,1) == np.array(v).reshape(1,-1))
    return np.concatenate([m, ~np.any(m, axis=1).reshape(-1,1)], axis=1)


def encode_structure(structure, with_chains=False, device=pt.device("cpu")):
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

    if with_chains:
        # atom to chain mapping
        if isinstance(structure['cid'], pt.Tensor):
            cids = structure['cid'].to(device)
        else:
            cids = pt.from_numpy(structure['cid']).to(device)
        Mc = (cids.unsqueeze(1) == pt.unique(cids).unsqueeze(0)).float()

        return X, Mr, Mc
    else:
        return X, Mr


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


def backbone_mask(qr, qn, std_backbone, std_aminoacids):
    # backbone atom mask
    m_std_bb = pt.from_numpy(np.isin(std_names,std_backbone)).to(qn.device)
    m_bb = pt.any(qn[:,:-1][:,m_std_bb] > 0.5, dim=1)

    # amino-acids or dna/rna residues
    m_std_aa = pt.from_numpy(np.isin(std_resnames, std_aminoacids)).to(qr.device)
    m_aa = pt.any(qr[:,:-1][:,m_std_aa] > 0.5, dim=1)

    # mask (backbone & polymer residue) or (not polymer residue)
    m = (~m_aa) | (m_aa & m_bb)

    return m


def locate_contacts(xyz_i, xyz_j, r_thr, device=pt.device("cpu")):
    with pt.no_grad():
        # send data to device
        if isinstance(xyz_i, pt.Tensor):
            X_i = xyz_i.to(device)
            X_j = xyz_j.to(device)
        else:
            X_i = pt.from_numpy(xyz_i).to(device)
            X_j = pt.from_numpy(xyz_j).to(device)

        # compute distance matrix between subunits
        D = pt.norm(X_i.unsqueeze(1) - X_j.unsqueeze(0), dim=2)

        # find contacts
        ids_i, ids_j = pt.where(D < r_thr)

        # get contacts distances
        d_ij = D[ids_i, ids_j]

    return ids_i.cpu(), ids_j.cpu(), d_ij.cpu()


def extract_all_contacts(subunits, r_thr, device=pt.device("cpu")):
    # get subunits names
    snames = list(subunits)

    # extract interfaces
    contacts_dict = {}
    for i in range(len(snames)):
        # current selection chain
        cid_i = snames[i]

        for j in range(i+1, len(snames)):
            # current selection chain
            cid_j = snames[j]

            # find contacts
            ids_i, ids_j, d_ij = locate_contacts(subunits[cid_i]['xyz'], subunits[cid_j]['xyz'], r_thr, device=device)

            # insert contacts
            if (ids_i.shape[0] > 0) and (ids_j.shape[0] > 0):
                if f'{cid_i}' in contacts_dict:
                    contacts_dict[f'{cid_i}'].update({f'{cid_j}': {'ids': pt.stack([ids_i,ids_j], dim=1), 'd': d_ij}})
                else:
                    contacts_dict[f'{cid_i}'] = {f'{cid_j}': {'ids': pt.stack([ids_i,ids_j], dim=1), 'd': d_ij}}

                if f'{cid_j}' in contacts_dict:
                    contacts_dict[f'{cid_j}'].update({f'{cid_i}': {'ids': pt.stack([ids_j,ids_i], dim=1), 'd': d_ij}})
                else:
                    contacts_dict[f'{cid_j}'] = {f'{cid_i}': {'ids': pt.stack([ids_j,ids_i], dim=1), 'd': d_ij}}

    return contacts_dict
