import h5py
import numpy as np
import torch as pt


from src.dataset import load_sparse_mask
from src.structure import data_to_structure, atom_select
# from src.data_encoding import std_names, std_backbone, std_resnames, std_aminoacids
from src.data_encoding import std_elements, std_resnames, std_names, encode_features, encode_structure, std_backbone, std_aminoacids


def add_virtual_cb(structure):
    # split structure by residue
    residues = [atom_select(structure, structure['resid']==i) for i in np.unique(structure['resid'])]

    # place virtual Cb using ideal angle and bond length (ref: ProteinMPNN)
    for res in residues:
        if (res['resname'][0] in std_resnames[:20]) and ('CA' in res['name']) and ('N' in res['name']) and ('C' in res['name']):
            # define vectors
            b = res['xyz'][res['name']=='CA'] - res['xyz'][res['name']=='N']
            c = res['xyz'][res['name']=='C'] - res['xyz'][res['name']=='CA']
            a = np.cross(b,c)

            if 'CB' in res['name']:
                # update Cb coordinates
                res['xyz'][res['name']=='CB'] = -0.58273431*a + 0.56802827*b - 0.54067466*c + res['xyz'][res['name']=='CA']
            else:
                if len(-0.58273431*a + 0.56802827*b - 0.54067466*c + res['xyz'][res['name']=='CA']) == 0:
                    print(res)
                # insert virtual Cb (GLY)
                virt_Cb = {
                    'name': 'CB',
                    'bfactor': 0.0,
                    'resid': res['resid'][-1],
                    'element': 'C',
                    'resname': 'GLY',
                    'xyz': (-0.58273431*a + 0.56802827*b - 0.54067466*c + res['xyz'][res['name']=='CA'])[0],
                    'het_flag': 'A',
                }
                for key in res:
                    res[key] = np.concatenate([res[key], [virt_Cb[key]]])

    # pack residues back
    return {key: np.concatenate([res[key] for res in residues]) for key in residues[0]}


def extract_backbone(X, qe, qr, qn, M, std_backbone, std_aminoacids):
    # backbone atom mask
    m_std_bb = pt.from_numpy(np.isin(std_names,std_backbone)).to(qn.device)
    m_bb = pt.any(qn[:,:-1][:,m_std_bb] > 0.5, dim=1)

    # amino-acids or dna/rna residues
    m_std_aa = pt.from_numpy(np.isin(std_resnames, std_aminoacids)).to(qr.device)
    m_aa = pt.any(qr[:,:-1][:,m_std_aa] > 0.5, dim=1)

    # mask (backbone & polymer residue) or (not polymer residue)
    m = (~m_aa) | (m_aa & m_bb)

    return X[m], qe[m], qr[m], qn[m], M[m]


def process_structure(X, qe, qr, qn, M, r):
    # create inital labels
    y = qr.clone()

    if r < 1.0:
        # randomly sample residues uniformly with ratio r
        nr_sel = max(int(np.ceil(M.shape[1]*r)), 1)
        ids_sel = pt.randperm(M.shape[1])[:nr_sel]
    else:
        ids_sel = pt.arange(M.shape[1])

    # randomly select residues and mask information
    m_sel = pt.any(M[:,ids_sel] > 0.5, dim=1)
    qr[m_sel] = 0.0

    # pack features
    q = pt.cat([qe, qr], dim=1)

    # atom to residue indexing
    _, rids_sel = pt.max(M[:,ids_sel], dim=0)

    return X, q, M, y[rids_sel], ids_sel


def random_atom_motion(X, r=0.75):
    # clip and resample random normal values
    rnv = np.sqrt(r*r/ 3.0) * pt.randn((X.shape[0]*10,X.shape[1]), device=X.device)
    m = (pt.norm(rnv, dim=1) <= r)
    rnv = rnv[m][:X.shape[0]]

    # compute displacement vectors
    dX = rnv * pt.rand((X.shape[0],1), device=X.device) * pt.rand((X.shape[0],1), device=X.device)
    dX = dX - pt.mean(dX, dim=0).unsqueeze(0)
    return dX


class Dataset(pt.utils.data.Dataset):
    def __init__(self, dataset_filepath, r_noise=0.0, virt_cb=False, partial=False):
        super(Dataset, self).__init__()
        # store dataset filepath
        self.dataset_filepath = dataset_filepath

        # store parameters
        self.r_noise = r_noise
        self.virt_cb = virt_cb
        self.partial = partial

        # preload data
        with h5py.File(dataset_filepath, 'r') as hf:
            # load keys, sizes and types
            self.keys = np.array(hf["metadata/keys"]).astype(np.dtype('U'))
            self.sizes = np.array(hf["metadata/sizes"])

            # load parameters to reconstruct data
            self.std_elements = np.array(hf["metadata/std_elements"]).astype(np.dtype('U'))
            self.std_resnames = np.array(hf["metadata/std_resnames"]).astype(np.dtype('U'))
            self.std_names = np.array(hf["metadata/std_names"]).astype(np.dtype('U'))

        # set default selection mask
        self.m = np.ones(len(self.keys), dtype=bool)

    def get_largest(self):
        i = np.argmax(self.sizes[self.m,0])
        return self[i]

    def __len__(self):
        return len(self.keys[self.m])

    def __getitem__(self, k):
        # get corresponding interface keys
        key = self.keys[self.m][k]

        # load data
        with h5py.File(self.dataset_filepath, 'r') as hf:
            # hdf5 group
            hgrp = hf['data/structures/'+key]

            # topology
            X = pt.from_numpy(np.array(hgrp['X']).astype(np.float32))
            M = load_sparse_mask(hgrp, 'M').float()

            # load features
            qe = load_sparse_mask(hgrp, 'qe')
            qr = load_sparse_mask(hgrp, 'qr')
            qn = load_sparse_mask(hgrp, 'qn')

        if self.virt_cb:
            # build structure back
            q = pt.cat([qe,qr,qn], dim=1)
            structure = data_to_structure(X.numpy(), q.numpy(), M.numpy(), std_elements, std_resnames, std_names)

            # add virtual Cb
            structure = add_virtual_cb(structure)

            # extract features, encode structure
            qe, qr, qn = encode_features(structure)
            X, M = encode_structure(structure)

            # extract backbone
            X, qe, qr, qn, M = extract_backbone(X, qe, qr, qn, M, np.concatenate([std_backbone, ['CB']]), std_aminoacids)
        else:
            # extract backbone
            X, qe, qr, qn, M = extract_backbone(X, qe, qr, qn, M, std_backbone, std_aminoacids)

        if self.partial:
            # randomly sample from X^2 with X uniform the percentage of residues to remove from the structure
            r = 1.0 - np.random.uniform(0.0, 1.0) * np.random.uniform(0.0, 1.0)
        else:
            r = 1.0

        # process structure
        m_std_aa = pt.from_numpy(np.isin(std_resnames, std_aminoacids)).to(qr.device)
        X, q, M, y, rids_sel = process_structure(X, qe, qr[:,:-1][:,m_std_aa], qn, M, r)

        # random atom motion
        if self.r_noise > 0.0:
            X = X + random_atom_motion(X, r=self.r_noise)

        return X, q, M, y, rids_sel
