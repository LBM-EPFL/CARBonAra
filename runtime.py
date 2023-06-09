import os
import sys
import h5py
import importlib
import numpy as np
import torch as pt
import blosum as bl
from scipy import signal

import src as sp


def aa_only(p: pt.Tensor, y: pt.Tensor):
    m = (pt.sum(y, dim=1) > 0.0)
    return p[m], y[m]


def recovery_rate(y: pt.Tensor, p: pt.Tensor):
    return pt.mean((pt.argmax(p, dim=1) == pt.argmax(y, dim=1)).float())


def maximum_recovery_rate(y: pt.Tensor, p: pt.Tensor):
    return pt.mean(pt.sum(pt.round(p) * y, dim=1))


def average_multiplicity(p: pt.Tensor):
    return pt.mean(pt.sum(pt.round(p), dim=1))


def average_maximum_confidence(p: pt.Tensor):
    return pt.mean(pt.max(p, dim=1)[0])


def max_pred_to_seq(p: pt.Tensor):
    return ''.join([sp.res3to1[r] for r in sp.std_resnames[:20][pt.argmax(p,dim=1).cpu().numpy()]])


def sample_pred_to_seq(p: pt.Tensor):
    seqm = ""
    for i in range(p.shape[0]):
        # locate positive predictions
        ids_p = pt.where(p[i] > 0.5)[0]

        # random sampling
        c = p[i][ids_p] - 0.5
        if len(c) > 0:
            k = np.random.choice(ids_p.cpu().numpy(), p=(c/pt.sum(c)).cpu().numpy())
        else:
            k = 0

        # update sequence
        seqm += sp.res3to1[sp.std_resnames[:20][k].item()]

    return seqm


def minimize_sequence_similarity(p, y):
    # sequence similarity criteria
    blm = bl.BLOSUM(62)
    seq_minsim = ""
    seq_ref = ""
    seq_score = []
    for i in range(p.shape[0]):
        # extract sequence from prediction and reference
        rs0 = sp.res3to1[sp.std_resnames[:20][pt.argmax(y[i]).cpu().numpy().item()]]
        ids_pr = pt.where(p[i] >= 0.5)[0]
        rsp_l = [sp.res3to1[r] for r in sp.std_resnames[:20][ids_pr.cpu().numpy()]]

        # compute sequence similarity and find minimum locations
        ss = np.array([blm[rs0][r] for r in rsp_l])
        ids_min = np.where(ss <= 0.0)[0]
        if len(ids_min) == 0:
            ids_min = np.where(ss == np.min(ss))[0]

        # find sequence with minimum sequence similarity and maximum probability
        k = pt.argmax(p[i][ids_pr][ids_min]).cpu().numpy().item()
        rs_ms = sp.res3to1[sp.std_resnames[:20][ids_pr[ids_min][k].cpu().item()]]

        # store results
        seq_minsim += rs_ms
        seq_ref += rs0
        seq_score.append(np.min(ss))

    return seq_minsim, seq_ref, np.array(seq_score)


def kstar(c: pt.Tensor):
    S = -pt.sum(c * pt.log2(c + 1e-6), dim=1)
    return pt.pow(pt.tensor(2.0), S)


def seq_to_features(seq):
    resnames = np.array([sp.res1to3[r] for r in list(seq)])
    return pt.from_numpy(sp.onehot(resnames, sp.std_resnames).astype(np.float32))


def sequence_identity(seq_ref, seq):
    return np.mean(np.array(list(seq_ref)) == np.array(list(seq)))


def sequence_similarity(seq_ref, seq):
    blm = bl.BLOSUM(62)
    return np.mean(np.array([blm[si][sj] for si,sj in zip(seq_ref,seq)]) > 0)


def write_fasta(filepath, seq, info=""):
    with open(filepath, 'w') as fs:
        fs.write(">{}\n{}".format(info, seq))


def read_fasta(fasta_filepath):
    # read content
    with open(fasta_filepath, 'r') as fs:
        fasta_content = fs.read().strip()[1:].split('\n')

    # parse content
    info = fasta_content[0]
    seq = ''.join(fasta_content[1:]).split(':')

    return info, seq


def traj_to_struct(traj):
    df = traj.topology.to_dataframe()[0]
    return {
        "xyz": np.transpose(traj.xyz, (1,0,2))*1e1,
        "name": df["name"].values,
        "element": df["element"].values,
        "resname": df["resName"].values,
        "resid": df["resSeq"].values,
        "het_flag": np.array(['A']*traj.xyz.shape[1]),
        "chain_name": df["chainID"].values,
        "icode": np.array([""]*df.shape[0]),
    }


def compute_lDDT(X, X0, r_thr=[0.5, 1.0, 2.0, 4.0], R0=15.0):
    # compute distance matrices
    D = pt.norm(X.unsqueeze(0) - X.unsqueeze(1), dim=2)
    D0 = pt.norm(X0.unsqueeze(0) - X0.unsqueeze(1), dim=2)

    # thresholds
    r_thr = pt.tensor(r_thr).to(D.device)

    # local selection mask
    M = ((D0 < R0) & (D0 > 0.0)).float()

    # compute score Local Distance Difference Test
    DD = (pt.abs(D0 - D).unsqueeze(0) < r_thr.view(-1,1,1)).float()
    lDD = pt.sum(DD * M.unsqueeze(0), dim=2) / pt.sum(M, dim=1).unsqueeze(0)
    lDDT = 1e2*pt.mean(lDD, dim=0)

    return lDDT


def process_structure(structure, rm_wat=True):
    # process structure
    structure = sp.clean_structure(structure, rm_wat=rm_wat)

    # update molecules chains
    structure = sp.tag_hetatm_chains(structure)

    # change chain name to chain index
    structure = sp.chain_name_to_index(structure)

    # split structure
    subunits = sp.split_by_chain(structure)

    # remove non atomic structures
    subunits = sp.filter_non_atomic_subunits(subunits)

    # remove duplicated molecules and ions
    subunits = sp.remove_duplicate_tagged_subunits(subunits)

    return sp.concatenate_chains(subunits)


def load_structure(pdb_filepath, rm_wat=True):
    # read structure
    structure = sp.read_pdb(pdb_filepath)

    # process structure
    structure = process_structure(structure, rm_wat=rm_wat)

    return structure


def split_by_residue(structure):
    uresids, ids = np.unique(structure['resid'], return_index=True)
    uresids = uresids[np.argsort(ids)]

    residues = [sp.atom_select(structure, structure['resid'] == resid) for resid in uresids]

    return residues


def cid_to_chain_name(structure):
    structure['chain_name'] = np.array(["ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"[i] for i in structure['cid']])
    return structure


def chain_masks(structure):
    _, Mr, Mc = sp.encode_structure(sp.chain_name_to_index(structure), with_chains=True)
    chain_names = [structure['chain_name'][i].split(':')[0] for i in pt.max(Mc, dim=0)[1]]
    mr_chains = (pt.matmul(Mr.T, Mc) / pt.sum(Mr, dim=0).unsqueeze(1) > 0.0)
    return mr_chains, chain_names


def subunit_type(subunit):
    if np.all([rn in sp.resname_to_categ for rn in subunit['resname']]):
        t = np.unique([sp.resname_to_categ[rn] for rn in subunit['resname'] if rn in sp.resname_to_categ])
        if len(t) == 1:
            return t.item()
        else:
            return "na"
    else:
        return "na"


def subunits_type(subunits):
    return {(subunit_type(subunits[cid]),cid) for cid in subunits}


def load_module(name, path):
    # load module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


class StructuresDataset(pt.utils.data.Dataset):
    def __init__(self, pdb_filepaths):
        super(StructuresDataset).__init__()
        # store dataset filepath
        self.pdb_filepaths = pdb_filepaths

    def __len__(self):
        return len(self.pdb_filepaths)

    def __getitem__(self, i):
        # find pdb filepath
        pdb_filepath = self.pdb_filepaths[i]

        # load structure
        structure = load_structure(pdb_filepath)

        return pdb_filepath, structure


class Dataset(pt.utils.data.Dataset):
    def __init__(self, dataset_filepath):
        super(Dataset, self).__init__()
        # store dataset filepath
        self.dataset_filepath = dataset_filepath

        # preload data
        with h5py.File(dataset_filepath, 'r') as hf:
            # load keys, sizes and types
            self.keys = np.array(hf["metadata/keys"]).astype(np.dtype('U'))
            self.sizes = np.array(hf["metadata/rsizes"])

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
            Mr = sp.load_sparse_mask(hgrp, 'Mr').float()
            Mc = sp.load_sparse_mask(hgrp, 'Mc').float()

            # load features
            qe = sp.load_sparse_mask(hgrp, 'qe')
            qr = sp.load_sparse_mask(hgrp, 'qr')
            qn = sp.load_sparse_mask(hgrp, 'qn')

        # convert data to structure
        structure = sp.data_to_structure(X.numpy(), pt.cat([qe, qr, qn], dim=1).numpy(), Mr.numpy(), sp.std_elements, sp.std_resnames, sp.std_names)
        structure['cid'] = pt.argmax(Mc, dim=1).numpy()

        return key, structure


class ConfidenceMap():
    def __init__(self, cdf_filepath):
        # load prediction CDF
        Z = np.loadtxt(cdf_filepath, delimiter=",")
        self.x = Z[0]
        self.C = Z[1:]

        # smooth raw mapping (finite sampling -> noise)
        for i in range(self.C.shape[0]):
            self.C[i] = signal.savgol_filter(self.C[i], 9, 3)

    def __call__(self, p):
        # interpolated confidence
        #return np.stack([np.interp(p[:,k], self.x, self.C[k]) * np.round(p[:,k]) for k in range(p.shape[1])], axis=1)
        return np.clip(np.stack([np.interp(p[:,k], self.x, self.C[k]) for k in range(p.shape[1])], axis=1), 0.0, 1.0)


class SequenceModel():
    def __init__(self, save_path, parameters_filename, device=pt.device("cpu")):
        # load module
        self.module = load_module(os.path.basename(save_path), os.path.join(save_path, "__init__.py"))

        # create and reload model
        self.device = device
        model_filepath = os.path.join(save_path, parameters_filename)
        self.model = self.module.Model(self.module.config_model)
        self.model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))
        self.model = self.model.eval().to(self.device)

    def __call__(self, structure, m_known=None, n_skip=1):
        # extract features, encode structure
        qe, qr, qn = self.module.encode_features(structure)
        X, Mr = self.module.encode_structure(structure)
        if m_known is None:
            mr_known = pt.zeros(Mr.shape[1]).bool()
        else:
            if np.any(m_known):
                mr_known = (pt.max(Mr[pt.from_numpy(m_known).to(Mr.device)], dim=0)[0] > 0.0)
            else:
                mr_known = pt.zeros(Mr.shape[1]).bool()

        # encode chain infromation
        chain_map, cids = np.unique(structure['chain_name'], return_inverse=True)

        # extract backbone with/out virtual C_beta
        if self.module.config_data['virt_cb']:
            # build structure back
            q = pt.cat([qe,qr,qn], dim=1)
            structure = self.module.data_to_structure(X.numpy(), q.numpy(), Mr.numpy(), self.module.std_elements, self.module.std_resnames, self.module.std_names)
            structure['cid'] = cids

            # add virtual Cb
            structure = self.module.add_virtual_cb(structure)

            # extract features, encode structure
            qe, qr, qn = self.module.encode_features(structure)
            X, Mr = self.module.encode_structure(structure)
            cids = structure['cid']

            # extract backbone mask
            m = sp.backbone_mask(qr, qn, np.concatenate([self.module.std_backbone, ['CB']]), self.module.std_aminoacids)
        else:
            # extract backbone mask
            m = sp.backbone_mask(qr, qn, self.module.std_backbone, self.module.std_aminoacids)

        # extract backbone
        X = X[m]
        qe = qe[m]
        qr = qr[m]
        qn = qn[m]
        Mr = Mr[m]
        cids = cids[m.numpy()]

        # get sequence
        m_std_aa = pt.from_numpy(np.isin(self.module.std_resnames, self.module.std_aminoacids)).to(qr.device)
        y = qr[:,:-1][:,m_std_aa][pt.max(Mr.float(), dim=0)[1]]

        # mask residue information
        yt = y.clone()
        yt[~mr_known] = 0.0

        # build structure back
        q = pt.cat([qe,qr,qn], dim=1)
        structure = self.module.data_to_structure(X.numpy(), q.numpy(), Mr.numpy(), self.module.std_elements, self.module.std_resnames, self.module.std_names)
        structure['chain_name'] = np.array([chain_map[i] for i in cids])
        structure = sp.encode_bfactor(structure, pt.sum(yt,dim=1).cpu().numpy())

        # apply mask and pack features
        qr = pt.matmul(Mr, yt)
        q = pt.cat([qe, qr], dim=1)

        # multiframe support
        if len(X.shape) < 3:
            X = X.unsqueeze(1)

        # run predictions
        P = []
        with pt.no_grad():
            for i in range(0, X.shape[1], n_skip):
                # send to device
                Xi, q, M = (v.to(self.device) for v in (X[:,i], q, Mr))

                # compute topology
                ids_topk, _, _, _, _ = self.module.extract_topology(Xi, 64)

                # run model
                z = self.model(Xi, ids_topk, q, M)

                # prediction
                p = pt.sigmoid(z).cpu()

                # store result
                P.append(p)

        return structure, pt.stack(P).squeeze(), y.cpu()
