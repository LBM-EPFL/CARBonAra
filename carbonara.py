import os
import sys
import argparse
import importlib
import numpy as np
import torch as pt
import blosum as bl
from functools import partial
from scipy import signal
from tqdm import tqdm

try:
    from .src.structure_io import read_pdb, save_pdb
    from .src.structure import encode_bfactor, res3to1, res1to3, clean_structure, split_by_chain, tag_hetatm_chains, chain_name_to_index, atom_select, add_virtual_cb, data_to_structure
    from .src.data_encoding import std_elements, std_resnames, std_names, std_aminoacids, std_backbone, onehot, encode_structure, encode_features, extract_topology
except ImportError:
    from src.structure_io import read_pdb, save_pdb
    from src.structure import encode_bfactor, res3to1, res1to3, clean_structure, split_by_chain, tag_hetatm_chains, chain_name_to_index, atom_select, add_virtual_cb, data_to_structure
    from src.data_encoding import std_elements, std_resnames, std_names, std_aminoacids, std_backbone, onehot, encode_structure, encode_features, extract_topology


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
    return ''.join([res3to1[r] for r in std_resnames[:20][pt.argmax(p,dim=1).cpu().numpy()]])


def seq_to_features(seq):
    resnames = np.array([res1to3[r] for r in list(seq)])
    return pt.from_numpy(onehot(resnames, std_resnames).astype(np.float32))


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


def process_structure(structure, rm_hetatm=False, rm_wat=True):
    # keep original resids
    structure['resid_orig'] = structure['resid'].copy()

    # process structure
    structure = clean_structure(structure, rm_hetatm=rm_hetatm, rm_wat=rm_wat)

    # update molecules chains
    structure = tag_hetatm_chains(structure)

    # change chain name to chain index
    structure = chain_name_to_index(structure)

    return structure


def load_structure(pdb_filepath, rm_hetatm=False, rm_wat=True):
    # read structure
    structure = read_pdb(pdb_filepath)

    # process structure
    structure = process_structure(structure, rm_hetatm=rm_hetatm, rm_wat=rm_wat)

    return structure


def chain_masks(structure):
    _, Mr, Mc = encode_structure(chain_name_to_index(structure))
    chain_names = [structure['chain_name'][i].split(':')[0] for i in pt.max(Mc, dim=0)[1]]
    mr_chains = (pt.matmul(Mr.T, Mc) / pt.sum(Mr, dim=0).unsqueeze(1) > 0.0)
    return mr_chains, chain_names


def load_module(name, path):
    # load module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

    return module


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
        # interpolated confidence to probability
        c =  np.clip(np.stack([np.interp(p[:,k], self.x, self.C[k]) for k in range(p.shape[1])], axis=1), 0.0, 1.0)
        return c / np.sum(c, axis=1).reshape(-1,1)


class CARBonAra():
    def __init__(self, model_name="s_v6_4_2022-09-16_11-51", parameters_filename="model.pt", device_name="cpu"):
        # locate model save path
        save_path = os.path.join(os.path.dirname(__file__), "model", "save", model_name)

        # load module
        self.config_module = load_module(os.path.basename(save_path), os.path.join(save_path, "config.py"))
        self.model_module = load_module(os.path.basename(save_path), os.path.join(save_path, "model.py"))

        # create and reload model
        self.device = pt.device("cuda" if pt.cuda.is_available() and device_name == "cuda" else "cpu")
        model_filepath = os.path.join(save_path, parameters_filename)
        self.model = self.model_module.Model(self.config_module.config_model)
        self.model.load_state_dict(pt.load(model_filepath, map_location=pt.device("cpu")))
        self.model = self.model.eval().to(self.device)

        # create confidence mapping
        conf_filepath = os.path.join(save_path, "{}_cdf.csv".format(os.path.basename(save_path)))
        self.conf = ConfidenceMap(conf_filepath)

    def process_structure(self, structure):
        # add virtual C_beta
        structure = add_virtual_cb(structure)

        # mask (backbone & polymer residue) or (not polymer residue)
        m_aa = np.isin(structure['resname'], std_aminoacids)
        m_bb = np.isin(structure['name'], np.concatenate([std_backbone, ['CB']]))
        m = (~m_aa) | (m_aa & m_bb)
        structure = atom_select(structure, m)
        
        # extract features, encode structure
        qe, qr, qn = encode_features(structure)
        X, Mr, Mc = encode_structure(structure)

        # extract sequence in one-hot encoding and amino acids mask at the residue level
        m_std_aa = pt.from_numpy(np.isin(std_resnames, std_aminoacids)).to(qr.device)
        yr = qr[:,:-1][:,m_std_aa][pt.max(Mr.float(), dim=0)[1]]
        mr_aa = (pt.sum(yr, dim=1) > 0.0)
        
        return X, qe, qr, qn, Mr, Mc, yr, mr_aa

    def apply_model(self, X, qe, M, yt=None):
        # mask
        if yt is None:
            qm = pt.zeros((qe.shape[0], std_aminoacids.shape[0]))
        else:
            qm = pt.matmul(M, yt.to(qe.device))

        # apply mask and pack features
        q = pt.cat([qe, qm], dim=1)

        # run predictions
        with pt.no_grad():
            # send to device
            X, q, M = (v.to(self.device) for v in (X, q, M))

            # compute topology
            ids_topk, _, _, _, _ = extract_topology(X, 64)

            # run model
            z = self.model(X, ids_topk, q, M)
            p = pt.sigmoid(z).cpu()

        return p


def imprint_sampling(
        carbonara, pdb_filepath, num_sample, imprint_ratio, b_sampled=True,
        known_chains=[], known_positions=[], unknown_positions=[], ignored_amino_acids=[],
        ignore_hetatm=False, ignore_wat=False,
    ):
    # load structure
    structure = load_structure(pdb_filepath, rm_hetatm=ignore_hetatm, rm_wat=ignore_wat)
    
    # process structure
    X, qe, qr, qn, Mr, Mc, y, mr_aa = carbonara.process_structure(structure)

    # known chains or residues
    m_known = pt.zeros_like(mr_aa)
    mr_chains, chain_names = chain_masks(structure)
    if len(known_chains) > 0:
        m_known |= pt.any(mr_chains[:,pt.from_numpy(np.isin(chain_names, known_chains))], dim=1)
    if len(known_positions) > 0:
        m_known |= pt.from_numpy(np.isin(np.arange(m_known.shape[0])+1, known_positions))
    if len(unknown_positions) > 0:
        m_known |= ~pt.from_numpy(np.isin(np.arange(m_known.shape[0])+1, unknown_positions))
    m_known = m_known.float().unsqueeze(1)

    # scaffold
    structure_scaffold = data_to_structure(
        X.numpy(), pt.concat([qe,qr,qn], dim=1).numpy(),
        Mr.numpy(), Mc.numpy(),
        std_elements, std_resnames, std_names
    )
    structure_scaffold = encode_bfactor(structure_scaffold, m_known.squeeze().numpy())
    
    # define mask for amino acids to ignore during sampling
    m_iaa = pt.tensor([res3to1[aa] in ignored_amino_acids for aa in std_aminoacids])
    
    # prediction without imprint
    yt = m_known * y
    p0 = carbonara.apply_model(X, qe, Mr, yt=yt)[mr_aa]
    p0[:,m_iaa] = 0.0
    p0 = (1.0 - m_known[mr_aa]) * p0 + m_known[mr_aa] * y[mr_aa]
    
    # confidence
    c0 = carbonara.conf(p0.numpy())
    
    # start sampling
    sequences, scores = [], []
    for _ in tqdm(range(num_sample)):
        # sample sequence from confidence
        if b_sampled:
            ids_sampled = np.array([np.random.choice(c0.shape[1], p=c0[i]) for i in range(c0.shape[0])])
        else:
            ids_sampled = np.argmax(p0.numpy(), axis=1)
    
        # convert to prior bias
        yt = pt.zeros((mr_aa.shape[0], 20))
        yt[pt.where(mr_aa)[0], ids_sampled] = 1.0
        
        # randomly mask imprint
        mm = (np.random.rand(yt.shape[0]) <= (1.0-imprint_ratio))
        yt[mm] = 0.0

        # prediction with imprint
        yt = (1.0 - m_known) * yt + m_known * y
        p = carbonara.apply_model(X, qe, Mr, yt=yt)[mr_aa]
        p[:,m_iaa] = 0.0
        p = (1.0 - m_known[mr_aa]) * p + m_known[mr_aa] * y[mr_aa]
        seq = max_pred_to_seq(p)

        # scoring
        c = carbonara.conf(p.numpy())
        alpha = m_known[mr_aa].squeeze().numpy()
        score = np.sum((1.0 - alpha) * c[np.arange(c.shape[0]), np.argmax(p.numpy(), axis=1)]) / np.sum(1.0 - alpha)

        # split sequence by chain
        seqs = [''.join(np.array(list(seq))[mr_chains[mr_aa,i]]) for i in range(mr_chains.shape[1])]
        seqs = list(filter(lambda seq: len(seq) > 0, seqs))

        # store results
        sequences.append(seqs)
        scores.append(score)

    return sequences, np.array(scores), p0.numpy(), structure_scaffold


def parse_list_args(arg_str, sep=',', dtype=str):
    return [dtype(e) for e in arg_str.split(sep)]


def main():
    # command line interface
    parser = argparse.ArgumentParser(description="generate sequences from a pdb scaffold using CARBonAra")

    # input arguments
    parser.add_argument("pdb_filepath", type=str, help="path to the input PDB file")

    # output arguments
    parser.add_argument("output_dir", type=str, help="directory to save all output files")

    # parameters
    parser.add_argument("--num_sequences", type=int, default=1, help="number of sequences to generate")
    parser.add_argument("--imprint_ratio", type=float, default=0.5, help="ratio of sequence imprint for sampling")
    parser.add_argument("--sampling_method", type=str, choices=["max", "sampled"], default="sampled", help="method to sample sequences")
    parser.add_argument("--known_chains", type=parse_list_args, default=[], help="comma-separated list of known chains")
    parser.add_argument("--known_positions", type=partial(parse_list_args, dtype=int), default=[], help="comma-separated list of known sequence position as indices")
    parser.add_argument("--unknown_positions", type=partial(parse_list_args, dtype=int), default=[], help="comma-separated list of unknown sequence position to design as indices")
    parser.add_argument("--ignored_amino_acids", type=parse_list_args, default=[], help="comma-separated list of one letter code of amino acids to ignore from the sequences sampling")
    parser.add_argument("--ignore_hetatm", action="store_true", help="ignore HETATM records in the input")
    parser.add_argument("--ignore_water", action="store_true", help="ignore water molecules in the input")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="method to sample sequences")

    # parse arguments
    args = parser.parse_args()

    # process inputs
    b_sampled = (args.sampling_method == 'sampled')

    # load model
    carbonara = CARBonAra(device_name=args.device)

    # sample sequences
    sequences, scores, pssm, structure_scaffold = imprint_sampling(
        carbonara, args.pdb_filepath, args.num_sequences, args.imprint_ratio, b_sampled=b_sampled,
        known_chains=args.known_chains, known_positions=args.known_positions, unknown_positions=args.unknown_positions,
        ignored_amino_acids=args.ignored_amino_acids,
        ignore_hetatm=args.ignore_hetatm, ignore_wat=args.ignore_water,
    )

    # save input scaffold
    os.makedirs(args.output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(args.pdb_filepath))[0]
    save_pdb(split_by_chain(structure_scaffold), os.path.join(args.output_dir, f"{name}_scaffold.pdb"))

    # save original pssm
    np.savetxt(os.path.join(args.output_dir, f"{name}_pssm.csv"), pssm, header=','.join(std_aminoacids), delimiter=",", comments='')

    # save all sequences
    for k in range(len(sequences)):
        # extract results
        seqs = sequences[k]
        score = scores[k]

        # write sequence to file
        info_str = f"imprint_ratio={args.imprint_ratio}, sampling_method={args.sampling_method}, score={score:.5f}"
        n = len(str(len(sequences)-1))
        write_fasta(os.path.join(args.output_dir, f"{name}_{k:0{n}d}.fasta"), ':'.join(seqs), info=info_str)


if __name__ == "__main__":
    main()
