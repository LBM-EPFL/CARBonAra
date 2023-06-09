import os
import numpy as np
import torch as pt
import pandas as pd
from tqdm import tqdm

import src as sp
import runtime as rt


def scoring(p, y, conf):
    # compute confidence probability
    c = pt.from_numpy(conf(p.numpy()))

    # get sequence
    seq_ref = rt.max_pred_to_seq(y)
    seq = rt.max_pred_to_seq(c)

    # assess predictions
    return {
        "size": p.shape[0],
        "recovery_rate": rt.recovery_rate(y, c).numpy().item(),
        "sequence_similarity": rt.sequence_similarity(seq_ref, seq),
        "maximum_recovery_rate": rt.maximum_recovery_rate(y, p).numpy().item(),
        "average_multiplicity": rt.average_multiplicity(p).numpy().item(),
        "average_maximum_confidence": rt.average_maximum_confidence(p).numpy().item(),
        "average_maximum_score": rt.average_maximum_confidence(c).numpy().item(),
    }


def main():
    # parameters
    device = pt.device("cuda")

    # results parameters
    output_dir = "results/benchmark"

    # model parameters
    # r6
    save_path = "model/save/s_v6_4_2022-09-16_11-51"  # virtual Cb & partial
    #save_path = "model/save/s_v6_5_2022-09-16_11-52"  # virtual Cb, partial & noise

    # r7
    #save_path = "model/save/s_v7_0_2023-04-25"  # partial chain
    #save_path = "model/save/s_v7_1_2023-04-25"  # partial secondary structure
    #save_path = "model/save/s_v7_2_2023-04-25"  # partial chain high coverage

    # create models
    model = rt.SequenceModel(save_path, "model.pt", device=device)

    # create confidence mapping
    conf = rt.ConfidenceMap("results/{}_cdf.csv".format(os.path.basename(save_path)))

    # parameters
    sids_selection_filepath = "datasets/subunits_validation_set_cath_subset.txt"
    sids_train_filepath = "datasets/subunits_train_set.txt"

    # load selected sids
    sids_sel = np.genfromtxt(sids_selection_filepath, dtype=np.dtype('U'))
    sids_sel = np.unique(np.array([s.split('_')[0] for s in sids_sel]))

    # mask partial in training set
    m_tr = np.isin(sids_sel, [s.split('_') for s in np.genfromtxt(sids_train_filepath, dtype=np.dtype('U'))])
    sids_sel = sids_sel[~m_tr]

    # find validation structure ids
    pdbids_sel = np.array([sid.split('_')[0].lower() for sid in sids_sel])

    # get filepaths
    pdb_filepaths = ['data/all_biounits/{}/{}.pdb1.gz'.format(pdbid[1:3], pdbid) for pdbid in pdbids_sel]
    pdb_filepaths = [fp for fp in pdb_filepaths if os.path.exists(fp)]
    pdb_filepaths = [fp for fp in pdb_filepaths if os.path.getsize(fp) < 1e6]

    # set up dataset
    dataset = rt.StructuresDataset(pdb_filepaths)

    # parameters
    N = len(dataset)

    # sample predictions
    for i in tqdm(np.random.choice(len(dataset), N, replace=False)):
        results = []
        try:
            # output file
            pdb_filepath = dataset.pdb_filepaths[i]
            out_filepath = os.path.join(output_dir, os.path.basename(pdb_filepath).split('.')[0]+".csv")
            if os.path.exists(out_filepath):
                continue

            # load structure
            key, structure = dataset[i]
            structure['chain_name'] = np.array([str(cid) for cid in structure['cid']])

            # max size
            if structure['xyz'].shape[0] > model.module.config_data['max_size']:
                continue

            # min size
            if len(np.unique(structure['resid'])) < model.module.config_data['min_num_res']:
                continue

            # apply model on full structure
            _, p0, y0 = model(structure)
            p0, y0 = rt.aa_only(p0, y0)

            # remove non-amino acids molecules
            structure = sp.atom_select(structure, np.isin(structure['resname'], sp.std_aminoacids))
            _, p1, y1 = model(structure)

            # split by chains
            subunits = sp.split_by_chain(structure)

            # predict for each chain
            ps, ys = {}, {}
            if len(subunits) > 1:
                for cid in subunits:
                    subunit = subunits[cid]
                    if len(np.unique(subunit['resid'])) >= model.module.config_data['min_num_res']:
                        subunit['chain_name'] = np.array([cid]*subunit['xyz'].shape[0])
                        _, ps[cid], ys[cid] = model(subunit)

            # assess predictions
            results.append({"key": key, "type": "all"})
            results[-1].update(scoring(p0, y0, conf))

            results.append({"key": key, "type": "aa"})
            results[-1].update(scoring(p1, y1, conf))

            for k, cid in enumerate(ps):
                results.append({"key": key, "type": "s{}".format(k)})
                results[-1].update(scoring(ps[cid], ys[cid], conf))

            # pack results
            if len(results) > 0:
                dfi = pd.DataFrame(results)
                dfi.to_csv(out_filepath, index=False)

        except Exception as e:
            print("ERROR", e, i)


if __name__ == '__main__':
    main()
