import numpy as np

from src.structure import split_by_residue
from src.data_encoding import std_resnames, std_aminoacids, std_element_radii


def initialize_residues(subunit, std_templates):
    # remove known key if existing
    if 'known' in subunit:
        subunit.pop('known')

    # create new starting fixed structure
    initial_subunit = {
        'xyz': [],
        'name': [],
        'element': [],
        'resname': [],
        'resid': [],
        'het_flag': [],
        'known': [],
    }
    residues = split_by_residue(subunit)
    for k in range(len(residues)):
        # get residue
        res = residues[k]

        # get center of point of existing atoms
        xyz_c = np.mean(res['xyz'], axis=0)

        # guess from current geometry if not template provided
        if res['resname'][0] not in std_templates:
            # debug print
            # print("WARNING: guessing template for {}".format(res['resname'][0]))

            # get internal connectivity
            D = np.linalg.norm(np.expand_dims(res['xyz'], 1) - np.expand_dims(res['xyz'], 0), axis=2)
            radii = np.array([std_element_radii[e.upper()] for e in res['element']]).reshape(-1,1)
            C = ((D < np.sqrt(1.2)*(radii + radii.T)) & (D > 0.0))

            # pack template
            std_templates[res['resname'][0]] = {
                'name': list(res['name']),
                'element': list(res['element']),
                'connectivity': [(int(i),int(j)) for i,j in np.stack(np.where(C), axis=1)],
            }

        # find template
        res_tmp = std_templates[res['resname'][0]]

        # iterate over template atoms
        for i in range(len(res_tmp['name'])):
            # skip phosphate backbone of DNA/RNA if first residue
            if (k == 0) and ((res_tmp['name'][i] == 'P') or (res_tmp['name'][i] == 'OP1') or (res_tmp['name'][i] == 'OP2')):
                continue

            # find if atom exists
            ids_exists = np.where(res['name'] == res_tmp['name'][i])[0]

            if len(ids_exists) == 1:
                # overwrite if existing
                for key in initial_subunit:
                    if key in res:
                        initial_subunit[key].append(res[key][ids_exists[0]])
                # flag existing
                initial_subunit['known'].append(True)
            else:
                # add new atom with random position around existing fragment
                initial_subunit['xyz'].append(xyz_c + np.random.randn(3))
                initial_subunit['name'].append(res_tmp['name'][i])
                initial_subunit['element'].append(res_tmp['element'][i])
                for key in ['resname', 'resid', 'het_flag']:
                    initial_subunit[key].append(res[key][0])
                # flag not existing
                initial_subunit['known'].append(False)

        # insert OXT
        if k == (len(residues) - 1):
            if np.all(np.isin(residues[-1]['resname'], std_resnames[:20])) and ('OXT' not in residues[-1]['name']):
                initial_subunit['xyz'].append(residues[-1]['xyz'][-1] + np.random.randn(3))
                initial_subunit['name'].append('OXT')
                initial_subunit['element'].append('O')
                initial_subunit['resname'].append(residues[-1]['resname'][0])
                initial_subunit['resid'].append(residues[-1]['resid'][0])
                initial_subunit['het_flag'].append(residues[-1]['het_flag'][-1])
                initial_subunit['known'].append(False)

    # reformat types
    for key in initial_subunit:
        initial_subunit[key] = np.array(initial_subunit[key])

    return initial_subunit


def assign_template_connectivity(subunit, std_templates):
    # connectivity
    C_ids = []

    # recover connectivity
    i_shift = 0
    for res in split_by_residue(subunit):
        resname = res['resname'][0]
        if resname in std_templates:
            # connectivity from template
            c = []
            for ti,tj in std_templates[resname]['connectivity']:
                vi = np.where(res['name'] == std_templates[resname]['name'][ti])[0]
                vj = np.where(res['name'] == std_templates[resname]['name'][tj])[0]
                if (len(vi) > 0) and (len(vj) > 0):
                    c.append((vi.item(),vj.item()))
                    c.append((vj.item(),vi.item()))
        else:
            # connectivity from structure
            pass

        # update connectivity if OXT present
        if 'OXT' in res['name']:
            i, j = np.where(res['name'] == 'C')[0].item(), np.where(res['name'] == 'OXT')[0].item()
            c.append((i,j))
            c.append((j,i))

        # insert connected indices
        C_ids.append(np.array(c) + i_shift)

        # atom index shift for full connectivity
        i_shift += len(res['xyz'])

    # pack connectivity
    return np.concatenate(C_ids)


def additional_inter_residue_connectivity(subunit):
    # add connection between amino-acids for each subunits
    C_bb_ids = []
    i_shift = 0

    # protein bond between amino-acids
    if np.all(np.isin(subunit['resname'], std_aminoacids)):
        # split structure by residue
        residues = split_by_residue(subunit)

        # connect successive residues
        for res1, res2 in zip(residues[:-1], residues[1:]):
            # locate connected atoms
            i = np.where(res1['name'] == 'C')[0].item() + i_shift
            i_shift += res1['xyz'].shape[0]
            j = np.where(res2['name'] == 'N')[0].item() + i_shift

            # update connectivity
            C_bb_ids.append((i,j))
            C_bb_ids.append((j,i))

    # dna/rna bond between bases
    if np.all(np.isin(subunit['resname'], std_resnames[20:28])):
        # split structure by residue
        residues = split_by_residue(subunit)

        # connect successive residues
        for res1, res2 in zip(residues[:-1], residues[1:]):
            # locate connected atoms
            i = np.where(res1['name'] == "O3'")[0].item() + i_shift
            i_shift += res1['xyz'].shape[0]
            j = np.where(res2['name'] == "P")[0].item() + i_shift

            # update connectivity
            C_bb_ids.append((i,j))
            C_bb_ids.append((j,i))

    # pack connectivity
    return np.array(C_bb_ids)
