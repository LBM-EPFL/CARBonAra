import numpy as np
import torch as pt

from .data_encoding import std_elements, std_element_radii


def superpose(xyz_ref, xyz):
    # centering
    t = pt.mean(xyz, dim=1).unsqueeze(1)
    t_ref = pt.mean(xyz_ref, dim=1).unsqueeze(1)

    # SVD decomposition
    U, _, Vt = pt.linalg.svd(pt.matmul(pt.transpose(xyz_ref-t_ref,1,2), xyz-t))

    # reflection matrix
    Z = pt.zeros(U.shape, device=xyz.device) + pt.eye(U.shape[1], U.shape[2], device=xyz.device).unsqueeze(0)
    Z[:,-1,-1] = pt.linalg.det(U) * pt.linalg.det(Vt)

    R = pt.matmul(pt.transpose(Vt,1,2), pt.matmul(Z, pt.transpose(U,1,2)))

    return pt.matmul(xyz-t, R)+t_ref


def extract_geometry(X):
    # compute displacement vectors
    R = X.unsqueeze(0) - X.unsqueeze(1)
    # compute distance matrix
    D = pt.norm(R, dim=2)
    # normalize displacement vectors
    R = R / (D.detach().unsqueeze(2) + (D < 1e-3).unsqueeze(2).float())
    return D, R


def extract_connectivity(qe, D, alpha=1.2):
    elements = np.concatenate([std_elements, ['X']])[pt.argmax(qe, dim=1).cpu().numpy()]
    radii = pt.from_numpy(np.array([std_element_radii[e.upper()] for e in elements])).to(D.device)
    return ((D < np.sqrt(alpha)*(radii.unsqueeze(0) + radii.unsqueeze(1))) & ~pt.eye(D.shape[0], device=D.device, dtype=pt.bool)).float()


def connected_distance_matrix(C):
    # initialize state
    S = C.clone()
    L = C.clone()
    I = pt.eye(C.shape[0], device=C.device)

    # iterate
    for i in range(C.shape[0]):
        # propagate information through graph
        S = pt.clip(pt.matmul(C,S), min=0.0, max=1.0)
        # deactivate already activated cells
        S = pt.clip(S-L-I, min=0.0, max=1.0)
        # update paths length
        L += (i+2)*S

        # check convergence
        if pt.sum(S) == 0.0:
            break

    return L


def follow_rabbit(M, i):
    ids_checked = {i}
    ids_checking = set(np.where(M[i])[0])
    while ids_checking:
        for j in ids_checking.copy():
            ids_checking.remove(j)
            ids_checking.update(set([i for i in np.where(M[j])[0] if i not in ids_checked]))
            ids_checked.add(j)

    return list(ids_checked)


def follow_rabbits(M):
    i = 0
    ids_checked = []
    ids_clust = []
    while len(ids_checked) < M.shape[0]:
        ids_connect = follow_rabbit(M, i)
        ids_checked.extend(ids_connect)
        ids_clust.append(ids_connect)
        for j in range(i,M.shape[0]):
            if j not in ids_checked:
                i = j
                break

    return ids_clust


def find_bonded_graph_neighborhood(L, D, num_bond):
    # find neighborhood in bonded graph space
    D1 = L + 0.999*(D.detach() / pt.max(D.detach())) + 2.0*pt.max(L)*(L < 0.5).float()
    _, ids_nn = pt.topk(D1, num_bond, dim=1, largest=False)

    # map unbonded atoms to itself
    m_bb_nn = pt.gather(L < 1.0, 1, ids_nn)
    m_bb_nn = m_bb_nn & ~pt.all(m_bb_nn, dim=1).reshape(-1,1)
    ids0, ids1 = pt.where(m_bb_nn)
    ids_nn[ids0, ids1] = ids0

    return ids_nn


def connected_paths(C, length):
    Mc = (C > 0.5)
    Gc = [np.array([])]+[np.where(Mc[i])[0]+1 for i in range(Mc.shape[0])]
    cids = [[i] for i in range(len(Gc))]
    for n in range(length):
        cids_next = []
        for k in range(len(cids)):
            ids_next = Gc[cids[k][-1]]
            if len(ids_next) > 0:
                for i in ids_next:
                    if i not in cids[k]:
                        cids_next.extend([cids[k].copy()+[i]])
                    else:
                        cids_next.extend([cids[k].copy()+[0]])
            else:
                cids_next.extend([cids[k].copy()+[0]])
        cids = cids_next
    return np.array(cids)-1


def topology_hash(C, qe, length):
    # get all connected paths up to length
    cpaths = connected_paths(C, length)

    # hash connections per atom
    #qs = np.array(["{}-{}-{}".format(ve,vr,vn) for ve,vr,vn in zip(np.argmax(qe, axis=1), np.argmax(qr, axis=1), np.argmax(qn, axis=1))])
    qs = np.array(["{}".format(ve) for ve in np.argmax(qe, axis=1)])
    hs = []
    for k in np.unique(cpaths[1:,0]):
        cpk = cpaths[cpaths[:,0] == k]
        hsk = []
        for i in range(cpk.shape[0]):
            hsi = []
            for j in range(cpk.shape[1]):
                if cpk[i,j] >= 0:
                    hsi.append(qs[cpk[i,j]])
                #else:
                    #hsi.append('_')
            hsk.append(':'.join(hsi))

        hs.append("+".join(sorted(hsk)))
    return np.array(hs)


def extract_context_map(C, qe, path_length=5):
    # hash topology and get indices of context
    hs = topology_hash(C.cpu().numpy(), qe.cpu().numpy(), path_length)
    hsu, ids_ctx = np.unique(hs, return_inverse=True)
    Mc = pt.stack([pt.from_numpy(ids_ctx == i) for i in range(hsu.shape[0])], dim=1)

    return Mc.to(C.device)
