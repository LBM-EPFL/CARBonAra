import torch as pt
from torch.utils.checkpoint import checkpoint


# >> UTILS
def unpack_state_features(X, ids_topk, q, max_nn):
    # compute displacement vectors
    R_nn = X[ids_topk] - X.unsqueeze(1)
    # compute distance matrix
    D_nn = pt.norm(R_nn, dim=2)
    # mask distances
    D_nn = D_nn + pt.max(D_nn)*(D_nn < 1e-2).float()
    # normalize displacement vectors
    R_nn = R_nn / D_nn.unsqueeze(2)

    # nearest neighbors with sink
    ids_topk_with_sink = pt.zeros((ids_topk.shape[0]+1, max_nn), device=ids_topk.device, dtype=pt.long)
    ids_topk_with_sink[1:, :ids_topk.shape[1]] = ids_topk+1

    # prepare sink
    q = pt.cat([pt.zeros((1, q.shape[1]), device=q.device), q], dim=0)

    # geometry sink
    D_nn = pt.cat([pt.zeros((1, D_nn.shape[1]), device=D_nn.device), D_nn], dim=0)
    R_nn = pt.cat([pt.zeros((1, R_nn.shape[1], R_nn.shape[2]), device=R_nn.device), R_nn], dim=0)

    return q, ids_topk_with_sink, D_nn, R_nn


# >>> OPERATIONS
class StateUpdate(pt.nn.Module):
    def __init__(self, Ns, Nh, Nk):
        super(StateUpdate, self).__init__()
        # operation parameters
        self.Ns = Ns
        self.Nh = Nh
        self.Nk = Nk

        # node query model
        self.nqm = pt.nn.Sequential(
            pt.nn.Linear(2*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 2*Nk*Nh),
        )

        # edges scalar keys model
        self.eqkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Nk),
        )

        # edges vector keys model
        self.epkm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, 4*Nk),
        )

        # edges value model
        self.evm = pt.nn.Sequential(
            pt.nn.Linear(6*Ns+1, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
            pt.nn.ELU(),
            pt.nn.Linear(2*Ns, 2*Ns),
        )

        # scalar projection model
        self.qpm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
            pt.nn.ELU(),
            pt.nn.Linear(Ns, Ns),
        )

        # vector projection model
        self.ppm = pt.nn.Sequential(
            pt.nn.Linear(Nh*Ns, Ns, bias=False),
        )

        # scaling factor for attention
        self.sdk = pt.nn.Parameter(pt.sqrt(pt.tensor(Nk).float()), requires_grad=False)

    def forward(self, q, p, q_nn, p_nn, d_nn, r_nn):
        # q: [N, S]
        # p: [N, 3, S]
        # q_nn: [N, n, S]
        # p_nn: [N, n, 3, S]
        # d_nn: [N, n]
        # r_nn: [N, n, 3]
        # N: number of nodes
        # n: number of nearest neighbors
        # S: state dimensions
        # H: number of attention heads

        # get dimensions
        N, n, S = q_nn.shape

        # node inputs packing
        X_n = pt.cat([
            q,
            pt.norm(p, dim=1),
        ], dim=1)  # [N, 2*S]

        # edge inputs packing
        X_e = pt.cat([
            d_nn.unsqueeze(2),                                  # distance
            X_n.unsqueeze(1).repeat(1,n,1),                     # centered state
            q_nn,                                               # neighbors states
            pt.norm(p_nn, dim=2),                               # neighbors vector states norms
            pt.sum(p.unsqueeze(1) * r_nn.unsqueeze(3), dim=2),  # centered vector state projections
            pt.sum(p_nn * r_nn.unsqueeze(3), dim=2),            # neighbors vector states projections
        ], dim=2)  # [N, n, 6*S+1]

        # node queries
        Q = self.nqm.forward(X_n).view(N, 2, self.Nh, self.Nk)  # [N, 2*S] -> [N, 2, Nh, Nk]

        # scalar edges keys while keeping interaction order inveriance
        Kq = self.eqkm.forward(X_e).view(N, n, self.Nk).transpose(1,2)  # [N, n, 6*S+1] -> [N, Nk, n]

        # vector edges keys while keeping bond order inveriance
        Kp = pt.cat(pt.split(self.epkm.forward(X_e), self.Nk, dim=2), dim=1).transpose(1,2)

        # edges values while keeping interaction order inveriance
        V = self.evm.forward(X_e).view(N, n, 2, S).transpose(1,2)  # [N, n, 6*S+1] -> [N, 2, n, S]

        # vectorial inputs packing
        Vp = pt.cat([
            V[:,1].unsqueeze(2) * r_nn.unsqueeze(3),
            p.unsqueeze(1).repeat(1,n,1,1),
            p_nn,
            pt.cross(p.unsqueeze(1).repeat(1,n,1,1), r_nn.unsqueeze(3).repeat(1,1,1,S), dim=2),
        ], dim=1).transpose(1,2)  # [N, 3, 4*n, S]

        # queries and keys collapse
        Mq = pt.nn.functional.softmax(pt.matmul(Q[:,0], Kq) / self.sdk, dim=2)  # [N, Nh, n]
        Mp = pt.nn.functional.softmax(pt.matmul(Q[:,1], Kp) / self.sdk, dim=2)  # [N, Nh, 4*n]

        # scalar state attention mask and values collapse
        Zq = pt.matmul(Mq, V[:,0]).view(N, self.Nh*self.Ns)  # [N, Nh*S]
        Zp = pt.matmul(Mp.unsqueeze(1), Vp).view(N, 3, self.Nh*self.Ns)  # [N, 3, Nh*S]

        # decode outputs
        qh = self.qpm.forward(Zq)  # [N, S]
        ph = self.ppm.forward(Zp)  # [N, 3, S]

        # update state with residual
        qz = q + qh
        pz = p + ph

        return qz, pz


class StatePoolLayer(pt.nn.Module):
    def __init__(self, N0, N1, Nh):
        super(StatePoolLayer, self).__init__()
        # state attention model
        self.sam = pt.nn.Sequential(
            pt.nn.Linear(2*N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, 2*Nh),
        )

        # attention heads decoding
        self.zdm = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N0),
            pt.nn.ELU(),
            pt.nn.Linear(N0, N1),
        )

        # vector attention heads decoding
        self.zdm_vec = pt.nn.Sequential(
            pt.nn.Linear(Nh * N0, N1, bias=False)
        )

    def forward(self, q, p, M):
        # create filter for softmax
        F = (1.0 - M + 1e-6) / (M - 1e-6)

        # pack features
        z = pt.cat([q, pt.norm(p, dim=1)], dim=1)

        # multiple attention pool on state
        Ms = pt.nn.functional.softmax(self.sam.forward(z).unsqueeze(1) + F.unsqueeze(2), dim=0).view(M.shape[0], M.shape[1], -1, 2)
        qh = pt.matmul(pt.transpose(q,0,1), pt.transpose(Ms[:,:,:,0],0,1))
        ph = pt.matmul(pt.transpose(pt.transpose(p,0,2),0,1), pt.transpose(Ms[:,:,:,1],0,1).unsqueeze(1))

        # attention heads decoding
        qr = self.zdm.forward(qh.view(Ms.shape[1], -1))
        pr = self.zdm_vec.forward(ph.view(Ms.shape[1], p.shape[1], -1))

        return qr, pr


# >>> LAYERS
class StateUpdateLayer(pt.nn.Module):
    def __init__(self, layer_params):
        super(StateUpdateLayer, self).__init__()
        # define operation
        self.su = StateUpdate(*[layer_params[k] for k in ['Ns', 'Nh', 'Nk']])
        # store number of nearest neighbors
        self.m_nn = pt.nn.Parameter(pt.arange(layer_params['nn'], dtype=pt.int64), requires_grad=False)

    def forward(self, Z):
        # unpack input
        q, p, ids_topk, D_topk, R_topk = Z

        # update q, p
        ids_nn = ids_topk[:,self.m_nn]
        # q, p = self.su.forward(q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])

        # with checkpoint
        q = q.requires_grad_()
        p = p.requires_grad_()
        q, p = checkpoint(self.su.forward, q, p, q[ids_nn], p[ids_nn], D_topk[:,self.m_nn], R_topk[:,self.m_nn])

        # sink
        q[0] = q[0] * 0.0
        p[0] = p[0] * 0.0

        return q, p, ids_topk, D_topk, R_topk
