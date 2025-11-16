import torch
import numpy as np
from src import utils
import os

import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from GraphRicciCurvature.FormanRicci import FormanRicci

class DummyExtraFeatures:
    def __init__(self):
        """ This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraFeatures:
    def __init__(self, extra_features_type, dataset_info, ricci_alpha=0.5):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        self.ricci_alpha = ricci_alpha
        
        # Initialize eigenfeatures if needed
        if extra_features_type in ['eigenvalues', 'all', 'all_olliver_ricci', 
                                   'all_forman_ricci', 'eigenvalues_olliver_ricci', 
                                   'eigenvalues_forman_ricci']:
            self.eigenfeatures = EigenFeatures(mode='eigenvalues' if 'eigenvalues' in extra_features_type and 'all' not in extra_features_type else 'all')

    def __call__(self, noisy_data):
        n = noisy_data['node_mask'].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(noisy_data)       # (bs, n_cycles)
        
        # Initialize edge features
        E = noisy_data['E_t']
        extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)

        if self.features_type == 'cycles':
            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles)))

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(noisy_data)
            n_components, batched_eigenvalues = eigenfeatures   # (bs, 1), (bs, k)
            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, 
                                    y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        
        elif self.features_type == 'all':
            eigenfeatures = self.eigenfeatures(noisy_data)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures
            return utils.PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
                                     E=extra_edge_attr,
                                     y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        
        # Ricci curvature features
        elif self.features_type in ['all_olliver_ricci', 'eigenvalues_olliver_ricci', 
                                    'cycle_olliver_ricci', 'olliver_ricci']:
            node_ricci, edge_ricci = self.compute_ricci_curvature(noisy_data, 'olliver')
            
            if self.features_type == 'olliver_ricci':
                # Only Ricci features
                return utils.PlaceHolder(X=node_ricci, E=edge_ricci, y=torch.hstack((n, y_cycles)))
            
            elif self.features_type == 'cycle_olliver_ricci':
                # Cycles + Ricci
                return utils.PlaceHolder(X=torch.cat((x_cycles, node_ricci), dim=-1),
                                       E=edge_ricci,
                                       y=torch.hstack((n, y_cycles)))
            
            elif self.features_type == 'eigenvalues_olliver_ricci':
                # Eigenvalues + Cycles + Ricci
                eigenfeatures = self.eigenfeatures(noisy_data)
                n_components, batched_eigenvalues = eigenfeatures
                return utils.PlaceHolder(X=torch.cat((x_cycles, node_ricci), dim=-1),
                                       E=edge_ricci,
                                       y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
            
            elif self.features_type == 'all_olliver_ricci':
                # All features + Ricci
                eigenfeatures = self.eigenfeatures(noisy_data)
                n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures
                return utils.PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec, node_ricci), dim=-1),
                                       E=edge_ricci,
                                       y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        
        elif self.features_type in ['all_forman_ricci', 'eigenvalues_forman_ricci', 
                                    'cycle_forman_ricci', 'forman_ricci']:
            node_ricci, edge_ricci = self.compute_ricci_curvature(noisy_data, 'forman')
            
            if self.features_type == 'forman_ricci':
                # Only Ricci features
                return utils.PlaceHolder(X=node_ricci, E=edge_ricci, y=torch.hstack((n, y_cycles)))
            
            elif self.features_type == 'cycle_forman_ricci':
                # Cycles + Ricci
                return utils.PlaceHolder(X=torch.cat((x_cycles, node_ricci), dim=-1),
                                       E=edge_ricci,
                                       y=torch.hstack((n, y_cycles)))
            
            elif self.features_type == 'eigenvalues_forman_ricci':
                # Eigenvalues + Cycles + Ricci
                eigenfeatures = self.eigenfeatures(noisy_data)
                n_components, batched_eigenvalues = eigenfeatures
                return utils.PlaceHolder(X=torch.cat((x_cycles, node_ricci), dim=-1),
                                       E=edge_ricci,
                                       y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
            
            elif self.features_type == 'all_forman_ricci':
                # All features + Ricci
                eigenfeatures = self.eigenfeatures(noisy_data)
                n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures
                return utils.PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec, node_ricci), dim=-1),
                                       E=edge_ricci,
                                       y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        
        else:
            raise ValueError(f"Features type {self.features_type} not implemented")

    def compute_ricci_curvature(self, noisy_data, ricci_type):
        """ Compute Ricci curvature for edges and aggregate to nodes.

        Args:
            noisy_data: Dictionary containing graph data
                - 'E_t': (bs, n, n, e_types) probabilistic edge tensor
                - 'node_mask': (bs, n)
            ricci_type: 'olliver' or 'forman'

        Returns:
            node_ricci: (bs, n, 1) - averaged Ricci curvature per node
            edge_ricci: (bs, n, n, 1) - Ricci curvature per edge
        """
        E_t = noisy_data['E_t']
        node_mask = noisy_data['node_mask']

        # Aggregate probabilities over all edge types except 'no-edge'
        adj_matrix = E_t[..., 1:].sum(dim=-1).float()  # (bs, n, n)
        bs, n, _ = adj_matrix.shape

        edge_curvatures = torch.zeros(bs, n, n, 1).type_as(adj_matrix)

        for b in range(bs):
            try:
                # Get adjacency matrix for this graph (as numpy)
                adj_b = adj_matrix[b].cpu().numpy()
                mask_b = node_mask[b].cpu().numpy().astype(bool)

                # Apply node mask
                adj_b = adj_b * mask_b[:, None] * mask_b[None, :]

                # Skip empty graphs
                if np.sum(adj_b) == 0:
                    continue

                # Build weighted graph for curvature computation
                G = nx.from_numpy_array(adj_b)
                G.remove_edges_from(nx.selfloop_edges(G))

                if G.number_of_edges() == 0:
                    continue

                # --- Compute Ricci curvature ---
                if ricci_type == "olliver":
                    orc = OllivierRicci(
                        G,
                        alpha=self.ricci_alpha,
                        method = "sinkhorn",
                        proc=os.cpu_count(),
                        verbose="ERROR"
                    )
                    orc.compute_ricci_curvature_edges()
                    G_ricci = orc.G
                    curvature_key = 'ricciCurvature'

                elif ricci_type == "forman":
                    frc = FormanRicci(
                        G,
                        verbose="ERROR"
                    )
                    frc.compute_ricci_curvature()
                    G_ricci = frc.G
                    curvature_key = 'formanCurvature'
                else:
                    raise ValueError(f"Unknown curvature type: {ricci_type}")

                # Extract curvature values
                curvatures = np.zeros((n, n))
                for (i, j) in G_ricci.edges():
                    if curvature_key in G_ricci[i][j]:
                        curv = G_ricci[i][j][curvature_key]
                        curvatures[i, j] = curv
                        curvatures[j, i] = curv  # symmetric

                # Clean and normalize curvature values (adaptive percentile-based)
                curvatures = np.nan_to_num(curvatures, nan=0.0, posinf=0.0, neginf=0.0)
                if np.any(curvatures != 0):
                    nonzero_curvs = curvatures[curvatures != 0]
                    # Percentile-based bounds with conservative clamps to [-2, 1]
                    min_curv = max(np.percentile(nonzero_curvs, 5), -2.0)
                    max_curv = min(np.percentile(nonzero_curvs, 95), 1.0)
                    if max_curv > min_curv:
                        curvatures_normalized = (curvatures - min_curv) / (max_curv - min_curv + 1e-12)
                        curvatures_normalized = np.clip(curvatures_normalized, 0, 1)
                    else:
                        # All curvatures effectively identical
                        curvatures_normalized = np.full_like(curvatures, 0.5)
                else:
                    # No curvature values present
                    curvatures_normalized = np.zeros_like(curvatures)

                # Convert back to tensor
                edge_curvatures[b, :, :, 0] = torch.from_numpy(curvatures_normalized).type_as(adj_matrix)

            except Exception as e:
                print(f"⚠️ Warning: Failed to compute {ricci_type} Ricci curvature for graph {b}: {e}")
                continue

        # --- Compute node-level Ricci curvature (average over edges) ---
        node_ricci = self.compute_node_ricci(edge_curvatures, adj_matrix, node_mask)

        return node_ricci, edge_curvatures

    
    def compute_node_ricci(self, edge_curvatures, adj_matrix, node_mask):
        """
        Compute node-level Ricci curvature by averaging edge curvatures.
        
        Args:
            edge_curvatures: (bs, n, n, 1) - edge Ricci curvatures
            adj_matrix: (bs, n, n) - adjacency matrix
            node_mask: (bs, n) - node mask
            
        Returns:
            node_ricci: (bs, n, 1) - average Ricci curvature per node
        """
        bs, n, _, _ = edge_curvatures.shape
        
        # Sum curvatures of incident edges for each node
        # Since the graph is undirected, we sum both outgoing and incoming (they're the same)
        edge_curv_squeezed = edge_curvatures.squeeze(-1)  # (bs, n, n)
        
        # Mask: only consider actual edges
        masked_curvatures = edge_curv_squeezed * adj_matrix  # (bs, n, n)
        
        # Sum curvatures of incident edges per node
        sum_curvatures = masked_curvatures.sum(dim=-1)  # (bs, n)
        
        # Compute node degree
        degree = adj_matrix.sum(dim=-1)  # (bs, n)
        
        # Avoid division by zero
        degree = degree.clamp(min=1.0)
        
        # Average curvature per node
        node_ricci = sum_curvatures / degree  # (bs, n)
        
        # Apply node mask
        node_ricci = node_ricci * node_mask  # (bs, n)
        
        return node_ricci.unsqueeze(-1)  # (bs, n, 1)


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)
        x_cycles = x_cycles.type_as(adj_matrix) * noisy_data['node_mask'].unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, mode):
        """ mode: 'eigenvalues' or 'all' """
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = torch.linalg.eigvalsh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == 'all':
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                               node_mask=noisy_data['node_mask'],
                                                                               n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)
    combinatorial = D - adjacency

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)
    D_norm = torch.diag_embed(diag_norm)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values
    mask = ~ (first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)
    indices = indices.expand(-1, n, -1)
    first_k_ev = torch.gather(vectors, dim=2, index=indices)
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy