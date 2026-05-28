use crate::{Laplacian, LaplacianType};

/// Build a Laplacian from a flat row-major transition matrix and a similarity kernel.
///
/// `transitions` is n*n row-major. `similarity` is Fn(usize, usize) -> f64.
pub fn build_laplacian<F>(
    transitions: &[f64],
    n: usize,
    similarity: F,
    laplacian_type: LaplacianType,
) -> Laplacian
where
    F: Fn(usize, usize) -> f64,
{
    // W[i,j] = transitions[i,j] * similarity(i,j)
    let mut w = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            w[i * n + j] = transitions[i * n + j] * similarity(i, j);
        }
    }

    // Degree: D[i] = sum_j W[i,j]
    let mut degree = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += w[i * n + j];
        }
        degree[i] = sum;
    }

    // Symmetrize W: W_sym = (W + W^T) / 2
    // This ensures the Laplacian is symmetric, required for spectral analysis.
    let mut w_sym = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            w_sym[i * n + j] = (w[i * n + j] + w[j * n + i]) / 2.0;
        }
    }

    // Recompute degree from symmetrized W
    let mut degree_sym = vec![0.0; n];
    for i in 0..n {
        degree_sym[i] = (0..n).map(|j| w_sym[i * n + j]).sum();
    }

    let normalized = laplacian_type != LaplacianType::Unnormalized;

    let matrix = match laplacian_type {
        LaplacianType::Unnormalized => {
            // L = D - W_sym
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        l[i * n + j] = degree_sym[i] - w_sym[i * n + j];
                    } else {
                        l[i * n + j] = -w_sym[i * n + j];
                    }
                }
            }
            l
        }
        LaplacianType::SymmetricNormalized => {
            // L_sym = I - D^{-1/2} W_sym D^{-1/2}
            let d_inv_sqrt: Vec<f64> = degree_sym
                .iter()
                .map(|d| if *d > 0.0 { 1.0 / d.sqrt() } else { 0.0 })
                .collect();
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        l[i * n + j] = 1.0 - d_inv_sqrt[i] * w_sym[i * n + j] * d_inv_sqrt[j];
                    } else {
                        l[i * n + j] = -d_inv_sqrt[i] * w_sym[i * n + j] * d_inv_sqrt[j];
                    }
                }
            }
            l
        }
        LaplacianType::RandomWalkNormalized => {
            // L_rw = I - D^{-1} W_sym
            let d_inv: Vec<f64> = degree_sym
                .iter()
                .map(|d| if *d > 0.0 { 1.0 / d } else { 0.0 })
                .collect();
            let mut l = vec![0.0; n * n];
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        l[i * n + j] = 1.0 - d_inv[i] * w_sym[i * n + j];
                    } else {
                        l[i * n + j] = -d_inv[i] * w_sym[i * n + j];
                    }
                }
            }
            l
        }
    };

    Laplacian {
        matrix,
        degree: degree_sym,
        weights: w_sym,
        normalized,
        num_vertices: n,
    }
}

/// Build Laplacian from a TensionGraph's adjacency, using unit similarity kernel.
pub fn build_laplacian_from_graph<V, A>(
    graph: &crate::TensionGraph<V, A>,
    laplacian_type: LaplacianType,
) -> Laplacian
where
    V: std::clone::Clone + std::cmp::Eq + std::hash::Hash,
    A: Clone + Into<f64>,
{
    let n = graph.vertex_count();
    // Build adjacency matrix from graph edges
    let mut transitions = vec![0.0; n * n];
    for (i, neighbors) in graph.adjacency.iter().enumerate() {
        let total: f64 = neighbors.iter().map(|(_, w)| Clone::clone(w).into()).sum();
        if total > 0.0 {
            for &(j, ref w) in neighbors {
                transitions[i * n + j] = Clone::clone(w).into() / total;
            }
        }
    }
    build_laplacian(&transitions, n, |_, _| 1.0, laplacian_type)
}

/// Convert Laplacian to a nalgebra DMatrix.
pub fn to_nalgebra(lap: &Laplacian) -> nalgebra::DMatrix<f64> {
    nalgebra::DMatrix::from_row_slice(lap.num_vertices, lap.num_vertices, &lap.matrix)
}
