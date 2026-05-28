use crate::{EigenDecomposition, Laplacian, LaplacianType};

/// Full symmetric eigendecomposition using nalgebra.
/// Returns all eigenvalues sorted ascending and corresponding eigenvectors.
pub fn eigendecompose(lap: &Laplacian) -> EigenDecomposition {
    let n = lap.num_vertices;
    let mat = crate::laplacian::to_nalgebra(lap);

    // nalgebra symmetric eigendecomposition
    let eig = mat.symmetric_eigen();
    // eig.eigenvalues is a VecN<nalgebra::U1, f64> effectively a DVector
    let mut pairs: Vec<(f64, Vec<f64>)> = Vec::with_capacity(n);

    for k in 0..n {
        let val = eig.eigenvalues[k];
        // Extract column k from eigenvectors matrix
        let col = eig.eigenvectors.column(k);
        let vec: Vec<f64> = col.iter().copied().collect();
        pairs.push((val, vec));
    }

    // Sort by eigenvalue ascending
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = pairs.iter().map(|(v, _)| *v).collect();
    let eigenvectors: Vec<Vec<f64>> = pairs.into_iter().map(|(_, v)| v).collect();

    EigenDecomposition {
        eigenvalues,
        eigenvectors,
        laplacian_type: if lap.normalized {
            LaplacianType::SymmetricNormalized
        } else {
            LaplacianType::Unnormalized
        },
        num_vertices: n,
    }
}

/// Power iteration for the single largest eigenvalue (by absolute value).
/// Returns (eigenvalue, eigenvector).
pub fn power_iteration(
    matrix: &[f64],
    n: usize,
    max_iters: usize,
    tolerance: f64,
) -> (f64, Vec<f64>) {
    let mut v = vec![1.0; n];
    // Normalize
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    for x in v.iter_mut() {
        *x /= norm;
    }

    let mut eigenvalue = 0.0;

    for _ in 0..max_iters {
        // w = M * v
        let mut w = vec![0.0; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i * n + j] * v[j];
            }
        }

        // Rayleigh quotient
        let new_eigenvalue: f64 = w.iter().zip(v.iter()).map(|(a, b)| a * b).sum();

        // Normalize w
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        for x in w.iter_mut() {
            *x /= norm;
        }

        // Check convergence
        if (new_eigenvalue - eigenvalue).abs() < tolerance {
            eigenvalue = new_eigenvalue;
            v = w;
            break;
        }
        eigenvalue = new_eigenvalue;
        v = w;
    }

    (eigenvalue, v)
}

/// Lanczos iteration for k smallest eigenvalues.
/// Works on a sparse-ish flat matrix. Returns k eigenvalue/eigenvector pairs.
pub fn lanczos_iteration(
    matrix: &[f64],
    n: usize,
    k: usize,
    max_iters: usize,
) -> EigenDecomposition {
    let k = k.min(n).max(1);
    let m = max_iters.min(n).max(k);

    // Initialize
    let mut q = vec![vec![0.0; n]; m + 1];
    let mut alpha = vec![0.0; m];
    let mut beta = vec![0.0; m];

    // q[0] = random start, normalized
    q[0] = (0..n).map(|i| (i as f64 + 1.0) / (n as f64).sqrt()).collect();

    let mut iterations = 0usize;

    for j in 0..m {
        // w = A * q[j]
        let mut w = vec![0.0; n];
        for i in 0..n {
            for col in 0..n {
                w[i] += matrix[i * n + col] * q[j][col];
            }
        }

        // alpha[j] = q[j]^T * w
        alpha[j] = q[j].iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f64>();

        // w = w - alpha[j]*q[j] - beta[j]*q[j-1]
        for i in 0..n {
            w[i] -= alpha[j] * q[j][i];
            if j > 0 {
                w[i] -= beta[j - 1] * q[j - 1][i];
            }
        }

        // Reorthogonalize (full)
        for _reorth in 0..2 {
            for l in 0..=j {
                let dot: f64 = q[l].iter().zip(w.iter()).map(|(a, b)| a * b).sum::<f64>();
                for i in 0..n {
                    w[i] -= dot * q[l][i];
                }
            }
        }

        iterations = j + 1;

        // beta[j] = ||w||
        let b: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if b < 1e-14 || j + 1 >= m {
            break;
        }
        beta[j] = b;

        // q[j+1] = w / beta[j]
        q[j + 1] = w.iter().map(|x| x / beta[j]).collect();
    }

    let m_actual = iterations;

    // Build tridiagonal matrix T (m_actual x m_actual) and solve it
    // Using dense eigendecomposition on the small tridiagonal matrix
    let mut t = vec![0.0; m_actual * m_actual];
    for i in 0..m_actual {
        t[i * m_actual + i] = alpha[i];
        if i > 0 {
            t[i * m_actual + (i - 1)] = beta[i - 1];
            t[(i - 1) * m_actual + i] = beta[i - 1];
        }
    }

    let t_mat = nalgebra::DMatrix::from_row_slice(m_actual, m_actual, &t);
    let t_eig = t_mat.symmetric_eigen();

    // Sort eigenvalues ascending, take k smallest
    let mut indices: Vec<usize> = (0..m_actual).collect();
    indices.sort_by(|&a, &b| {
        t_eig.eigenvalues[a]
            .partial_cmp(&t_eig.eigenvalues[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let k_actual = k.min(m_actual);
    let mut eigenvalues = Vec::with_capacity(k_actual);
    let mut eigenvectors = Vec::with_capacity(k_actual);

    for &idx in indices.iter().take(k_actual) {
        eigenvalues.push(t_eig.eigenvalues[idx]);
        // Recover eigenvector in original space: v = Q * s
        let s = t_eig.eigenvectors.column(idx);
        let mut v = vec![0.0; n];
        for j in 0..m_actual {
            let sj = s[j];
            for i in 0..n {
                v[i] += sj * q[j][i];
            }
        }
        // Normalize
        let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for x in v.iter_mut() {
                *x /= norm;
            }
        }
        eigenvectors.push(v);
    }

    EigenDecomposition {
        eigenvalues,
        eigenvectors,
        laplacian_type: LaplacianType::Unnormalized,
        num_vertices: n,
    }
}
