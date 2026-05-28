use crate::{
    Anomaly, AnomalyType, ConservationRatio, ConservationReport, EigenDecomposition, Fix,
    SpectralFingerprint, TensionGraph,
};
use std::hash::Hash;

/// Conservation ratio for an attribute along the k-th eigenvector.
/// Measures variance of the gradient of the attribute projected onto the eigenvector.
/// Low ratio = high conservation in this mode.
pub fn conservation_ratio(eigen: &EigenDecomposition, attribute: &[f64], k: usize) -> f64 {
    if k >= eigen.eigenvectors.len() {
        return f64::INFINITY;
    }
    let phi = &eigen.eigenvectors[k];
    let n = phi.len().min(attribute.len());

    // Projection: p = phi * attribute
    let _projection: f64 = (0..n).map(|i| phi[i] * attribute[i]).sum();

    // Gradient: consecutive differences of (phi * projection contribution)
    // We compute the "mode-weighted attribute": phi_i * attribute_i
    let weighted: Vec<f64> = (0..n).map(|i| phi[i] * attribute[i]).collect();

    if weighted.len() < 2 {
        return 0.0;
    }

    let gradients: Vec<f64> = weighted.windows(2).map(|w| w[1] - w[0]).collect();
    let gn = gradients.len() as f64;
    if gn == 0.0 {
        return 0.0;
    }

    let mean = gradients.iter().sum::<f64>() / gn;
    let variance = gradients.iter().map(|g| (g - mean).powi(2)).sum::<f64>() / gn;

    // Scale by eigenvalue (small eigenvalue = smooth mode, expect small gradients)
    let lambda = eigen.eigenvalues.get(k).copied().unwrap_or(1.0);
    if lambda.abs() < 1e-15 {
        return variance;
    }
    variance / lambda.abs()
}

/// Batch conservation ratios for all eigenvectors.
pub fn conservation_ratios(
    eigen: &EigenDecomposition,
    attribute: &[f64],
    attribute_name: &str,
) -> Vec<ConservationRatio> {
    (0..eigen.eigenvalues.len())
        .map(|k| ConservationRatio {
            eigenvector_index: k,
            eigenvalue: eigen.eigenvalues[k],
            ratio: conservation_ratio(eigen, attribute, k),
            attribute_name: attribute_name.to_string(),
        })
        .collect()
}

/// Detect anomalies: vertices where eigenvector component deviates significantly.
/// Uses z-score based detection on the Fiedler vector (eigenvector 1, the second smallest).
pub fn detect_anomalies<V, A>(graph: &TensionGraph<V, A>, eigen: &EigenDecomposition, threshold: f64) -> Vec<Anomaly>
where
    V: Clone + Eq + Hash,
    A: Clone,
{
    let n = graph.vertex_count();
    if n == 0 || eigen.eigenvectors.is_empty() {
        return vec![];
    }

    let mut anomalies = Vec::new();

    // Check each significant eigenvector (skip eigenvalue 0 — constant vector)
    for k in 1..eigen.eigenvectors.len().min(n) {
        let phi = &eigen.eigenvectors[k];
        let mean = phi.iter().sum::<f64>() / n as f64;
        let std = {
            let var = phi.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            var.sqrt()
        };

        if std < 1e-15 {
            continue;
        }

        for (i, &val) in phi.iter().enumerate() {
            let z_score = (val - mean) / std;
            if z_score.abs() > threshold {
                anomalies.push(Anomaly {
                    vertex_index: i,
                    score: z_score,
                    anomaly_type: if k == 1 {
                        AnomalyType::ConservationViolation
                    } else {
                        AnomalyType::SpectralOutlier
                    },
                    description: format!(
                        "Vertex {} has z-score {:.3} in eigenvector {} (eigenvalue {:.4})",
                        i, z_score, k, eigen.eigenvalues[k]
                    ),
                });
            }
        }
    }

    anomalies
}

/// Compute spectral fingerprint from eigendecomposition.
pub fn spectral_fingerprint(eigen: &EigenDecomposition) -> SpectralFingerprint {
    let evals = &eigen.eigenvalues;
    let n = evals.len();

    if n == 0 {
        return SpectralFingerprint {
            eigenvalue_histogram: vec![],
            spectral_entropy: 0.0,
            effective_dimension: 0.0,
            gap_profile: vec![],
            conservation_profile: vec![],
        };
    }

    // Eigenvalue histogram (binned into sqrt(n) bins)
    let n_bins = (n as f64).sqrt().ceil() as usize;
    let emin = evals.iter().cloned().fold(f64::INFINITY, f64::min);
    let emax = evals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = emax - emin;
    let bin_width = if range > 1e-15 { range / n_bins as f64 } else { 1.0 };
    let histogram: Vec<f64> = (0..n_bins)
        .map(|b| {
            evals
                .iter()
                .filter(|&&e| {
                    e >= emin + b as f64 * bin_width
                        && (b + 1 == n_bins || e < emin + (b + 1) as f64 * bin_width)
                })
                .count() as f64
        })
        .collect();

    // Spectral entropy: H = -Σ p_i log(p_i) where p_i = |λ_i| / Σ |λ_i|
    let total: f64 = evals.iter().map(|e| e.abs()).sum();
    let entropy = if total > 1e-15 {
        evals
            .iter()
            .map(|&e| {
                let p = e.abs() / total;
                if p > 1e-15 {
                    -p * p.ln()
                } else {
                    0.0
                }
            })
            .sum()
    } else {
        0.0
    };

    let effective_dim = entropy.exp();

    // Gap profile
    let gaps: Vec<f64> = evals.windows(2).map(|w| w[1] - w[0]).collect();

    SpectralFingerprint {
        eigenvalue_histogram: histogram,
        spectral_entropy: entropy,
        effective_dimension: effective_dim,
        gap_profile: gaps,
        conservation_profile: vec![], // filled by caller
    }
}

/// Approximate Cheeger constant from spectral gap (λ₂).
/// Cheeger inequality: λ₂/2 ≤ h ≤ √(2λ₂).
/// We use λ₂/2 as a lower bound approximation.
pub fn cheeger_constant(eigen: &EigenDecomposition) -> f64 {
    if eigen.eigenvalues.len() < 2 {
        return 0.0;
    }
    // λ₂ is the second smallest eigenvalue (index 1 after sorting)
    let lambda2 = eigen.eigenvalues[1];
    lambda2 / 2.0
}

/// Suggest corrections for detected anomalies.
pub fn suggest_correction<V, A>(
    graph: &TensionGraph<V, A>,
    anomaly: &Anomaly,
) -> Vec<Fix>
where
    V: Clone + Eq + Hash,
    A: Clone + Into<f64>,
{
    let mut fixes = Vec::new();

    // Look at edges incident to the anomalous vertex
    let vi = anomaly.vertex_index;
    if vi < graph.adjacency.len() {
        let neighbors = &graph.adjacency[vi];
        if neighbors.is_empty() {
            fixes.push(Fix {
                description: format!("Vertex {} is isolated — consider adding edges", vi),
                confidence: 0.8,
                suggested_weight: Some(1.0),
                edge: None,
            });
        } else {
            // Suggest adjusting edge weights to reduce the anomaly
            let total_weight: f64 = neighbors.iter().map(|(_, w)| Clone::clone(w).into()).sum();
            let avg_weight = total_weight / neighbors.len() as f64;

            for &(j, ref w) in neighbors {
                let w_f64: f64 = Clone::clone(w).into();
                if (w_f64 - avg_weight).abs() > avg_weight * 0.5 {
                    fixes.push(Fix {
                        description: format!(
                            "Edge {}→{} weight {:.3} deviates significantly from average {:.3}",
                            vi, j, w_f64, avg_weight
                        ),
                        confidence: 0.6,
                        suggested_weight: Some(avg_weight),
                        edge: Some((vi, j)),
                    });
                }
            }
        }
    }

    if fixes.is_empty() {
        fixes.push(Fix {
            description: format!(
                "Anomaly at vertex {} with score {:.3} — investigate manually",
                anomaly.vertex_index, anomaly.score
            ),
            confidence: 0.3,
            suggested_weight: None,
            edge: None,
        });
    }

    fixes
}

/// Full analysis pipeline: graph → Laplacian → eigendecomposition → report.
pub fn analyze<V, A>(
    graph: &TensionGraph<V, A>,
    attribute_name: &str,
) -> ConservationReport
where
    V: Clone + Eq + Hash,
    A: Clone + Into<f64>,
{
    let n = graph.vertex_count();
    if n == 0 {
        return ConservationReport {
            ratios: vec![],
            anomalies: vec![],
            spectral_gap: 0.0,
            cheeger_constant: 0.0,
            fingerprint: SpectralFingerprint {
                eigenvalue_histogram: vec![],
                spectral_entropy: 0.0,
                effective_dimension: 0.0,
                gap_profile: vec![],
                conservation_profile: vec![],
            },
        };
    }

    // Build Laplacian from graph adjacency
    let lap = crate::laplacian::build_laplacian_from_graph(graph, crate::LaplacianType::Unnormalized);

    // Full eigendecomposition
    let eigen = crate::eigen::eigendecompose(&lap);

    // Get attribute or use uniform
    let attribute = graph
        .vertex_attributes
        .get(attribute_name)
        .cloned()
        .unwrap_or_else(|| vec![1.0; n]);

    let ratios = conservation_ratios(&eigen, &attribute, attribute_name);

    // Spectral gap = largest gap between consecutive eigenvalues
    let spectral_gap = if eigen.eigenvalues.len() >= 2 {
        eigen
            .eigenvalues
            .windows(2)
            .map(|w| w[1] - w[0])
            .fold(f64::NEG_INFINITY, f64::max)
    } else {
        0.0
    };

    let cheeger = cheeger_constant(&eigen);
    let anomalies = detect_anomalies(graph, &eigen, 2.0);

    let mut fingerprint = spectral_fingerprint(&eigen);
    fingerprint.conservation_profile = ratios.iter().map(|r| r.ratio).collect();

    ConservationReport {
        ratios,
        anomalies,
        spectral_gap,
        cheeger_constant: cheeger,
        fingerprint,
    }
}
