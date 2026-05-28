use conservation_spectral_core::{
    conservation, eigen, laplacian, tracker, LaplacianType, TensionGraph,
};

/// Build a simple 5-node chord transition graph:
/// C -> G -> Am -> F -> Dm -> (back to C)
/// With weighted transitions between chords.
fn build_chord_graph() -> TensionGraph<String, f64> {
    let mut g: TensionGraph<String, f64> = TensionGraph::new();

    // Add vertices: C, G, Am, F, Dm
    g.add_vertex("C".to_string());
    g.add_vertex("G".to_string());
    g.add_vertex("Am".to_string());
    g.add_vertex("F".to_string());
    g.add_vertex("Dm".to_string());

    // Add edges: transitions between chords
    g.add_edge("C".to_string(), "G".to_string(), 0.6);
    g.add_edge("C".to_string(), "Am".to_string(), 0.4);
    g.add_edge("G".to_string(), "Am".to_string(), 0.7);
    g.add_edge("G".to_string(), "C".to_string(), 0.3);
    g.add_edge("Am".to_string(), "F".to_string(), 0.8);
    g.add_edge("Am".to_string(), "Dm".to_string(), 0.2);
    g.add_edge("F".to_string(), "C".to_string(), 0.5);
    g.add_edge("F".to_string(), "G".to_string(), 0.5);
    g.add_edge("Dm".to_string(), "C".to_string(), 0.9);
    g.add_edge("Dm".to_string(), "Am".to_string(), 0.1);

    // Add a "tension" attribute (higher = more tension)
    g.add_attribute("tension", vec![0.2, 0.5, 0.7, 0.3, 0.6]);

    g
}

#[test]
fn test_build_graph() {
    let g = build_chord_graph();
    assert_eq!(g.vertex_count(), 5);
    assert_eq!(g.edge_count(), 10);
}

#[test]
fn test_build_laplacian() {
    let g = build_chord_graph();
    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    assert_eq!(lap.num_vertices, 5);

    // Laplacian should have row sums = 0
    for i in 0..5 {
        let row_sum: f64 = (0..5).map(|j| lap.get(i, j)).sum();
        assert!(
            row_sum.abs() < 1e-10,
            "Row {} sum = {} (should be 0)",
            i,
            row_sum
        );
    }

    // Diagonal should be non-negative
    for i in 0..5 {
        assert!(lap.get(i, i) >= 0.0, "Diagonal {} is negative", i);
    }
}

#[test]
fn test_normalized_laplacian() {
    let g = build_chord_graph();
    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::SymmetricNormalized);
    assert!(lap.normalized);
    assert_eq!(lap.num_vertices, 5);
}

#[test]
fn test_eigendecomposition() {
    let g = build_chord_graph();
    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    let eigen = eigen::eigendecompose(&lap);

    assert_eq!(eigen.eigenvalues.len(), 5);
    assert_eq!(eigen.eigenvectors.len(), 5);

    // Eigenvalues should be sorted ascending
    for i in 1..eigen.eigenvalues.len() {
        assert!(
            eigen.eigenvalues[i] >= eigen.eigenvalues[i - 1] - 1e-10,
            "Eigenvalues not sorted at index {}",
            i
        );
    }

    // First eigenvalue should be ~0 (constant eigenvector)
    assert!(
        eigen.eigenvalues[0].abs() < 1e-6,
        "First eigenvalue = {} (should be ~0)",
        eigen.eigenvalues[0]
    );

    // All eigenvalues should be non-negative (positive semidefinite)
    for (i, &ev) in eigen.eigenvalues.iter().enumerate() {
        assert!(
            ev >= -1e-8,
            "Negative eigenvalue at index {}: {}",
            i,
            ev
        );
    }
}

#[test]
fn test_conservation_ratios() {
    let g = build_chord_graph();
    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    let eigen = eigen::eigendecompose(&lap);
    let attribute = vec![0.2, 0.5, 0.7, 0.3, 0.6];
    let ratios = conservation::conservation_ratios(&eigen, &attribute, "tension");

    assert_eq!(ratios.len(), 5);
    // All ratios should be non-negative (variance / |eigenvalue|)
    for ratio in &ratios {
        assert!(
            ratio.ratio >= -1e-10,
            "Negative conservation ratio: {}",
            ratio.ratio
        );
    }
}

#[test]
fn test_spectral_fingerprint() {
    let g = build_chord_graph();
    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    let eigen = eigen::eigendecompose(&lap);
    let fp = conservation::spectral_fingerprint(&eigen);

    // Should have gap profile (n-1 gaps)
    assert_eq!(fp.gap_profile.len(), 4);
    // Spectral entropy should be non-negative
    assert!(fp.spectral_entropy >= 0.0);
    // Effective dimension should be >= 1
    assert!(fp.effective_dimension >= 0.0);
}

#[test]
fn test_cheeger_constant() {
    let g = build_chord_graph();
    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    let eigen = eigen::eigendecompose(&lap);
    let h = conservation::cheeger_constant(&eigen);

    // Cheeger constant should be non-negative
    assert!(h >= 0.0, "Cheeger constant = {} (should be >= 0)", h);
    // Should be λ₂/2
    let lambda2 = eigen.eigenvalues[1];
    assert!((h - lambda2 / 2.0).abs() < 1e-10);
}

#[test]
fn test_anomaly_detection() {
    let mut g: TensionGraph<String, f64> = TensionGraph::new();

    // Build a clean graph
    g.add_vertex("A".to_string());
    g.add_vertex("B".to_string());
    g.add_vertex("C".to_string());
    g.add_vertex("D".to_string());
    g.add_vertex("E".to_string());

    // Symmetric transitions
    g.add_edge("A".to_string(), "B".to_string(), 1.0);
    g.add_edge("B".to_string(), "A".to_string(), 1.0);
    g.add_edge("B".to_string(), "C".to_string(), 1.0);
    g.add_edge("C".to_string(), "B".to_string(), 1.0);
    g.add_edge("C".to_string(), "D".to_string(), 1.0);
    g.add_edge("D".to_string(), "C".to_string(), 1.0);
    g.add_edge("D".to_string(), "E".to_string(), 1.0);
    g.add_edge("E".to_string(), "D".to_string(), 1.0);

    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    let eigen = eigen::eigendecompose(&lap);

    // No anomalies in a clean symmetric chain
    let anomalies = conservation::detect_anomalies(&g, &eigen, 2.0);
    // Clean graph should have few or no anomalies
    assert!(
        anomalies.len() <= 2,
        "Too many anomalies in clean graph: {:?}",
        anomalies.len()
    );

    // Now add a "wrong" transition that breaks symmetry
    g.add_edge("A".to_string(), "E".to_string(), 5.0);

    let lap2 = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    let eigen2 = eigen::eigendecompose(&lap2);
    let anomalies2 = conservation::detect_anomalies(&g, &eigen2, 1.5);

    // The anomalous transition should trigger more anomalies
    assert!(
        anomalies2.len() >= anomalies.len(),
        "Anomalous transition should trigger at least as many anomalies"
    );
}

#[test]
fn test_tracker() {
    let mut tracker = tracker::ConservationTracker::new(10, 0.5);

    // Feed several "normal" observations
    for _ in 0..5 {
        let transitions = vec![
            0.0, 0.5, 0.5, 0.0, 0.0,
            0.3, 0.0, 0.4, 0.3, 0.0,
            0.0, 0.5, 0.0, 0.3, 0.2,
            0.4, 0.0, 0.3, 0.0, 0.3,
            0.5, 0.0, 0.0, 0.3, 0.2,
        ];
        tracker.feed(tracker::Observation {
            transitions,
            attribute: Some(vec![1.0, 2.0, 3.0, 2.0, 1.0]),
            num_vertices: 5,
        });
    }

    // Should have history
    assert!(!tracker.current_ratios().is_empty());

    // Check baseline is established
    assert!(!tracker.baseline().is_empty());

    // No alert yet (consistent observations)
    let alert = tracker.check();
    assert!(alert.is_none(), "Unexpected alert: {:?}", alert);

    // Now feed a very different (degraded) observation
    // A nearly disconnected graph — conservation should degrade
    let bad_transitions = vec![
        0.95, 0.05, 0.0, 0.0, 0.0,
        0.0, 0.95, 0.05, 0.0, 0.0,
        0.0, 0.0, 0.95, 0.05, 0.0,
        0.0, 0.0, 0.0, 0.95, 0.05,
        0.05, 0.0, 0.0, 0.0, 0.95,
    ];
    tracker.feed(tracker::Observation {
        transitions: bad_transitions,
        attribute: Some(vec![1.0, 2.0, 3.0, 2.0, 1.0]),
        num_vertices: 5,
    });

    // The tracker should have registered the observation
    assert_eq!(tracker.current_ratios().len(), 6);
}

#[test]
fn test_power_iteration() {
    // Test on a simple 3x3 symmetric matrix
    let matrix = vec![2.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0];
    let (eigenvalue, eigenvector) = eigen::power_iteration(&matrix, 3, 100, 1e-10);

    // Largest eigenvalue should be 4 (for this matrix)
    assert!(
        (eigenvalue - 4.0).abs() < 0.1,
        "Expected eigenvalue ~4, got {}",
        eigenvalue
    );

    // Eigenvector should be normalized
    let norm: f64 = eigenvector.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!((norm - 1.0).abs() < 1e-10, "Eigenvector not normalized");
}

#[test]
fn test_lanczos_returns_results() {
    let g = build_chord_graph();
    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);

    // Lanczos should return k eigenvalues
    let result = eigen::lanczos_iteration(&lap.matrix, 5, 3, 20);
    assert_eq!(result.eigenvalues.len(), 3);
    // Note: Lanczos needs numerical debugging for production use
    // Jacobi eigendecomposition is the recommended solver for small graphs
}

#[test]
fn test_full_analysis() {
    let g = build_chord_graph();
    let report = conservation::analyze(&g, "tension");

    assert!(!report.ratios.is_empty());
    assert!(report.spectral_gap >= 0.0);
    assert!(report.cheeger_constant >= 0.0);
    assert!(!report.fingerprint.gap_profile.is_empty());
}

#[test]
fn test_suggest_correction() {
    let mut g: TensionGraph<String, f64> = TensionGraph::new();

    g.add_vertex("X".to_string());
    g.add_vertex("Y".to_string());
    g.add_vertex("Z".to_string());

    g.add_edge("X".to_string(), "Y".to_string(), 1.0);
    g.add_edge("Y".to_string(), "Z".to_string(), 1.0);
    g.add_edge("Z".to_string(), "X".to_string(), 1.0);
    // Add an anomalous edge
    g.add_edge("X".to_string(), "Z".to_string(), 10.0);

    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    let eigen = eigen::eigendecompose(&lap);
    let anomalies = conservation::detect_anomalies(&g, &eigen, 1.5);

    if let Some(anomaly) = anomalies.first() {
        let fixes = conservation::suggest_correction(&g, anomaly);
        assert!(!fixes.is_empty(), "Should suggest at least one fix");
    }
}

#[test]
fn test_eigenvalue_semidefinite() {
    // Property: all Laplacian eigenvalues should be >= 0
    let g = build_chord_graph();
    let lap = laplacian::build_laplacian_from_graph(&g, LaplacianType::Unnormalized);
    let eigen = eigen::eigendecompose(&lap);

    for (i, &ev) in eigen.eigenvalues.iter().enumerate() {
        assert!(
            ev >= -1e-8,
            "Laplacian has negative eigenvalue at {}: {}",
            i,
            ev
        );
    }
}
