# conservation-spectral-core

Spectral analysis of tension graphs for anomaly detection, fingerprinting, and structural health — implemented in Rust.

## Features

- **Tension graphs** — generic weighted graphs with vertex attributes
- **Laplacian computation** — unnormalized, symmetric-normalized, and random-walk-normalized
- **Eigen decomposition** — eigenvalues and eigenvectors for spectral analysis
- **Conservation ratios** — measure how well attributes are conserved across spectral modes
- **Anomaly detection** — conservation violations, structural breaks, spectral outliers, transition anomalies
- **Spectral fingerprinting** — eigenvalue histograms, spectral entropy, effective dimension, gap profiles
- **Cheeger constant** — graph connectivity / expansion measure

## Usage

```rust
use conservation_spectral_core::{TensionGraph, LaplacianType};

fn main() {
    // Build a tension graph
    let mut graph: TensionGraph<&str, f64> = TensionGraph::new();
    graph.add_edge("a", "b", 1.0);
    graph.add_edge("b", "c", 2.0);
    graph.add_edge("c", "a", 1.5);

    // Add vertex attributes
    graph.add_attribute("importance", vec![1.0, 2.0, 1.5]);

    // Compute Laplacian
    let laplacian = conservation_spectral_core::laplacian::compute(&graph, LaplacianType::SymmetricNormalized);

    // Full conservation report with anomalies and fingerprint
    let report = conservation_spectral_core::conservation::full_report(&graph, &laplacian);

    println!("Spectral gap: {}", report.spectral_gap);
    println!("Cheeger constant: {}", report.cheeger_constant);
    println!("Anomalies found: {}", report.anomalies.len());
}
```

## License

MIT
