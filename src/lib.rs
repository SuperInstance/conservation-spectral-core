use std::collections::HashMap;
use std::hash::Hash;

pub mod laplacian;
pub mod eigen;
pub mod conservation;
pub mod tracker;

// ── Core Types ──────────────────────────────────────────────────────────────

/// Generic weighted graph with vertex attributes.
pub struct TensionGraph<V, A>
where
    V: Clone + Eq + Hash,
    A: Clone,
{
    pub vertices: Vec<V>,
    vertex_index: HashMap<V, usize>,
    pub adjacency: Vec<Vec<(usize, A)>>,
    pub vertex_attributes: HashMap<String, Vec<f64>>,
    pub directed: bool,
}

impl<V, A> TensionGraph<V, A>
where
    V: Clone + Eq + Hash,
    A: Clone,
{
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            vertex_index: HashMap::new(),
            adjacency: Vec::new(),
            vertex_attributes: HashMap::new(),
            directed: true,
        }
    }

    pub fn with_capacity(n: usize) -> Self {
        Self {
            vertices: Vec::with_capacity(n),
            vertex_index: HashMap::with_capacity(n),
            adjacency: vec![Vec::new(); n],
            vertex_attributes: HashMap::new(),
            directed: true,
        }
    }

    pub fn add_vertex(&mut self, v: V) -> usize {
        if let Some(&idx) = self.vertex_index.get(&v) {
            return idx;
        }
        let idx = self.vertices.len();
        self.vertices.push(v.clone());
        self.vertex_index.insert(v, idx);
        self.adjacency.push(Vec::new());
        idx
    }

    pub fn add_edge(&mut self, source: V, target: V, weight: A) {
        let si = self.add_vertex(source);
        let ti = self.add_vertex(target);
        self.adjacency[si].push((ti, weight));
    }

    pub fn add_attribute(&mut self, name: &str, values: Vec<f64>) {
        self.vertex_attributes.insert(name.to_string(), values);
    }

    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    pub fn edge_count(&self) -> usize {
        self.adjacency.iter().map(|a| a.len()).sum()
    }

    pub fn get_vertex_index(&self, v: &V) -> Option<usize> {
        self.vertex_index.get(v).copied()
    }
}

impl<V, A> Default for TensionGraph<V, A>
where
    V: Clone + Eq + Hash,
    A: Clone,
{
    fn default() -> Self {
        Self::new()
    }
}

// ── Laplacian ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum LaplacianType {
    Unnormalized,
    SymmetricNormalized,
    RandomWalkNormalized,
}

#[derive(Debug, Clone)]
pub struct Laplacian {
    pub matrix: Vec<f64>,        // n*n row-major
    pub degree: Vec<f64>,        // diagonal of D
    pub weights: Vec<f64>,       // n*n row-major W
    pub normalized: bool,
    pub num_vertices: usize,
}

impl Laplacian {
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.matrix[i * self.num_vertices + j]
    }

    pub fn identity(n: usize) -> Self {
        let mut m = vec![0.0; n * n];
        for i in 0..n {
            m[i * n + i] = 1.0;
        }
        Laplacian {
            matrix: m,
            degree: vec![1.0; n],
            weights: vec![0.0; n * n],
            normalized: false,
            num_vertices: n,
        }
    }
}

// ── Eigen Decomposition ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EigenDecomposition {
    pub eigenvalues: Vec<f64>,
    pub eigenvectors: Vec<Vec<f64>>, // eigenvectors[i] = i-th eigenvector (length n)
    pub laplacian_type: LaplacianType,
    pub num_vertices: usize,
}

// ── Conservation Report ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ConservationRatio {
    pub eigenvector_index: usize,
    pub eigenvalue: f64,
    pub ratio: f64,
    pub attribute_name: String,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AnomalyType {
    ConservationViolation,
    StructuralBreak,
    SpectralOutlier,
    TransitionAnomaly,
}

#[derive(Debug, Clone)]
pub struct Anomaly {
    pub vertex_index: usize,
    pub score: f64,
    pub anomaly_type: AnomalyType,
    pub description: String,
}

#[derive(Debug, Clone)]
pub struct Fix {
    pub description: String,
    pub confidence: f64,
    pub suggested_weight: Option<f64>,
    pub edge: Option<(usize, usize)>,
}

#[derive(Debug, Clone)]
pub struct SpectralFingerprint {
    pub eigenvalue_histogram: Vec<f64>,
    pub spectral_entropy: f64,
    pub effective_dimension: f64,
    pub gap_profile: Vec<f64>,
    pub conservation_profile: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct ConservationReport {
    pub ratios: Vec<ConservationRatio>,
    pub anomalies: Vec<Anomaly>,
    pub spectral_gap: f64,
    pub cheeger_constant: f64,
    pub fingerprint: SpectralFingerprint,
}
