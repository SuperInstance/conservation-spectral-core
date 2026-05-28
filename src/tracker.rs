use crate::conservation;

/// An observation to feed into the tracker.
#[derive(Debug, Clone)]
pub struct Observation {
    /// Transition matrix as flat row-major (n*n)
    pub transitions: Vec<f64>,
    /// Optional attribute vector
    pub attribute: Option<Vec<f64>>,
    /// Number of vertices
    pub num_vertices: usize,
}

/// Alert triggered when conservation drops below threshold.
#[derive(Debug, Clone)]
pub struct Alert {
    pub message: String,
    pub conservation_ratios: Vec<f64>,
    pub dropped_by: f64,
}

/// Real-time conservation tracking with sliding window.
pub struct ConservationTracker {
    pub window_size: usize,
    pub history: Vec<f64>,
    observations: Vec<Observation>,
    baseline_ratios: Vec<f64>,
    threshold: f64,
}

impl ConservationTracker {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            window_size,
            history: Vec::with_capacity(window_size),
            observations: Vec::with_capacity(window_size),
            baseline_ratios: Vec::new(),
            threshold,
        }
    }

    /// Feed a new observation. Computes conservation ratios and checks for alerts.
    pub fn feed(&mut self, obs: Observation) {
        let n = obs.num_vertices;
        if n == 0 {
            return;
        }

        // Build Laplacian and decompose
        let lap = crate::laplacian::build_laplacian(
            &obs.transitions,
            n,
            |_, _| 1.0,
            crate::LaplacianType::Unnormalized,
        );
        let eigen = crate::eigen::eigendecompose(&lap);

        // Compute aggregate conservation (mean of conservation ratios for attribute)
        let attribute = obs.attribute.clone().unwrap_or_else(|| vec![1.0; n]);
        let ratios: Vec<f64> = (0..eigen.eigenvalues.len())
            .map(|k| conservation::conservation_ratio(&eigen, &attribute, k))
            .collect();

        let mean_ratio = if ratios.is_empty() {
            0.0
        } else {
            ratios.iter().sum::<f64>() / ratios.len() as f64
        };

        // Store
        self.history.push(mean_ratio);
        self.observations.push(obs);

        // Establish baseline from first observations
        if self.baseline_ratios.is_empty() && self.history.len() >= 3 {
            self.baseline_ratios = ratios.clone();
        }

        // Trim to window
        while self.history.len() > self.window_size {
            self.history.remove(0);
            self.observations.remove(0);
        }
    }

    /// Check if conservation has dropped below threshold. Returns alert if so.
    pub fn check(&self) -> Option<Alert> {
        if self.history.len() < 3 {
            return None;
        }

        let recent = &self.history;
        let n = recent.len();

        // Compare recent mean to historical mean
        let historical_mean: f64 = recent[..n - 2].iter().sum::<f64>() / (n - 2) as f64;
        let recent_val = recent[n - 1];

        // Conservation DROPPED means the ratio INCREASED (higher variance = less conserved)
        let increase = recent_val - historical_mean;
        if increase > self.threshold * historical_mean.abs().max(1e-10) {
            return Some(Alert {
                message: format!(
                    "Conservation degraded: ratio jumped from {:.4} to {:.4} ({:.1}% increase)",
                    historical_mean,
                    recent_val,
                    increase / historical_mean.abs().max(1e-10) * 100.0
                ),
                conservation_ratios: recent.clone(),
                dropped_by: increase,
            });
        }

        None
    }

    pub fn baseline(&self) -> &[f64] {
        &self.baseline_ratios
    }

    pub fn current_ratios(&self) -> &[f64] {
        &self.history
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.observations.clear();
        self.baseline_ratios.clear();
    }
}
