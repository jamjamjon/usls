/// Kalman Filter for 8D state space (x, y, a, h, vx, vy, va, vh)
///
/// Implements a Kalman filter for tracking objects with position (x, y), aspect ratio (a),
/// height (h), and their corresponding velocities. Used in multi-object tracking systems.
#[derive(Debug, Clone)]
pub struct KalmanFilterXYAH {
    std_weight_position: f32,
    std_weight_velocity: f32,
    motion_mat: [[f32; 8]; 8], // 8x8
    update_mat: [[f32; 8]; 4], // 4x8
}

impl Default for KalmanFilterXYAH {
    fn default() -> Self {
        Self::new(1.0 / 20.0, 1.0 / 160.0)
    }
}

impl KalmanFilterXYAH {
    /// Creates a new Kalman filter with custom noise parameters
    pub fn new(std_weight_position: f32, std_weight_velocity: f32) -> Self {
        let ndim = 4;
        let dt = 1.0;
        let mut motion_mat = mat8x8_identity();
        let mut update_mat = mat4x8_zeros();
        update_mat[0][0] = 1.0;
        update_mat[1][1] = 1.0;
        update_mat[2][2] = 1.0;
        update_mat[3][3] = 1.0;

        for (i, row) in motion_mat.iter_mut().enumerate().take(ndim) {
            row[i + ndim] = dt;
        }

        Self {
            std_weight_position,
            std_weight_velocity,
            motion_mat,
            update_mat,
        }
    }

    /// Initializes a new track from a detection measurement
    pub fn initiate(&self, measurement: &DetectBox) -> (StateMean, StateCov) {
        let mut mean = [0.0; 8];

        let mean_vel = [0.0; 4];
        let mean_pos = measurement;
        mean[0..4].copy_from_slice(mean_pos);
        mean[4..8].copy_from_slice(&mean_vel);

        let mut std = [0.0; 8];
        let mesure_val = measurement[3];
        std[0] = 2.0 * self.std_weight_position * mesure_val;
        std[1] = 2.0 * self.std_weight_position * mesure_val;
        std[2] = 1e-2;
        std[3] = 2.0 * self.std_weight_position * mesure_val;
        std[4] = 10.0 * self.std_weight_velocity * mesure_val;
        std[5] = 10.0 * self.std_weight_velocity * mesure_val;
        std[6] = 1e-5;
        std[7] = 10.0 * self.std_weight_velocity * mesure_val;

        let tmp = vec8_mul(&std, &std);
        let covariance = mat8x8_from_diagonal(&tmp);

        (mean, covariance)
    }

    /// Predicts the next state based on current state and motion model
    pub fn predict(&self, mean: &StateMean, covariance: &StateCov) -> (StateMean, StateCov) {
        let mut new_mean = *mean;
        let mut new_covariance = *covariance;
        let mut std = [0.0; 8];
        std[0] = self.std_weight_position * new_mean[3];
        std[1] = self.std_weight_position * new_mean[3];
        std[2] = 1e-1;
        std[3] = self.std_weight_position * new_mean[3];
        std[4] = self.std_weight_velocity * new_mean[3];
        std[5] = self.std_weight_velocity * new_mean[3];
        std[6] = 1e-5;
        std[7] = self.std_weight_velocity * new_mean[3];

        let tmp = vec8_mul(&std, &std);
        let motion_cov = mat8x8_from_diagonal(&tmp);
        new_mean = mat8x8_mul_vec8(&self.motion_mat, &new_mean);

        let tmp1 = mat8x8_mul(&self.motion_mat, &new_covariance);
        let tmp2 = mat8x8_transpose(&self.motion_mat);
        let tmp = mat8x8_mul(&tmp1, &tmp2);
        new_covariance = mat8x8_add(&tmp, &motion_cov);

        (new_mean, new_covariance)
    }

    /// Updates the state estimate using a new measurement
    pub fn update(
        &self,
        mean: &StateMean,
        covariance: &StateCov,
        measurement: &DetectBox,
    ) -> (StateMean, StateCov) {
        let mut new_mean = *mean;
        let mut new_covariance = *covariance;
        let (projected_mean, projected_covariance) = self.project(&new_mean, &new_covariance);

        let update_mat_t = mat4x8_transpose(&self.update_mat);
        let b_t = mat8x8_mul_mat8x4(&new_covariance, &update_mat_t);
        let b = mat8x4_transpose(&b_t);

        // kalman_gain: 4x8
        let kalman_gain = cholesky_solve_4x4(&projected_covariance, &b).unwrap();

        // innovation: 1x4
        let innovation = vec4_sub(measurement, &projected_mean);

        // tmp: 1x8
        let tmp = vec4_mul_mat4x8(&innovation, &kalman_gain);
        new_mean = vec8_add(&new_mean, &tmp);

        let kg_t = mat4x8_transpose(&kalman_gain);
        let tmp1 = mat8x4_mul_mat4x4(&kg_t, &projected_covariance);
        let tmp2 = mat8x4_mul_mat4x8(&tmp1, &kalman_gain);
        new_covariance = mat8x8_sub(&new_covariance, &tmp2);

        (new_mean, new_covariance)
    }

    /// Projects the state distribution to measurement space
    pub fn project(&self, mean: &StateMean, covariance: &StateCov) -> (StateHMean, StateHCov) {
        let std = [
            self.std_weight_position * mean[3],
            self.std_weight_position * mean[3],
            1e-1,
            self.std_weight_position * mean[3],
        ];

        // update_mat: 4x8, mean: 1x8
        // projected_mean: 1x4
        let update_mat_t = mat4x8_transpose(&self.update_mat);
        let projected_mean = vec8_mul_mat8x4(mean, &update_mat_t);

        // 4x4
        let diag = mat4x4_from_diagonal(&std);
        let innovation_cov = mat4x4_component_mul(&diag, &diag);
        let tmp1 = mat4x8_mul_mat8x8(&self.update_mat, covariance);
        let update_mat_t = mat4x8_transpose(&self.update_mat);
        let cov = mat4x8_mul_mat8x4(&tmp1, &update_mat_t);
        let projected_covariance = mat4x4_add(&cov, &innovation_cov);

        (projected_mean, projected_covariance)
    }
}

// 1x4
pub type DetectBox = [f32; 4];
// 1x8
pub type StateMean = [f32; 8];
// 8x8
pub type StateCov = [[f32; 8]; 8];
// 1x4
pub type StateHMean = [f32; 4];
// 4x4
pub type StateHCov = [[f32; 4]; 4];

#[inline]
fn mat8x8_identity() -> [[f32; 8]; 8] {
    let mut m = [[0.0; 8]; 8];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

#[inline]
fn mat4x8_zeros() -> [[f32; 8]; 4] {
    [[0.0; 8]; 4]
}

// 8x8 diagonal matrix from 1x8 vector
#[inline]
fn mat8x8_from_diagonal(v: &[f32; 8]) -> [[f32; 8]; 8] {
    let mut m = [[0.0; 8]; 8];
    for i in 0..8 {
        m[i][i] = v[i];
    }
    m
}

// 4x4 diagonal matrix from 1x4 vector
#[inline]
fn mat4x4_from_diagonal(v: &[f32; 4]) -> [[f32; 4]; 4] {
    let mut m = [[0.0; 4]; 4];
    for i in 0..4 {
        m[i][i] = v[i];
    }
    m
}

// Element-wise multiplication of two 1x8 vectors
#[inline]
fn vec8_mul(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = a[i] * b[i];
    }
    result
}

// Element-wise multiplication of two 4x4 matrices
#[inline]
fn mat4x4_component_mul(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = a[i][j] * b[i][j];
        }
    }
    result
}

// 8x8 matrix multiplication
#[inline]
fn mat8x8_mul(a: &[[f32; 8]; 8], b: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut result = [[0.0; 8]; 8];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            for (k, &a_val) in a[i].iter().enumerate() {
                *cell += a_val * b[k][j];
            }
        }
    }
    result
}

// 8x8 matrix + 8x8 matrix
#[inline]
fn mat8x8_add(a: &[[f32; 8]; 8], b: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut result = [[0.0; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

// 8x8 matrix - 8x8 matrix
#[inline]
fn mat8x8_sub(a: &[[f32; 8]; 8], b: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut result = [[0.0; 8]; 8];
    for i in 0..8 {
        for j in 0..8 {
            result[i][j] = a[i][j] - b[i][j];
        }
    }
    result
}

// 8x8 matrix transpose
#[inline]
fn mat8x8_transpose(m: &[[f32; 8]; 8]) -> [[f32; 8]; 8] {
    let mut result = [[0.0; 8]; 8];
    for (i, row) in m.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[j][i] = val;
        }
    }
    result
}

// 4x8 matrix transpose to 8x4
#[inline]
fn mat4x8_transpose(m: &[[f32; 8]; 4]) -> [[f32; 4]; 8] {
    let mut result = [[0.0; 4]; 8];
    for (i, row) in m.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[j][i] = val;
        }
    }
    result
}

// 8x8 matrix * 1x8 vector (returns 1x8)
#[inline]
fn mat8x8_mul_vec8(m: &[[f32; 8]; 8], v: &[f32; 8]) -> [f32; 8] {
    let mut result = [0.0; 8];
    for (i, row) in m.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[i] += val * v[j];
        }
    }
    result
}

// 1x8 vector * 8x4 matrix (returns 1x4)
#[inline]
fn vec8_mul_mat8x4(v: &[f32; 8], m: &[[f32; 4]; 8]) -> [f32; 4] {
    let mut result = [0.0; 4];
    for (j, cell) in result.iter_mut().enumerate() {
        for (i, &val) in m.iter().enumerate() {
            *cell += v[i] * val[j];
        }
    }
    result
}

// 4x8 matrix * 8x8 matrix (returns 4x8)
#[inline]
fn mat4x8_mul_mat8x8(a: &[[f32; 8]; 4], b: &[[f32; 8]; 8]) -> [[f32; 8]; 4] {
    let mut result = [[0.0; 8]; 4];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            for (k, &a_val) in a[i].iter().enumerate() {
                *cell += a_val * b[k][j];
            }
        }
    }
    result
}

// 4x8 matrix * 8x4 matrix (returns 4x4)
#[inline]
fn mat4x8_mul_mat8x4(a: &[[f32; 8]; 4], b: &[[f32; 4]; 8]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            for (k, &a_val) in a[i].iter().enumerate() {
                *cell += a_val * b[k][j];
            }
        }
    }
    result
}

// 8x8 matrix * 8x4 matrix (returns 8x4)
#[inline]
fn mat8x8_mul_mat8x4(a: &[[f32; 8]; 8], b: &[[f32; 4]; 8]) -> [[f32; 4]; 8] {
    let mut result = [[0.0; 4]; 8];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            for (k, &a_val) in a[i].iter().enumerate() {
                *cell += a_val * b[k][j];
            }
        }
    }
    result
}

// 1x4 vector - 1x4 vector
#[inline]
fn vec4_sub(a: &[f32; 4], b: &[f32; 4]) -> [f32; 4] {
    let mut result = [0.0; 4];
    for (i, &a_val) in a.iter().enumerate() {
        result[i] = a_val - b[i];
    }
    result
}

// 1x8 vector + 1x8 vector
#[inline]
fn vec8_add(a: &[f32; 8], b: &[f32; 8]) -> [f32; 8] {
    let mut result = [0.0; 8];
    for i in 0..8 {
        result[i] = a[i] + b[i];
    }
    result
}

// 1x4 vector * 4x8 matrix (returns 1x8)
#[inline]
fn vec4_mul_mat4x8(v: &[f32; 4], m: &[[f32; 8]; 4]) -> [f32; 8] {
    let mut result = [0.0; 8];
    for (j, cell) in result.iter_mut().enumerate() {
        for (i, &val) in m.iter().enumerate() {
            *cell += v[i] * val[j];
        }
    }
    result
}

// 8x4 matrix * 4x4 matrix (returns 8x4)
#[inline]
fn mat8x4_mul_mat4x4(a: &[[f32; 4]; 8], b: &[[f32; 4]; 4]) -> [[f32; 4]; 8] {
    let mut result = [[0.0; 4]; 8];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            for (k, &a_val) in a[i].iter().enumerate() {
                *cell += a_val * b[k][j];
            }
        }
    }
    result
}

// 8x4 matrix transpose to 4x8
#[inline]
fn mat8x4_transpose(m: &[[f32; 4]; 8]) -> [[f32; 8]; 4] {
    let mut result = [[0.0; 8]; 4];
    for (i, row) in m.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            result[j][i] = val;
        }
    }
    result
}

// 8x4 matrix * 4x8 matrix (returns 8x8)
#[inline]
fn mat8x4_mul_mat4x8(a: &[[f32; 4]; 8], b: &[[f32; 8]; 4]) -> [[f32; 8]; 8] {
    let mut result = [[0.0; 8]; 8];
    for (i, row) in result.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            for (k, &a_val) in a[i].iter().enumerate() {
                *cell += a_val * b[k][j];
            }
        }
    }
    result
}

// 4x4 matrix + 4x4 matrix
#[inline]
fn mat4x4_add(a: &[[f32; 4]; 4], b: &[[f32; 4]; 4]) -> [[f32; 4]; 4] {
    let mut result = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            result[i][j] = a[i][j] + b[i][j];
        }
    }
    result
}

// Cholesky decomposition for 4x4 positive definite matrix
// Returns lower triangular matrix L such that A = L * L^T
#[inline]
fn cholesky_4x4(a: &[[f32; 4]; 4]) -> Option<[[f32; 4]; 4]> {
    let mut l = [[0.0; 4]; 4];

    for i in 0..4 {
        for j in 0..=i {
            let mut sum = 0.0;
            for (k, &l_val) in l[i].iter().enumerate().take(j) {
                sum += l_val * l[j][k];
            }

            if i == j {
                let val = a[i][i] - sum;
                if val <= 0.0 {
                    return None; // Not positive definite
                }
                l[i][j] = val.sqrt();
            } else {
                if l[j][j].abs() < 1e-10 {
                    return None;
                }
                l[i][j] = (a[i][j] - sum) / l[j][j];
            }
        }
    }

    Some(l)
}

// Solve L * y = b for y (forward substitution)
#[inline]
fn forward_substitution_4x4(l: &[[f32; 4]; 4], b: &[[f32; 8]; 4]) -> [[f32; 8]; 4] {
    let mut y = [[0.0; 8]; 4];

    for i in 0..4 {
        for j in 0..8 {
            let mut sum = 0.0;
            for (k, &l_val) in l[i].iter().enumerate().take(i) {
                sum += l_val * y[k][j];
            }
            y[i][j] = (b[i][j] - sum) / l[i][i];
        }
    }

    y
}

// Solve L^T * x = y for x (backward substitution)
#[inline]
fn backward_substitution_4x4(l: &[[f32; 4]; 4], y: &[[f32; 8]; 4]) -> [[f32; 8]; 4] {
    let mut x = [[0.0; 8]; 4];

    for i in (0..4).rev() {
        for j in 0..8 {
            let mut sum = 0.0;
            for (k, &l_val) in l.iter().enumerate().skip(i + 1) {
                sum += l_val[i] * x[k][j];
            }
            x[i][j] = (y[i][j] - sum) / l[i][i];
        }
    }

    x
}

// Solve A * X = B using Cholesky decomposition, where A is 4x4 and B is 4x8
// Returns X (4x8 matrix)
#[inline]
fn cholesky_solve_4x4(a: &[[f32; 4]; 4], b: &[[f32; 8]; 4]) -> Option<[[f32; 8]; 4]> {
    let l = cholesky_4x4(a)?;
    let y = forward_substitution_4x4(&l, b);
    let x = backward_substitution_4x4(&l, &y);
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initiate() {
        let kf = KalmanFilterXYAH::default();
        let measurement = [1.0, 2.0, 3.0, 4.0];
        let (mean, covariance) = kf.initiate(&measurement);

        // Check mean initialization
        assert_eq!(mean[0..4], measurement);
        assert_eq!(mean[4..8], [0.0; 4]);

        // Check covariance is positive definite
        for (i, row) in covariance.iter().enumerate() {
            assert!(row[i] > 0.0);
        }
    }

    #[test]
    fn test_predict() {
        let kf = KalmanFilterXYAH::default();
        let mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let covariance = [
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.000001, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0],
        ];

        let (new_mean, new_covariance) = kf.predict(&mean, &covariance);

        // Check position update with velocity
        assert_eq!(new_mean[0], mean[0] + mean[4]); // x + vx
        assert_eq!(new_mean[1], mean[1] + mean[5]); // y + vy
        assert_eq!(new_mean[2], mean[2] + mean[6]); // a + va
        assert_eq!(new_mean[3], mean[3] + mean[7]); // h + vh

        // Check covariance remains positive definite
        for (i, row) in new_covariance.iter().enumerate() {
            assert!(row[i] > 0.0);
        }
    }

    #[test]
    fn test_project() {
        let kf = KalmanFilterXYAH::default();
        let mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let covariance = [
            [4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 1.01e-2, 0.0, 0.0, 0.0, 1.0e-6, 0.0],
            [0.0, 0.0, 0.0, 4.24, 0.0, 0.0, 0.0, 4.0],
            [4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0],
            [0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.000625],
        ];

        let (projected_mean, projected_covariance) = kf.project(&mean, &covariance);

        // Check projection extracts position components
        assert_eq!(projected_mean, [1.0, 2.0, 3.0, 4.0]);

        // Check projected covariance is positive definite
        for (i, row) in projected_covariance.iter().enumerate() {
            assert!(row[i] > 0.0);
        }
    }

    #[test]
    fn test_update() {
        let kf = KalmanFilterXYAH::default();
        let mean = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let covariance = [
            [4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0],
            [0.0, 4.24, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0],
            [0.0, 0.0, 1.01e-2, 0.0, 0.0, 0.0, 1.0e-6, 0.0],
            [0.0, 0.0, 0.0, 4.24, 0.0, 0.0, 0.0, 4.0],
            [4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0, 0.0],
            [0.0, 4.0, 0.0, 0.0, 0.0, 4.000625, 0.0, 0.0],
            [0.0, 0.0, 1.0e-6, 0.0, 0.0, 0.0, 1.0e-6, 0.0],
            [0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.000625],
        ];

        let measurement = [1.0, 2.0, 3.0, 4.0];
        let (new_mean, new_covariance) = kf.update(&mean, &covariance, &measurement);

        // Check covariance remains positive definite after update
        for (i, row) in new_covariance.iter().enumerate() {
            assert!(row[i] > 0.0);
        }

        // Check that update doesn't cause numerical instability
        for val in &new_mean {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_full_cycle() {
        let kf = KalmanFilterXYAH::default();
        let measurement = [100.0, 200.0, 1.5, 50.0];

        // Initialize track
        let (mut mean, mut covariance) = kf.initiate(&measurement);

        // Run multiple predict-update cycles
        for i in 0..5 {
            let (new_mean, new_covariance) = kf.predict(&mean, &covariance);
            mean = new_mean;
            covariance = new_covariance;

            let new_measurement = [
                100.0 + (i as f32) * 2.0,
                200.0 + (i as f32) * 2.0,
                1.5,
                50.0,
            ];
            let (new_mean, new_covariance) = kf.update(&mean, &covariance, &new_measurement);
            mean = new_mean;
            covariance = new_covariance;
        }

        // Check convergence properties
        assert!((mean[0] - 108.0).abs() < 10.0);
        assert!((mean[1] - 208.0).abs() < 10.0);

        // Check covariance remains positive definite
        for (i, row) in covariance.iter().enumerate() {
            assert!(row[i] > 0.0);
        }
    }

    #[test]
    fn test_constructor() {
        let kf = KalmanFilterXYAH::default();
        assert_eq!(kf.std_weight_position, 0.05);
        assert_eq!(kf.std_weight_velocity, 0.00625);
        assert_eq!(kf.motion_mat.len(), 8);
        assert_eq!(kf.update_mat.len(), 4);
    }
}
