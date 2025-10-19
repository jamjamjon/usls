use anyhow::Result;

const LARGE: f64 = 1_000_000.0;

/// Linear Assignment Problem solver using Jonker-Volgenant algorithm
///
/// Solves the assignment problem: minimize sum of costs where each row is assigned to exactly one column
/// and each column is assigned to exactly one row.
///
/// # Arguments
/// * `cost` - n×n cost matrix
///
/// # Returns
/// * `Ok((Vec<u64>, Vec<u64>))` - (row_indices, col_indices) if successful
/// * `Err` if input validation fails or algorithm doesn't converge
pub fn lapjv(cost: &[Vec<f64>]) -> Result<(Vec<u64>, Vec<u64>)> {
    let n = cost.len();
    if n == 0 {
        anyhow::bail!("cost.len() must be greater than 0, got: {}", n);
    }

    // Create mutable copy of cost matrix for computation
    let cost_mut = cost.to_vec();
    let mut x = vec![-1; n];
    let mut y = vec![-1; n];
    let mut free_rows = vec![0; n];
    let mut v = vec![0.0; n];

    // Phase 1: Initial assignment
    let mut ret = ccrt_dense(n, &cost_mut, &mut free_rows, &mut x, &mut v, &mut y);

    // Phase 2: Row reduction (up to 2 iterations)
    let mut i = 0;
    while ret > 0 && i < 2 {
        ret = carr_dense(n, &cost_mut, ret, &mut free_rows, &mut x, &mut y, &mut v);
        i += 1;
    }

    // Phase 3: Column augmentation (if needed)
    if ret > 0 {
        ret = ca_dense(n, &cost_mut, ret, &mut free_rows, &mut x, &mut y, &mut v);
    }

    if ret > 0 {
        anyhow::bail!("Algorithm failed to converge, ret = {}", ret);
    }

    // Convert assignment to separate row and column vectors
    let mut row_indices = Vec::with_capacity(n);
    let mut col_indices = Vec::with_capacity(n);
    for (i, &col) in x.iter().enumerate() {
        if col >= 0 {
            row_indices.push(i as u64);
            col_indices.push(col as u64);
        }
    }

    Ok((row_indices, col_indices))
}

/// Initial assignment phase - finds initial feasible solution
fn ccrt_dense(
    n: usize,
    cost: &[Vec<f64>],
    free_rows: &mut [usize],
    x: &mut [isize],
    v: &mut [f64],
    y: &mut [isize],
) -> usize {
    debug_assert!(cost.len() == n, "cost.len() must be equal to {}", n);
    debug_assert!(x.len() == n, "x.len() must be equal to {}", n);
    debug_assert!(y.len() == n, "y.len() must be equal to {}", n);
    debug_assert!(v.len() == n, "v.len() must be equal to {}", n);

    // Initialize dual variables and assignments
    x.fill(-1);
    v.fill(LARGE);
    y.fill(0);

    // Find minimum cost for each column
    for (i, cost_row) in cost.iter().enumerate().take(n) {
        for j in 0..n {
            let c = cost_row[j];
            if c < v[j] {
                v[j] = c;
                y[j] = i as isize;
            }
        }
    }

    // Handle conflicts and find free rows
    let mut unique = vec![true; n];
    for j in (0..n).rev() {
        let i = y[j] as usize;
        if x[i] < 0 {
            x[i] = j as isize;
        } else {
            unique[i] = false;
            y[j] = -1;
        }
    }

    let mut n_free_rows = 0;
    for i in 0..n {
        if x[i] < 0 {
            free_rows[n_free_rows] = i;
            n_free_rows += 1;
        } else if unique[i] {
            let j = x[i] as usize;
            let mut min = LARGE;
            for (j2, &v_j2) in v.iter().enumerate().take(n) {
                if j2 != j {
                    let c = cost[i][j2] - v_j2;
                    if c < min {
                        min = c;
                    }
                }
            }
            v[j] -= min;
        }
    }
    n_free_rows
}

/// Augmenting row reduction phase - improves assignment for free rows
fn carr_dense(
    n: usize,
    cost: &[Vec<f64>],
    n_free_rows: usize,
    free_rows: &mut [usize],
    x: &mut [isize],
    y: &mut [isize],
    v: &mut [f64],
) -> usize {
    let mut current = 0;
    let mut new_free_rows = 0;
    let mut rr_cnt = 0;

    while current < n_free_rows {
        rr_cnt += 1;
        let free_i = free_rows[current];
        current += 1;

        let mut j1 = 0;
        let mut j2 = -1;
        let mut v1 = cost[free_i][0] - v[0];
        let mut v2 = LARGE;

        // Find two best assignments for this row
        for (j, &v_j) in v.iter().enumerate().skip(1) {
            let c = cost[free_i][j] - v_j;
            if c < v2 {
                if c >= v1 {
                    v2 = c;
                    j2 = j as isize;
                } else {
                    v2 = v1;
                    v1 = c;
                    j2 = j1;
                    j1 = j as isize;
                }
            }
        }

        let mut i0 = y[j1 as usize];
        let v1_new = v[j1 as usize] - (v2 - v1);
        let v1_lowers = v1_new < v[j1 as usize];

        if rr_cnt < current * n {
            if v1_lowers {
                v[j1 as usize] = v1_new;
            } else if i0 >= 0 && j2 >= 0 {
                j1 = j2;
                i0 = y[j2 as usize];
            }

            if i0 >= 0 {
                if v1_lowers {
                    current -= 1;
                    free_rows[current] = i0 as usize;
                } else {
                    free_rows[new_free_rows] = i0 as usize;
                    new_free_rows += 1;
                }
            }
        } else if i0 >= 0 {
            free_rows[new_free_rows] = i0 as usize;
            new_free_rows += 1;
        }
        x[free_i] = j1;
        y[j1 as usize] = free_i as isize;
    }
    new_free_rows
}

/// Find columns with minimum distance values
fn find_dense(n: usize, lo: usize, d: &[f64], cols: &mut [usize]) -> usize {
    debug_assert!(d.len() == n, "d.len() must be equal to n");
    debug_assert!(cols.len() == n, "cols.len() must be equal to n");

    let hi = lo + 1;
    let mut mind = d[cols[lo]];

    let mut current_hi = hi;
    for k in hi..n {
        let j = cols[k];
        debug_assert!(j < d.len(), "j must be less than d.len()");

        if d[j] <= mind {
            if d[j] < mind {
                current_hi = lo;
                mind = d[j];
            }
            debug_assert!(current_hi <= cols.len(), "hi must be less than cols.len()");
            debug_assert!(k <= cols.len(), "k must be less than cols.len()");
            cols[k] = cols[current_hi];
            cols[current_hi] = j;
            current_hi += 1;
        }
    }
    current_hi
}

/// Scan columns to find augmenting paths
#[allow(clippy::too_many_arguments)]
fn scan_dense(
    n: usize,
    cost: &[Vec<f64>],
    plo: &mut usize,
    phi: &mut usize,
    d: &mut [f64],
    cols: &mut [usize],
    pred: &mut [usize],
    y: &mut [isize],
    v: &mut [f64],
) -> isize {
    let mut lo = *plo;
    let mut hi = *phi;
    let mut h: f64;
    let mut cred_ij: f64;

    while lo != hi {
        debug_assert!(lo < cols.len(), "lo must be less than cols.len()");
        let mut j = cols[lo];
        lo += 1;

        debug_assert!(j < y.len(), "j must be less than y.len()");
        debug_assert!(j < d.len(), "j must be less than d.len()");
        debug_assert!(j < v.len(), "j must be less than v.len()");
        let i = y[j] as usize;
        let mind = d[j];

        debug_assert!(y[j] >= 0, "y[j] must be greater than or equal to 0");
        debug_assert!(i < cost.len(), "i must be less than cost.len()");
        h = cost[i][j] - v[j] - mind;

        let mut current_hi = hi;
        for k in hi..n {
            j = cols[k];
            cred_ij = cost[i][j] - v[j] - h;
            if cred_ij < d[j] {
                d[j] = cred_ij;
                pred[j] = i;
                if cred_ij == mind {
                    if y[j] < 0 {
                        return j as isize;
                    }
                    cols[k] = cols[current_hi];
                    cols[current_hi] = j;
                    current_hi += 1;
                }
            }
        }
        hi = current_hi;
    }
    *plo = lo;
    *phi = hi;
    -1
}

/// Find augmenting path from a free row
fn find_path_dense(
    n: usize,
    cost: &[Vec<f64>],
    start_i: usize,
    y: &mut [isize],
    v: &mut [f64],
    pred: &mut [usize],
) -> isize {
    let mut lo = 0;
    let mut hi = 0;
    let mut final_j = -1;
    let mut n_ready = 0;
    let mut cols = vec![0; n];
    let mut d = vec![0.0; n];

    // Initialize columns and distances
    for i in 0..n {
        cols[i] = i;
        pred[i] = start_i;
        d[i] = cost[start_i][i] - v[i];
    }

    while final_j == -1 {
        if lo == hi {
            n_ready = lo;
            hi = find_dense(n, lo, &d, &mut cols);
            for &j in cols.iter().take(hi).skip(lo) {
                if y[j] < 0 {
                    final_j = j as isize;
                }
            }
        }
        if final_j == -1 {
            final_j = scan_dense(n, cost, &mut lo, &mut hi, &mut d, &mut cols, pred, y, v);
        }
    }

    // Update dual variables
    let mind = d[cols[lo]];
    for &j in cols.iter().take(n_ready) {
        v[j] += d[j] - mind;
    }
    final_j
}

/// Column augmentation phase - final assignment optimization
fn ca_dense(
    n: usize,
    cost: &[Vec<f64>],
    n_free_rows: usize,
    free_rows: &mut [usize],
    x: &mut [isize],
    y: &mut [isize],
    v: &mut [f64],
) -> usize {
    let mut pred = vec![0; n];

    for &free_row in free_rows.iter().take(n_free_rows) {
        let mut i = -1isize;
        let mut k = 0;

        let mut j = find_path_dense(n, cost, free_row, y, v, &mut pred);
        debug_assert!(j >= 0, "j must be greater than or equal to 0");
        debug_assert!(j < n as isize, "j must be less than n as isize");

        while i != free_row as isize {
            i = pred[j as usize] as isize;
            y[j as usize] = i;

            // Swap x[i] and j
            std::mem::swap(&mut j, &mut x[i as usize]);

            k += 1;
            debug_assert!(k <= n, "k must be less than or equal to n");
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lapjv_3x3() {
        let cost = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let res = lapjv(&cost);
        assert!(res.is_ok(), "expected Ok, got {:?}", res);
        let (row_indices, col_indices) = res.unwrap();
        assert_eq!(row_indices, vec![0, 1, 2]);
        assert_eq!(col_indices, vec![2, 0, 1]);
    }

    #[test]
    fn test_lapjv_4x4() {
        let cost = vec![
            vec![1., 2., 3., 4.],
            vec![5., 6., 7., 8.],
            vec![9., 10., 11., 12.],
            vec![13., 14., 15., 16.],
        ];
        let res = lapjv(&cost);
        assert!(res.is_ok(), "expected Ok, got {:?}", res);
        let (row_indices, col_indices) = res.unwrap();
        assert_eq!(row_indices, vec![0, 1, 2, 3]);
        assert_eq!(col_indices, vec![3, 0, 1, 2]);
    }

    #[test]
    fn test_lapjv_5x5() {
        let cost = vec![
            vec![1., 2., 3., 4., 1.],
            vec![5., 6., 7., 8., 2.],
            vec![9., 10., 11., 12., 3.],
            vec![13., 14., 15., 16., 4.],
            vec![17., 18., 19., 20., 5.],
        ];
        let res = lapjv(&cost);
        assert!(res.is_ok(), "expected Ok, got {:?}", res);
        let (row_indices, col_indices) = res.unwrap();
        assert_eq!(row_indices, vec![0, 1, 2, 3, 4]);
        assert_eq!(col_indices, vec![0, 2, 1, 3, 4]);
    }

    #[test]
    fn test_lapjv_10x10_1() {
        let cost = vec![
            vec![
                0.84612522, 0.38549337, 0.27955776, 0.76146103, 0.85084611, 0.02021263, 0.05006527,
                0.40961263, 0.19081828, 0.26665063,
            ],
            vec![
                0.09142041, 0.39511703, 0.5287497, 0.43743945, 0.30997663, 0.76304532, 0.37178294,
                0.73159998, 0.59313334, 0.86550584,
            ],
            vec![
                0.03684529, 0.27024986, 0.1672481, 0.14402541, 0.76511803, 0.94059419, 0.22349045,
                0.51600746, 0.61480263, 0.6346781,
            ],
            vec![
                0.68874528, 0.98444085, 0.33925711, 0.83052401, 0.71814185, 0.62298001, 0.76450538,
                0.03825611, 0.50084776, 0.46314705,
            ],
            vec![
                0.05084691, 0.89486244, 0.87147786, 0.64935965, 0.72806465, 0.05434427, 0.03566491,
                0.73072368, 0.94922003, 0.01400043,
            ],
            vec![
                0.20976728, 0.50350434, 0.83373798, 0.15936914, 0.97320944, 0.00213279, 0.72815469,
                0.17278318, 0.87271939, 0.19039888,
            ],
            vec![
                0.24818255, 0.52879636, 0.22082257, 0.69962464, 0.85367808, 0.0130662, 0.12319754,
                0.01034406, 0.44409775, 0.31241999,
            ],
            vec![
                0.2413349, 0.91244109, 0.67805999, 0.84944587, 0.02873252, 0.60521235, 0.06197102,
                0.33353023, 0.01528123, 0.17659061,
            ],
            vec![
                0.84612522, 0.38549337, 0.27955776, 0.76146103, 0.85084611, 0.02021263, 0.05006527,
                0.40961263, 0.19081828, 0.26665063,
            ],
            vec![
                0.09142041, 0.39511703, 0.5287497, 0.43743945, 0.30997663, 0.76304532, 0.37178294,
                0.73159998, 0.59313334, 0.86550584,
            ],
        ];
        let res = lapjv(&cost);
        assert!(res.is_ok(), "expected Ok, got {:?}", res);
        let (row_indices, col_indices) = res.unwrap();
        assert_eq!(row_indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(col_indices, vec![8, 0, 2, 7, 9, 3, 5, 4, 6, 1]);
    }

    #[test]
    fn test_lapjv_10x10_2() {
        let cost = vec![
            vec![
                0.84612522, 0.38549337, 0.27955776, 0.76146103, 0.85084611, 0.02021263, 0.05006527,
                0.40961263, 0.19081828, 0.26665063,
            ],
            vec![
                0.09142041, 0.39511703, 0.5287497, 0.43743945, 0.30997663, 0.76304532, 0.37178294,
                0.73159998, 0.59313334, 0.86550584,
            ],
            vec![
                0.03684529, 0.27024986, 0.1672481, 0.14402541, 0.76511803, 0.94059419, 0.22349045,
                0.51600746, 0.61480263, 0.6346781,
            ],
            vec![
                0.68874528, 0.98444085, 0.33925711, 0.83052401, 0.71814185, 0.62298001, 0.76450538,
                0.03825611, 0.50084776, 0.46314705,
            ],
            vec![
                0.05084691, 0.89486244, 0.87147786, 0.64935965, 0.72806465, 0.05434427, 0.03566491,
                0.73072368, 0.94922003, 0.01400043,
            ],
            vec![
                0.20976728, 0.50350434, 0.83373798, 0.15936914, 0.97320944, 0.00213279, 0.72815469,
                0.17278318, 0.87271939, 0.19039888,
            ],
            vec![
                0.24818255, 0.52879636, 0.22082257, 0.69962464, 0.85367808, 0.0130662, 0.12319754,
                0.01034406, 0.44409775, 0.31241999,
            ],
            vec![
                0.2413349, 0.91244109, 0.67805999, 0.84944587, 0.02873252, 0.60521235, 0.06197102,
                0.33353023, 0.01528123, 0.17659061,
            ],
            vec![
                0.2413349, 0.91244109, 0.67805999, 0.84944587, 0.02873252, 0.60521235, 0.06197102,
                0.33353023, 0.01528123, 0.17659061,
            ],
            vec![
                0.2413349, 0.91244109, 0.67805999, 0.84944587, 0.02873252, 0.60521235, 0.06197102,
                0.33353023, 0.01528123, 0.17659061,
            ],
        ];
        let res = lapjv(&cost);
        assert!(res.is_ok(), "expected Ok, got {:?}", res);
        let (row_indices, col_indices) = res.unwrap();
        assert_eq!(row_indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(col_indices, vec![5, 0, 1, 7, 9, 3, 2, 8, 4, 6]);
    }

    #[test]
    fn test_lapjv_2x2() {
        let cost = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let res = lapjv(&cost);
        assert!(res.is_ok());
        let (row_indices, col_indices) = res.unwrap();
        assert_eq!(row_indices, vec![0, 1]);
        assert_eq!(col_indices, vec![1, 0]);
    }

    #[test]
    fn test_lapjv_empty() {
        let cost: Vec<Vec<f64>> = vec![];
        let res = lapjv(&cost);
        assert!(res.is_err());
    }

    // Validation tests using reference data
    #[cfg(test)]
    mod validation_tests {
        use super::*;

        const EPSILON: f64 = 1e-10; // Tolerance for floating point comparison

        // Embedded test data constants
        const SIMPLE_2X2_COST: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
        const SIMPLE_2X2_TOTAL_COST: f64 = 5.0;

        const ZEROS_3X3_COST: [[f64; 3]; 3] = [[0.0, 1.0, 2.0], [1.0, 0.0, 3.0], [2.0, 3.0, 0.0]];
        const ZEROS_3X3_TOTAL_COST: f64 = 0.0;

        const NEGATIVE_4X4_COST: [[f64; 4]; 4] = [
            [-1.0, 2.0, 3.0, 4.0],
            [1.0, -2.0, 3.0, 4.0],
            [1.0, 2.0, -3.0, 4.0],
            [1.0, 2.0, 3.0, -4.0],
        ];
        const NEGATIVE_4X4_TOTAL_COST: f64 = -10.0;

        const LARGE_5X5_COST: [[f64; 5]; 5] = [
            [100.0, 200.0, 300.0, 400.0, 500.0],
            [150.0, 250.0, 350.0, 450.0, 550.0],
            [200.0, 300.0, 400.0, 500.0, 600.0],
            [250.0, 350.0, 450.0, 550.0, 650.0],
            [300.0, 400.0, 500.0, 600.0, 700.0],
        ];
        const LARGE_5X5_TOTAL_COST: f64 = 2000.0;

        const DECIMAL_6X6_COST: [[f64; 6]; 6] = [
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            [0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            [0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
        ];
        const DECIMAL_6X6_TOTAL_COST: f64 = 3.6;

        const ALL_ZEROS_3X3_COST: [[f64; 3]; 3] =
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
        const ALL_ZEROS_3X3_TOTAL_COST: f64 = 0.0;

        const ALL_SAME_4X4_COST: [[f64; 4]; 4] = [
            [5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0],
            [5.0, 5.0, 5.0, 5.0],
        ];
        const ALL_SAME_4X4_TOTAL_COST: f64 = 20.0;

        const IDENTITY_5X5_COST: [[f64; 5]; 5] = [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ];
        const IDENTITY_5X5_TOTAL_COST: f64 = 0.0;

        const LARGE_VALUES_3X3_COST: [[f64; 3]; 3] = [
            [1000000.0, 2000000.0, 3000000.0],
            [2000000.0, 3000000.0, 4000000.0],
            [3000000.0, 4000000.0, 5000000.0],
        ];
        const LARGE_VALUES_3X3_TOTAL_COST: f64 = 9000000.0;

        fn assert_cost_nearly_eq(actual: f64, expected: f64, name: &str) {
            assert!(
                (actual - expected).abs() < EPSILON,
                "{}: {} vs {} (diff: {})",
                name,
                actual,
                expected,
                (actual - expected).abs()
            );
        }

        fn calculate_total_cost_from_assignments(
            cost_matrix: &[[f64; 2]],
            row_indices: &[u64],
            col_indices: &[u64],
        ) -> f64 {
            let mut total_cost = 0.0;
            for (&row, &col) in row_indices.iter().zip(col_indices.iter()) {
                total_cost += cost_matrix[row as usize][col as usize];
            }
            total_cost
        }

        fn calculate_total_cost_from_assignments_3x3(
            cost_matrix: &[[f64; 3]],
            row_indices: &[u64],
            col_indices: &[u64],
        ) -> f64 {
            let mut total_cost = 0.0;
            for (&row, &col) in row_indices.iter().zip(col_indices.iter()) {
                total_cost += cost_matrix[row as usize][col as usize];
            }
            total_cost
        }

        fn calculate_total_cost_from_assignments_4x4(
            cost_matrix: &[[f64; 4]],
            row_indices: &[u64],
            col_indices: &[u64],
        ) -> f64 {
            let mut total_cost = 0.0;
            for (&row, &col) in row_indices.iter().zip(col_indices.iter()) {
                total_cost += cost_matrix[row as usize][col as usize];
            }
            total_cost
        }

        fn calculate_total_cost_from_assignments_5x5(
            cost_matrix: &[[f64; 5]],
            row_indices: &[u64],
            col_indices: &[u64],
        ) -> f64 {
            let mut total_cost = 0.0;
            for (&row, &col) in row_indices.iter().zip(col_indices.iter()) {
                total_cost += cost_matrix[row as usize][col as usize];
            }
            total_cost
        }

        fn calculate_total_cost_from_assignments_6x6(
            cost_matrix: &[[f64; 6]],
            row_indices: &[u64],
            col_indices: &[u64],
        ) -> f64 {
            let mut total_cost = 0.0;
            for (&row, &col) in row_indices.iter().zip(col_indices.iter()) {
                total_cost += cost_matrix[row as usize][col as usize];
            }
            total_cost
        }

        #[test]
        fn test_simple_2x2_matches_reference() {
            let cost = SIMPLE_2X2_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();

            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for simple_2x2");
            let (row_indices, col_indices) = res.unwrap();

            // Verify total cost
            let actual_total_cost =
                calculate_total_cost_from_assignments(&SIMPLE_2X2_COST, &row_indices, &col_indices);
            assert_cost_nearly_eq(
                actual_total_cost,
                SIMPLE_2X2_TOTAL_COST,
                "simple_2x2 total_cost",
            );
        }

        #[test]
        fn test_zeros_3x3_matches_reference() {
            let cost = ZEROS_3X3_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();

            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for zeros_3x3");
            let (row_indices, col_indices) = res.unwrap();

            // Verify total cost
            let actual_total_cost = calculate_total_cost_from_assignments_3x3(
                &ZEROS_3X3_COST,
                &row_indices,
                &col_indices,
            );
            assert_cost_nearly_eq(
                actual_total_cost,
                ZEROS_3X3_TOTAL_COST,
                "zeros_3x3 total_cost",
            );
        }

        #[test]
        fn test_negative_4x4_matches_reference() {
            let cost = NEGATIVE_4X4_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();

            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for negative_4x4");
            let (row_indices, col_indices) = res.unwrap();

            // Verify total cost
            let actual_total_cost = calculate_total_cost_from_assignments_4x4(
                &NEGATIVE_4X4_COST,
                &row_indices,
                &col_indices,
            );
            assert_cost_nearly_eq(
                actual_total_cost,
                NEGATIVE_4X4_TOTAL_COST,
                "negative_4x4 total_cost",
            );
        }

        #[test]
        fn test_large_5x5_matches_reference() {
            let cost = LARGE_5X5_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for large_5x5");
            let (row_indices, col_indices) = res.unwrap();

            // Verify total cost
            let actual_total_cost = calculate_total_cost_from_assignments_5x5(
                &LARGE_5X5_COST,
                &row_indices,
                &col_indices,
            );
            assert_cost_nearly_eq(
                actual_total_cost,
                LARGE_5X5_TOTAL_COST,
                "large_5x5 total_cost",
            );
        }

        #[test]
        fn test_decimal_6x6_matches_reference() {
            let cost = DECIMAL_6X6_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for decimal_6x6");
            let (row_indices, col_indices) = res.unwrap();

            // Verify total cost
            let actual_total_cost = calculate_total_cost_from_assignments_6x6(
                &DECIMAL_6X6_COST,
                &row_indices,
                &col_indices,
            );
            assert_cost_nearly_eq(
                actual_total_cost,
                DECIMAL_6X6_TOTAL_COST,
                "decimal_6x6 total_cost",
            );
        }

        #[test]
        fn test_edge_cases_match_reference() {
            // Test all_zeros_3x3
            let cost = ALL_ZEROS_3X3_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for all_zeros_3x3");
            let (row_indices, col_indices) = res.unwrap();
            let actual_total_cost = calculate_total_cost_from_assignments_3x3(
                &ALL_ZEROS_3X3_COST,
                &row_indices,
                &col_indices,
            );
            assert_cost_nearly_eq(
                actual_total_cost,
                ALL_ZEROS_3X3_TOTAL_COST,
                "all_zeros_3x3 total_cost",
            );

            // Test all_same_4x4
            let cost = ALL_SAME_4X4_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for all_same_4x4");
            let (row_indices, col_indices) = res.unwrap();
            let actual_total_cost = calculate_total_cost_from_assignments_4x4(
                &ALL_SAME_4X4_COST,
                &row_indices,
                &col_indices,
            );
            assert_cost_nearly_eq(
                actual_total_cost,
                ALL_SAME_4X4_TOTAL_COST,
                "all_same_4x4 total_cost",
            );

            // Test identity_5x5
            let cost = IDENTITY_5X5_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for identity_5x5");
            let (row_indices, col_indices) = res.unwrap();
            let actual_total_cost = calculate_total_cost_from_assignments_5x5(
                &IDENTITY_5X5_COST,
                &row_indices,
                &col_indices,
            );
            assert_cost_nearly_eq(
                actual_total_cost,
                IDENTITY_5X5_TOTAL_COST,
                "identity_5x5 total_cost",
            );

            // Test large_values_3x3
            let cost = LARGE_VALUES_3X3_COST
                .iter()
                .map(|row| row.to_vec())
                .collect::<Vec<_>>();
            let res = lapjv(&cost);
            assert!(res.is_ok(), "LAPJV failed for large_values_3x3");
            let (row_indices, col_indices) = res.unwrap();
            let actual_total_cost = calculate_total_cost_from_assignments_3x3(
                &LARGE_VALUES_3X3_COST,
                &row_indices,
                &col_indices,
            );
            assert_cost_nearly_eq(
                actual_total_cost,
                LARGE_VALUES_3X3_TOTAL_COST,
                "large_values_3x3 total_cost",
            );
        }
    }
}
