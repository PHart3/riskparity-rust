// small test suite

use super::*;
use std::fs;
use tempfile::tempdir;

fn solve_test_case(risk: &[f64], budget: &[f64]) -> Vector {
    let dim = budget.len();
    assert_eq!(risk.len(), dim * (dim + 1) / 2, "risk matrix has the wrong packed length");

    let dir = tempdir().unwrap();

    let mut risk_data = String::new();
    let mut start = 0;
    for row_len in 1..=dim {
        let row = &risk[start..start + row_len];
        risk_data.push_str(&row.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" "));
        risk_data.push('\n');
        start += row_len;
    }
    let risk_path = dir.path().join("risk_matrix.csv").to_str().unwrap().to_string();
    fs::write(&risk_path, risk_data).unwrap();
    
    let budget_data = budget.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ") + "\n";
    let budget_path = dir.path().join("budget.csv").to_str().unwrap().to_string();
    fs::write(&budget_path, budget_data).unwrap();

    solve_from_csv(&budget_path, &risk_path).unwrap()
}

fn vector_equal(actual: &Vector, expected: &Vector, tol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!((a - e).abs() < tol, "entry {i}: expected {e}, got {a}");
    }
}

#[test]
fn identity_matrix() {
    let weights = solve_test_case(
        &[
            1.0,
            0.0, 1.0,
            0.0, 0.0, 1.0,
        ],
        &[
            25.0 / 38.0,
            9.0 / 38.0,
            4.0 / 38.0,
        ],
    );
    vector_equal(&weights, &vec![0.5, 0.3, 0.2], 1e-7);
}

#[test]
fn diagonal_matrix() {
    let weights = solve_test_case(
        &[
            1.0,
            0.0, 4.0,
            0.0, 0.0, 9.0,
        ],
        &[
            25.0 / 97.0,
            36.0 / 97.0,
            36.0 / 97.0,
        ],
    );
    vector_equal(&weights, &vec![0.5, 0.3, 0.2], 1e-7);
}

#[test]
fn correlated_matrix() {
    let weights = solve_test_case(
        &[
            2.0,
            0.3, 1.5,
            0.2, 0.1, 1.0,
        ],
        &[
            122.0 / 659.0,
            420.0 / 659.0,
            117.0 / 659.0,
        ],
    );
    vector_equal(&weights, &vec![0.2, 0.5, 0.3], 1e-7);
}

#[test]
fn two_asset_correlated_matrix() {
    let budget = vec![
        3.0 / 4.0,
        1.0 / 4.0,
    ];

    let weights = solve_test_case(
        &[
            2.0,
            0.5, 1.0,
        ],
        &budget,
    );

    vector_equal(&weights, &vec![0.6, 0.4], 1e-7);
}

#[test]
fn four_asset_diagonal_matrix() {
    let budget = vec![
        2.0 / 13.0,
        9.0 / 26.0,
        9.0 / 26.0,
        2.0 / 13.0,
    ];
    let weights = solve_test_case(
        &[
            1.0,
            0.0, 4.0,
            0.0, 0.0, 9.0,
            0.0, 0.0, 0.0, 16.0,
        ],
        &budget,
    );
    
    vector_equal(&weights, &vec![0.4, 0.3, 0.2, 0.1], 1e-7);
}

#[test]
fn four_asset_correlated_matrix() {
    let budget = vec![
        31.0 / 452.0,
        43.0 / 226.0,
        111.0 / 452.0,
        56.0 / 113.0,
    ];

    let weights = solve_test_case(
        &[
            2.0,
            0.2, 1.5,
            0.1, 0.1, 1.0,
            0.1, 0.2, 0.1, 1.2,
        ],
        &budget,
    );
    
    vector_equal(&weights, &vec![0.1, 0.2, 0.3, 0.4], 1e-7);
}

#[test]
fn five_asset_identity_matrix() {
    let budget = vec![
        2.0 / 5.0,
        5.0 / 18.0,
        8.0 / 45.0,
        1.0 / 10.0,
        2.0 / 45.0,
    ];

    let weights = solve_test_case(
        &[
            1.0,
            0.0, 1.0,
            0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0, 1.0,
        ],
        &budget,
    );

    vector_equal(&weights, &vec![0.3, 0.25, 0.2, 0.15, 0.1], 1e-7);
}

#[test]
fn five_asset_equal_correlation_matrix() {
    let budget = vec![
        38.0 / 605.0,
        141.0 / 1210.0,
        112.0 / 605.0,
        65.0 / 242.0,
        222.0 / 605.0,
    ];

    let weights = solve_test_case(
        &[
            1.0,
            0.1, 1.0,
            0.1, 0.1, 1.0,
            0.1, 0.1, 0.1, 1.0,
            0.1, 0.1, 0.1, 0.1, 1.0,
        ],
        &budget,
    );

    vector_equal(&weights, &vec![0.1, 0.15, 0.2, 0.25, 0.3], 1e-7);
}

#[test]
fn six_asset_diagonal_matrix() {
    let budget = vec![
        5.0 / 37.0,
        32.0 / 185.0,
        48.0 / 185.0,
        36.0 / 185.0,
        4.0 / 37.0,
        24.0 / 185.0,
    ];

    let weights = solve_test_case(
        &[
            1.0,
            0.0, 2.0,
            0.0, 0.0, 3.0,
            0.0, 0.0, 0.0, 4.0,
            0.0, 0.0, 0.0, 0.0, 5.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 6.0,
        ],
        &budget,
    );

    vector_equal(&weights, &vec![0.25, 0.2, 0.2, 0.15, 0.1, 0.1], 1e-7);
}

#[test]
fn six_asset_tridiagonal_matrix() {
    let budget = vec![
        1.0 / 77.0,
        4.0 / 77.0,
        9.0 / 77.0,
        16.0 / 77.0,
        7.0 / 22.0,
        45.0 / 154.0,
    ];

    let weights = solve_test_case(
        &[
            2.0,
            0.25, 2.0,
            0.0, 0.25, 2.0,
            0.0, 0.0, 0.25, 2.0,
            0.0, 0.0, 0.0, 0.25, 2.0,
            0.0, 0.0, 0.0, 0.0, 0.25, 2.0,
        ],
        &budget,
    );

    vector_equal(&weights, &vec![0.05, 0.1, 0.15, 0.2, 0.25, 0.25], 1e-7);
}

#[test]
fn eight_asset_dense_mixed_covariance() {
    let budget = vec![
	0.0312762445,
	0.0566855118,
	0.1081433001,
	0.1306501834,
	0.2116144699,
	0.1884297202,
	0.1132358793,
	0.1599646908,
    ];

    let weights = solve_test_case(
        &[
            1.2625,
            0.1700, 1.0625,
           -0.1300, 0.2125, 0.9075,
            0.4150, -0.0950, 0.1025, 0.8050,
            0.1900, 0.3150, -0.0500, 0.2725, 1.0550,
            0.0925, 0.1400, 0.4375, -0.0075, 0.1675, 0.9375,
           -0.1700, 0.1375, 0.1175, 0.2325, 0.0275, 0.1375, 0.7250,
            0.3350, 0.0100, 0.1125, 0.1975, 0.2700, -0.1850, 0.0700, 0.8675,
        ],
        &budget,
    );

    vector_equal(&weights, &vec![0.04, 0.07, 0.11, 0.13, 0.16, 0.18, 0.14, 0.17], 1e-7);
}
