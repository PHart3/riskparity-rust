// portfolio computation via cyclic coordinate descent (ccd)

use riskparity_rust::lin_alg::core::*;

/*
We compute the weights of a risk-parity portfolio from a given risk matrix and positive budget vector.
The risk matrix is a covariance matrix, i.e., a Gram matrix R^TR with positive diagonal entries and R centered.
Here, the columns of R represent the assets of the portfolio.
To ensure theoretical convergence, we assume that Rx is nonzero for all x > 0.
*/

// maximum number of iterations to run
const TOTAL_STEPS: usize = 5000;
const STAG: usize = 20;
const TOL_W: f64 = 1e-8;
const TOL_O: f64 = 1e-10;
const TOL_G: f64 = 1e-8;
const TOL_G_STAG: f64 = 1e-6;

fn ccd_riskparity(risk: &Vector, budget: &Vector) -> Vector {
    let dim = budget.len();
    // extract variances from risk matrix
    let var : Vector = (0..dim).map(|n| { let r = risk[(n * n + 3 * n) / 2]; assert!(r > 0.0, "variance of asset {} is zero", n); r }).collect();
    // initialize weights (dim will be positive here)
    let mut weights : Vector = vec![1.0 / (dim as f64); dim];
    let mut portfolio_var : f64 = quad_form(risk, &weights);
    let mut risk_weights_rows = vec![0.0; dim];
    for n in 0..dim {
	let row_idx = (n * n + n) / 2;
	for m in 0..=n {
	    risk_weights_rows[n] += weights[m] * risk[row_idx + m];
	}
	for m in n + 1..dim {
	    risk_weights_rows[n] += weights[m] * risk[(m * m + m) / 2 + n];
	}
    }
    let mut obj_new : f64 = 0.5 * portfolio_var - (0..dim).map(|n| budget[n] * weights[n].ln()).sum::<f64>();
    
    // ccd
    let (mut count, mut stag) = (0, 0);
    let mut weights_best_found = Vec::with_capacity(dim);
    let mut grad_accuracy_best_found = f64::INFINITY;
    loop {
	count += 1;

	// perform one sweep
	let weights_old : Vector = weights.clone();
	let obj_old : f64 = obj_new; 
	for n in 0..dim {
	    weights[n] =
		(-risk_weights_rows[n] +
		 weights[n] * var[n] +
		 ((risk_weights_rows[n] - weights[n] * var[n]).powi(2) + 4.0 * var[n] * budget[n]).sqrt()) /
		(2.0 * var[n]);
	    let risk_weight_old = risk_weights_rows[n];
	    for m in 0..dim {
		let idx : usize = if m >= n { m * (m + 1) / 2 + n } else { n * (n + 1) / 2 + m };
		risk_weights_rows[m] += (weights[n] - weights_old[n]) * risk[idx];
	    }
	    portfolio_var += var[n] * (weights_old[n].powi(2) - weights[n].powi(2)) + 2.0 * (weights[n] * risk_weights_rows[n] - weights_old[n] * risk_weight_old);
	}
	obj_new = 0.5 * portfolio_var - (0..dim).map(|n| budget[n] * weights[n].ln()).sum::<f64>();

	// check for convergence
	let grad_accuracy : f64 = (0..dim).map(|n| (weights[n] * risk_weights_rows[n] - budget[n]).abs()).max_by(|a, b| a.total_cmp(b)).
	    expect("dim is zero");
	if grad_accuracy < TOL_G {
	    let scale = weights.iter().sum::<f64>();
	    for w in weights.iter_mut() {
		*w /= scale; 
	    }
	    println!("ccd successfully converged");
	    println!("returned weights: {:#?}", weights);
	    return weights;
	}
	
	if grad_accuracy < grad_accuracy_best_found {
	    grad_accuracy_best_found = grad_accuracy;
	    weights_best_found.clone_from(&weights);
	}
	
	// detect stagnation
	let weights_max_change = (0..dim).map(|n| (weights[n] - weights_old[n]).abs()).max_by(|a, b| a.total_cmp(b)).expect("dim is zero");
	let weights_old_max = (weights_old.iter().map(|w| w.abs())).max_by(|a, b| a.total_cmp(b)).expect("dim is zero").max(1.0);
	let obj_max_change = (obj_new - obj_old).abs();
	let obj_old_max = obj_old.abs().max(1.0);
	if weights_max_change / weights_old_max < TOL_W && obj_max_change / obj_old_max < TOL_O {
	    stag += 1 ;
	    if stag >= STAG && grad_accuracy < TOL_G_STAG {
		let scale = weights_best_found.iter().sum::<f64>();
		for w in weights_best_found.iter_mut() {
		    *w /= scale;
		}
		println!("ccd stagnated near the optimal point, with peak accuracy {:.3e}", grad_accuracy_best_found);
		println!("most accurate weights found: {:#?}", weights_best_found);
		return weights_best_found
	    }
	} else {
	    stag = 0;
	}

	// return if iteration limit is reached
	if count == TOTAL_STEPS {
	    let scale = weights_best_found.iter().sum::<f64>();
	    for w in weights_best_found.iter_mut() {
		*w /= scale;
	    }
	    println!("ccd failed to converge with peak accuracy {:.3e}", grad_accuracy_best_found);
	    println!("most accurate weights found: {:#?}", weights_best_found);
	    return weights_best_found;
	}
    }
}

use std::fs;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    // the user supplies a positive risk budget for each asset
    let budget_path = format!("{}/input/budget.csv", manifest_dir);
    // the user supplies the lower part of a symmetric risk matrix as a list of its rows
    let risk_path = format!("{}/input/risk_matrix.csv", manifest_dir);
    // compute portfolio weights
    let weights : Vector = solve_from_csv(&budget_path, &risk_path)?;
    let weights_data = weights.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(" ") + "\n";
    let output_dir = format!("{}/output", manifest_dir);
    fs::create_dir_all(&output_dir)?;
    let output_path = format!("{}/computed_weights.csv", output_dir);
    fs::write(&output_path, weights_data)?;
    Ok(())
}

use csv::ReaderBuilder;
use std::num::ParseFloatError;

pub fn solve_from_csv(budget_path: &String, risk_path: &String) -> Result<Vector, Box<dyn std::error::Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(false).delimiter(b' ').from_path(budget_path);
    let budget : Vector = rdr?.records().last().ok_or("you have not provided a budget vector")??.into_iter().
	map(|s| s.trim().parse::<f64>().map_err(<ParseFloatError as Into<Box<dyn std::error::Error>>>::into)).collect::<Result<Vector, Box<dyn std::error::Error>>>()?;
    if budget.is_empty() {
	panic!("you have not provided any budget data");
    }
    let mut sum : f64 = 0.0;
    for b in &budget {
	assert!(*b > 0.0, "every budget entry must be positive");
	sum += *b;
	if sum > 1.0 {
	    panic!("budget sums to greater than 1");
	}
    }
    let diff : f64 = (1.0 - sum).abs();
    assert!(diff <= TOL_O || diff <= TOL_W * sum.abs().max(1.0), "budget does not sum to 1");
    
    rdr = ReaderBuilder::new().has_headers(false).delimiter(b' ').trim(csv::Trim::All).flexible(true).from_path(risk_path);
    let mut risk_len = 0;
    let mut risk_matrix : SymMatrix = Vec::with_capacity(budget.len() * (budget.len() + 1) / 2);
    for (n, row_result) in rdr?.records().enumerate() {
	let row : Vector = row_result?.into_iter().
	    map(|s| s.trim().parse::<f64>().map_err(<ParseFloatError as Into<Box<dyn std::error::Error>>>::into)).collect::<Result<Vector, Box<dyn std::error::Error>>>()?;
	assert_eq!(row.len(), n + 1, "row {} has {} elements but should have {}", n, row.len(), n + 1);
	risk_matrix.extend(row);
	risk_len += 1;
    }
    assert_eq!(budget.len(), risk_len, "budget vector has dim {}, while risk matrix has dim {}", budget.len(), risk_len);
    
    println!("running ccd_riskparity...");
    Ok(ccd_riskparity(&risk_matrix, &budget))
}

// testing
#[cfg(test)]
#[path = "tests.rs"]
mod tests;
