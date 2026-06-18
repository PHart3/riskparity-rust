// core linear algebra machinery

pub type Vector = Vec<f64>;
// we represent symmetric matrices as lower row-major packed vectors
pub type SymMatrix = Vec<f64>;

// quadratic form x^TSx for a symmetric matrix S
pub fn quad_form(mat : &SymMatrix, vec : &Vector) -> f64 {
    let dim : usize = vec.len();
    let mut sum = 0.0;
    for i in 0..dim {
	let mut row_vec_sum = 0.0;
	let row_idx = (i * i + i) / 2;
	for n in 0..=i {
	    row_vec_sum += vec[n] * mat[row_idx + n];
	}
	for n in i + 1..dim {
	    row_vec_sum += vec[n] * mat[(n * n + n) / 2 + i];
	}
	sum += vec[i] * row_vec_sum;
    }
    sum
}

