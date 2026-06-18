# Risk Parity Portfolio Computation in Rust

This repo provides efficient risk parity portfolio computation in Rust, following the main algorithm of the following paper:

Griveau-Billion, Théophile, Jean-Charles Richard, and Thierry Roncalli. 2013. “A Fast Algorithm for Computing High-Dimensional Risk Parity Portfolios.” *SSRN Electronic Journal*. https://doi.org/10.2139/ssrn.2325255.

The algorithm computes the weights for a risk parity portfolio relative to a given budget vector $b$, which must be positive and sum to one.
The only notable change we make to this algorithm is minimizing $\frac{x^T \Sigma x}{2} - \sum_{i=1}^n b_i\ln(x_i)$ over $\mathbb{R}^n_{++}$
where $\Sigma$ is the risk matrix, defined as a Gram matrix $R^TR$ with positive diagonal and each column of $R$ mean-centered. 
This change is harmless: If $x^{\ast}$ is a stationary point of this convex function, then it is a stationary point of the original Lagrangian (with $\lambda = 1$), namely $\sqrt{x^T\Sigma x} - \sum_{i=1}^n b_i\ln(x_i)$. It has the benefits of simplifying the component update equation and removing some square roots.

This library is essentially self-contained, only using external crates for csv parsing and for creating temporary folders during testing.

## Running the solver

1. Form a budget vector and a risk matrix:
   ```bash 
   touch ./input/budget.csv ./input/risk_matrix.csv
   ```
   The budget is a bare *space-separated* list of floats (so just one row).
   The risk matrix is the lower part of a symmetric matrix, with each row a bare *space-separated* list of floats.
   (One technical point not addressed in the above paper: to ensure convergence, the vector $Rx$ should be nonzero 
   for every nonzero $x \geq 0$, so that Tseng's boundedness condition on sublevel sets is satisfied.)
   See the current `./input/` folder for examples.
   
   The main function will use these two csv files as inputs to the solver.
   
2. To run the solver on the given budget and risk matrix, run
   ```bash
   cargo run
   ```
3. The output weights will be written to a new file: `./output/computed_weights.csv`.
   They will appear as space-separated values in the same order as the columns of $R$ (the asset matrix).

## License

This project is licensed under the Mozilla Public License 2.0. See [LICENSE](LICENSE.txt) for details.
