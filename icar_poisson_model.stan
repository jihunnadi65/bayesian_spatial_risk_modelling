data {
  int<lower=0> N; // number of spatial units or neighbourhoods
  int<lower=0> N_edges; // number of edges connecting adjacent areas using Queen's contiguity
  array[N_edges] int<lower=1, upper=N> node1; // list of index areas showing which spatial units are neighbours
  array[N_edges] int<lower=1, upper=N> node2; // list of neighbouring areas showing the connection to index spatial unit
  array[N] int<lower=0> Y; // dependent variable (suicide count)
  
  int<lower=1> K; // number of independent variables (4)
  matrix[N, K] X; // matrix of independent variables (N * K
  
  vector<lower=0>[N] Off_set; // offset variable (expected number of cases)
}

transformed data {
  vector[N] log_Offset = log(Off_set); // log-transform the expected cases for offset
}

parameters {
  real alpha; // intercept
  vector[K] beta; // coefficients for independent variables
  real<lower=0> sigma; // overall standard deviation
  real<lower=0, upper=1> rho; // proportion of unstructured vs spatially structured variance
  vector[N] theta; // unstructured random effects
  vector[N] phi; // structured spatial random effects
}

transformed parameters {
  vector[N] combined; // combined spatial random effect
  combined = sqrt(1 - rho) * theta + sqrt(rho) * phi; // weighted sum of structured and unstructured effects
}

model {
  // Likelihood function: Spatial Poisson Reression with Multiple Predictors
  Y ~ poisson_log(log_Offset + alpha + X * beta + combined * sigma);
  
  // priors
  alpha ~ normal(0.0, 1.0); // prior for alpha (intercept)
  beta ~ normal(0.0, 1.0); // prior for each beta coefficient (weakly informative)
  theta ~ normal(0.0, 1.0); // prior for unstructured random effects
  sigma ~ normal(0.0, 1.0); // prior for standard deviation
  rho ~ beta(0.5, 0.5); // prior for proportion of structured spatial variation
  
  // spatial dependency priors for 'phi' (ICAR model)
  target += -0.5 * dot_self(phi[node1] - phi[node2]); // calculates the spatial weights
  sum(phi) ~ normal(0, 0.001 * N); // priors for phi
}

generated quantities {
  vector[N] eta = alpha + X * beta + combined * sigma; // compute linear predictor (eta)
  vector[N] rr_mu = exp(eta); // compute relative risks for each area
  vector[K] rr_beta = exp(beta); // compute relative riks for each independent variable
  real rr_alpha = exp(alpha); // compute the relative risk for the intercept
}
