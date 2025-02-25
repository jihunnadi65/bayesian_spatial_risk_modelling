data {
  int<lower=0> N; // number of spatial units (municipalities_)
  int<lower=0> N_edges; // number of adjacency edges (neighbouring relationships)
  array[N_edges] int<lower=1, upper=N> node1; // index regions (first node in each edge pair)
  array[N_edges] int<lower=1, upper=N> node2; // corresponding neighbouring regions
  
  array[N] int<lower=0> suicide; // dependent variable: suicide count
  
  vector[N] sph_ratio; // independent variable: single-person household ratio
  vector[N] stress_rate; // independent variable: stress awareness rate
  vector[N] unemp_rate; // independent variable: unemployment rate
  vector[N] unmet_med; // independent variable: unmet medical needs rate
  vector[N] pop_dens; // independent variable: population density
  
  vector<lower=0>[N] off_set; // offset variabe (expected count of suicides)
  real overdispersion_parameter; // overdispersion parameter for negative binomial model
}

transformed data {
  vector[N] log_off_set = log(off_set); // log-transformed offset for regression model
}

parameters{
  real alpha; // intercept (baseline log-rate of suicides)
  
  vector[5] beta; // coeficients for independent variables (sph_ratio, stress_rate, unemp_rate, unmet_med, pop_dens)
  
  real<lower=0> sigma; // standard deviation for overall variability
  real<lower=0, upper=1> rho; // proportion of unstructured vs spatially structured variance
  
  vector[N] theta; // unstructured random effects (normal erros)
  vector[N] phi; // structured spatial random effects (ICAR model)
  real<lower=0> chi; // estimated overdispersion parameter
}

transformed parameters {
  vector[N] combined; // combined random effect (unstructured + spatially structured)
  combined = sqrt(1 - rho) * theta + sqrt(rho) * phi; //   formulation for the combined random effect
}

model {
  // likelihood function: multivariable negative binomial ICAR regression model
  suicide ~ neg_binomial_2_log(
    log_off_set + alpha
    + sph_ratio * beta[1]
    + stress_rate * beta[2]
    + unemp_rate * beta[3]
    + unmet_med * beta[4]
    + pop_dens * beta[5]
    + combined * sigma,
    overdispersion_parameter // over-dispersion parameter
  );
  
  // priors
  alpha ~ normal(0, 1); // prior for alpha: weakly informative
  beta ~ normal(0, 1); // prior for betas: weakly informative
  theta ~ normal(0, 1); // prior for theta: weakly informative
  sigma ~ normal(0, 1); // prior for sigma: weakly informative
  rho ~ beta(0.5, 0.5); // prior for rho

  // spatial ICAR prior for structured random effects (phi)
  target += 0.5 * dot_self(phi[node1] - phi[node2]); // calculates the spatial weights
  sum(phi) ~ normal(0, 0.001 * N); // priors for phi
  
  chi ~ normal(overdispersion_parameter, 0.2); // prior centered at 1.28 with moderate variation
}

generated quantities {
  vector[N] eta; // linear predictor
  vector[N] rr_mu; // neighborhood-specific relative risks
  vector[5] rr_beta; // risk ratios for each independent variable
  real rr_alpha; // risk ratio for intercept

  // compute the linear predictor (eta)
  eta = alpha 
      + sph_ratio * beta[1] 
      + stress_rate * beta[2] 
      + unemp_rate * beta[3] 
      + unmet_med * beta[4] 
      + pop_dens * beta[5] 
      + combined * sigma;

  // compute relative risks (exp transformation)
  rr_mu = exp(eta);    // neighbourhood-specific relative risks
  rr_beta = exp(beta); // risk ratios for each predictor
  rr_alpha = exp(alpha); // risk ratio for the intercept
}