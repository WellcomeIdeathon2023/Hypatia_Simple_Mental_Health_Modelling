// The 'data' block list all input variables that are given to Stan from R. You need to specify the size of the arrays
data {
  int ntrials;  // number of trials per participant; "int" means that the values are integers
  int nsub;     // number of subjects
  matrix [ntrials,nsub] reward;     // if rewarded when chose shape A
  int choices[ntrials,nsub];     // if chose shape A
}


// The 'parameters' block defines the parameter that we want to fit
parameters {
  // Stan syntax explanation:
  // real : parameters are real numbers
  // <lower=0,upper=1> : parameter is in the range of 0 to 1
  // alpha : name of the parameter
  // [nsub,2] : size of the parameter (number of rows, number of columns)
  // Group level parameters
  // parameter 1
  real alpha_a;
  real beta_a;

  //parameter 2
  real <lower=0> alpha_b;
  real <lower=0> beta_b;

  // Single subject parameters
  real alpha_raw[nsub];
  real beta_raw[nsub];
}

transformed parameters{
  vector<lower=0,upper=1>[nsub] alpha;
  vector<lower=0> [nsub] beta;

  for (p in 1:nsub){
    alpha[p] = Phi_approx(alpha_a + alpha_b*alpha_raw[p]);
    beta[p] = exp(beta_a + beta_b*beta_raw[p]);
  }
}

// This block runs the actual model
model {
  // Priors
  alpha_a ~ normal(0,5);
  alpha_b ~ cauchy(0,5);
  beta_a ~ normal(0,5);
  beta_b ~ cauchy(0,5);

  // Priors for the individual subjects are the group (pat or con)
  alpha_raw ~ std_normal();
  beta_raw ~ std_normal();


  // The learning model: the aim is to define how the input data (i.e. the reward outcomes, the reward magnitudes) and parameters relate to the behavior
  // The basic structure of the model is exactly as in Matlab before:
  // The first lines define the learning of reward probabilities, then these are combined with magnitudes to give utilities
  // Then the choice utilities are linked to the actual choice using a softmax function
  for (p in 1:nsub){ // run the model for each subject
    vector [2] Q;
    real lr;

    Q[1]=0; // first trial, best guess is that Qs are at 0
    Q[2]=0;

    for (t in 1:ntrials){
      // action model
      choices[t,p] ~ categorical_logit(Q*beta[p]);

      // learning model
      Q[choices[t,p]] = Q[choices[t,p]] + alpha[p] * (reward[t,p] - Q[choices[t,p]]); // Q learning
    }
  }
}

generated quantities { //does the same calculations again for the fitted Qs
  real loglik[nsub];
  matrix [ntrials,nsub] genchoices;
  {

    //this code is basically a copy of the model block
    for (p in 1:nsub){ // run the model for each subject
      vector [2] Q;

      Q[1] = 0; // first trial, best guess is that Qs are at 0
      Q[2] = 0;
      loglik[p]=0;

      for (t in 1:ntrials){
        // action model
        genchoices[t,p] = categorical_rng(softmax(Q*beta[p]));
        loglik[p]+=categorical_logit_lpmf(choices[t,p]|Q*beta[p]);

        // learning model
        Q[choices[t,p]] = Q[choices[t,p]] + alpha[p] * (reward[t,p] - Q[choices[t,p]]); // Q learning
      }
    }
  }
}
