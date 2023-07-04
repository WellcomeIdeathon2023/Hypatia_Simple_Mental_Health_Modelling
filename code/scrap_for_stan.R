#test out rstan

library('rstan')

model<-stan_model(file = 'fit_1a1b.stan')

saveRDS(model,'model_1a1b.rds')

data<-read.csv('../data/data.csv')

reward<-data%>%
  select(id,trial,reward)%>%
  pivot_wider(id_cols = trial,names_from = id,values_from = reward)%>%
  select(-c(trial))

choice<-data%>%
  select(id,trial,choice)%>%
  pivot_wider(id_cols = trial,names_from = id,values_from = choice)%>%
  select(-c(trial))

data<-list(
  nsub=length(unique(data$id)),
  ntrials=max(data$trial),
  reward=reward,
  choices=choice)

fit<-sampling(model,data)

parameters<-summary(fit, pars=c('alpha','beta'))$summary[,1]
loglik<-summary(fit, pars=c('loglik'))$summary[,1]
fit_summary<-list(pars=parameters,value=loglik)
