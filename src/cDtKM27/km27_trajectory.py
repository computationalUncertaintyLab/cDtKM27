#mcandrew

class cDtKM27(object):

    def __init__(self,
                 r                      = None
                 ,R0                    = None
                 ,phi                   = None
                 ,sigma                 = None
                 ,serial_interval_m     = None
                 ,serial_interval_v     = None
                 ,vacc_rate             = None
                 ,init_vacc             = None
                 , N                    = None
                 , T                    = None
                 , S                    = None
                 , K                    = None
                 , lag                  = 14
                 , R                    = None):
        
        #--THE USER SETS THESE AS CONSTANTS-----------------------------
        self.T   = T
        self.N   = N

        self.R = R     
        self.K = K

        self.S   = S
        self.LAG = lag

    def extract_from_long_format_data(self,d
                                      , week_col        = "MMWRWK"
                                      , season_col      = "season"
                                      , flucount_column = "ttl_flu"
                                      , strata_column   = "strat"):
        import pandas as pd
        import numpy as np
        from patsy import dmatrix

        #--PIVOT TO ONE SEASON PER COLUMN------------------------------------------------------------------------------------
        cases_per_season = pd.pivot_table( index = [week_col], columns = [season_col], values = [flucount_column], data = d )

        #--WE ASSUME THAT THE INTEREST IS IN THE FLU SEASON FROM EPIWEEK 40 TO EPIWEEK 20------------------------------------
        order = pd.DataFrame({"MMWRWK": list(np.arange(40,52+1)) + list(np.arange(1,20+1)) })

        cases_per_season.columns = [y for (x,y) in cases_per_season.columns]
        cases_per_season = order.merge(cases_per_season, left_on = ["MMWRWK"], right_index=True)

        self.Y = cases_per_season.to_numpy()[:,1:]

        #--EXTRACT SEASON STRATIFICATION-------------------------------------------------------------------------------------
        season2_strat = d[ [season_col,strata_column] ].drop_duplicates()

        R = dmatrix("C({:s})-1".format(strata_column),data=season2_strat)
        self.R = R

        K = R.shape[-1]
        self.K = K

        return R,K
        
    #--estimate serial interval based on Cowling here = https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3057478/ 
    def serial_interval_compute(self,m,v):
        import jax
        import jax.numpy as jnp
        
        #--The below serial interval is in days
        #m,v = 3.6,2.9 ##from Cowling
        k   = (m**2)/v
        b   = v/m
        return jnp.diff(jax.scipy.stats.gamma.cdf(jnp.arange(self.LAG+1), a=k, scale=b, loc=0  ))

    def generate_trajectory(self,from_param_name_2_value):
        import jax
        import jax.numpy as jnp
        from jax.scipy.special import expit
        from jax.scipy.special import logit
        
        #--LOAD PARAMETER VALUES TO PRODUCE K KM27 TRAJECTORIES
        r           = expit( jnp.array(from_param_name_2_value["r"]) )            #--this is a tuple of length K
        R0          = jnp.array(from_param_name_2_value["R0"])          #--this is a tuple of length K

        phi         = expit( jnp.array(from_param_name_2_value["phi"]) )

        m           = jnp.array(from_param_name_2_value["serial_interval_m"] )  #--this is a tuple of length K
        v           = jnp.array(from_param_name_2_value["serial_interval_v"] )  #--this is a tuple of length K

        vacc        = expit( jnp.array(from_param_name_2_value["vacc"])      )       #--This is a tuple of length K
        init_vacc   = expit( jnp.array(from_param_name_2_value["init_vacc"]) )        #--This is a tuple of length K

        tref        = jnp.array(from_param_name_2_value["tref"])
        b           = jnp.array(from_param_name_2_value["b"])
    
        #--NUMBER OF WEEKS (T), NUMBER OF (S)EASONS AND NUMBER OF CLUSTERS (K)
        T   = self.T
        S   = self.S
        K   = self.K
        N   = self.N
        lag = self.LAG

        #--COMPUTE K SERIAL INTERVALS---------------------------------------------------------------
        serial_interval_compute = self.serial_interval_compute
        serial_interval         = jax.vmap( lambda m,v: serial_interval_compute(m,v) )(m,v) #--K X Lag #(INPUT IMPLICIT IS LAG)


        #--add in a term that allows the case count to be dormant awhile
        cos_value = jnp.cos( 2*jnp.pi*(( jnp.arange(T*7).reshape(-1,1)-tref)/(T*7)))
        betas     = R0*( 1 + (b*0.5)*cos_value ) 
        
        #--DISCRETE KM27 BY DIEKMANN HTTPS://WWW.PNAS.ORG/DOI/10.1073/PNAS.2106332118#SEC-6---------
        def one_step(past_si, array, serial_interval,N,vacc):
            past_s,past_i = past_si
            t,beta        = array

            change = jnp.exp( -jnp.sum(beta*serial_interval[::-1]*past_i/N ) )*(1-vacc)

            s_vacc = past_s[-1]
            s = past_s[-1]*change
            i = past_s[-1]*(1-change)

            states = jnp.array([s,i])
            new_si = jnp.concatenate( [jnp.delete( past_si, obj=0,axis=1 ), states.reshape(2,1)], axis=1)

            return  new_si, states

        #--SET INITIAL CONDITIONS------------------------------------------------------------
        initial_incs  = jax.vmap( lambda r: jnp.exp( r*jnp.arange(lag) ) )(r) #--K X Lag
        initial_s     = (N - initial_incs)*init_vacc
        Yhat          = jnp.dstack([initial_s, initial_incs]) #--K X lag X states 

        #--EVOLVE TRAJECTORIES OVER TIME-----------------------------------------------------
        _,states =  jax.vmap( lambda Yhat,serial_interval,beta: jax.lax.scan( lambda x,y,: one_step(x,y,serial_interval,N,vacc)
                                                                            , init = Yhat.T, xs = (jnp.arange(T*7),beta) )
                          , in_axes = (0,0,1)  ) (Yhat, serial_interval, betas )

        #--States is K X T X States
        s_days = states[...,0]
        i_days = states[...,1]

        #--AGGREGATE TO WEEKS
        i = i_days.reshape(K,-1,7).sum(2)

        #--ADD IN PHI WHICH MAY ACCOUNT FOR REPORTING BIAS AND DETECTION--------------------
        i_hat = (i.T)*phi #--T by S

        self.incident_curves = i_hat
        return i_hat

    def from_vector_to_parameter_values(self,x,sigma=False):
        import numpyro
        numpyro.enable_x64(True)
        
        K = self.K

        if self.R is None:
            
            from_vector_to_parameter_values                      = {}
            from_vector_to_parameter_values["r"]                 = x[0:K]
            from_vector_to_parameter_values["R0"]                = x[K:2*K]
            from_vector_to_parameter_values["serial_interval_m"] = x[2*K:3*K]
            from_vector_to_parameter_values["serial_interval_v"] = x[3*K:4*K]

            from_vector_to_parameter_values["theta"]             = x[4*K:5*K]           

            from_vector_to_parameter_values["tref"]              = x[5*K:6*K]
            from_vector_to_parameter_values["b"]                 = x[6*K:7*K]
 
            
            from_vector_to_parameter_values["vacc"]              = x[7*K]
            from_vector_to_parameter_values["init_vacc"]         = x[7*K+1]

            from_vector_to_parameter_values["phi"]               = x[7*K+2]

            if sigma:
                from_vector_to_parameter_values["sigma"]             = x[(7*K+2)+1:(7*K+2)+(K+1)]
           
        else:
            from_vector_to_parameter_values                      = {}
            from_vector_to_parameter_values["r"]                 = x[0:K]
            from_vector_to_parameter_values["R0"]                = x[K:2*K]
            from_vector_to_parameter_values["serial_interval_m"] = x[2*K:3*K]
            from_vector_to_parameter_values["serial_interval_v"] = x[3*K:4*K]

            from_vector_to_parameter_values["tref"]              = x[4*K:5*K]
            from_vector_to_parameter_values["b"]                 = x[5*K:6*K]
 
            
            from_vector_to_parameter_values["vacc"]              = x[6*K]
            from_vector_to_parameter_values["init_vacc"]         = x[6*K+1]

            from_vector_to_parameter_values["phi"]               = x[6*K+2]
            if sigma:
                from_vector_to_parameter_values["sigma"]         = x[(6*K+2)+1:(6*K+2) + (K+1)]

        self.from_param_name_2_value = from_vector_to_parameter_values
        return from_vector_to_parameter_values
    
    def evaluate_nloglik(self,parameter_vector=None,from_param_name_2_value=None,Y=None):
        import numpy as np
        
        import jax
        import jax.numpy as jnp
        from jax.random import PRNGKey
        from jax.scipy.special import logsumexp
        from jax.scipy.special import expit

        import numpyro
        import numpyro.distributions as dist

        S = self.S

        if from_param_name_2_value is None:
            from_param_name_2_value = self.from_vector_to_parameter_values(parameter_vector,sigma=False)
        incident_curves         = self.generate_trajectory( from_param_name_2_value  )
        
        #--Add weights Season X K
        theta   = expit(from_param_name_2_value["theta"])
        weights = dist.Dirichlet(jnp.repeat(jnp.array(theta).reshape(1,-1),S,axis=0) ).mean

        try:
            L   = jax.vmap( lambda O: dist.Poisson(incident_curves+10**-10).mask( (~jnp.isnan(O)).reshape(-1,1) ).log_prob(O.reshape(-1,1)).sum(0))(Y.T)
            W   = jnp.log(weights)
            sol = sum(logsumexp( (W+L), axis=1))
            self.loglik = sol

        except RuntimeWarning:
            self.loglik = jnp.inf
            return jnp.inf
        self.loglik = sol
        return -1*self.loglik


    def evaluate_nloglik_mcmc(self,parameter_vector=None,from_param_name_2_value=None,Y=None):
        import numpy as np
        
        import jax
        import jax.numpy as jnp
        from jax.random import PRNGKey
        from jax.scipy.special import logsumexp
        from jax.scipy.special import expit

        import numpyro
        import numpyro.distributions as dist

        S = self.S

        if from_param_name_2_value is None:
            from_param_name_2_value = self.from_vector_to_parameter_values(parameter_vector)
        incident_curves             = self.generate_trajectory( from_param_name_2_value  ) + 10**-10
        
        #--Add weights Season X K
        theta   = expit(from_param_name_2_value["theta"])
        weights =  numpyro.sample( "weights", dist.Dirichlet(jnp.repeat(jnp.array(theta).reshape(1,-1),S,axis=0) ) )

        sigma = from_param_name_2_value["sigma"]
        try:
            L   = jax.vmap( lambda O: dist.Poisson(incident_curves).mask( (~jnp.isnan(O)).reshape(-1,1) ).log_prob(O.reshape(-1,1)).sum(0))(Y.T+10**-10)
            W   = jnp.log(weights)
            
            Q = (W+L)
            sol = sum(logsumexp( Q, axis=1))

            responsabilities = jnp.exp(Q - logsumexp(Q,axis=1).reshape(-1,1))
            
            self.loglik = sol

        except:
            self.loglik = jnp.inf
            return jnp.inf
        
        self.loglik = sol
        return -1*self.loglik, responsabilities

    def evaluate_nloglik_fixed(self,parameter_vector,Y):
        import numpy as np
        
        import jax
        import jax.numpy as jnp
        from jax.random import PRNGKey
        from jax.scipy.special import logsumexp

        import numpyro
        import numpyro.distributions as dist

        S = self.S
        R = self.R
        
        from_param_name_2_value = self.from_vector_to_parameter_values(parameter_vector,sigma=False)
        incident_curves         = self.generate_trajectory( from_param_name_2_value  )

        incident_curves_across_strata = jnp.matmul(incident_curves, R.T)
        
        #--Add weights Season X K
        try:
            sol   = dist.Poisson(incident_curves_across_strata+10**-10).mask( (~jnp.isnan(Y)) ).log_prob(Y).sum()
            self.loglik = sol
        except RuntimeWarning:
            self.loglik = np.inf
            return np.inf
        if np.isnan(sol):
            
            self.loglik = np.inf
        else:
            self.loglik = sol
        return -1*self.loglik

    def fit(self, Y,  method="GA", polish=False):
        from pymoo.core.problem           import ElementwiseProblem
        from pymoo.operators.sampling.lhs import LHS
        from pymoo.optimize               import minimize
        from pymoo.termination.default    import DefaultSingleObjectiveTermination
        from pymoo.problems.functional import FunctionalProblem

        from multiprocessing.pool import ThreadPool
        from pymoo.core.problem import StarmapParallelization

        import numpy as np

        n_threads = 1
        pool      = ThreadPool(n_threads)
        runner    = StarmapParallelization(pool.starmap)

        R = self.R
        
        #--DEFINE OPTIMIZATION PROBLEM------------------------------------
        if R is None:
            class MyProblem(ElementwiseProblem):
                def __init__(self,K,T,eval_func,**kwargs):
                    import numpy as np
                    self.K         = K
                    self.evaluator = eval_func

                    lower_bound = np.array( [-2]*K + [0.5]*K  + [3.55]*K   + [2.85]*K  + [-10]*K  +   [0]*K + [0]*K  + [-10] + [-10] + [-10] )
                    upper_bound = np.array( [ 2]*K +   [3]*K  + [3.65]*K   + [2.95]*K  +  [10]*K  + [T*7]*K + [1]*K  + [10]  + [10]  + [10]  )

                    super().__init__(n_var        = 7*K+3,
                                     n_obj        = 1,
                                     n_eq_constr  = 0,
                                     n_ieq_constr = 0,
                                     xl           = lower_bound,
                                     xu           = upper_bound)

                def _evaluate(self, x, out, *args, **kwargs):
                    out["F"] = [self.evaluator(x)]

            # define the problem by passing the starmap interface of the thread pool
            problem = MyProblem(elementwise_runner=runner,K=self.K, T=self.T, eval_func = lambda x: self.evaluate_nloglik(x,Y=Y))
        else: #--NO THETA
            class MyProblem(ElementwiseProblem):
                def __init__(self,K,T,eval_func,**kwargs):
                    import numpy as np
                    self.K         = K
                    self.evaluator = eval_func

                    lower_bound = np.array( [-2]*K + [0.5]*K  + [3.55]*K   + [2.85]*K  + [0]*K        + [0]*K    + [-10] + [-10] + [-10])
                    upper_bound = np.array( [ 2]*K +   [3]*K  + [3.65]*K   + [2.95]*K  + [T*7]*K      + [1]*K    + [10]  + [10]  + [10] )

                    super().__init__(n_var        = 6*K+3,
                                     n_obj        = 1,
                                     n_eq_constr  = 0,
                                     n_ieq_constr = 0,
                                     xl           = lower_bound,
                                     xu           = upper_bound)

                def _evaluate(self, x, out, *args, **kwargs):
                    out["F"] = [self.evaluator(x)]

            # define the problem by passing the starmap interface of the thread pool
            problem = MyProblem(elementwise_runner=runner,K=self.K, T=self.T, eval_func = lambda x: self.evaluate_nloglik_fixed(x,Y=Y))
            
        #--ALGORITHMS TO SOLVE PROBLEM----------------------------------------------------
        if method=="GA":
            from pymoo.algorithms.soo.nonconvex.ga import GA

            algorithm = GA(
                pop_size             = 200,
                eliminate_duplicates = True
            )
            termination = DefaultSingleObjectiveTermination(
                xtol      = 1e-8,
                cvtol     = 1e-6,
                ftol      = 1e-6,
                period    = 20,
                n_max_gen = 125,
            )
            res = minimize(problem,
                           algorithm,
                           termination,
                           seed    = 20200320,
                           verbose = True)
            #--SOLUTION----------------------------------------------------------------
            solution, solution_fitness = res.X,res.F

            from_param_name_2_value      = self.from_vector_to_parameter_values(solution)

            self.from_param_name_2_value = from_param_name_2_value
            self.vector_solution         = solution
            self.nloglik                 = solution_fitness                

            if polish:
                from pymoo.algorithms.soo.nonconvex.nelder import NelderMead
                algorithm = NelderMead( x0 = solution)
                
                res = minimize(problem,
                               algorithm,
                               termination,
                               seed    = 20200320,
                               verbose = True)
                #--SOLUTION----------------------------------------------------------------
                solution, solution_fitness = res.X,res.F
                from_param_name_2_value      = self.from_vector_to_parameter_values(solution)
                
            self.from_param_name_2_value = from_param_name_2_value
            self.vector_solution         = solution
            self.nloglik                 = solution_fitness                
           
            return from_param_name_2_value, solution_fitness
            #--------------------------------------------------------------------------

    def best_fit_centers(self):
        return self.generate_trajectory(self.from_param_name_2_value)

    def best_fit_uncertainty(self,Y):
        import numpyro
        import numpyro.distributions as dist
        from   numpyro.infer import Predictive
        from   numpyro.infer import MCMC, NUTS, init_to_value
        
        from jax.random import PRNGKey
        import jax.numpy as jnp

        numpyro.enable_x64(True)

        def mcmcmodel(params=None,Y=None,forecast=False,fixed=False):
            if params is None:
                r                 = numpyro.sample("r"                 , dist.Normal(  0*jnp.ones(self.K), 10         ) )
                R0                = numpyro.sample("R0"                , dist.Normal(  2*jnp.ones(self.K), 5          ) )
                phi               = numpyro.sample("phi"               , dist.Normal(  0     , 10) )
                
                sigma             = numpyro.sample("sigma"             , dist.HalfCauchy(1*jnp.ones(self.K)      ) )
                
                serial_interval_m = numpyro.sample("serial_interval_m" , dist.Normal(   3.6*jnp.ones(self.K)      , 10 ) )
                serial_interval_v = numpyro.sample("serial_interval_v" , dist.Normal(   2.9*jnp.ones(self.K)      , 10 ) )

                vacc              = numpyro.sample("vacc"              , dist.Normal( 0, 10                       ) )
                init_vacc         = numpyro.sample("init_vacc"         , dist.Normal( 0, 10                       ) )

                tref              = numpyro.sample("tref"              , dist.Uniform( 0, self.T*7*jnp.ones(K)    ) )
                b                 = numpyro.sample("b"                 , dist.Uniform( 0, 2*jnp.ones(self.K)           ) )
                
                theta             = numpyro.sample("theta"             , dist.Dirichlet(jnp.ones(K))   )  #--This is a tuple of length k
            else:
                r                 = numpyro.sample("r"                 , dist.Normal(  params["r"]       , 10) )
                R0                = numpyro.sample("R0"                , dist.Normal(  params["R0"]      , 5)  )
                phi               = numpyro.sample("phi"               , dist.Normal(  params["phi"]     , 10) )
               
                sigma             = numpyro.sample("sigma"             , dist.Gamma(100*jnp.ones(self.K),1.                  ) )
                
                serial_interval_m = numpyro.sample("serial_interval_m" , dist.Normal(   params["serial_interval_m"]      , 1 ) )
                serial_interval_v = numpyro.sample("serial_interval_v" , dist.Normal(   params["serial_interval_v"]      , 1 ) )
               
                vacc              = numpyro.sample("vacc"              , dist.Normal( params["vacc"]     , 10) )
                init_vacc         = numpyro.sample("init_vacc"         , dist.Normal( params["init_vacc"], 10) )

                tref              = numpyro.sample("tref"              , dist.Uniform( 0, self.T*7*jnp.ones(self.K) ) )
                b                 = numpyro.sample("b"                 , dist.Uniform( 0, 2*jnp.ones(self.K)        ) )

                if fixed == False:
                    theta             = numpyro.sample("theta"             , dist.Gamma( params["theta"],1           ) )
                else:
                    theta             = numpyro.sample("theta"             , dist.Gamma(jnp.ones(K),1.)   )  #--This is a tuple of length k
                    
            params       = {  "r"                :r
                            , "R0"               :R0
                            , "phi"              :phi
                            , "theta"            :theta
                            , "serial_interval_m":serial_interval_m
                            , "serial_interval_v":serial_interval_v
                            , "vacc"             :vacc
                            , "init_vacc"        :init_vacc
                            , "tref"             :tref
                            , "b"                :b
                            , "sigma"            :sigma }
            trajectories = self.generate_trajectory(params)

            numpyro.deterministic("trajectories", trajectories)
            
            #--observational error and weights
            ll,responsabilities = self.evaluate_nloglik_mcmc( from_param_name_2_value = params, Y=Y)
            numpyro.factor("ll", -1*ll)

            numpyro.deterministic("responsabilities", responsabilities)
            
            if forecast:
                yhat_unrestricted = numpyro.sample("yhat_unrestricted",  dist.Laplace(trajectories,sigma) )
                numpyro.deterministic("ypred", jnp.clip(yhat_unrestricted, 0, jnp.inf) )

        self.from_param_name_2_value["sigma"] = 100*jnp.ones(self.K)
        init_values = self.from_param_name_2_value
           
        num_warmup  = 6*10**3
        num_samples = 5*10**3
        num_chains  = 1

        mcmc = MCMC(
            NUTS(mcmcmodel, dense_mass=False, max_tree_depth=5 , init_strategy = init_to_value(values = init_values))
            , num_warmup  = num_warmup
            , num_samples = num_samples
            , num_chains  = num_chains
            , thinning    = 2
        )
        #try:
        mcmc.run(PRNGKey(20200320), params = init_values,Y=Y, forecast=False)
        samples = mcmc.get_samples()
        self.posterior_samples = samples

        predictive  = Predictive(mcmcmodel, posterior_samples=samples)
        predictions = predictive(PRNGKey(20201017), Y=Y, forecast=True, params = self.from_param_name_2_value )

        self.predictions = predictions
        
        return samples, predictions

    
          
        
    #     L   = jax.vmap( lambda O: dist.NegativeBinomial2(i_hat+10**-10,sigma).mask( (~jnp.isnan(O)).reshape(-1,1) ).log_prob(O.reshape(-1,1)).sum(0))(obs.T)
    #     responsabilities = jnp.exp(L - logsumexp(L,axis=1).reshape(-1,1)) #--exp( log(p1) - log(p1+p2)  )
    #     return i_hat, responsabilities, jnp.matmul(responsabilities, i_hat.T)

