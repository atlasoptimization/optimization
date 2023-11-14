def Mixed_pixel_resolution(observations, wavelengths, phase_variances,n_mix, optim_opts):

    """ 
    The goal of this function is to solve the mixed pixel unmixing problem 
    in which a sequence of (potentially noisy) measurements with different wavelengths
    has been acquired. It is assumed that all these measurements have been made to
    several surfaces S_1, ..., S_n_mix synchronously resulting in the received signal
    being a mixture of signals scattered back from distances d_1, ... d_n_mix to the 
    measuring instrument.
    The sequence d_1, ... ,d_n_mix is to be extracted from the observations. 
    This is done via formulating the optimal estimation of d_1 and associated 
    numbers N_1 of full wavecycles as a mixed integer linear program emulating 
    an l1 norm minimization on the phase residuals. Due to robustness, the extracted
    distance corresponds to the one to the most dominant scatterer - iterative
    estimation and elimination from the data yields the desired sequence with
    high probability. However, subsequent optimizations contain nonlinear terms
    due to the involved weights being undetermined and subject to estimation as 
    well.
    
    For this, do the following:
        1. Definitions and imports
        2. Assemble required matrices
        3. Perform initial optimization
        4. Iterate: Optimization of weights and distances
        5. Assemble results
        
    INPUTS
    The inputs consist in three vectors, all of which have length equal to the number
    n_obs of observations. The vector "observations" contains the observed complex
    values whereas the sequence of n_obs wavelengths used to perform the 
    observations is stored in the m-dim vector "wavelengths". A vector "phase_
    variances" documents the assumed variances of the phase measurements; in the
    setting of Multiwavelength-EDM they are typically all equal. 
    A dictionary "optim_opts" collects further information pertaining to the 
    optimization - like bounds and convergence criteria.
    
    Name                 Interpretation                             Type
    observations        Observations in the form of complex         c-vector [1,n_obs]
                        numbers representing phases and amplitudes
                        observed during the measurements
    wavelengths         Wavelengths of the waves used to perform    vector [1,n_obs]
                        the measurements.
    phase_variances     Phase variances of the noise added onto     vector [1,n_obs]
                        the superposition of backscatter
    n_mix               The number of suspected backscattering      integer > 0
                        surfaces contributing to the mixing
    optim_options       The options for optimization                dictionary
                        
                        
    OUTPUTS
    The outputs consist in the distance minimizing the l1 norm of residuals as well 
    as the vector N of full wavecycles and the vector of residuals
    
    Name                 Interpretation                             Type
    d                  The optimally estimated distances            vector [n_mix]
    N                  A matrix containing estimated full           integer matrix [n_obs,n_mix]
                       wavecycles
    r                  A matrix containing weighted residuals       matrix [n_obs,n_mix]
    z_sol              A matrix containt the estimated indi         matrix [n_obs,n_mix]
                       vidual backscatters from the surfaces
                          
    """
    
    
    
    """
        1. Definitions and imports -------------------------------------------
    """
    
    
    # i) Import numerical and optimization libraries
    
    import numpy as np
    import cvxpy as cp
    
    
    # ii) Define other quantities
    
    n_obs=len(observations)
    
    
    # iii) Extract qunatities
    
    phi_obs=np.angle(observations)
    max_iter=optim_opts['max_iter']
    constraints=optim_opts['constraints']
    z_obs=observations
    
    
    """
        2. Assemble required matrices ----------------------------------------
    """
    
    
    # i) Coefficient matrices
    
    lambda_mat=np.diag(wavelengths)
    lambda_mat_pinv=np.linalg.pinv(lambda_mat)
    lambda_vec_pinv=np.diag(lambda_mat_pinv)
    
    phase_std=np.sqrt(phase_variances)
    phase_std_pinv=np.linalg.pinv(np.diag(phase_std))
    
    
    
    
    """
        3. Perform initial optimization --------------------------------------
    """
    
    
    # i) Objective function and constraints
    
    d_opt=cp.Variable(nonneg=True)
    N_opt=cp.Variable(n_obs,integer=True)
    
    objective=cp.Minimize(cp.norm(phase_std_pinv@(2*np.pi*(2*d_opt*lambda_vec_pinv-N_opt)-phi_obs),p=1))
    
    cons=[]
    for cstr in constraints:
        cons=cons+[eval(cstr)]
    
    
    # ii) Solve optimization
    
    Optim_problem=cp.Problem(objective,constraints=cons)
    Optim_solution=Optim_problem.solve(solver='GLPK_MI',max_iters=max_iter, verbose=True)
    
    
    # iii) Assemble solution
    
    d=np.zeros(n_mix)
    N=np.zeros([n_obs,n_mix])
    r=np.zeros([n_obs,n_mix])
    z_sol=np.zeros([n_obs,n_mix])
    
    d[0]=d_opt.value
    N[:,0]=N_opt.value
    r[:,0]=phase_std_pinv@(2*np.pi*(2*d[0]*lambda_vec_pinv-N[:,0])-phi_obs)
    
    
    
    """
        4. Iterate: Optimization of weights and distances ---------------------
    """
    
    
    
    # i) General setup
    
    n_disc_base= 21
    n_disc=n_disc_base+2*n_obs
    
    z_sol_temp=np.exp(4*np.pi*1j*d[0]*lambda_vec_pinv)
    
    
    # ii) Set up linearization
        
    w_cut=np.logspace(-1,1,n_disc_base)
    intersections=np.zeros(n_obs)
    
    for k in range(n_obs):
        w_intersect_b= np.abs(np.imag(z_obs[k])/np.imag(z_sol_temp))-0.01
        w_intersect_u= np.abs(np.imag(z_obs[k])/np.imag(z_sol_temp))+0.01
        w_cut=np.append(w_cut,[w_intersect_b,w_intersect_u])
        
    w_cut=np.sort(w_cut)
    
    
    # iii) Create optimization variables
    
    
    
    d_opt=cp.Variable(1,pos=True)
    N_opt=cp.Variable(n_obs,integer=True)
    w_opt=cp.Variable(n_disc,pos=True)
    
    
    # for m in range(1,n_mix):
    
    #     # ii) Set up linear inequalities
        
    #     n_disc= 20
    #     z_sol_temp=np.exp(4*np.pi*1j*d[m]*lambda_vec_pinv)
        
        
        
        
        
    #     # iii) Formulate optimization problem
        
        
    #     # iv) Solve and extract values
        
    #     w_est[m]=np.sum(w_opt.value)
    #     d[m]=d_opt.value
    #     N[:,m]=N_opt.value
    #     r[:,m]=phase_std_pinv@(2*np.pi*(2*d[m]*lambda_vec_pinv-N[:,m])-phi_obs)
        
    #     # v) Prepare next iteration
        
    #     z_sol[:,m-1]=w_est[m-1]*z_sol_temp
    #     z_resid_temp=z_resid-z_sol[:,m]
    
    
    """
        5. Assemble results --------------------------------------------------
    """
    
    
    # i) Values of optimization variables
    
    d=d_opt.value
    N=N_opt.value
    
    
    # ii) Residuals
    
    r=phase_std_pinv@(2*np.pi*(2*d*lambda_vec_pinv-N)-phi_obs)
    
    return d, N, r
    
    
    
    
    
    
    
    
    
    
    