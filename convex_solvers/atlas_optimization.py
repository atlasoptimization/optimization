"""
------------------------------------------------------------------------------
LP solver 
------------------------------------------------------------------------------
"""


def lp_solve(c,A,b,G,h,max_iter=40, tol=10**(-6)):
    
    """
    The goal of this function is to provide an algorithm that solves the linear
    programming problem
        min_x c^T x
        s.t.  Ax=b
              Gx>=h
    We will use a primal dual path following scheme as laid out in p. 612 
    of Convex optimization, Boyd 2009. We adapt it for
    linear programming to solve the standard-form LP
        min_(x_s) c_s^T x_s
        s.t.  A_sx_s=b_s
              x_s>=0
    Every LP can be written in this way. Additional steps include backtracking 
    and centering steps as described in Mehrotras 1992 paper "On the implementation
    of a primal–dual interior point method" in the SIAM Journal of optimization.
    
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Solve standard-form LP
        4. Assemble solutions
        
    INPUTS
    The inputs consist (among others) in a cost vector c, a constraint matrix A 
    with associated constraint vector b representing the m_1 equality constraints
    Ax=b. The matrix G and the vector h represent the m_2 inequality constraints 
    of the form G x >=h. All matrices and vectors need to be numpy nd arrays.
    
    Name                 Interpretation                             Type
    c                   Cost vector such that c^T x is to be       Vector [n,1]
                        minimized. Has same dimension as x
    A                   Eq. constraint matrix s.t Ax=b             Matrix [m_1,n] 
    b                   Eq. constraint vector s.t. Ax=b            Vector [m_1]
    G                   Ineq. constraint matrix s.t Gx>=h.         Matrix [m_2,n]
    h                   Ineq. constraint vector s.t. Gx>=h         Vector [m_2,1]
    max_iter            Maximum number of Newton steps before      Positive integer
                        iteration is stopped
    tol                 Maximally allowed deviation of duality     Small positive number
                        gap from zero (which is the optimum)
                        
    OUTPUTS
    The outputs consist in the optimal values for the optimization variable x 
    and some diagnostics certificates.
    
    Name                 Interpretation                             Type
    x_opt               The optimal values of the optimi-          Vector [n,1]
                        zation variable        
    opt_logfile         Dictionary containing details of the       Dictionary
                        optimization
    
    """
    
    
        
        
    """
        1. Definitions and imports -----------------------------------------------
    """
    
    
    # i) Imports
    
    import numpy as np
    
    
    # ii) Definitions
    
    n=np.shape(c)[0]
    m_1=np.shape(b)[0]
    m_2=np.shape(h)[0]
    
    
    
    """
        2. Set up problem matrices -----------------------------------------------
    """
    
    
    # i) Transform problem to standard form (s)
    
    c_s=np.vstack((c,-c,np.zeros([m_2,1])))
    b_s=np.vstack((b,h))
    A_s=np.block([[A,-A,np.zeros([m_1,m_2])],[G,-G,-np.eye(m_2)]])
    
    dim_x_s=2*n+m_2         # x_s=[x_plus, x_minus, x_slack]
    
    
    
    """
        3.Solve standard-form LP -----------------------------------------------
    """
    
    
    x_s, opt_logfile=lp_solve_standard(c_s,A_s,b_s,max_iter=max_iter,tol=tol)
    
    
    
    """
        4. Assemble solutions ----------------------------------------------------
    """
    
    
    # i) x_plus- x_minus= x_opt
    
    x_p=x_s[0:n]
    x_m=x_s[n:2*n]
    
    x_opt=x_p-x_m
    
    return x_opt, opt_logfile

   

    
"""
------------------------------------------------------------------------------
Standard-form LP solver 
------------------------------------------------------------------------------
"""



def lp_solve_standard(c,A,b,max_iter=40, tol=10**(-6)):
    
    """
    The goal of this function is to provide an algorithm that solves the linear
    standard-form LP
        min_x c^T x
        s.t.  Ax=b
              x>=0
    Every LP can be written in this way. Additional steps include backtracking 
    and centering steps as described in Mehrotras 1992 paper "On the implementation
    of a primal–dual interior point method" in the SIAM Journal of optimization. 
    A further source is Convex optimization, Boyd 2009 p. 612.
    
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Iteration
        4. Assemble solutions
        
    INPUTS
    The inputs consist in a cost vector c, a constraint matrix A with associated 
    constraint vector b representing the m_1 equality constraints Ax=b. All 
    matrices and vectors need to be numpy nd arrays.
    
    Name                 Interpretation                             Type
    c                   Cost vector such that c^T x is to be       Vector [n,1]
                        minimized. Has same dimension as x
    A                   Eq. constraint matrix s.t Ax=b             Matrix [m_1,n] 
    b                   Eq. constraint vector s.t. Ax=b            Vector [m_1]
    max_iter            Maximum number of Newton steps before      Positive integer
                        iteration is stopped
    tol                 Maximally allowed deviation of duality     Small positive number
                        gap from zero (which is the optimum)
                        
    OUTPUTS
    The outputs consist in the optimal values for the optimization variable x 
    and some diagnostics certificates.
    
    Name                 Interpretation                             Type
    x_opt               The optimal values of the optimi-          Vector [n,1]
                        zation variable        
    opt_logfile         Dictionary containing details of the       Dictionary
                        optimization
    
    """


    """
        1. Definitions and imports --------------------------------------------
    """


    # i) Imports
    
    import numpy as np
    import copy
    import scipy.sparse.linalg as scsl
    import scipy.sparse as scs
    import timeit

 
    # ii) Definitions
    
    n=np.shape(c)[0]        # dim optimization variable
    m_1=np.shape(b)[0]      # dim equality constraints
    m_2=n                   # dim inequality constraints
    
    
    # iii) Optimization metadata
    
    mu=10
    eps_feas=10**(-8)           # norm of optimization residuals
    eps=10**(-8)                # termination criteria on duality gap
    r_pri=1                     # initial residuals are above stopping threshold
    r_dual=1
    
    
        
    """
        2. Set up problem matrices --------------------------------------------
    """
    
    
    # i) Initialize optimization variables
    
    x=10**(-0)*np.ones([n,1])
    l=10**(-0)*np.ones([n,1])
    nu=np.zeros([m_1,1])
    
    
    # ii) Auxiliary functions
    
    def build_residuals(x, l, nu, c, A, b, t):
        residuals=-np.block([[c-l+A.T@nu],[np.diag(l.squeeze())@x -(1/t)*np.ones([n,1])],[A@x-b]])
        return residuals
        
    def build_newton_matrix(x, l, A):
        form='csr'
        newton_matrix=scs.bmat([[ np.zeros([n,n]), -np.eye(n), A.T],[np.diag(l.squeeze()), np.diag(x.squeeze()), 
                                                                     np.zeros([n, m_1])],[A, np.zeros([m_1,n]),np.zeros([m_1,m_1])]],format=form)
        return newton_matrix
    
    def backtracking_linesearch(x, l, nu, c, A, b, t, d_x, d_l, d_nu):
        
        # Define heuristic quantities
        alpha=0.05
        beta=0.8
        
        # Calculate s_max
        d_l_neg=d_l[d_l<0]
        l_neg=l[d_l<0]
        l_dl=-np.diag(np.diag(l_neg)@np.linalg.pinv(np.diag(d_l_neg)))
        s_max=np.min([1, np.min(np.hstack((l_dl,np.array([1]))))])
        
        # Update steplengths to ensure feasibility
        s=0.99*s_max
        while any(x+s*d_x<=0):
            s=s*beta
        while np.linalg.norm(build_residuals(x+s*d_x,l+s*d_l, nu+s*d_nu,c,A,b,t)) > (1-alpha*s)*np.linalg.norm(build_residuals(x, l, nu, c, A, b, t)):
            s=s*beta
        
        return s
        
    
    
    """
        3. Perform iterations ----------------------------------------------------
    """
    
    
    # i) Set up iteration
    
    r_pri_vec=[]; r_dual_vec=[]; s_vec=[]; dgap_vec=[]; 
    k_vec=[]; dx_vec=[]; dl_vec=[]; dnu_vec=[]
    
    k=0
    while np.max(np.array([r_pri, r_dual])) >=eps_feas or x.T@l>=eps:
        k=k+1
        if k>= max_iter:
            print('Algorithm did not converge during permissible iteration count. Handle results with care')
            break
        t=(mu*m_2)/(x.T@l)      

        
        # ii) Matrices and vectors
        
        H=build_newton_matrix(x, l, A)
        r=build_residuals(x, l, nu, c, A, b, t)
        Delta=np.expand_dims(scsl.spsolve(H,r),1)
        
        d_x=Delta[0:n]
        d_l=Delta[n:2*n]
        d_nu=Delta[2*n:]
        
        
        # iii) Backtracking
        
        s=backtracking_linesearch(x, l, nu, c, A, b, t, d_x, d_l, d_nu)
        
        x=x+s*d_x
        l=l+s*d_l
        nu=nu+s*d_nu
        
        r_pri=np.linalg.norm(r[2*n:])
        r_dual=np.linalg.norm(r[0:n])
        
        
        # iv) Update logfile
        
        k_vec=np.hstack((k_vec,k))                          # Step count
        s_vec=np.hstack((s_vec,s))                          # Step lengths
        dgap_vec=np.hstack((dgap_vec,(x.T@l).squeeze()))    # Surrogate duality gaps
        r_pri_vec=np.hstack((r_pri_vec,r_pri))              # Norm of primal residuals
        r_dual_vec=np.hstack((r_dual_vec,r_dual))           # Norm of dual residuals
        dx_vec=np.hstack((dx_vec,np.linalg.norm(s*d_x)))    # Norm of d_x
        dl_vec=np.hstack((dl_vec,np.linalg.norm(s*d_l)))    # Norm of d_l
        dnu_vec=np.hstack((dnu_vec,np.linalg.norm(s*d_nu))) # Norm of d_nu
        
        
        
    """
        4. Assemble solutions ----------------------------------------------------
    """
    
    
    # i) Function output
    
    opt_logfile=dict(x_opt=x, l_opt=l, nu_opt=nu, step=k_vec, steplength=s_vec, 
                     dgap= dgap_vec, r_pri=r_pri_vec, r_dual=r_dual_vec, dx_norm=dx_vec, dl_norm=dl_vec, dnu_norm=dnu_vec)
    return x, opt_logfile







"""
------------------------------------------------------------------------------
QP solver 
------------------------------------------------------------------------------
"""


def qp_solve(P,c,A,b,G,h,max_iter=40, tol=10**(-6)):
    
    """
    The goal of this function is to provide an algorithm that solves the quadratic
    programming problem
        min_x x^TPx+c^T x
        s.t.  Ax=b
              Gx>=h
    We will use a primal dual path following scheme as laid out in p. 612 
    of Convex optimization, Boyd 2009. We adapt it for
    quadratic programming to solve the standard-form QP
        min_(x_s) x_s^TPx_s+c_s^T x_s
        s.t.  A_sx_s=b_s
              x_s>=0
    Every QP can be written in this way. Additional steps include backtracking 
    and centering steps.
    
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Solve standard-form QP
        4. Assemble solutions
        
    INPUTS
    The inputs consist (among others) in a cost vector c, a cost Matrix P, a 
    constraint matrix A with associated constraint vector b representing the 
    m_1 equality constraints Ax=b. The matrix G and the vector h represent the
    m_2 inequality constraints of the form G x >=h. All matrices and vectors 
    need to be numpy nd arrays.
    
    Name                 Interpretation                             Type
    P                   Cost matrix such that x^TPx is a           Matrix [n,n]
                        component of total cost x^TPx+c^Tx
    c                   Cost vector such that c^Tx is a            Vector [n,1]
                        component of total cost x^TPx+c^Tx
    A                   Eq. constraint matrix s.t Ax=b             Matrix [m_1,n] 
    b                   Eq. constraint vector s.t. Ax=b            Vector [m_1]
    G                   Ineq. constraint matrix s.t Gx>=h.         Matrix [m_2,n]
    h                   Ineq. constraint vector s.t. Gx>=h         Vector [m_2,1]
    max_iter            Maximum number of Newton steps before      Positive integer
                        iteration is stopped
    tol                 Maximally allowed deviation of duality     Small positive number
                        gap from zero (which is the optimum)
                        
    OUTPUTS
    The outputs consist in the optimal values for the optimization variable x 
    and some diagnostics certificates.
    
    Name                 Interpretation                             Type
    x_opt               The optimal values of the optimi-          Vector [n,1]
                        zation variable        
    opt_logfile         Dictionary containing details of the       Dictionary
                        optimization
    
    """
    
    
        
        
    """
        1. Definitions and imports -----------------------------------------------
    """
    
    
    # i) Imports
    
    import numpy as np
    
    
    # ii) Definitions
    
    n=np.shape(c)[0]
    m_1=np.shape(b)[0]
    m_2=np.shape(h)[0]
    
    
    
    """
        2. Set up problem matrices -----------------------------------------------
    """
    
    
    # i) Transform problem to standard form (s)
    
    c_s=np.vstack((c,-c,np.zeros([m_2,1])))
    P_s=np.block([[P,-P, np.zeros([n,m_2])],[-P,P, np.zeros([n,m_2])],
                  [np.zeros([m_2,n]), np.zeros([m_2,n]), np.zeros([m_2,m_2])]])
    b_s=np.vstack((b,h))
    A_s=np.block([[A,-A,np.zeros([m_1,m_2])],[G,-G,-np.eye(m_2)]])
    
    dim_x_s=2*n+m_2         # x_s=[x_plus, x_minus, x_slack]
    
    
    
    """
        3.Solve standard-form LP -----------------------------------------------
    """
    
    
    x_s, opt_logfile=qp_solve_standard(P_s,c_s,A_s,b_s,max_iter=max_iter,tol=tol)
    
    
    
    """
        4. Assemble solutions ----------------------------------------------------
    """
    
    
    # i) x_plus- x_minus= x_opt
    
    x_p=x_s[0:n]
    x_m=x_s[n:2*n]
    
    x_opt=x_p-x_m
    
    return x_opt, opt_logfile

  

"""
------------------------------------------------------------------------------
Standard-form QP solver 
------------------------------------------------------------------------------
"""  


def qp_solve_standard(P,c,A,b,max_iter=40, tol=10**(-6)):
    
    """
    The goal of this function is to provide an algorithm that solves the quadratic
    standard-form QP
        min_x x^TPx+c^T x
        s.t.  Ax=b
              x>=0
    Every QP can be written in this way. Additional steps include backtracking 
    and centering steps as described in Convex optimization, Boyd 2009 p. 612.
    
    For this, do the following:
        1. Definitions and imports
        2. Set up problem matrices
        3. Iteration
        4. Assemble solutions
        
    INPUTS
    The inputs consist in a positive semidefinite cost matrix P and a cost 
    vector c, a constraint matrix A with associated constraint vector b 
    representing the m_1 equality constraints Ax=b. All matrices and vectors 
    need to be numpy nd arrays.
    
    Name                 Interpretation                             Type
    P                   Cost matrix such that x^TPx is a           Matrix [n,n]
                        component of total cost x^TPx+c^Tx
    c                   Cost vector such that c^Tx is a            Vector [n,1]
                        component of total cost x^TPx+c^Tx
    A                   Eq. constraint matrix s.t Ax=b             Matrix [m_1,n] 
    b                   Eq. constraint vector s.t. Ax=b            Vector [m_1]
    max_iter            Maximum number of Newton steps before      Positive integer
                        iteration is stopped
    tol                 Maximally allowed deviation of duality     Small positive number
                        gap from zero (which is the optimum)
                        
    OUTPUTS
    The outputs consist in the optimal values for the optimization variable x 
    and some diagnostics certificates.
    
    Name                 Interpretation                             Type
    x_opt               The optimal values of the optimi-          Vector [n,1]
                        zation variable        
    opt_logfile         Dictionary containing details of the       Dictionary
                        optimization
    
    """


    """
        1. Definitions and imports --------------------------------------------
    """


    # i) Imports
    
    import numpy as np
    import copy
    import scipy.sparse.linalg as scsl
    import scipy.sparse as scs
    import timeit

 
    # ii) Definitions
    
    n=np.shape(c)[0]        # dim optimization variable
    m_1=np.shape(b)[0]      # dim equality constraints
    m_2=n                   # dim inequality constraints
    
    
    # iii) Optimization metadata
    
    mu=10
    eps_feas=10**(-8)           # norm of optimization residuals
    eps=10**(-8)                # termination criteria on duality gap
    r_pri=1                     # initial residuals are above stopping threshold
    r_dual=1
    
    
        
    """
        2. Set up problem matrices --------------------------------------------
    """
    
    
    # i) Initialize optimization variables
    
    x=10**(-0)*np.ones([n,1])
    l=10**(-0)*np.ones([n,1])
    nu=np.zeros([m_1,1])
    
    
    # ii) Auxiliary functions
    
    def build_residuals(x, l, nu, P, c, A, b, t):
        residuals=-np.block([[2*P@x+c-l+A.T@nu],[np.diag(l.squeeze())@x -(1/t)*np.ones([n,1])],[A@x-b]])
        return residuals
        
    def build_newton_matrix(x, l, P, A):
        form='csr'
        newton_matrix=scs.bmat([[ 2*P, -np.eye(n), A.T],[np.diag(l.squeeze()), np.diag(x.squeeze()), 
                                                                     np.zeros([n, m_1])],[A, np.zeros([m_1,n]),np.zeros([m_1,m_1])]],format=form)
        return newton_matrix
    
    def backtracking_linesearch(x, l, nu, P, c, A, b, t, d_x, d_l, d_nu):
        
        # Define heuristic quantities
        alpha=0.05
        beta=0.8
        
        # Calculate s_max
        d_l_neg=d_l[d_l<0]
        l_neg=l[d_l<0]
        l_dl=-np.diag(np.diag(l_neg)@np.linalg.pinv(np.diag(d_l_neg)))
        s_max=np.min([1, np.min(np.hstack((l_dl,np.array([1]))))])
        
        # Update steplengths to ensure feasibility
        s=0.99*s_max
        while any(x+s*d_x<=0):
            s=s*beta
        while np.linalg.norm(build_residuals(x+s*d_x,l+s*d_l, nu+s*d_nu,P,c,A,b,t)) > (1-alpha*s)*np.linalg.norm(build_residuals(x,l,nu,P,c,A,b,t)):
            s=s*beta
        
        return s
        
    
    
    """
        3. Perform iterations ----------------------------------------------------
    """
    
    
    # i) Set up iteration
    
    r_pri_vec=[]; r_dual_vec=[]; s_vec=[]; dgap_vec=[]; 
    k_vec=[]; dx_vec=[]; dl_vec=[]; dnu_vec=[]
    
    k=0
    while np.max(np.array([r_pri, r_dual])) >=eps_feas or x.T@l>=eps:
        k=k+1
        if k>= max_iter:
            print('Algorithm did not converge during permissible iteration count. Handle results with care')
            break
        t=(mu*m_2)/(x.T@l)      

        
        # ii) Matrices and vectors
        
        H=build_newton_matrix(x, l, P, A)
        r=build_residuals(x, l, nu, P, c, A, b, t)
        Delta=np.expand_dims(scsl.spsolve(H,r),1)
        # Delta=np.expand_dims(scsl.lsqr(H,r)[0],1)
        
        d_x=Delta[0:n]
        d_l=Delta[n:2*n]
        d_nu=Delta[2*n:]
        
        
        # iii) Backtracking
        
        s=backtracking_linesearch(x, l, nu, P, c, A, b, t, d_x, d_l, d_nu)
        
        x=x+s*d_x
        l=l+s*d_l
        nu=nu+s*d_nu
        
        r_pri=np.linalg.norm(r[2*n:])
        r_dual=np.linalg.norm(r[0:n])
        
        
        # iv) Update logfile
        
        k_vec=np.hstack((k_vec,k))                          # Step count
        s_vec=np.hstack((s_vec,s))                          # Step lengths
        dgap_vec=np.hstack((dgap_vec,(x.T@l).squeeze()))    # Surrogate duality gaps
        r_pri_vec=np.hstack((r_pri_vec,r_pri))              # Norm of primal residuals
        r_dual_vec=np.hstack((r_dual_vec,r_dual))           # Norm of dual residuals
        dx_vec=np.hstack((dx_vec,np.linalg.norm(s*d_x)))    # Norm of d_x
        dl_vec=np.hstack((dl_vec,np.linalg.norm(s*d_l)))    # Norm of d_l
        dnu_vec=np.hstack((dnu_vec,np.linalg.norm(s*d_nu))) # Norm of d_nu
        
        
        
    """
        4. Assemble solutions ----------------------------------------------------
    """
    
    
    # i) Function output
    
    opt_logfile=dict(x_opt=x, l_opt=l, nu_opt=nu, step=k_vec, steplength=s_vec, 
                     dgap= dgap_vec, r_pri=r_pri_vec, r_dual=r_dual_vec, dx_norm=dx_vec, dl_norm=dl_vec, dnu_norm=dnu_vec)
    return x, opt_logfile











































    