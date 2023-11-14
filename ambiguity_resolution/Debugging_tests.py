


# import sys

# sys.path.insert(0, './AR_compilation')

# import Support_funs_AR as sf
# import numpy as np
# import Ambiguity_resolution as AR

# n_obs=10

# weights=np.ones([1])
# distances=np.ones([1])
# wavelengths=np.linspace(0.01,0.05,n_obs)

# observations, C_mat=sf.Generate_data(weights,distances,wavelengths)


# cons=['d_opt>=0.5', 'd_opt<=1.5']
# optim_opts=sf.Setup_optim_options(n_obs, constraints=cons)

# phase_variances=np.ones([n_obs])


# d,N,r=AR.Ambiguity_resolution(observations, wavelengths, phase_variances,optim_opts)




import sys

sys.path.insert(0, './AR_compilation/')

import Support_funs_AR as sf
import numpy as np
import Ambiguity_resolution as AR

n_obs=10

weights=np.ones([1])
distances=np.ones([1])
wavelengths=np.linspace(0.01,0.05,n_obs)

phase_variances=np.ones([n_obs])*0.01

observations, C_mat=sf.Generate_data_noisy(weights,distances,wavelengths,phase_variances)


cons=['d_opt>=0.3', 'd_opt<=1.5']
optim_opts=sf.Setup_optim_options(n_obs, constraints=cons)



d,N,r=AR.Ambiguity_resolution(observations, wavelengths, phase_variances,optim_opts)
print(d)


























