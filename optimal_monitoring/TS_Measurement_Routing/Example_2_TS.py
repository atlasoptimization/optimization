"""
The goal of this script is to demonstrate optimal measurement routing for two
total stations measuring concurrently. We want to derive a route traversing 
all measurement points in the least amount of 
total time. Two total stations are used in this example, which indices difficulty
in the objective function via several distance matrices and penalization of the
max time effort.. The problem features n randomly generated  points in 3 D 
space which are used to generate the distance matrices needed to evaluate total
measurement times.
For this, do the following:
    1. Imports and definitions
    2. Generate problem data
    3. Formulate the Optimization problem
    4. Assemble the solution
    5. Try different strategy
    6. Plots and illustratons

More Information can be found e.g. on pp. 151-156 in the Handbook on modelling 
for discrete optimization by G. Appa, L. Pitsoulis, and H. P. Williams,
Springer Science & Business Media (2006).
The problem is formulated and solved OR-Tools, a Python framework for posing and
solving problems from operations research by Google. More information on 
OR-Tools 7.2. by Laurent Perron and Vincent Furnon can be found on
https://developers.google.com/optimization/.


The script is meant solely for educational and illustrative purposes. Adapted 
from the OR-Tools tutorial https://developers.google.com/optimization/routing/vrp
on the vehicle routing problem. Written by Jemil Avers Butt, ETH Zuerich

"""


"""
    1. Imports and definitions -----------------------------------------------
"""


# i) Imports

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


# ii) Definitions

n_locations=45                  # Should be multiple of 3 for cluster simulation
n_clusterpoints=np.round(n_locations/3).astype(int)



"""
    2. Generate problem data --------------------------------------------------
"""


# i) Random locations

np.random.seed(1)

mu_1=np.array([[1],[1],[1]])
mu_2=np.array([[-1],[1],[1]])
mu_3=np.array([[0],[-1],[0]])

x_1=np.random.multivariate_normal(mu_1.flatten(), 3.1*np.eye(3), size=[n_clusterpoints])
x_2=np.random.multivariate_normal(mu_2.flatten(), 3.1*np.eye(3), size=[n_clusterpoints])
x_3=np.random.multivariate_normal(mu_3.flatten(), 3.1*np.eye(3), size=[n_clusterpoints])

x=np.vstack((x_1,x_2,x_3))


# ii) Total station location and relative coordinates

x_TS_1=np.array([[0],[2],[0]])
x_TS_2=np.array([[0],[-2],[0]])

x_centered_on_TS_1=x-np.tile(np.reshape(x_TS_1[0:3],[1,3]),[n_locations,1])
x_centered_on_TS_2=x-np.tile(np.reshape(x_TS_2[0:3],[1,3]),[n_locations,1])

# horizontal coordinate differences
x_proj_hor_1=x_centered_on_TS_1[:,0:2] #drop the height
hor_diff_angle_1=np.zeros([n_locations,n_locations])

x_proj_hor_2=x_centered_on_TS_2[:,0:2] #drop the height
hor_diff_angle_2=np.zeros([n_locations,n_locations])

# vertical coordinate differences
x_proj_range_1=np.linalg.norm(x_centered_on_TS_1[:,0:2],ord=2,axis=1)
x_height_1=x_centered_on_TS_1[:,2]
x_proj_vert_1=(np.vstack((x_proj_range_1,x_height_1))).T
vert_diff_angle_1=np.zeros([n_locations,n_locations])

x_proj_range_2=np.linalg.norm(x_centered_on_TS_2[:,0:2],ord=2,axis=1)
x_height_2=x_centered_on_TS_2[:,2]
x_proj_vert_2=(np.vstack((x_proj_range_2,x_height_2))).T
vert_diff_angle_2=np.zeros([n_locations,n_locations])


# iii) Construct distance matrix 1

dist_mat_1=np.zeros([n_locations,n_locations])
for k in range(n_locations):
    for l in range(n_locations):
        
        # horizontal angle
        
        x1_temp=x_proj_hor_1[k,:]
        x2_temp=x_proj_hor_1[l,:]
        quot_temp=np.round((x1_temp.T@x2_temp)/(np.linalg.norm(x1_temp)*np.linalg.norm(x2_temp)), decimals=5)
        hor_diff_angle_1[k,l]=np.arccos(quot_temp)
        
        # vertical angle
        x1_temp=x_proj_vert_1[k,:]
        x2_temp=x_proj_vert_1[l,:]
        quot_temp=np.round((x1_temp.T@x2_temp)/(np.linalg.norm(x1_temp)*np.linalg.norm(x2_temp)), decimals=5)
        vert_diff_angle_1[k,l]=np.arccos(quot_temp)
        
        # As distance take the max of the angle difference max(horizontal, vertical).
        dist_mat_1[k,l]=np.max([vert_diff_angle_1[k,l], hor_diff_angle_1[k,l]])
    
    
# iv) Construct distance matrix 2

dist_mat_2=np.zeros([n_locations,n_locations])
for k in range(n_locations):
    for l in range(n_locations):
        
        # horizontal angle
        
        x1_temp=x_proj_hor_2[k,:]
        x2_temp=x_proj_hor_2[l,:]
        quot_temp=np.round((x1_temp.T@x2_temp)/(np.linalg.norm(x1_temp)*np.linalg.norm(x2_temp)), decimals=5)
        hor_diff_angle_2[k,l]=np.arccos(quot_temp)
        
        # vertical angle
        x1_temp=x_proj_vert_2[k,:]
        x2_temp=x_proj_vert_2[l,:]
        quot_temp=np.round((x1_temp.T@x2_temp)/(np.linalg.norm(x1_temp)*np.linalg.norm(x2_temp)), decimals=5)
        vert_diff_angle_2[k,l]=np.arccos(quot_temp)
        
        # As distance take the max of the angle difference max(horizontal, vertical).
        dist_mat_2[k,l]=np.max([vert_diff_angle_2[k,l], hor_diff_angle_2[k,l]])
    
    

# iii) Data model

n_discretization=1000
data = {}
data['distance_matrix_1'] = n_discretization*dist_mat_1  # x100 since only integers are allowed
data['distance_matrix_2'] = n_discretization*dist_mat_2  # x100 since only integers are allowed
data['num_vehicles'] = 2
data['depot'] = 0



"""
    3. Formulate the Optimization problem ------------------------------------
"""


# i) Set up the subroutines

manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix_1']),data['num_vehicles'], data['depot'])
routing = pywrapcp.RoutingModel(manager)

def distance_callback_1(from_index, to_index):
    return data['distance_matrix_1'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

def distance_callback_2(from_index, to_index):
    return data['distance_matrix_2'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

transit_callback_index_1 = routing.RegisterTransitCallback(distance_callback_1)
transit_callback_index_2 = routing.RegisterTransitCallback(distance_callback_2)
transit_callback_vector=[transit_callback_index_1,transit_callback_index_2]

# ii) Add info and constraints

# Angle difference costs
routing.SetArcCostEvaluatorOfVehicle(transit_callback_index_1, 0)
routing.SetArcCostEvaluatorOfVehicle(transit_callback_index_2, 1)

# Add Distance constraint
# dimension_name = 'Angle'
# routing.AddDimension(transit_callback_index, 0, 100, True, dimension_name)
# distance_dimension = routing.GetDimensionOrDie(dimension_name)
# distance_dimension.SetGlobalSpanCostCoefficient(100)

# Add Distance constraint
dimension_name = 'Angle'
routing.AddDimensionWithVehicleTransits(transit_callback_vector, 0, 100000, True, 'Angle')
distance_dimension = routing.GetDimensionOrDie(dimension_name)
distance_dimension.SetGlobalSpanCostCoefficient(100)

# Setting first solution heuristic
search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)


# ii) Solve the problem

solution = routing.SolveWithParameters(search_parameters)



"""
    4. Assemble the solution -------------------------------------------------
"""


# i) Assemble the tours

route_list=[]

for vehicle_id in range(data['num_vehicles']):
    temp_route=[0]
    index = routing.Start(vehicle_id)
    plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
    route_distance = 0
    
    
    # ii) Loop through the tour
    
    while not routing.IsEnd(index):
        plan_output += ' {} -> '.format(manager.IndexToNode(index))
        previous_index = index
        index = solution.Value(routing.NextVar(index))
        temp_route+=[index]
        route_distance += routing.GetArcCostForVehicle(
            previous_index, index, vehicle_id)
        
        
    # iii) Print out results
    
    plan_output += '{}\n'.format(manager.IndexToNode(index))
    plan_output += 'Distance of the route: {}rad\n'.format(route_distance/n_discretization)
    temp_route[-1]=0
    route_list+=[temp_route]
    print(plan_output)


# iv) Coordinaes of the routes

full_tour=[]

for k in range(data['num_vehicles']):
    temp_location=np.empty([0,3])
    for l in range(len(route_list[k])):
        temp_location=np.vstack((temp_location, x[route_list[k][l],:]))
    full_tour+=[temp_location]



"""
    5. Try different strategy ------------------------------------------------
"""



# # i) Nearest Neighbor method - find indices

# # Start at arbitrary point, then go to next closest one
# route_nn=[0]

# # small_dist_mat= dist_mat
# old_index=0
# distance_total_nn=0

# for k in range(n_locations):
    
#     # Sort distances
#     dist_col=dist_mat_1[:,old_index]
#     dist_sorted=np.sort(dist_col)
#     indices_sorting=np.argsort(dist_col)
    
#     # Find first untraversed index
#     for xx in indices_sorting:
#         if xx in route_nn:
#             pass
#         else:
#             next_index=xx
#             distance_total_nn+=dist_col[next_index]
#             break
        
#     if k==n_locations-1:
#         next_index=0
#         distance_total_nn+=dist_mat_1[0,old_index]
        
#     # Append the index to the routing list
#     route_nn+=[next_index]
#     old_index=next_index

# # ii) Nearest Neighbor method - construct tour

# temp_location=np.empty([0,3])
# full_tour_nn=[]
# for l in range(len(route_nn)):
#     temp_location=np.vstack((temp_location, x[route_nn[l],:]))

# full_tour_nn+=[temp_location]
    





"""
    6. Plots and illustratons ------------------------------------------------
"""


# i) Figure displaying distance marix

plt.figure(1,dpi=300)
plt.imshow(dist_mat_1)
plt.title('The distance matrix 1')
plt.xlabel('Point nr')
plt.ylabel('Point nr')

plt.figure(2,dpi=300)
plt.imshow(dist_mat_2)
plt.title('The distance matrix 2')
plt.xlabel('Point nr')
plt.ylabel('Point nr')



# ii) Figure displaying the final routes - optimized

fig=plt.figure(3,dpi=300)
ax = fig.add_subplot(projection='3d')
ax.scatter(x[:,0],x[:,1], x[:,2],color='b')
ax.scatter(x_TS_1[0],x_TS_1[1], x_TS_1[2], color='r', linewidths=5)
ax.scatter(x_TS_2[0],x_TS_2[1], x_TS_2[2], color='r', linewidths=5)
plt.title('Point distribution and routes - VRP')
plt.xlabel('x-coordinate')
plt.ylabel('y-coordinate')

plt.plot(full_tour[0][:,0], full_tour[0][:,1], full_tour[0][:,2], color='k',linestyle='-',label='Tour 1')
plt.plot(full_tour[1][:,0], full_tour[1][:,1], full_tour[1][:,2], color='k',linestyle=':',label='Tour 2')
plt.legend()


# # iii) Figure displaying the final routes - nearest neighbor

# fig=plt.figure(3,dpi=300)
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x[:,0],x[:,1], x[:,2],color='b')
# ax.scatter(x_TS[0],x_TS[1], x_TS[2], color='r', linewidths=5)
# plt.title('Point distribution and routes -NN')
# plt.xlabel('x-coordinate')
# plt.ylabel('y-coordinate')

# plt.plot(full_tour_nn[0][:,0], full_tour_nn[0][:,1], full_tour_nn[0][:,2], color='k',linestyle='-',label='Tour 1')
# plt.legend()

# print('Nearest neighbor tour has length {} rad'.format(distance_total_nn))
