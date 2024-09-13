import numpy as np 
import cvxpy as cp
np.set_printoptions(suppress=True)
import random 
random.seed(10)


def mpc_solver(price_all_loc, water_all_loc, carbon_all_loc,workload_trace, num_ins,
               historical_e_cost,historical_w_cost,historical_c_cost,l_0 = 1, l_1 = 100,
                l_2 = 100, max_cap =1, verbose=True, f_type = "MAX"):
    # print(l_0,l_1,l_2)
    '''
    Solve the offline problem with cumulative workload constraint
    Args:
        price_all_loc   : Energy price of all locations [10, num_ins]
        water_all_loc   : Water WUE of all locations [10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10, num_ins]
        workload_trace  : Workload trace
        num_ins         : Number of timesteps to solve
    Return:
        optimal_cost:,y:
    '''
    y            = cp.Variable([10, num_ins],nonneg = True) # 10 number of data centers

    energy_cost  = cp.sum(cp.multiply(y, price_all_loc)) + historical_e_cost
    water_cost   = cp.sum(cp.multiply(y, water_all_loc), axis=1) + historical_w_cost
    carbon_cost  = cp.sum(cp.multiply(y, carbon_all_loc), axis=1) + historical_c_cost

    if f_type == "MAX":
        # Max Price
        water_cost   = cp.norm(water_cost, p="inf")
        carbon_cost  = cp.norm(carbon_cost, p="inf")
    elif f_type == "AVG":
        # Average Price
        water_cost   = cp.sum(water_cost, axis=0)/10 # 10 number of data centers
        carbon_cost  = cp.sum(carbon_cost, axis=0)/10
    else:
        raise NotImplementedError
        
    total_cost   = l_0*energy_cost + l_1*water_cost + l_2*carbon_cost

    constraints = []
    
    for i in range(num_ins):
        for j in range(10):
            constraints += [y[j,i] <= max_cap]

    # for cumulative constraints # discusse with Jianyi on Aug 24th at lab that without this constraint would be okay
    for i in range(num_ins):#1+window, num_ins-1 
        constraints+= [cp.sum(y[:,:i+1])<=cp.sum (workload_trace[:,:i+1]) ]

    constraints += [cp.sum(y) ==  cp.sum(workload_trace)  ]

    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    optimal_cost   = prob.value
    # print(l_0*energy_cost.value,l_1*water_cost.value,l_2*carbon_cost.value)
    return optimal_cost,y.value


def offline_solver_cc(price_all_loc, water_all_loc, carbon_all_loc,
                   workload_trace, num_ins, 
                   l_0 = 1, l_1 = 100, l_2 = 100, max_cap = 1, 
                   verbose=True, f_type = "MAX"):
    '''
    Solve the offline problem with cumulative workload constraint
    Args:
        price_all_loc   : Energy price of all locations [10, num_ins]
        water_all_loc   : Water WUE of all locations [10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10, num_ins]
        workload_trace  : Workload trace
        num_ins         : Number of timesteps to solve
    Return:
        optimal_cost:,y:
    '''
    y            = cp.Variable([10, num_ins],nonneg = True) # 10 number of data centers

    energy_cost  = cp.sum(cp.multiply(y, price_all_loc))
    water_cost   = cp.sum(cp.multiply(y, water_all_loc), axis=1)
    carbon_cost  = cp.sum(cp.multiply(y, carbon_all_loc), axis=1)

    if f_type == "MAX":
        # Max Price
        water_cost   = cp.norm(water_cost, p="inf")
        carbon_cost  = cp.norm(carbon_cost, p="inf")
    elif f_type == "AVG":
        # Average Price
        water_cost   = cp.sum(water_cost, axis=0)/10 # 10 number of data centers
        carbon_cost  = cp.sum(carbon_cost, axis=0)/10
    else:
        raise NotImplementedError
        
    total_cost   = l_0*energy_cost + l_1*water_cost + l_2*carbon_cost

    constraints = []
    
    for i in range(num_ins):
        for j in range(10):
            constraints += [y[j,i] <= max_cap]

    # for cumulative constraints # discusse with Jianyi on Aug 24th at lab that without this constraint would be okay
    for i in range(num_ins):
        constraints+= [cp.sum(y[:,:i+1])<=cp.sum (workload_trace[:,:i+1]) ]


    constraints += [cp.sum(y) ==  cp.sum(workload_trace)  ]

    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    optimal_cost   = prob.value
    # print(l_0*energy_cost.value,l_1*water_cost.value,l_2*carbon_cost.value)

    return optimal_cost,y.value



def evaluate_single(action_mask, price_all_loc):
    '''
    Evaluate the price of single resource
    '''
    # price_tensor  = price_all_loc.reshape([10,1,-1])
    price_res     = np.multiply(price_all_loc, action_mask)
    price_res     = price_res.sum(axis=(1))
    
    return price_res

def evaluate_total(action_mask, price_all_loc, carbon_all_loc, 
                   water_all_loc, l_1, l_2, 
                   verbose = True, l_0 = 1):
    '''
    Evaluate the total cost of the policy
    '''    
    price_res   = evaluate_single(action_mask, price_all_loc)
    carbon_res  = evaluate_single(action_mask, carbon_all_loc)
    water_res   = evaluate_single(action_mask, water_all_loc)
    
    
    price_cost  = l_0*np.sum(price_res)
    water_cost  = l_1*np.linalg.norm(water_res, ord=np.inf)
    carbon_cost = l_2*np.linalg.norm(carbon_res, ord=np.inf)
    
    total_cost  = price_cost + carbon_cost + water_cost
    
    if verbose:
        print("Electric price  :  {:.3f}".format(price_cost))
        print("Total water     :  {:.3f}".format(water_cost))
        print("Total carbon    :  {:.3f}".format(carbon_cost))
        print("-----")
        print("Overall Cost    :  {:.3f}".format(total_cost))
    
    return total_cost
