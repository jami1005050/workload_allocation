import numpy as np 
import cvxpy as cp
np.set_printoptions(suppress=True)
import random 
random.seed(10)
def offline_solver_y(price_all_loc, water_all_loc, carbon_all_loc,
                   workload_trace, num_ins, 
                   l_0 = 1, l_1 = 100, l_2 = 100, max_cap = 1, 
                   verbose=True, f_type = "MAX"):
    '''
    Solve the offline problem
    Args:
        price_all_loc   : Energy price of all locations [10, num_ins]
        water_all_loc   : Water WUE of all locations [10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10, num_ins]
        workload_trace  : Workload trace
        num_ins         : Number of timesteps to solve
    Return:
        optimal_cost:,y:
    '''
    y            = cp.Variable([10, num_ins],nonneg = True)

    energy_cost  = cp.sum(cp.multiply(y, price_all_loc))
    water_cost   = cp.sum(cp.multiply(y, water_all_loc), axis=1)
    carbon_cost  = cp.sum(cp.multiply(y, carbon_all_loc), axis=1)

    if f_type == "MAX":
        # Max Price
        water_cost   = cp.norm(water_cost, p="inf")
        carbon_cost  = cp.norm(carbon_cost, p="inf")
    elif f_type == "AVG":
        # Average Price
        water_cost   = cp.sum(water_cost, axis=0)/10
        carbon_cost  = cp.sum(carbon_cost, axis=0)/10
    else:
        raise NotImplementedError
        
    total_cost   = l_0*energy_cost + l_1*water_cost + l_2*carbon_cost

    constraints = []
    
    for i in range(num_ins):
        for j in range(10):
            constraints += [y[j,i] <= max_cap]

    constraints += [cp.sum(y) ==  cp.sum(workload_trace)  ]

    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    optimal_cost   = prob.value

    return optimal_cost,y.value

def offline_solver_unconstrained(price_all_loc, water_all_loc, carbon_all_loc,workload_trace, num_ins,l_md, 
                                 l_0 = 1, l_1 = 100, l_2 = 100, max_cap = 1, verbose=True, f_type = "MAX"):
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
    y            = cp.Variable([10, num_ins],nonneg = True)

    energy_cost  = cp.sum(cp.multiply(y, price_all_loc))
    water_cost   = cp.sum(cp.multiply(y, water_all_loc), axis=1)
    carbon_cost  = cp.sum(cp.multiply(y, carbon_all_loc), axis=1)

    if f_type == "MAX":
        # Max Price
        water_cost   = cp.norm(water_cost, p="inf")
        carbon_cost  = cp.norm(carbon_cost, p="inf")
    elif f_type == "AVG":
        # Average Price
        water_cost   = cp.sum(water_cost, axis=0)/10
        carbon_cost  = cp.sum(carbon_cost, axis=0)/10
    else:
        raise NotImplementedError
        
    total_cost   = l_0*energy_cost + l_1*water_cost + l_2*carbon_cost  + l_md*cp.maximum(0, (cp.sum(workload_trace)-cp.sum(y)))

    constraints = []
    
    for i in range(num_ins):
        for j in range(10):
            constraints += [y[j,i] <= max_cap]

    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    optimal_cost   = prob.value

    return optimal_cost,y.value




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

def offline_solver(price_all_loc, water_all_loc, carbon_all_loc,
                   workload_trace, mask_array, num_ins, 
                   l_0 = 1, l_1 = 100, l_2 = 100, max_cap = 1, 
                   verbose=True, f_type = "MAX"):
    '''
    Solve the offline problem
    Args:
        price_all_loc   : Energy price of all locations [10, num_ins]
        water_all_loc   : Water WUE of all locations [10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10, num_ins]
        workload_trace  : Workload trace
        mask_array      : Array with size of [10, 1, num_ins]
        num_ins         : Number of timesteps to solve
    Return:
        optimal_cost:
        action_mask:
    '''
    x            = cp.Variable([10*1, num_ins],nonneg = True)
    x_masked     = cp.multiply(x, mask_array)
    y            = cp.Variable([10, num_ins],nonneg = True)

    energy_cost  = cp.sum(cp.multiply(y, price_all_loc))

    water_cost   = cp.sum(cp.multiply(y, water_all_loc), axis=1)
    # water_cost   = cp.reshape(water_cost, [10,1], order="C")
    # water_cost   = cp.sum(water_cost, axis=1)

    carbon_cost  = cp.sum(cp.multiply(y, carbon_all_loc), axis=1)
    # carbon_cost  = cp.reshape(carbon_cost, [10,1], order="C")
    # carbon_cost  = cp.sum(carbon_cost, axis=1)

    
    if f_type == "MAX":
        # Max Price
        water_cost   = cp.norm(water_cost, p="inf")
        carbon_cost  = cp.norm(carbon_cost, p="inf")
    elif f_type == "AVG":
        # Average Price
        water_cost   = cp.sum(water_cost, axis=0)/10
        carbon_cost  = cp.sum(carbon_cost, axis=0)/10
    else:
        raise NotImplementedError
        
    total_cost   = l_0*energy_cost + l_1*water_cost + l_2*carbon_cost

    constraints = []
    for i in range(num_ins):
        for j in range(1):
            c_i = [
                cp.sum(x_masked[j::1, i]) ==  workload_trace[j,i] # 
            ]
            constraints += c_i

    for i in range(num_ins):
        for j in range(10):
            constraints += [y[j,i] <= max_cap]

    for j in range(10):
        constraints += [cp.sum(x_masked[1*j:1*j+1, :]) <=  cp.sum(y[j, :])]

    for i in range(num_ins):
        for j in range(10):
            constraints += [x[j,i] >= 0]


    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    
    optimal_cost   = prob.value
    action_optimal = x.value
    action_mask    = np.multiply(action_optimal, mask_array)
    
    return optimal_cost, action_mask,y.value


def online_solver(price_all_loc, water_all_loc, carbon_all_loc,
                   workload_trace, mask_array, num_ins,water_off,carbon_off,
                  l_1 = 100, l_2 = 100, max_cap = 1, 
                   verbose=True,v = 200):
    '''
    Solve the online problem
    Args:
        price_all_loc   : Energy price of all locations [10, num_ins]
        water_all_loc   : Water WUE of all locations [10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10, num_ins]
        workload_trace  : Workload trace
        mask_array      : Array with size of [10, 1, num_ins]
        num_ins         : Number of timesteps to solve
    Return:
        optimal_cost:
        action_mask:
    '''

    x_array = []
    y_array = [] 
    a_array = []
    b_array = []
    
    H = np.zeros((10,num_ins))
    G = np.zeros((10,num_ins))
    J = np.zeros((10,num_ins)) 

    for t in range(num_ins):
        # print(H[:,t],G[:,t],J[:,t])
        x_t,y_t = resource_optimizer(H[:,t],G[:,t],J[:,t],price_all_loc[:,t],
                                     water_all_loc[:,t],carbon_all_loc[:,t],
                                     workload_trace[:,t],v,max_cap,verbose,
                                     mask_array[:,t])
        # x_t = workload_optimizer(J[:,t],workload_trace[:,t],mask_array[:,t],verbose)
        # print(x_t.shape,y_t.shape)
        l_w = np.max(water_off[:,t])*0.80
        u_w = np.max(water_off[:,t])
        l_c = np.max(carbon_off[:,t])*0.80
        u_c = np.max(carbon_off[:,t])

        a_t = water_optimizer(H[:,t],v,l_1,l_w,u_w)
        b_t = carbon_optimizer(G[:,t],v,l_2,l_c,u_c)
        
        x_array.append(x_t)
        y_array.append(y_t)
        a_array.append(a_t)
        b_array.append(b_t)

        w_t = np.multiply(water_all_loc[:,t],y_t)
        c_t = np.multiply(carbon_all_loc[:,t],y_t)
        
        if(t == (num_ins-1)): break
        x_t=np.sum(x_t,axis = 1)
        # print( y_t)
        for i in range (10):
            H[i][t + 1] = np.maximum((H[i][t] - a_t + w_t[i]),0)
            G[i][t + 1] = np.maximum((G[i][t] - b_t + c_t[i]),0)
            J[i][t + 1] = np.maximum((J[i][t] - y_t[i]),0)  + x_t[i]

    x_array  = np.stack(x_array,  axis=2)
    y_array = np.stack(y_array, axis=1)

    a_array = np.array(a_array)
    b_array = np.array(b_array)
    return x_array, y_array, a_array, b_array

def water_optimizer(h,v,kappa_w,lb,ub): # optimizer for the water fairness

    coe_diff = v*kappa_w  - np.sum(h)
    # print(v*kappa_w,np.sum(h))
    if(coe_diff>0):
        return lb#random.uniform(lb,.80) 
    else:
        return ub#random.uniform(1,ub)

def carbon_optimizer(g,v,kappa_c,lb,ub): # optimizer for the water fairness

    coe_diff = v*kappa_c - np.sum(g)
    if(coe_diff>0): 
        return lb#random.uniform(lb,.80)  
    else: 
        return ub#random.uniform(1,ub)

def workload_optimizer(J,workload_trace,mask_array,verbose): # optimizer for the workload fairness
    x            = cp.Variable([10,1],nonneg = True)
    x_masked     = cp.multiply(x, mask_array.reshape([10,1]))
    workload_cost = cp.sum(cp.multiply(cp.sum(x_masked,axis = 1),J))
    constraints = []
    for j in range(1):#1 gateway
        constraints += [cp.sum(x_masked[:,j]) ==  workload_trace[j]]

    for i in range(10):# 10 data center
        for j in range(1): #1 gateway
            constraints += [x[i,j] >= 0]
    
    objective = cp.Minimize(workload_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    
    optimal_cost   = prob.value
    action_optimal = x.value
    action_mask    = np.multiply(action_optimal, mask_array.reshape([10,1]))
    # print(optimal_cost)
    return action_mask

def resource_optimizer(H,G,J,price_all_loc,water_all_loc,carbon_all_loc,
                       workload_trace,v,max_cap,verbose,mask_array):

    x            = cp.Variable([10,1],nonneg = True)
    x_masked     = cp.multiply(x, mask_array.reshape([10,1]))
    y            = cp.Variable([10],nonneg = True)

    energy_cost  = cp.sum(cp.multiply(y, price_all_loc))

    water_cost   = cp.sum(cp.multiply(cp.multiply(y, water_all_loc),H))# w_i*y(t)*h_i(t),for all i
    carbon_cost  = cp.sum(cp.multiply(cp.multiply(y, carbon_all_loc),G))

    workload_cost = cp.sum(cp.multiply(cp.sum(x_masked,axis = 1),J))
    resource_cost = cp.sum(cp.multiply(y,J))

    # when v is small energy cost will be large vice versa
    total_cost   = v*energy_cost + water_cost + carbon_cost  - resource_cost + workload_cost

    constraints = []
    for j in range(1):#1 gateway
        constraints += [cp.sum(x_masked[:,j]) ==  workload_trace[j]]

    for j in range(10):# 10 datacenter
        constraints += [y[j] <= max_cap,y[j] >= 0]

    for i in range(10):# 10 data center
        for j in range(1): #1 gateway
            constraints += [x[i,j] >= 0]


    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    
    optimal_cost   = prob.value
    action_optimal = x.value
    action_mask    = np.multiply(action_optimal, mask_array.reshape([10,1]))
    # print(optimal_cost)
    return action_mask,y.value#



def online_solver_y(price_all_loc, water_all_loc, carbon_all_loc,
                   workload_trace, num_ins,water_off,carbon_off,
                    l_0 = 1,l_1 = 100, l_2 = 100, max_cap = 1, 
                   verbose=True,v = 200,f_type = 'MAX'):
    '''
    Solve the online problem
    Args:
        price_all_loc   : Energy price of all locations [10, num_ins]
        water_all_loc   : Water WUE of all locations [10, num_ins]
        carbon_all_loc  : Carbon consumption of all places [10, num_ins]
        workload_trace  : Workload trace
        mask_array      : Array with size of [10, 1, num_ins]
        num_ins         : Number of timesteps to solve
    Return:
        optimal_cost:
        action_mask:
    '''

    y_array = [] 
    a_array = []
    b_array = []
    
    H = np.zeros((10,num_ins))
    G = np.zeros((10,num_ins))
    J = np.zeros(num_ins) 
    y_sum = 0
    # print("Max W: ",np.max(water_off),"Min W: ",np.min(water_off))
    # print("Max C: ",np.max(carbon_off),"Min W: ",np.min(carbon_off))
    if(f_type == 'MAX'):
        l_w = np.max(water_off)*0.4  #np.max(water_off[:,t])
        u_w = np.max(water_off)*1.3  #7.462507177574032# np.max(water_off[:,t])
        l_c = np.max(carbon_off)*0.4 #np.max(carbon_off[:,t])*0.80
        u_c = np.max(carbon_off)*1.3  # np.max(carbon_off[:,t])
    else:
        l_w = np.mean(water_off)*0.40 #np.max(water_off[:,t])
        u_w = np.mean(water_off)*1.2#7.462507177574032# np.max(water_off[:,t])
        l_c = np.mean(carbon_off)*0.40 #np.max(carbon_off[:,t])*0.80
        u_c = np.mean(carbon_off)*1.2 # np.max(carbon_off[:,t])
    for t in range(num_ins):
        rem_workload = np.sum (workload_trace[:,:t+1]) - y_sum
        

        y_t = resource_optimizer_y(H[:,t],G[:,t],J[t],price_all_loc[:,t],water_all_loc[:,t],
                                   carbon_all_loc[:,t],workload_trace[:,t],v,
                                   max_cap,verbose,rem_workload,l_0)
        y_sum += np.sum(y_t)

        a_t = water_optimizer(H[:,t],v,l_1,l_w,u_w)
        a_array.append(a_t)

        # if(l_2!=0):
        b_t = carbon_optimizer(G[:,t],v,l_2,l_c,u_c)
        b_array.append(b_t)

        y_array.append(y_t)
        w_t = np.multiply(water_all_loc[:,t],y_t)
        c_t = np.multiply(carbon_all_loc[:,t],y_t)
        if(t == (num_ins-1)): break
        for i in range (10):
            # print(H[i][t],a_t,w_t[i],l_w,u_w)
            # if(l_1!=0):
            H[i][t + 1] = np.maximum((H[i][t] - a_t + w_t[i]),0)
            
            # if(l_2!=0):
            G[i][t + 1] = np.maximum((G[i][t] - b_t + c_t[i]),0)

        # if(l_0!=0):
        J[t + 1] = J[t] - np.sum(y_t) + np.sum(workload_trace[:,t])
            # J[t + 1] = J[t] - np.sum(y_t) + rem_workload


    y_array = np.stack(y_array, axis=1)

    a_array = np.array(a_array)
    b_array = np.array(b_array)
    return y_array, a_array, b_array,J[num_ins-1]

def resource_optimizer_y(H,G,J,price_all_loc,water_all_loc,carbon_all_loc,
                       workload_trace,v,max_cap,verbose,rem_workload,l_0):

    y            = cp.Variable([10],nonneg = True)

    energy_cost  = cp.sum(cp.multiply(y, price_all_loc))


    water_cost   = cp.norm(cp.multiply(cp.multiply(y, water_all_loc),H), p="inf")
    carbon_cost  = cp.norm(cp.multiply(cp.multiply(y, carbon_all_loc),G), p="inf")

    workload_cost = J*cp.sum(workload_trace)
    resource_cost = J*cp.sum(y)

    # when v is small energy cost will be large vice versa
    total_cost   = l_0*v*energy_cost + water_cost + carbon_cost  - resource_cost + workload_cost

    constraints = []

    for j in range(10):# 10 datacenter
        constraints += [y[j] <= max_cap,y[j] >= 0]

    constraints += [cp.sum(y) <= rem_workload]
    
    objective = cp.Minimize(total_cost)
    prob      = cp.Problem(objective, constraints)
    if verbose: print("Start Solving...")
    prob.solve(verbose = verbose)
    # print(water_cost.value,carbon_cost.value,resource_cost.value, workload_cost.value)
    return y.value




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


def distribute_rem_wl(data,remwl):
    i  = 0
    while remwl!=0:
        # print(data.shape)
        if(i>=data.shape[1]):
            break
        data[0,i]+= remwl
        if(data[0,i]>10):
            remwl = data[0,i]  - 10
            data[0,i]  = 10
            i+=1
            # print(remwl)
        else:
            remwl = 0
    return data,remwl   


def mpc_allocation(l0,l1,l2,f_type,price,water,carbon,workload,
                   workload_trace,price_all_loc,water_all_loc,carbon_all_loc):
    optimal_action_off = []
    historical_e_cost = 0
    historical_w_cost = np.zeros(10)
    historical_c_cost = np.zeros(10)
    action_sum = 0
    rem_t = 0
    for i in range(workload_trace.shape[1]):

        rem_wl = np.sum(workload_trace[:,:i]) - action_sum+rem_t
        workload[i],rem_t =  distribute_rem_wl(workload[i] ,rem_wl)
        optimal_cost,y = mpc_solver(price[i], water[i],carbon[i],workload[i],price[i].shape[1],
                                    historical_e_cost,historical_w_cost,historical_c_cost,
                                            verbose=False,l_0 = l0, l_1 = l1, l_2 = l2,f_type = f_type)
        if(y is None):
            print(np.sum(workload[i]),workload[i],i)
        historical_e_cost += np.sum(np.multiply(y[:,0],price_all_loc[:,i])) # single value 
        historical_w_cost += np.multiply(y[:,0],water_all_loc[:,i]) #[10,0:i-1]
        historical_c_cost += np.multiply(y[:,0],carbon_all_loc[:,i])##[10,0:i-1]
        action_sum += np.sum(y[:,0])
        optimal_action_off.append(y[:,0])
    optimal_action_off = np.stack(optimal_action_off,axis = 1)
    return optimal_action_off


def get_const_predicted_data_for_GLB(data,window,const):
    predicted_data_array = []
    predicted_data = np.full((data.shape[0],window),const)
    # print(predicted_data)
    for i in range(data.shape[1]):
        if((i+window) > (data.shape[1]-1)):  
            length = data.shape[1]-1-i-1
            future_data =predicted_data[:,0:length+1]
        else:
            future_data = predicted_data
        # print(future_data.shape)
        combined_data = np.concatenate([data[:,[i]],future_data],axis = 1)
        predicted_data_array.append(combined_data)
    return predicted_data_array



def add_noise(data, percentage_error,is_wl = False):# 1-24
    true_average = np.mean(data)
    # true_means = np.mean(data.T, axis=0)
    # print(true_means)
    noise_vector = percentage_error * true_average#.1,.2,.3,.4-.7
    error = np.random.normal(0, noise_vector, size=data.shape)
    noisy_data = data+ error#np.random.normal(0, noise, size=data.shape)# 10, 457
    print(np.mean(noisy_data),np.std(noisy_data))

    if(is_wl):
        noisy_data = np.clip(noisy_data,0, 10)  # Clip values to the valid range (0-255 for images)
    else:
        noisy_data = np.clip(noisy_data,0, None)  # Clip values to the valid range (0-255 for images)

    return noisy_data


def mpc_without_prediction(l0,l1,l2,f_type,price,water,carbon,workload,
                            workload_trace,price_all_loc,water_all_loc,carbon_all_loc):
    optimal_action_off = []
    historical_e_cost = 0
    historical_w_cost = np.zeros(10)
    historical_c_cost = np.zeros(10)
    action_sum = 0
    rem_t = 0
    for i in range(workload_trace.shape[1]):
        # if(i==0):print(np.sum(workload_trace[:,:i]),'First Sum')
        rem_wl = np.sum(workload_trace[:,:i]) - action_sum + rem_t
        workload[:,i]+=rem_wl
        if(workload[:,i]>10):
            rem_t = workload[:,i] - 10
            workload[:,i] = 10
        optimal_cost,y = mpc_solver(price[:,[i]], water[:,[i]],carbon[:,[i]],workload[:,[i]],price[:,[i]].shape[1],
                                    historical_e_cost,historical_w_cost,historical_c_cost,
                                    verbose=False,l_0 = l0, l_1 = l1, l_2 = l2,f_type = f_type)
        if(y is None):
            print(np.sum(workload[i]),workload[i],i)
        # print(y.shape,price[:,[i]].shape,water[:,[i]].shape,carbon[:,[i]].shape,workload[:,[i]].shape)
        historical_e_cost += np.sum(np.multiply(y[:,0],price_all_loc[:,i])) # single value 
        # print('historical_e_cost: ',historical_e_cost)
        historical_w_cost += np.multiply(y[:,0],water_all_loc[:,i]) #[10,0:i-1]
        historical_c_cost += np.multiply(y[:,0],carbon_all_loc[:,i])##[10,0:i-1]
        # print('historical_w_cost: ',historical_w_cost)
        # print('historical_c_cost: ',historical_c_cost)

        action_sum += np.sum(y[:,0])
        optimal_action_off.append(y[:,0])
    optimal_action_off = np.stack(optimal_action_off,axis = 1)
    return optimal_action_off


def compute_cost(action,l1,l2,price_all_loc,water_all_loc,carbon_all_loc):

    price_res   = evaluate_single(action, price_all_loc)
    water_res   = evaluate_single(action, water_all_loc)
    carbon_res  = evaluate_single(action, carbon_all_loc)

    price_cost  = np.sum(price_res)
    water_cost  = l1*np.linalg.norm(water_res, ord=np.inf)
    carbon_cost = l2*np.linalg.norm(carbon_res, ord=np.inf)
    total_cost  = price_cost + carbon_cost + water_cost
    # print(price_cost,water_cost,carbon_cost)
    return total_cost,np.max(price_res),np.mean(price_res),np.max(water_res),\
        np.mean(water_res),np.max(carbon_res),np.mean(carbon_res),price_cost



def get_exact_predicted_data_for_GLB(data,window_size):
    predicted_data_array = []
    for i in range(data.shape[1]): #124,4 [122-124]+4 = 126
        if((i+window_size) > (data.shape[1]-1)):  
            future_data = data[:,i+1:]
        else:
            future_data = data[:,i+1:i+window_size+1]
            
        combined_data = np.concatenate([data[:,[i]],future_data],axis = 1)
        predicted_data_array.append(combined_data)
    return predicted_data_array


def get_predicted_data(data,noisy_data,window_size):
    predicted_data_array = []
    for i in range(data.shape[1]):
        if((i+window_size) > (data.shape[1]-1)):  
            future_data = noisy_data[:,i+1:]
        else:
            future_data = noisy_data[:,i+1:i+window_size+1]
        combined_data = np.concatenate([data[:,[i]],future_data],axis = 1)
        predicted_data_array.append(combined_data)
    return predicted_data_array