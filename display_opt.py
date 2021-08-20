'''
Created on Nov 24, 2017

@author: aliaouad
'''
import pandas as pd
import numpy as np
import math
import time
from random import *
#from gurobipy import Model,GRB,quicksum
from scipy.stats import poisson,norm
from itertools import combinations
import scipy.misc as sc
from numpy import dtype


def swap(permt,i,j):
    '''
    Swapping i-th and j-th elements of a permutation
    '''
    a = permt[i]
    permt[i] = permt[j]
    permt[j] = a
    return()
    
                
def vector_from_combination(t,k,m,ub):
        '''
        Converts an element of the combination into 
        
        t <- element of the combinations; number of items are captured by 
             differences in consecutive elements in t ; skipped classes are the
             elements above i (memory - 1)
        k <- number of locations filled so far
        m <- number of classes (memory -1)
        ub <- number of products per class
        '''
        tuple_cons = ()
        last_ind = -1
        for j in range(1,m+1):
            if (k + j) in t:
                # the class is not picked if selected above m              
                tuple_cons += (0,)
            elif (last_ind > -1) & (t[last_ind+1]-t[last_ind] <= ub[j-1]):
                # difference of t-s is  the number of items in current class               
                tuple_cons += (t[last_ind+1]-t[last_ind],)
                last_ind += 1 
            elif (last_ind == -1) & (t[last_ind+1] <= ub[j-1]):
                # we start by all products in the union class
                tuple_cons += (t[last_ind+1],)
                last_ind += 1
            else: 
                return("ERROR")
        if (last_ind > -1) & (k - t[last_ind] <= ub[m]):
            # adding the last class
            return(tuple_cons + (k - t[last_ind],))
        elif (last_ind == -1) & (k <= ub[m]):
            # case of a single class
            return(tuple_cons + (k,))
        else: 
            return("ERROR")
            

class DisplayOpt(object):
    '''
    Experiments for Display Optimzation
    '''


    def __init__(self, params):
        '''
        Instance constructor, specifies  parameters of the generative model
        
        n <- number of products
        weight_scale <- total weight
        price_scale <- scale of log-normal price generator
        n_classes <- number of weight classes
        memory <- number of consecutive weight classes
        mode < average # consideration sets relative to n
        '''
        self.n = params["n"]
        self.weight_scale = params["weight_scale"] # unused
        self.weight_total = params["weight_total"]
        self.price_scale = params["price_scale"]
        self.n_classes = params["n_classes"]
        self.memory = params["memory"]
        self.mode = params["mode"]
        
        
    def reset_instance(self):
        '''
        Reset random instance instances
        '''
        self.weights = self.weight_total/\
                        self.n*np.sort((np.random.random(size = self.n))/0.5)
        self.weights = self.weight_total/\
                        self.n*np.sort(np.random.choice(\
                            [0.2*i for i in range(1,10)],
                            size = self.n, 
                            replace = True))
        self.weights = self.weight_total/\
                        self.n*2.0*np.sort(np.random.rand(self.n))        
        self.prices = np.sort(np.exp(\
                    (np.random.normal(scale = self.price_scale,\
                                      size = self.n))))[::-1]
        print "Price range: ",min(self.prices),max(self.prices)
        self.rhos = np.multiply(self.prices,self.weights)
        self.lambda_dis = np.zeros(self.n)
        n_pick = min(self.n,int(self.mode*self.n) + 1)
        selection_ind = sample(range(self.n),n_pick) 
        self.lambda_dis[selection_ind] = np.random.random(n_pick)
        self.lambda_dis = self.lambda_dis/np.sum(self.lambda_dis) 
        self.permt = np.arange(self.n)
        
        
    def estimate_revenue(self,permt = None,m = None):
        """
        Estimates the expected revenue for a given (partial) permutation
        
        perm <- permutation over products
        m <- shape of the permutation
        """
        if type(permt) == type(None):
            permt = self.permt
        if type(m) == type(None):
            m = permt.shape[0]
        if m > 0:
            val = np.dot(self.lambda_dis[:m], 
                         np.divide(np.cumsum(np.multiply(self.weights[permt[:m]],
                                                         self.prices[permt[:m]])),
                                   1+np.cumsum(self.weights[permt[:m]]))
                         )
        else:
            val = 0
        return(val)
    
    
    def greedy_solve(self,start = np.array([])):
        '''
        Implements the greedy algorithm
        
        start <- initial assortment
        '''
        remaining_items = range(self.n)
        remaining_items = filter(lambda x: x not in start,remaining_items)
        permt = np.zeros(self.n,dtype =int)
        permt[:len(start)] = start
        for i in range(len(start),self.n):
            best = -1
            val = - 1
            for j in remaining_items:
                permt[i] = j
                new_val = self.estimate_revenue(permt = permt,m = int(i+1))
                if new_val >= val:
                    best = j      
                    val = new_val              
            permt[i] = best
            remaining_items.remove(best)

        return (val,permt)
    
        
    def sort_rho(self):
        '''
        Rank by decreasing rho-quantities
        '''
        permt = np.argsort(np.multiply(self.weights,self.prices))[::-1]
        val = self.estimate_revenue(permt)
        return (val,permt)
   
    
    def sort_prices(self):
        '''
        Rank by decreasing prices
        '''
        permt = np.argsort(self.prices)[::-1]
        val = self.estimate_revenue(permt)
        return (val,permt)
   
     
    def local_search(self, start =None):
        '''
        Local search algorithm
        
        start <- initial assortment        
        '''
        if start == None:
            permt = np.arange(self.n)
            np.random.shuffle(permt)
        else:
            permt = start
        val = self.estimate_revenue(permt)
        t = 0
        incr = 1
        while (incr > 0.001):
            t +=1
            best = (-1,-1)
            incr = - 1
            val_loop = val
            for i in range(self.n-1):
                for j in range(i+1,self.n):
                        swap(permt,i,j)
                        #print permt
                        new_val = self.estimate_revenue(permt)
                        if new_val > val_loop:
                            best = (i,j)      
                            incr = new_val/val -1.0
                            val_loop = new_val             
                        swap(permt,i,j)
            if incr >0:
                swap(permt, best[0], best[1]) 
                val = val_loop        
        return (val,permt)
    
    
    def heuristic_gallego_al(self):
        '''
        Algorithm NEST+        
        '''
        val = 0
        list_vals = [0.0]
        permt = np.zeros(self.n)
        for i in range(self.n-1):
            for j in range(i+1,self.n):
                if self.weights[i] != self.weights[j]: 
                    list_vals.append((self.prices[i]*\
                                      self.weights[i]-\
                                      self.prices[j]*self.weights[j])\
                                    /(self.weights[i]-self.weights[j]))
        list_vals = sorted(np.unique(filter(lambda x: x>=0, list_vals)))
        t=0
        for c in range(0,self.n+1):
                bin_val = 1
                ordered_assortment = []
                t = 0
                while (bin_val == 1) & (t < len(list_vals)): 
                    l = list_vals[t]
                    bin_val = 0
                    min_rho = np.multiply(self.prices - l,self.weights)
                    assortment = np.argsort(min_rho)[::-1]
                    if np.sum(min_rho[assortment[:c].tolist()]) >= l:
                        ordered_assortment = np.argsort(\
                                        self.rhos[assortment[:c].tolist()])[::-1]
                        ordered_assortment = (assortment[:c]\
                                              [ordered_assortment.tolist()])\
                                              .tolist()
                        bin_val = 1
                        t += 1                              
                if c < self.n:
                    v,p = self.greedy_solve(start = ordered_assortment)
                else:
                    v = self.estimate_revenue(np.array(ordered_assortment))
                    p = ordered_assortment
                if v> val:
                    val = v
                    permt = np.array(p)

        print "Heuristic Gallego et. al.", val,permt              
        return (val,permt)
                                                
            
    def generate_classes(self,eps = 0.4):
        '''
        Generates weight classes
        
        eps <- accuracy of weight classes
        '''
        
        sorted_weights_vals = np.sort(self.weights)
        sorted_weights = np.argsort(self.weights)
        #finding indices of products delimiting the weight classes        
        weight_indices = [np.argmax(sorted_weights_vals >= eps*np.power(1+eps,K)\
                                    *self.weight_total/self.n) 
                          for K in range(int(np.ceil(np.log(5/eps)/np.log(1+eps))))]
        v = np.argmax(weight_indices)
        if v > 0:
            weight_indices = weight_indices[:(np.argmax(weight_indices)+1)]
        weight_indices = np.unique(weight_indices).tolist()
        print "Weight class indices",weight_indices
        print "Partition size", len(weight_indices)
        
        #union of classes below a certain weight threshold
        union_classes = [sorted_weights[:i].tolist() for i in weight_indices] + \
                        [sorted_weights.tolist()]
        union_classes =np.unique(union_classes).tolist()

        #actual weight classes
        classes = [np.array(sorted_weights[:weight_indices[0]]).tolist()]\
                  + [np.array(sorted_weights[weight_indices[i]:w]).tolist() \
                     for (i,w) in enumerate(weight_indices[1:])]\
                  + [np.array(sorted_weights[weight_indices[-1]:]).tolist()]
         
        union_classes = list(filter(lambda a: len(a) != 0, union_classes))
        classes = list(filter(lambda a: len(a) != 0, classes))        
        
        # reranking by decreasing rho-quantities
        classes = [[classes[i][j] for j in np.argsort(self.rhos[classes[i]])[::-1]] \
                    for i in range(len(classes))]
        union_classes = [[union_classes[i][j] for j in np.argsort(\
                          self.rhos[union_classes[i]])[::-1]] \
                            for i in range(len(union_classes))]
        print "Length classes",len(classes)
        
        return(classes,union_classes)
        
        
    def ADP(self):
        '''
        Approximate Dynamic Program in modified form
        
        Two main modifications:
            Fixed number of consecutive weight classes (memory)
            Assortments are directly encoded with # stocked items per class
        '''
        
        #initialization
        classes,union_classes = self.generate_classes()
        n_classes = len(classes)
        t_max = max(map(lambda x: len(x),classes))
        memory = min(self.memory,n_classes)
        DP = -10000000*np.zeros((n_classes-memory+1,) + (self.n+1,) + \
                                tuple(1+t_max for i in range(memory-1)))
        DP[:,tuple(0 for i in range(memory))] = 0
        DP_product = {(q,)+tuple(0 for i in range(memory)):[] \
                      for q in range(DP.shape[0])}        
        evaluated = {(q,)+tuple(0 for i in range(memory)):0 \
                     for q in range(DP.shape[0])}
        
        #DP recursion        
        for c in range(DP.shape[0]):
            #max number of items per classes
            ub = [len(union_classes[c])] + \
                 [len(classes[c+i]) for i in range(1,memory)]
            for k in range(1,self.n+1):
                #k is current location
                for t in combinations(range(1,k+memory),memory - 1):
                    #generating all vectors of number of stocked items per class
                    #over #memory previous classes
                    config = vector_from_combination(t,k,memory - 1,ub)
                    if config != "ERROR":
                        
                        # computing the immediate payoff
                        assortment  = union_classes[c][:min(config[0],\
                                                    len(union_classes[c]))]\
                                      + reduce(lambda x,i: x + \
                                               classes[c+i][:min(config[i],\
                                                           len(classes[c+i]))],\
                                               range(1,memory),[])
                        lambda_immediate = self.lambda_dis[len(assortment) -1]
                        current_rev = self.evaluate_assortment_revenue(assortment)
                        evaluated[(c,)+config] = current_rev
                        
                        # computing the reward-to-go
                        # Case 1: deleting one product from class j without changing 
                        # the subset of consecutive classes
                        vals = []
                        residual_assortments = []
                        for j in range(memory):
                            if ((j != memory - 1) | (config[-1] != 1) | (c==0)) & (config[j]>0):                                
                                config2 = tuple([config[alpha] if alpha != j \
                                                 else config[j] - 1 for alpha 
                                                 in range(memory)])
                                #print j,current_rev,evaluated[(c,)+config2],lambda_immediate,(c,)+config2
                                vals += [DP[(c,)+config2] + \
                                            current_rev*lambda_immediate]
                                if j == 0:
                                    residual_assortments += [DP_product[(c,)+config2] \
                                                           + [union_classes[c][config2[j]]]]
                                else:
                                    residual_assortments += [DP_product[(c,)+config2] \
                                                            + [classes[c+j][config2[j]]]]
                                    
                        # Case 2: deleting one product from last class while changing 
                        # the subset of consecutive classes                                                                        
                        if (config[-1] == 1) & (c>0):
                            if config[0]>0:
                                # Case 2a: current union of classes should be split
                                # into current class and previous union of classes
                                thresh = self.rhos[union_classes[c][config[0]-1]]
                                ind_union = np.argmax(self.rhos[union_classes[c-1]] < thresh)
                                if (ind_union == 0) & (self.rhos[union_classes[c-1][-1]] >= thresh):
                                    ind_union = len(union_classes[c-1])
                                    
                                vals += [DP[(c-1,)+(ind_union,config[0]-ind_union,) \
                                            + config[1:(memory-1)]] \
                                          + lambda_immediate*\
                                         current_rev]
                                residual_assortments += [DP_product[(c-1,)+\
                                                        (ind_union,config[0]-ind_union,) \
                                                        + config[1:(memory-1)]] +\
                                                        [classes[c+memory-1][0]]]
                            else:
                                # Case 2b: current union of classes is empty                             
                                vals += [DP[(c-1,)+(0,0,) + config[1:(memory-1)]] \
                                            + lambda_immediate*current_rev]
                                residual_assortments += [DP_product[(c-1,)+(0,0,) \
                                                          + config[1:(memory-1)]] +\
                                                        [classes[c+memory-1][0]]]
                                
                        # picking the best local DP action
                        max_winner = np.argmax(np.array(vals))
                        DP[(c,)+config] = vals[max_winner]
                        DP_product[(c,)+config] = residual_assortments[max_winner]
                        
                                
        # picking the best global DP action                                
        max_val = np.max(DP)
        max_winner = tuple([ np.where(DP == max_val)[j][0] for j in range(len(DP.shape))])
        print len(DP_product[max_winner]),DP_product[max_winner] 
        print max_val,self.estimate_revenue(np.array(DP_product[max_winner])),DP_product[max_winner]
        return(self.estimate_revenue(np.array(DP_product[max_winner])),DP_product[max_winner])
                    
                    
                    
    def evaluate_assortment_revenue(self,A):
        '''
        Expected revenue of an individual assortment
        
        A <- assortment
        '''
        return np.divide(np.sum(np.multiply(self.weights[A],self.prices[A])),\
                         1+np.sum(self.weights[A]))
            
if __name__ == '__main__':        
    
    path = "results-ranking/"
    version = "_v2"
    params = {
                  "n":20,
                  "lambda_scale": 0.5,
                  "weight_scale": 1.0,
                  "weight_total": 8.0,
                  "price_scale": 1.0,
                  "n_classes": 8,
                  "memory":5,
                  "mode": 1.0
                  }
    
    R =  DisplayOpt(params)
    R.reset_instance()
    
    list_n = [30]
    list_m = [5]
    list_s = [20]
    list_mode = [0.33,0.66]
    list_wt = [1.0,5.0,10.0]
    list_ps = [0.3,1.0]
        
    
    for i in range(len(list_n)):
            for wt in list_wt:
                for ps in list_ps:
                    for md in list_mode:
                        params["n"] = list_n[i]
                        params["memory"] =  list_m[i]
    #                     params["lambda_scale"] = ls
                        params["weight_total"] = wt
                        params["price_scale"] = ps
                        params["mode"] = md
                        print params.values(),params.keys()
                        sol = {"local_search":[], "greedy_solve":[],
                               "heuristic_gallego_al":[], "ADP":[],
                               "sort_rho":[], "sort_prices":[]}
                        tim = {"local_search":[], "greedy_solve":[],
                               "heuristic_gallego_al":[], "ADP":[],
                               "sort_rho":[], "sort_prices":[]}
                        for q in range(list_s[i]):
                            print "\n"
                            R =  DisplayOpt(params)
                            R.reset_instance()
                            try:
                                t = time.time()
                                a,b =  R.local_search()
                                t = time.time() - t
                                sol["local_search"].append(a)
                                tim["local_search"].append(t)
                            except:
                                sol["local_search"].append("ERROR")
                                tim["local_search"].append("ERROR")
                            try:
                                t = time.time()
                                a,b =  R.greedy_solve()
                                t = time.time() - t
                                sol["greedy_solve"].append(a)
                                tim["greedy_solve"].append(t)
                            except:
                                sol["greedy_solve"].append("ERROR")
                                tim["greedy_solve"].append("ERROR")
                            try:
                                t = time.time()
                                a,b =  R.heuristic_gallego_al()
                                t = time.time() - t
                                sol["heuristic_gallego_al"].append(a)
                                tim["heuristic_gallego_al"].append(t)
                            except:
                                sol["heuristic_gallego_al"].append("ERROR")
                                tim["heuristic_gallego_al"].append("ERROR")
                            try:
                                t = time.time()
                                a,b =  R.ADP()
                                t = time.time() - t
                                sol["ADP"].append(a)
                                tim["ADP"].append(t)
                            except:
                                sol["ADP"].append("ERROR")
                                tim["ADP"].append("ERROR")
     
                            try:
                                t = time.time()
                                a,b =  R.sort_rho()
                                t = time.time() - t
                                sol["sort_rho"].append(a)
                                tim["sort_rho"].append(t)
                            except:
                                sol["sort_rho"].append("ERROR")
                                tim["sort_rho"].append("ERROR")
                            try:
                                t = time.time()
                                a,b =  R.sort_prices()
                                t = time.time() - t
                                sol["sort_prices"].append(a)
                                tim["sort_prices"].append(t)
                            except:
                                sol["sort_prices"].append("ERROR")
                                tim["sort_prices"].append("ERROR")
                            pd.DataFrame(sol).to_csv(path +"sol" +\
                                        ",".join(map(lambda x: str(x),params.values()))+\
                                        version + ".csv")
                            pd.DataFrame(tim).to_csv(path +"tim" +\
                                        ",".join(map(lambda x: str(x),params.values()))+\
                                        version + ".csv")
        
        
            