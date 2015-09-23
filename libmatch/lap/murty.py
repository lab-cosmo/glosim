#!/usr/bin/python
from copy import copy, deepcopy
import sys
from lap import best_pairs
import numpy as np

__all__ = [ "cost_list" ]

myINF=1e100
def main(filename='tmp', mxbest=20, mxdelta=50):
  a=np.loadtxt(filename)
  costs=best_costs(a,mxbest=int(mxbest), mxdelta=float(mxdelta))
  for x in costs:
    print x

def factorial(n):
  f = 1
  for i in xrange(2,n+1):
     f*=i
  return f

def cost_list(matrix, mxdelta=None, mxbest=None):
  #nbest: number of best costs 
  best_costs=[] #output array containing costs
  node_rule_list=[] # the node gen rule list
  cost_list=[] #list containing total costs
  rsrv_list=[] # list containing the costs which are not included in subset

  mxfac = factorial(len(matrix))
  if mxbest is None or mxbest > mxfac: 
     mxbest = mxfac

  # The best cost  by hugarian method
  hun=best_pairs(matrix)
  cost=0.0
  for pair in hun:
     cost+=matrix[pair[0],pair[1]]
  verybest = cost  
  best_costs.append(cost)
  if mxdelta == 0.0: return best_costs

  # Murty's algorithm for finding next k best costs 
  partition_list(matrix,node_rule_list,cost_list,rsrv_list)
  for k in range(1,mxbest):
    min_cost=min(cost_list)
    if not mxdelta is None:
       if min_cost - verybest > mxdelta: break
       for i in xrange(len(cost_list)):
          if cost_list[i] - verybest > mxdelta: cost_list[i]=myINF
    best_costs.append(min_cost)
    min_index=cost_list.index(min_cost)
    partition_list(matrix,node_rule_list,cost_list,rsrv_list)
    cost_list[min_index]=myINF
  return best_costs


def partition_list(matrix_orig,node_rule_list,cost_list,rsrv_list):
  try:
    min_index=cost_list.index(min(cost_list))
    rsrv_cost=rsrv_list[min_index]
  except:
    rsrv_cost=0.0
  try:
    node_rule=node_rule_list[min_index]
  except:
    node_rule=copy(node_rule_list)

  matrix,rsrv_cost=gen_partition(matrix_orig,node_rule)
  pair=[]
  assignments=lap(matrix)

  for m in range (0,len(assignments)-1):
    rule=copy(node_rule)
    pair=assignments[m]
    rule.append([-pair[0]-1,-pair[1]-1]) # - sign to denote the inf value position. -1 added to avoid confusion of +/- 0. 

    for x in range(0,m):
      rule.append([assignments[x][0],assignments[x][1]])

    node,rsrv_cost=gen_partition(matrix_orig,rule)
    hun=best_pairs(node)
    node_rule_list.append(rule)
    cost=0.0
    for pair in hun:
      cost+=node[pair[0],pair[1]]
    cost_list.append(cost+rsrv_cost)
    rsrv_list.append(rsrv_cost)
  return 

def gen_partition(matrix,rule):
  node=copy(matrix)
  rsrv_cost=0.0
#  print rule
  if (len(rule)==0): 
    return node,rsrv_cost # node is the main matrix itself
  # indices: indices of the negative pairs. Number of the appearance 
  # of negative indicies in the rule suggests the steps in gen the partition.
  indices=[i for i in range(len(rule)) if rule[i][0]<0]
#  print "indices=",indices
  nstep=len(indices)
  indices.append(len(rule))
  cost=0.0
  for step in range(nstep):
     step_rule=rule[indices[step]:indices[step+1]]
     #print "step=",step_rule
     node, cost=gen_node(node,step_rule)
     rsrv_cost+=cost
  return node,rsrv_cost


def gen_node(matrix,rule):
 node=copy(matrix)
 rsrv_cost=0.0
 pair=rule[0]
 node[-1-pair[0],-1-pair[1]]=myINF
 row_list=[]
 col_list=[]
 for pair in rule[1:]:
     row_list.append(pair[0])
     col_list.append(pair[1])
     rsrv_cost+=node[pair[0],pair[1]]
 node=remove_row_col(node,row_list,col_list)
 return node,rsrv_cost 

def remove_row_col(matrix,row,col):
#sorting in descending order so that it is easy to remove
  row.sort(reverse=True)
  col.sort(reverse=True)
  for r in row:
    matrix=np.delete(matrix,(r), axis=0)
  for c in col:
    matrix=np.delete(matrix,(c), axis=1)
  return matrix

if __name__ == "__main__":
    main(*sys.argv[1:])
