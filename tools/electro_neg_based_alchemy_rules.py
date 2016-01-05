#!/usr/bin/env python

import sys, time,ast
from copy import copy, deepcopy
import numpy as np
import argparse

class alchemy:
   def getpair(self, sa, sb):
      if len(self.rules)==0: # special case when the alchemical matrix is default
         if sa==sb: return 1
         else: return 0  
      else:
          if sa<=sb and (sa,sb) in self.rules:            
             return self.rules[(sa,sb)]
          elif sa>sb and (sb,sa) in self.rules:
             return self.rules[(sb,sa)] 
          else: 
             if sa==sb: return 1
             else: return 0  
   
   def __init__(self, rules={}, mu=0):            
      self.rules = rules.copy()
      self.mu = mu

class alchemy_electro:
   def getpair(self, sa, sb,delta):
      if sa<=sb and (sa,sb) in self.rules:
         return self.rules[(sa,sb)]
      elif sa>sb and (sb,sa) in self.rules:
         return self.rules[(sb,sa)]
      else:
         Elec_neg={1:2.1,6:2.5,7:3.0,8:3.5,16:2.5,17:3.0}
         deltaE=Elec_neg[sa] -Elec_neg[sb]
         p=np.exp(-0.5*(deltaE/delta)**2)
         return p
   def __init__(self, rules={},delta=1, mu=0):            
      self.rules = rules.copy()
      self.mu = mu
      self.delta= delta

def main(delta,splist,alchemyrules):
   if (alchemyrules=="none"):
      alchem=alchemy_electro(delta=delta,rules={})
   else:
       r=alchemyrules.replace('"', '').strip()
       r=alchemyrules.replace("'", '').strip()
       r=ast.literal_eval(r)
       print >> sys.stderr, "Using Alchemy rules: ", r,"\n"
       alchem=alchemy_electro(delta=delta,rules=r)
   rule={}
   for sa in splist:
      for sb in splist:
        if (sb >sa): 
           rule[(sa,sb)]=alchem.getpair(sa,sb,delta)
   print rule
if __name__ == '__main__':
      parser = argparse.ArgumentParser(description="""Utility code to generate alchemy rules based on electronegetivity.""")

      parser.add_argument("--delta",type=float,default="1.0", help="delta value in  exp(-0.5*(Elec_neg[sa] -Elec_neg[sb]/delta)**2) ")
      parser.add_argument("--species", type=str, default="", help="list of species (e.g. --species 1,6  for H and C )")
      parser.add_argument("--rules", type=str, default="none", help='Dictionary-style rule specification in quote (e.g. --rules "{(6,7):1,(6,8):1}"')
      args = parser.parse_args()
      if args.species == "":
         splist = []
      else:
         splist = sorted(map(int,args.species.split(',')))
      main(delta=args.delta,splist=splist,alchemyrules=args.rules)
