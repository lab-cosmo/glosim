#!/usr/bin/env python

import sys, time,ast
from copy import copy, deepcopy
import numpy as np
import argparse
import gc 
import cPickle as pickle

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
         Elec_neg={1: 2.2, 2: 0, 3: 0.98, 4: 1.57, 5: 2.04, 6: 2.55, 7: 3.04, 8: 3.44, 9: 3.98, 10: 0, 11: 0.93, 12: 1.31, 13: 1.5, 14: 1.8, 15: 2.19, 16: 2.58, 17: 3.16, 18: 0, 19: 0.82, 20: 1, 21: 1.36, 22: 1.54, 23: 1.63, 24: 1.66, 25: 1.55, 26: 1.83, 27: 1.88, 28: 1.91, 29: 1.9, 30: 1.65, 31: 1.81, 32: 2.01, 33: 2.18, 34: 2.55, 35: 2.96, 36: 0, 37: 0.82, 38: 0.95, 39: 1.22, 40: 1.33, 41: 1.6, 42: 2.16, 43: 1.9, 44: 2.2, 45: 2.28, 46: 2.2, 47: 1.93, 48: 1.69, 49: 1.78, 50: 1.96, 51: 2.05, 52: 2.1, 53: 2.66, 54: 0, 55: 0.79, 56: 0.89, 57: 1.1, 58: 1.12, 59: 1.13, 60: 1.14, 61: 1.13, 62: 1.17, 63: 1.2, 64: 1.2, 65: 1.2, 66: 1.22, 67: 1.23, 68: 1.24, 69: 1.25, 70: 1.1, 71: 1.27, 72: 1.3, 73: 1.5, 74: 2.36, 75: 1.9, 76: 2.2, 77: 2.2, 78: 2.28, 79: 2.54, 80: 2, 81: 2.04, 82: 2.33, 83: 2.02, 84: 2, 85: 2.2, 86: 0, 87: 0.7, 88: 0.9, 89: 1.1, 90: 1.3, 91: 1.5, 92: 1.38, 93: 1.36, 94: 1.28, 95: 1.3, 96: 1.3, 97: 1.3, 98: 1.3, 99: 1.3, 100: 1.3, 101: 1.3, 102: 1.3, 103: 10.0, 104: 10.0, 105:10.0, 106: 10.0, 107: 10.0, 108: 10.0, 109: 10.0}
         #Elec_neg={1:2.1,6:2.5,7:3.0,8:3.5,16:2.5,17:3.0}
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
   f="alchemy_elecneg_delta"+str(delta)+".pickle"
   file = open(f,"wb")
   gc.disable()
   pickle.dump(rule, file,protocol=pickle.HIGHEST_PROTOCOL) # HIGHEST_PROTOCOL is 2 in py 2.7
   file.close()
   gc.enable()
if __name__ == '__main__':
      parser = argparse.ArgumentParser(description="""Utility code to generate alchemy rules based on electronegetivity.""")

      parser.add_argument("--delta",type=float,default="1.0", help="delta value in  exp(-0.5*((Elec_neg[sa] -Elec_neg[sb])/delta)**2) ")
      parser.add_argument("--species", type=str, default="all", help="list of species (e.g. --species 1,6  for H and C )")
      parser.add_argument("--rules", type=str, default="none", help='Dictionary-style rule specification in quote (e.g. --rules "{(6,7):1,(6,8):1}"')
      args = parser.parse_args()
      if args.species == "all":
         splist = []
         for i in range(1,109):
            splist.append(i)
      else:
         splist = sorted(map(int,args.species.split(',')))
      main(delta=args.delta,splist=splist,alchemyrules=args.rules)
