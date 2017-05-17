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
         Elec_aff={1: 0.75420375, 2: 0.0, 3: 0.618049, 4: 0.0, 5: 0.279723, 6: 1.262118, 7: -0.07, 8: 1.461112, 9: 3.4011887, 10: 0.0, 11: 0.547926, 12: 0.0, 13: 0.43283, 14: 1.389521, 15: 0.7465, 16: 2.0771029, 17: 3.612724, 18: 0.0, 19: 0.501459, 20: 0.02455, 21: 0.188, 22: 0.084, 23: 0.525, 24: 0.67584, 25: 0.0, 26: 0.151, 27: 0.6633, 28: 1.15716, 29: 1.23578, 30: 0.0, 31: 0.41, 32: 1.232712, 33: 0.814, 34: 2.02067, 35: 3.363588, 36: 0.0, 37: 0.485916, 38: 0.05206, 39: 0.307, 40: 0.426, 41: 0.893, 42: 0.7472, 43: 0.55, 44: 1.04638, 45: 1.14289, 46: 0.56214, 47: 1.30447, 48: 0.0, 49: 0.404, 50: 1.112066, 51: 1.047401, 52: 1.970875, 53: 3.059038, 54: 0.0, 55: 0.471626, 56: 0.14462, 57: 0.47, 58: 0.5, 59: 0.5, 60: 0.5, 61: 0.5, 62: 0.5, 63: 0.5, 64: 0.5, 65: 0.5, 66: 0.5, 67: 0.5, 68: 0.5, 69: 0.5, 70: 0.5, 71: 0.5, 72: 0.0, 73: 0.322, 74: 0.815, 75: 0.15, 76: 1.0778, 77: 1.56436, 78: 2.1251, 79: 2.30861, 80: 0.0, 81: 0.377, 82: 0.364, 83: 0.942363, 84: 1.9, 85: 2.8, 86: 0.0, 87: 0.0, 88: 0.0, 89: 0.0, 90: 0.0, 91: 0.0, 92: 0.0, 93: 0.0, 94: 0.0, 95: 0.0, 96: 0.0, 97: 0.0, 98: 0.0, 99: 0.0, 100: 0.0, 101: 0.0, 102: 0.0, 103: 0.0, 104: 0.0, 105: 0.0, 106: 0.0, 107: 0.0, 108: 0.0, 109: 0.0}
         Ion_Nrg={1: 13.5984, 2: 24.5874, 3: 5.3917, 4: 9.3227, 5: 8.298, 6: 11.2603, 7: 14.5341, 8: 13.6181, 9: 17.4228, 10: 21.5645, 11: 5.1391, 12: 7.6462, 13: 5.9858, 14: 8.1517, 15: 10.4867, 16: 10.36, 17: 12.9676, 18: 15.7596, 19: 4.3407, 20: 6.1132, 21: 6.5615, 22: 6.8281, 23: 6.7462, 24: 6.7665, 25: 7.434, 26: 7.9024, 27: 7.881, 28: 7.6398, 29: 7.7264, 30: 9.3942, 31: 5.9993, 32: 7.8994, 33: 9.7886, 34: 9.7524, 35: 11.8138, 36: 13.9996, 37: 4.1771, 38: 5.6949, 39: 6.2173, 40: 6.6339, 41: 6.7589, 42: 7.0924, 43: 7.28, 44: 7.3605, 45: 7.4589, 46: 8.3369, 47: 7.5762, 48: 8.9938, 49: 5.7864, 50: 7.3439, 51: 8.6084, 52: 9.0096, 53: 10.4513, 54: 12.1298, 55: 3.8939, 56: 5.2117, 57: 5.5769, 58: 5.5387, 59: 5.473, 60: 5.525, 61: 5.582, 62: 5.6437, 63: 5.6704, 64: 6.1498, 65: 5.8638, 66: 5.9389, 67: 6.0215, 68: 6.1077, 69: 6.1843, 70: 6.2542, 71: 5.4259, 72: 6.8251, 73: 7.5496, 74: 7.864, 75: 7.8335, 76: 8.4382, 77: 8.967, 78: 8.9588, 79: 9.2255, 80: 10.4375, 81: 6.1082, 82: 7.4167, 83: 7.2855, 84: 8.414, 85: -1, 86: 10.7485, 87: 4.0727, 88: 5.2784, 89: 5.17, 90: 6.3067, 91: 5.89, 92: 6.1941, 93: 6.2657, 94: 6.026, 95: 5.9738, 96: 5.9914, 97: 6.1979, 98: 6.2817, 99: 6.42, 100: 6.5, 101: 6.58, 102: 6.65, 103: 4.9, 104: 6.0, 105: -1, 106: -1, 107: -1, 108: -1, 109: -1}
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
