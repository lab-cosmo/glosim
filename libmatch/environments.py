#!/usr/bin/env python
# Computes the matrix of similarities between structures in a xyz file
# by first getting SOAP descriptors for all environments, finding the best
# match between environments using the Hungarian algorithm, and finally
# summing up the environment distances.
# Supports periodic systems, matching between structures with different
# atom number and kinds, and sports the infrastructure for introducing an
# alchemical similarity kernel to match different atomic species

import sys, time
from copy import copy, deepcopy
import numpy as np

__all__ = [ "environ", "alchemy", "envk" ]

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
      

class environ:   
   def getpair(self, sa, sb):
      # siab = sidx(sa, sb)
      siab = (sa,sb) # the power spectra are not fully symmetric with respect to exchange of species index, unless one also exchanges n1 and n2, which is a mess.
      if siab in self.soaps:               
         return self.soaps[siab]
      else: 
         if len(self.soaps)==0: # dummy atoms environments just returned as isolated species!            
            if sa==sb and sa==self.z: return self.dummy1
         return self.dummy0  
         
   def __init__(self, nmax, lmax, salchem=None, specie=0):            
      self.alchem = salchem
      if self.alchem is None: self.alchem=alchemy()
      
      self.z = specie
      
      self.nmax = nmax
      self.lmax = lmax      
      self.dummy0 = np.zeros((self.nmax*self.nmax)*(self.lmax+1), float); 
      self.dummy1 = self.dummy0.copy(); self.dummy1[0]=1.0;
      
      self.soaps = {}        
      if self.z == 0:
        self.nspecies = 0
        self.zspecies = []
        self.natoms = 0
      else:
        self.nspecies = 1
        self.zspecies = [self.z]
        self.natoms = 1
      
   
   def add(self, nenv):
      # combine the SOAPS in the nenv environment to this one
      self.zspecies = sorted(list(set(self.zspecies+nenv.zspecies)))  # merges the list of environment species
      self.nspecies = len(self.zspecies)
      if self.z>0 and self.z!= nenv.z: self.z=-1  # self species is not defined for a sum of different centers of environments
      if self.nmax != nenv.nmax or self.lmax != nenv.lmax: raise ValueError("Cannot combine environments with different expansion channels")
      if len(self.soaps)==0 and self.z>0:  # we need an explicit description 
        self.soaps[(self.z,self.z)] = self.dummy1
      if len(nenv.soaps)==0 and nenv.z>0:
        if (nenv.z,nenv.z) in self.soaps:
            self.soaps[(nenv.z,nenv.z)] += self.dummy1
        else:
            self.soaps[(nenv.z,nenv.z)] = self.dummy1.copy()
      else:
        for k,w in nenv.soaps.items():     
            if k in self.soaps:
                self.soaps[k] = self.soaps[k] + nenv.soaps[k] 
            else:
                self.soaps[k] = w.copy()
        
      self.natoms += nenv.natoms            

   def normalize(self):       
      nrm = np.sqrt( envk(self, self, self.alchem) )
      for sij in self.soaps:  self.soaps[sij]*=1.0/nrm
   # @profile
   def convert(self, specie, species, rawsoap):      
      self.z = specie
      self.zspecies = sorted(species)
      self.nspecies = len(species)
      self.natoms = 1
      
      self.soaps = {}
      ipair = {}
      for s1 in range(self.nspecies):
         for s2 in range(self.nspecies): #  range(s1+1): we actually need to store also the reverse pairs if we want to go alchemical
            self.soaps[(self.zspecies[s2],self.zspecies[s1])] = np.zeros((self.nmax*self.nmax)*(self.lmax+1), float) 
            ipair[(self.zspecies[s2],self.zspecies[s1])] = 0
      
      isoap = 0
      isqrttwo = 1.0/np.sqrt(2.0)
      for s1 in xrange(self.nspecies):
         for n1 in xrange(self.nmax):         
            for s2 in xrange(s1+1):
               selpair = self.soaps[(self.zspecies[s2],self.zspecies[s1])]
               # we need to reconstruct the spectrum for the inverse species order, that also swaps n1 and n2. 
               # This is again only needed to enable alchemical combination of e.g. alpha-beta and beta-alpha. Shit happens
               revpair = self.soaps[(self.zspecies[s1],self.zspecies[s2])]                  
               isel = ipair[(self.zspecies[s2],self.zspecies[s1])]
               for n2 in xrange(self.nmax if s2<s1 else n1+1): 
                  for l in xrange(self.lmax+1):
                     #print s1, s2, n1, n2, isel, l+(self.lmax+1)*(n2+self.nmax*n1), l+(self.lmax+1)*(n1+self.nmax*n2)            
                     #selpair[isel] = rawsoap[isoap] 
                     if (s1 != s2):
                        selpair[isel] = rawsoap[isoap] * isqrttwo  # undo the normalization since we will actually sum over all pairs in all directions!                        
                        revpair[l+(self.lmax+1)*(n1+self.nmax*n2)] = selpair[isel]
                     else: 
                        # diagonal species (s1=s2) have only half of the elements.          
                        # this is tricky. we need to duplicate diagonal blocks "repairing" these to be full.
                        # this is necessary to enable alchemical similarity matching, where we need to combine 
                        # alpha-alpha and alpha-beta environment fingerprints
                        selpair[l+(self.lmax+1)*(n2+self.nmax*n1)] = selpair[l+(self.lmax+1)*(n1+self.nmax*n2)] = rawsoap[isoap] * (1 if n1==n2 else isqrttwo)                                             
                     isoap+=1
                     isel+=1
               ipair[(self.zspecies[s2],self.zspecies[s1])] = isel
      
      # alchemy-aware normalization
      self.normalize()
      
# SOAP kernel between environments (with possibly alchemical similarity matrix)
# @profile
def envk(envA, envB, alchem=alchemy()):
   dotp = 0.0
   
   #union of atom kinds present in the two environments
   #zspecies = sorted(list(set(envA.zspecies+envB.zspecies)))
   #zspecies = envA.zspecies
   #zspecies = envA.zspecies
   #print "ENV CHECK A", envA.zspecies
   #'print "ENV CHECK B", sorted(list(set(envA.zspecies+envB.zspecies)))
         
   
   if len(alchem.rules) == 0 : # special case, only sum over diagonal bits
       # only species that are in both environments will give nonzero contributions
       zspecies = sorted(list(set(envA.zspecies).intersection(envB.zspecies))) 
       for s1 in zspecies:
           for s2 in zspecies:
               ndot = np.dot(envA.getpair(s1,s2), envB.getpair(s1,s2))
               dotp+=ndot
   else:
       #union of atom kinds present in the two environments   
       zspecies = sorted(list(set(envA.zspecies+envB.zspecies)))
       nsp = len(zspecies)
   
       # alchemical matrix for species
       alchemAB = np.zeros((nsp,nsp), float)
       for sA in xrange(nsp):
           for sB in xrange(sA+1):
               alchemAB[sA,sB] = alchem.getpair(zspecies[sB],zspecies[sA])
               alchemAB[sB,sA] = alchemAB[sA,sB]
       
       # prepares the lists of pairs to avoid calling many times getpair further down the line
       eB = []
       for iB1 in xrange(nsp):
           sB1 = zspecies[iB1]
           eB.append([])
           for iB2 in xrange(nsp):
               sB2 = zspecies[iB2] 
               eB[iB1].append(envB.getpair(sB1,sB2))
        
       for iA1 in xrange(nsp):
          sA1 = zspecies[iA1]
          for iA2 in xrange(nsp):
             sA2 = zspecies[iA2] 
             eA = envA.getpair(sA1, sA2)                
             for iB1 in xrange(nsp):          
                if alchemAB[iA1,iB1] == 0.0: continue
                sB1 = zspecies[iB1]
                for iB2 in xrange(nsp):                              
                    if alchemAB[iA2,iB2] == 0.0: continue
                    sB2 = zspecies[iB2]
                    dotp += np.dot(eA, eB[iB1][iB2]) * alchemAB[iA1,iB1] * alchemAB[iA2,iB2]
       
   return dotp
