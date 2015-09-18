#!/usr/bin/env python
# Computes the matrix of similarities between structures in a xyz file
# by first getting SOAP descriptors for all environments, finding the best
# match between environments using the Hungarian algorithm, and finally
# summing up the environment distances.
# Supports periodic systems, matching between structures with different
# atom number and kinds, and sports the infrastructure for introducing an
# alchemical similarity kernel to match different atomic species

import quippy
import sys, time
from multiprocessing import Process, Value, Array
import argparse
from lap.lap import best_pairs
import numpy as np

#def sidx(i1,i2): return ( (i1,i2) if i1<=i2 else (i2,i1) ) 

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
      for k,w in nenv.soaps.items():
         if k in self.soaps:
            self.soaps[k] = self.soaps[k]*self.natoms + nenv.soaps[k]*nenv.natoms
         else:
            self.soaps[k] = w.copy() * nenv.natoms
        # print k, self.soaps[k] 
      self.natoms += nenv.natoms      
      self.normalize()
         
   def normalize(self):
      nrm = np.sqrt( envk(self, self, self.alchem) )
      for sij in self.soaps:  self.soaps[sij]*=1.0/nrm
   
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
      
   
   
class structure:
   def __init__(self, salchem=None):
      self.env={}
      self.species={}
      self.zspecies = []
      self.nenv=0  
      self.alchem=salchem
      if self.alchem is None: self.alchem=alchemy()
      self.globenv = None
      
   def getnz(self, sp):
      if sp in self.species:
         return self.species[sp]
      else: return 0
      
   def getenv(self, sp, i):
      if sp in self.species and i<self.species[sp]:
         return self.env[sp][i]
      else: 
         return environ(self.nmax,self.lmax,self.alchem,sp)  # missing atoms environments just returned as isolated species!
         
   def ismissing(self, sp, i):
      if sp in self.species and i<self.species[sp]:
         return False
      else: return True
      
   def parse(self, fat, coff=5.0, nmax=4, lmax=3, gs=0.5, cw=1.0, nocenter=[], noatom=[]):
      """ Takes a frame in the QUIPPY format and computes a list of its environments. """
      
      # removes atoms that are to be ignored
      at = fat.copy()
      nol = []
      for s in range(1,at.z.size+1):
         if at.z[s] in noatom: nol.append(s)
      if len(nol)>0: at.remove_atoms(nol)
      
      self.nmax = nmax
      self.lmax = lmax
      self.species = {}
      for z in at.z:      
         if z in self.species: self.species[z]+=1
         else: self.species[z] = 1
   
      self.zspecies = self.species.keys();
      self.zspecies.sort(); 
      lspecies = 'n_species='+str(len(self.zspecies))+' species_Z={ '
      for z in self.zspecies: lspecies = lspecies + str(z) + ' '
      lspecies = lspecies + '}'
   
      at.set_cutoff(coff);
      at.calc_connect();
      
      self.nenv = 0
      for sp in self.species:
         if sp in nocenter: 
            self.species[sp]=0
            continue # Option to skip some environments
         
         # first computes the descriptors of species that are present
         desc = quippy.descriptors.Descriptor("soap central_weight="+str(cw)+"  covariance_sigma0=0.0 atom_sigma="+str(gs)+" cutoff="+str(coff)+" n_max="+str(nmax)+" l_max="+str(lmax)+' '+lspecies+' Z='+str(sp) )   
         psp=np.asarray(desc.calc(at,desc.dimensions(),self.species[sp])).T;
      
         # now repartitions soaps in environment descriptors
         lenv = []
         for p in psp:
            nenv = environ(nmax, lmax, self.alchem)
            nenv.convert(sp, self.zspecies, p)
            lenv.append(nenv)
         self.env[sp] = lenv
         self.nenv += self.species[sp]
         
      # also compute the global (flattened) fingerprint
      self.globenv = environ(nmax, lmax, self.alchem)
      for k, se in self.env.items():
         for e in se:
            self.globenv.add(e)
      
         

# SOAP kernel between environments (with possibly alchemical similarity matrix)
def envk(envA, envB, alchem=alchemy()):
   dotp = 0.0
   
   #union of atom kinds present in the two environments
   zspecies = sorted(list(set(envA.zspecies+envB.zspecies)))
      
   nsp = len(zspecies)
   
   if len(alchem.rules) == 0 : # special case, only sum over diagonal bits
       for s1 in zspecies:
           for s2 in zspecies:
               ndot = np.dot(envA.getpair(s1,s2), envB.getpair(s1,s2))
               dotp+=ndot
   else:
       # alchemical matrix for species
       alchemAB = np.zeros((nsp,nsp), float)
       for sA in xrange(nsp):
           for sB in xrange(sA+1):
               alchemAB[sA,sB] = alchem.getpair(zspecies[sB],zspecies[sA])
               alchemAB[sB,sA] = alchemAB[sA,sB]       
       for iA1 in xrange(nsp):
          sA1 = zspecies[iA1]    
          for iB1 in xrange(nsp):          
             if alchemAB[iA1,iB1] == 0.0: continue
             sB1 = zspecies[iB1]
             for iA2 in xrange(nsp):
                sA2 = zspecies[iA2] 
                for iB2 in xrange(nsp):                              
                   if alchemAB[iA2,iB2] == 0.0: continue
                   sB2 = zspecies[iB2]
                   ndot = np.dot(envA.getpair(sA1,sA2), envB.getpair(sB1,sB2)) * alchemAB[iA1,iB1] * alchemAB[iA2,iB2]
                   dotp += ndot   
   #! SHOULD CHECK IF THE ALCHEMY IN THE ENVIRONMENTS IS THE SAME AS THIS, TO RENORMALIZE IF NEEDED
   return dotp

def gcd(a,b):
   if (b>a): a,b = b, a
   
   while (b):  a, b = b, a%b
   
   return a
   
def lcm(a,b):
   return a*b/gcd(b,a)

def gstructk(strucA, strucB, alchem=alchemy()):
   return envk(strucA.globenv, strucB.globenv, alchem) 

def structk(strucA, strucB, alchem=alchemy(), periodic=False, mode="sumlog", fout=None):
   nenv = 0
   
   if periodic: # replicate structures to match structures of different periodicity
      # we do not check for compatibility at this stage, just assume that the 
      # matching will be done somehow (otherwise it would be exceedingly hard to manage in case of non-standard alchemy)
      nspeciesA = []
      nspeciesB = []
      for z in strucA.zspecies:
         nspeciesA.append( (z, strucA.getnz(z)) )
      for z in strucB.zspecies:
         nspeciesB.append( (z, strucB.getnz(z)) )
      nenvA = strucA.nenv
      nenvB = strucB.nenv      
   else:   
      # top up missing atoms with isolated environments
      # first checks whic atoms are present
      zspecies = sorted(list(set(strucB.zspecies+strucA.zspecies)))
      nspecies = []
      for z in zspecies:
         nz = max(strucA.getnz(z),strucB.getnz(z))
         nspecies.append((z,nz)) 
         nenv += nz
      nenvA = nenvB = nenv
      nspeciesA = nspeciesB = nspecies
   
   np.set_printoptions(linewidth=500,precision=4)

   kk = np.zeros((nenvA,nenvB),float)
   ika = 0
   ikb = 0   
   for za, nza in nspeciesA:
      for ia in xrange(nza):
         envA = strucA.getenv(za, ia)
         
         ikb = 0
         for zb, nzb in nspeciesB:
            acab = alchem.getpair(za,zb)
            for ib in xrange(nzb):
               envB = strucB.getenv(zb, ib)
               if alchem.mu > 0 and (strucA.ismissing(za, ia) ^ strucB.ismissing(zb, ib)):
                  if mode == "kdistance" or mode == "nkdistance":  # includes a penalty dependent on "mu", in a way that is consistent with the definition of kernel distance
                     kk[ika,ikb] = acab - alchem.mu/2  
                  else:
                     kk[ika,ikb] = np.exp(-alchem.mu) * acab 
               else:
                  kk[ika,ikb] = envk(envA, envB, alchem) * acab
               ikb+=1
         ika+=1

   if fout != None:
      # prints out similarity information for the environment pairs
      fout.write("# atomic species in the molecules (possibly topped up with dummy isolated atoms): \n")      
      for za, nza in nspeciesA:
         for ia in xrange(nza): fout.write(" %d " % (za) )
      fout.write("\n");
      for zb, nzb in nspeciesB:
         for ib in xrange(nzb): fout.write(" %d " % (zb) )
      fout.write("\n");
      
      fout.write("# environment kernel matrix: \n")      
      for r in kk:
         for e in r:
            fout.write("%8.4e " % (e) )
         fout.write("\n")
       
   if periodic: 
      # now we replicate the (possibly rectangular) matrix to make
      # a square matrix for matching
      nenv = lcm(nenvA, nenvB)
      skk = np.zeros((nenv, nenv), float)
      for i in range(nenv/nenvA):
         for j in range(nenv/nenvB):
            skk[i*nenvA:(i+1)*nenvA,j*nenvB:(j+1)*nenvB] = kk
      kk = skk
      
   # Now we have the matrix of scalar products. 
   # We can first find the optimal scalar product kernel
   # we must find the maximum "cost"
   if mode == "logsum":
      hun=best_pairs(1.0-kk)
   
      dotcost = 0.0
      for pair in hun:
         dotcost+=kk[pair[0],pair[1]]
      cost = -np.log(dotcost/nenv)
   elif mode == "kdistance" or mode == "nkdistance":
      dk = 2*(1-kk)
      hun=best_pairs(dk)
      distcost = 0.0
      for pair in hun:
         distcost+=dk[pair[0],pair[1]]
      if periodic or mode == "nkdistance":
         # kdistance is extensive, nkdistance (or periodic matching) is intensive
         distcost*=1.0/nenv
      cost = distcost # this is really distance ^ 2      
   elif mode == "sumlog":   
      dk = (kk +1e-100)/(1+1e-100)  # avoids inf...
      dk = -np.log(dk)
      #print dk
      hun=best_pairs(dk)
      distcost = 0.0
      for pair in hun:
         distcost+=dk[pair[0],pair[1]]
      if periodic:    
         # here we normalize by number of environments only when doing 
         # periodic matching, so that otherwise the distance is extensive
         distcost*=1.0/nenv
      cost = distcost
   elif mode == "permanent":
      from permanent import permanent
      perm=permanent(np.array(kk,dtype=complex))       
      cost=-np.log(perm.real) 
      print cost
   else: raise ValueError("Unknown global fingerprint mode ", mode)
   
   if fout != None:
      fout.write("# optimal environment list: \n")      
      for pair in hun:
         fout.write("%d  %d  \n" % (pair[0],pair[1]) )
      fout.close()
   
   if cost<0.0: 
      print >> sys.stderr, "\n WARNING: negative squared distance ", cost, "\n"
      return 0.0 
   else: return np.sqrt(cost)

def main(filename, nd, ld, coff, gs, mu, centerweight, periodic, dmode, nocenter, noatom, nprocs, verbose=False, envij=None):
   
   print >> sys.stderr, "Reading input file", filename
   al = quippy.AtomsList(filename);
   
   print >> sys.stderr, "Computing SOAPs"

   alchem = alchemy(mu=mu)
#   alchem = alchemy(mu=mu, rules={ (6,7) : 1, (6,8): 1, (6, 17):1, (7, 8): 1  })  #hard coded
      
   sl = []
   iframe = 0   
   
   if verbose:
      qlog=quippy.AtomsWriter("log.xyz")
      slog=open("log.soap", "w")
   for at in al:
      if envij == None or iframe in envij:
         sys.stderr.write("Frame %d                              \r" %(iframe) )
         if verbose: qlog.write(at)

         si = structure(alchem)
         si.parse(at, coff, nd, ld, gs, centerweight, nocenter, noatom )
         sl.append(si)
         if verbose:
          slog.write("# Frame %d \n" % (iframe))
          for sp, el in si.env.iteritems():
            ik=0
            for ii in el:
                slog.write("# Species %d Environment %d \n" % (sp, ik))
                ik+=1
                for p, s in ii.soaps.iteritems():
                    slog.write("%d %d   " % p)
                    for si in s:
                        slog.write("%8.4e " %(si))                        
                    slog.write("\n")
          
      iframe +=1; 
   nf = len(sl)   
      
   print >> sys.stderr, "Computing distance matrix"
              
   sim=np.zeros((nf,nf))
   
   if dmode=="average":
      # use quick & dirty global fingerprints to compute distances
      for iframe in range (0, nf):      
         for jframe in range(0,iframe):
            sij = -np.log(gstructk(sl[iframe], sl[jframe], alchem))
            sim[iframe][jframe]=sim[jframe][iframe]=sij
         sys.stderr.write("Matrix row %d                           \r" % (iframe))
   else:
      if (nprocs<=1):
         # no multiprocess
         for iframe in range (0, nf):   
            for jframe in range(0,iframe):
               if verbose:
                  fij = open("environ-"+str(iframe)+"-"+str(jframe)+".dat", "w")
               else: fij = None
               
               sij = structk(sl[iframe], sl[jframe], alchem, periodic, mode=dmode, fout=fij)          
               sim[iframe][jframe]=sim[jframe][iframe]=sij
            sys.stderr.write("Matrix row %d                           \r" % (iframe))
      else:      
         # multiple processors
         def dorow(irow, nf, sl, psim): 
            for jframe in range(0,irow):
               sij = structk(sl[iframe], sl[jframe], alchem, periodic, mode=dmode)          
               psim[irow*nf+jframe]=sij  
               
         proclist = []   
         psim = Array('d', nf*nf, lock=False)      
         for iframe in range (0, nf):
            while(len(proclist)>=nprocs):
               for ip in proclist:
                  if not ip.is_alive(): proclist.remove(ip)            
                  time.sleep(0.01)
            sp = Process(target=dorow, name="doframe proc", kwargs={"irow":iframe, "nf":nf, "psim": psim, "sl":sl})  
            proclist.append(sp)
            sp.start()
            sys.stderr.write("Matrix row %d, %d active processes     \r" % (iframe, len(proclist)))
            
         # waits for all threads to finish
         for ip in proclist:
            while ip.is_alive(): ip.join(0.1)  
         
         # copies from the shared memory array to Sim.
         for iframe in range (0, nf):      
            for jframe in range(0,iframe):
               sim[iframe,jframe]=sim[jframe,iframe]=psim[iframe*nf+jframe]
   
   if dmode=="permanent" :
      # must fix the normalization of the similarity matrix!
      sys.stderr.write("Normalizing permanent kernels           \n")
      for iframe in range (0, nf):   
         sii = structk(sl[iframe], sl[iframe], alchem, periodic, mode=dmode, fout=None)
         sim[iframe,iframe]=sii
         for jframe in range (0, nf): 
            sim[iframe,jframe]-=0.5*sii
            sim[jframe,iframe]-=0.5*sii
      
   #print "final check", -np.log( np.dot( pdummy[6][0], pdummy[6][0] ) ), -np.log( np.dot( pdummy[6][0], pdummy[1][0] ) )
   print "# Similarity matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Distance: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, dmode, str(noatom), str(nocenter))
   for iframe in range(0,nf):
      for x in sim[iframe][0:nf]:
         print x,
      print ""   

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description="""Computes the similarity matrix between a set of atomic structures 
						   based on SOAP descriptors and an optimal assignment of local environments.""")
      parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
      parser.add_argument("--periodic", action="store_true", help="Matches structures with different atom numbers by replicating the environments")
      parser.add_argument("--exclude", type=str, default="", help="Comma-separated list of atom Z to be removed from the input structures (e.g. --exclude 96,101)")
      parser.add_argument("--nocenter", type=str, default="", help="Comma-separated list of atom Z to be ignored as environment centers (e.g. --nocenter 1,2,4)")
      parser.add_argument("--verbose",  action="store_true", help="Writes out diagnostics for the optimal match assignment of each pair of environments")   
      parser.add_argument("-n", type=int, default='8', help="Number of radial functions for the descriptor")
      parser.add_argument("-l", type=int, default='6', help="Maximum number of angular functions for the descriptor")
      parser.add_argument("-c", type=float, default='5.0', help="Radial cutoff")
      parser.add_argument("-g", type=float, default='0.5', help="Atom Gaussian sigma")
      parser.add_argument("-cw", type=float, default='1.0', help="Center atom weight")
      parser.add_argument("--mu", type=float, default='0.0', help="Extra penalty for comparing to missing atoms")
      parser.add_argument("--dotdistance", action="store_true", help="Use dot product distance for the global metric")
      parser.add_argument("--kdistance", action="store_true", help="Use kernel distance sqrt(2-2k(x,x')) for the global metric")
      parser.add_argument("--nkdistance", action="store_true", help="Use normalized kernel distance sqrt(2-2k(x,x')/natoms) for the global metric") 
      parser.add_argument("--permanent", action="store_true", help="Use permanent kernel for the global metric")
      parser.add_argument("--average",  action="store_true", help="Use average fingerprints to compute similarity")      
      parser.add_argument("--np", type=int, default='1', help="Use multiple processes to compute the similarity matrix")
      parser.add_argument("--ij", type=str, default='', help="Compute and print diagnostics for the environment similarity between frames i,j (e.g. --ij 3,4)")
	   
	   
	   
      args = parser.parse_args()

      if args.exclude == "":
         noatom = []
      else: 
         noatom = map(int,args.exclude.split(','))
	   
      if args.nocenter == "":
         nocenter = []
      else: 
         nocenter = map(int,args.nocenter.split(','))   
		    
      nocenter = sorted(list(set(nocenter+noatom)))
	   
      if (args.verbose and args.np>1): raise ValueError("Cannot write out diagnostics when running parallel jobs") 
	   
      if args.dotdistance:
         dmode="logsum"
      elif args.permanent:
         dmode="permanent"
      elif args.average:
         dmode="average"
      elif args.kdistance:
         dmode="kdistance"
      elif args.nkdistance:
	      dmode="nkdistance"
      else: # default
         dmode="sumlog"
	      
      if args.ij == "":
         envij=None
      else:
         envij=tuple(map(int,args.ij.split(",")))

      main(args.filename, nd=args.n, ld=args.l, coff=args.c, gs=args.g, mu=args.mu, centerweight=args.cw, periodic=args.periodic, dmode=dmode, noatom=noatom, nocenter=nocenter, nprocs=args.np, verbose=args.verbose, envij=envij)
