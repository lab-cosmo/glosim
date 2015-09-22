#!/usr/bin/env python
import quippy
import sys, time
from random import randint
from multiprocessing import Process, Value, Array
import argparse
from lap.lap import best_pairs
from glosim import *
import numpy as np
def main(filename, nd, ld, coff, gs, mu, centerweight, periodic, dmode, nocenter, noatom, nprocs, verbose=False, envij=None,minmax=False,nlandmark=0):
   
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
   
   if (minmax):
     print >> sys.stderr, "##### FARTHEST POINT SAMPLING ######"
     print >> sys.stderr, "Selecting",nlandmark,"Frames from",nf, "Frames"
     print >> sys.stderr, "##### Additional output files=sim-rect.dat, landmarks.xyz ######"
     
     m=nlandmark
     sim=np.zeros((m,m))
     sim_rect=np.zeros((m,nf))
     dist_list=[]
     landmarks=[]
     landxyz=quippy.AtomsWriter("landmarks.xyz")
     fsim=open("sim-rect.dat","w")
     iframe=randint(0,nf-1)
     iland=0
     landmarks.append(iframe)
     landxyz.write(al[iframe])
     for jframe in range(nf):
       if verbose:
         fij = open("environ-"+str(iframe)+"-"+str(jframe)+".dat", "w")
       else: fij = None    
       sij = structk(sl[iframe], sl[jframe], alchem, periodic, mode=dmode, fout=fij)
       sim_rect[iland][jframe]=sij
       dist_list.append(sij)
     for x in sim_rect[iland][:]:
       fsim.write("%8.4e " %(x))
     fsim.write("\n")
     maxd=0.0
     for iland in range(1,m):
       maxd=0.0
       for iframe in range(nf):
         if(dist_list[iframe]>maxd):
            maxd=dist_list[iframe]
            maxj=iframe
       landmarks.append(maxj)
       landxyz.write(al[maxj])
       iframe=maxj
       for jframe in range(nf):
         if verbose:
           fij = open("environ-"+str(iframe)+"-"+str(jframe)+".dat", "w")
         else: fij = None
         sij = structk(sl[iframe], sl[jframe], alchem, periodic, mode=dmode, fout=fij)
         sim_rect[iland][jframe]=sij
         if(sij<dist_list[jframe]): dist_list[jframe]=sij
         	   
       for x in sim_rect[iland][:]:
        fsim.write("%8.4e " %(x))
       fsim.write("\n")
       sys.stderr.write("Matrix row %d                           \r" % (iland))
     sys.stderr.write("Indices of the selected land marks \n")
     for iland in range(m):
       sys.stderr.write(" %d \n" % (landmarks[iland]))
       for jland in range(iland):
          sim[iland][jland]=sim[jland][iland]=sim_rect[iland][landmarks[jland]]
 

   else:           
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
   if (minmax):nf=nlandmark
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
      parser.add_argument("--minmax", type=int,default='0',help="Use farthest point sampling method to select n landmarks. std output is n x n matrix. The n x N rectangular matrix is stored in file sim-rect.dat and the selected landmark frames are stored in landmarks.xyz file") 	   
	   
	   
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
      if (args.minmax > 0):
           minmax=True
      else: minmax=False

      main(args.filename, nd=args.n, ld=args.l, coff=args.c, gs=args.g, mu=args.mu, centerweight=args.cw, periodic=args.periodic, dmode=dmode, noatom=noatom, nocenter=nocenter, nprocs=args.np, verbose=args.verbose, envij=envij,minmax=minmax,nlandmark=args.minmax)
