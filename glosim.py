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
from random import randint
from libmatch.environments import alchemy, environ
from libmatch.structures import structk, structure
import numpy as np

def main(filename, nd, ld, coff, gs, mu, centerweight, periodic, kmode, nocenter, noatom, nprocs, verbose=False, envij=None, usekit=False, kit="auto", prefix="",nlandmark=0, printsim=False):
   
    filename = filename[0]
    # sets a few defaults 
    if prefix=="": prefix=filename

    # reads input file using quippy
    print >> sys.stderr, "Reading input file", filename
    al = quippy.AtomsList(filename);
   
    print >> sys.stderr, "Computing SOAPs"

    # sets alchemical matrix
    alchem = alchemy(mu=mu)
    sl = []
    iframe = 0      
    if verbose:
        qlog=quippy.AtomsWriter("log.xyz")
        slog=open("log.soap", "w")
      
    # determines reference kit    
    if usekit:
        if kit == "auto": 
            kit = {} 
            iframe=0
            for at in al:
                if envij == None or iframe in envij: 
                    sp = {} 
                    for z in at.z:     
                        if z in noatom or z in nocenter: continue  
                        if z in sp: sp[z]+=1
                        else: sp[z] = 1
                    for s in sp:
                        if not s in kit:
                            kit[s]=sp[s]
                        else:
                            kit[s]=max(kit[s], sp[s])
                iframe+=1
        iframe=0
        print >> sys.stderr, "Using kit: ", kit
    else: kit=None
     
    for at in al:
        if envij == None or iframe in envij:
            sys.stderr.write("Frame %d                              \r" %(iframe) )
            if verbose: qlog.write(at)

            # parses one of the structures, topping up atoms with isolated species if requested
            si = structure(alchem)
            si.parse(at, coff, nd, ld, gs, centerweight, nocenter, noatom, kit = kit)
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
            
    print >> sys.stderr, "Computing kernel matrix"
    # must fix the normalization of the similarity matrix!
    sys.stderr.write("Computing kernel normalization           \n")
    nrm = np.zeros(nf,float)
    for iframe in range (0, nf):           
        sii = structk(sl[iframe], sl[iframe], alchem, periodic, mode=kmode, fout=None)        
        nrm[iframe]=sii        
    
    if (nlandmark>0):
        print >> sys.stderr, "##### FARTHEST POINT SAMPLING ######"
        print >> sys.stderr, "Selecting",nlandmark,"Frames from",nf, "Frames"
        print >> sys.stderr, "##### Additional output files=sim-rect.dat, landmarks.xyz ######"
         
        m = nlandmark
        sim = np.zeros((m,m))
        sim_rect=np.zeros((m,nf))
        dist_list=[]
        landmarks=[]         
        
        iframe=randint(0,nf-1)  # picks a random frame
        iland=0
        landmarks.append(iframe)
        for jframe in range(nf):            
            sij = structk(sl[iframe], sl[jframe], alchem, periodic, mode=kmode, fout=None)
            sim_rect[iland][jframe]=sij/np.sqrt(nrm[iframe]*nrm[jframe])
            dist_list.append(np.sqrt(max(0,2-2*sij))) # use kernel metric
        #for x in sim_rect[iland][:]:
        #    fsim.write("%8.4e " %(x))
        #fsim.write("\n")
        maxd=0.0
        for iland in range(1,m):
            maxd=0.0
            for jframe in range(nf):
                if(dist_list[jframe]>maxd):
                    maxd=dist_list[jframe]
                    maxj=jframe
            landmarks.append(maxj)
            
            sys.stderr.write("Landmark %5d    maxd %f                          \r" % (iland, maxd))
            iframe=maxj
            for jframe in range(nf):                
                sij = structk(sl[iframe], sl[jframe], alchem, periodic, mode=kmode, fout=None)
                sim_rect[iland][jframe]=sij/np.sqrt(nrm[iframe]*nrm[jframe])
                dij = np.sqrt(max(0,2-2*sij))
                if(dij<dist_list[jframe]): dist_list[jframe]=dij
                
        for iland in range(0,m):
            for jland in range(0,m):
                sim[iland,jland] = sim_rect[iland, landmarks[jland] ]
        
        landxyz=quippy.AtomsWriter(prefix+".landmarks.xyz")         
        landlist=open(prefix+".landmarks","w")
        landlist.write("# Landmark list\n")
        for iland in landmarks: 
            landxyz.write(al[iland])
            landlist.write("%d\n" % (iland))
            
        fkernel = open(prefix+".landmarks.k", "w")  
        fkernel.write("# Kernel matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
        if (usekit): fkernel.write( " [ Using reference kit: %s\n" % (str(kit)) )
        else: fkernel.write("\n")
        for iframe in range(0,m):
            for x in sim[iframe][0:m]:
                fkernel.write("%16.8e " % (x))
            fkernel.write("\n")   
        fkernel.close()
        
        fkernel = open(prefix+".oos.k", "w")  
        for jframe in range(0,nf):
            for x in sim_rect[:,jframe]:
                fkernel.write("%16.8e " % (x))
            fkernel.write("\n")   
        fkernel.close()            
            
        if printsim:
            fsim = open(prefix+".landmarks.sim", "w")  
            fsim.write("# Distance matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
            if (usekit): fsim.write( " [ Using reference kit: %s\n" % (str(kit)) )
            else: fsim.write("\n")
            for iframe in range(0,m):
                for x in sim[iframe][0:m]:
                    fsim.write("%16.8e " % (np.sqrt(max(2-2*x,0))))
                fsim.write("\n")   
            fsim.close()
            
            fsim = open(prefix+".oos.sim", "w")  
            for jframe in range(0,nf):
                for x in sim_rect[:,jframe]:
                    fsim.write("%16.8e " % (np.sqrt(max(2-2*x,0))))
                fsim.write("\n")   
            fsim.close()
            
    else:
        sim=np.zeros((nf,nf))

        if (nprocs<=1):
            # no multiprocess
            for iframe in range (0, nf):
                sim[iframe,iframe]=1.0
                for jframe in range(0,iframe):
                    if verbose:
                        fij = open(prefix+".environ-"+str(iframe)+"-"+str(jframe)+".dat", "w")
                    else: fij = None
               
                    sij = structk(sl[iframe], sl[jframe], alchem, periodic, mode=kmode, fout=fij)          
                    sim[iframe][jframe]=sim[jframe][iframe]=sij/np.sqrt(nrm[iframe]*nrm[jframe])
                sys.stderr.write("Matrix row %d                           \r" % (iframe))
        else:      
            # multiple processors
            def dorow(irow, nf, sl, psim): 
                for jframe in range(0,irow):
                    sij = structk(sl[iframe], sl[jframe], alchem, periodic, mode=kmode)          
                    psim[irow*nf+jframe]=sij/np.sqrt(nrm[irow]*nrm[jframe])  
               
            proclist = []   
            psim = Array('d', nf*nf, lock=False)      
            for iframe in range (0, nf):
                sim[iframe,iframe]=1.0
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
     
        fkernel = open(prefix+".k", "w")  
        fkernel.write("# Kernel matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
        if (usekit): fkernel.write( " [ Using reference kit: %s\n" % (str(kit)) )
        else: fkernel.write("\n")
        for iframe in range(0,nf):
            for x in sim[iframe][0:nf]:
                fkernel.write("%16.8e " % (x))
            fkernel.write("\n")   
            
        if printsim:
            fsim = open(prefix+".sim", "w")  
            fsim.write("# Distance matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
            if (usekit): fsim.write( " [ Using reference kit: %s\n" % (str(kit)) )
            else: fsim.write("\n")
            for iframe in range(0,nf):
                for x in sim[iframe][0:nf]:
                    fsim.write("%16.8e " % (np.sqrt(max(2-2*x,0))))
                fsim.write("\n")   

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
      parser.add_argument("--usekit", action="store_true", help="Computes the least commond denominator of all structures and uses that as a reference state")      
      parser.add_argument("--kit", type=str, default="auto", help="Dictionary-style kit specification (e.g. --kit '{4:1,6:10}'")
      parser.add_argument("--kernel", type=str, default="match", help="Global kernel mode (e.g. --kernel average")      
      parser.add_argument("--distance", action="store_true", help="Also prints out similarity (as kernel distance)")
      parser.add_argument("--np", type=int, default='1', help="Use multiple processes to compute the kernel matrix")
      parser.add_argument("--ij", type=str, default='', help="Compute and print diagnostics for the environment similarity between frames i,j (e.g. --ij 3,4)")
      parser.add_argument("--nlandmarks", type=int,default='0',help="Use farthest point sampling method to select n landmarks. std output is n x n matrix. The n x N rectangular matrix is stored in file sim-rect.dat and the selected landmark frames are stored in landmarks.xyz file")     
      parser.add_argument("--prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
      
           
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
          
      if args.ij == "":
         envij=None
      else:
         envij=tuple(map(int,args.ij.split(",")))
               
      main(args.filename, nd=args.n, ld=args.l, coff=args.c, gs=args.g, mu=args.mu, centerweight=args.cw, periodic=args.periodic, usekit=args.usekit, kit=args.kit, kmode=args.kernel, noatom=noatom, nocenter=nocenter, nprocs=args.np, verbose=args.verbose, envij=envij, prefix=args.prefix, nlandmark=args.nlandmarks, printsim=args.distance)
