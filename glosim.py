#!/usr/bin/env python
# Computes the matrix of similarities between structures in a xyz file
# by first getting SOAP descriptors for all environments, finding the best
# match between environments using the Hungarian algorithm, and finally
# summing up the environment distances.
# Supports periodic systems, matching between structures with different
# atom number and kinds, and sports the infrastructure for introducing an
# alchemical similarity kernel to match different atomic species

import quippy
import sys, time, ast
from multiprocessing import Process, Value, Array
import argparse
from random import randint
from libmatch.environments import alchemy, environ
from libmatch.structures import structk, structure, structurelist
import os
import numpy as np
from copy import copy 
from time import ctime
from datetime import datetime
import gc
import cPickle as pickle

# tries really hard to flush any buffer to disk!
def flush(stream):
    stream.flush()
    os.fsync(stream)
   

def main(filename, nd, ld, coff, gs, mu, centerweight, periodic, kmode, nonorm, permanenteps, reggamma, nocenter, envsim, noatom, nprocs, verbose=False, envij=None, usekit=False, kit="auto", alchemyrules="none",prefix="",nlandmark=0, printsim=False,ref_xyz="",partialsim=False,lowmem=False,restartflag=False, kcsi=1.0):
    start_time = datetime.now()
    print >>sys.stderr, "          TIME:  ", ctime() ;
    print >>sys.stderr, "        ___  __    _____  ___  ____  __  __ ";
    print >>sys.stderr, "       / __)(  )  (  _  )/ __)(_  _)(  \/  )";
    print >>sys.stderr, "      ( (_-. )(__  )(_)( \__ \ _)(_  )    ( ";
    print >>sys.stderr, "       \___/(____)(_____)(___/(____)(_/\/\_)";
    print >>sys.stderr, "                                            ";
    print >>sys.stderr, "                                             ";
    filename = filename[0]
    # sets a few defaults
    if ( (ref_xyz !="") and envij != None): 
       print >> sys.stderr,"--ij option is not compitable with --refxyz "
       return 
    if (restartflag and  (not lowmem)):  
       print >> sys.stderr,"lowmem option has to be activated for restart flag"
       return 
    if lowmem :
       print >> sys.stderr,"!!!!!!!!!!!!!!!!!!! LOW MEMORY OPTION ACTIVATED !!!!!!!!!!!!!!!!!!!!!!!!"
       print >> sys.stderr,"Descriptors will be written in tmpstructures dir instead of storing in memory"
       print >> sys.stderr,"       Things are about to get SLOOOOOOOOOOOOOOOOOW !!! "
       print >> sys.stderr," ========================================================================="
       print >> sys.stderr,"   "
    if prefix=="": prefix=filename
    if prefix.endswith('.xyz'): prefix=prefix[:-4]

    # Reads input file using quippy
    print >> sys.stderr, "Reading input file", filename
    al = quippy.AtomsList(filename);
    print >> sys.stderr, len(al.n) , " Configurations Read"
    if (ref_xyz !=""):
        print >> sys.stderr, "================================REFERENCE XYZ FILE GIVEN=====================================\n",
        print >> sys.stderr, "Only Rectangular Matrix Containing Distances Between Two Sets of Input Files Will be Computed.\n",
        print >> sys.stderr, "Reading Referance xyz file: ", ref_xyz
        alref = quippy.AtomsList(ref_xyz);
        print >> sys.stderr, len(alref.n) , " Configurations Read"
    if restartflag:
       print >> sys.stderr, "Restart run: Reading SOAPs"
    else:
      print >> sys.stderr, "Computing SOAPs"

    # Sets alchemical matrix
    if (alchemyrules=="none"):
       alchem = alchemy(mu=mu)
    elif (alchemyrules=="read"):
       try:
            file = open("alchemy.pickle","rb")
       except IOError:
            raise IOError("alchemy.pickle file is not present")
       gc.disable()
       r=pickle.load(file)
       file.close()
       gc.enable()
       print >> sys.stderr, "Using Alchemy rules: ", r,"\n"
       alchem = alchemy(mu=mu,rules=r)
    else:
       r=alchemyrules.replace('"', '').strip()
       r=alchemyrules.replace("'", '').strip()
       r=ast.literal_eval(r)
       print >> sys.stderr, "Using Alchemy rules: ", r,"\n"
       alchem = alchemy(mu=mu,rules=r)
      
    if lowmem:
       sl = structurelist()
    else:
       sl=[]
    iframe = 0      
    if verbose:
        qlog=quippy.AtomsWriter("log.xyz")
        slog=open("log.soap", "w")

    # set flag for the envsim mode    
    fl_envsim = 0
    if np.sum(envsim)!=0:
      fl_envsim = 1
      nocenter = range(1,119) # generating the list of all atoms to ignore from SOAP algo
      nocenter.remove(envsim)

    # Determines reference kit     
    if usekit:
        if kit == "auto": 
            kit = {} 
            iframe=0
            # al is the structure containing all frames info 
            for at in al:
                if envij == None or iframe in envij: 
                    sp = {} 
                    for z in at.z:
                        # noatom corresponds to excluded atoms     
                        if z in noatom or z in nocenter: continue  
                        if z in sp: sp[z]+=1
                        else: 
                          sp[z] = 1
                    # select the composition of the largest molecule from the frames      
                    for s in sp:
                        if not s in kit:
                            kit[s]=sp[s]
                        else:
                            kit[s]=max(kit[s], sp[s])
                iframe+=1
            if ref_xyz != "" : # also looks into reference xyz if given
                for at in alref:
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
        else: # kit specified manually on command-line
            kit = ast.literal_eval(kit)
        iframe=0
        print >> sys.stderr, "Using kit: ", kit
    else: kit=None
    if (envij==None):
       nf = len(al.n) 
    else:
       nf=len(envij)
 
    nf_ref=nf 
    iframe=0 
    icount=0
    nrm = np.zeros(nf,float)
    
    # get the number of selected atoms and initialize the output matrix .env.sim
    if fl_envsim: # works only for one atom
      nenv = kit[envsim]
      simenv=np.zeros((nenv*nf,nenv*nf))
      print nenv

    for at in al:
        if envij == None or iframe in envij:
            if verbose: qlog.write(at)

            # parses one of the structures, topping up atoms with isolated species if requested
            if restartflag: #if we are restarting from previously calculated soap files
              if sl.exists(iframe):
                 sys.stderr.write ("Reading SOAP for frame %d       \r " %(iframe))
                 si = sl[iframe]
              else: 
                print >> sys.stderr, "\n Could not Find file for frame: ", iframe ,"\n" # At the moment the way it is implemented if it does not exist and recalculate
                return                                                                  #  it will store the missing frame as first frame file.
             
            else:
                si = structure(alchem)
                sys.stderr.write("Frame %d                              \r" %(iframe) )
                si.parse(at, coff, nd, ld, gs, centerweight, nocenter, noatom, kit = kit)       
                
                # discard the list of all environments if they are not needed for this calculation
                if kmode == "fastavg" and not verbose: 
                    si.env = []
                sl.append(si)
            if verbose:
                slog.write("# Frame %d \n" % (iframe))
                fii = open(prefix+".environ-"+str(iframe)+"-"+str(iframe)+".dat", "w")
                for sp, el in si.env.iteritems():
                    ik=0
                    for ii in el:
                        slog.write("# Species %d Environment %d \n" % (sp, ik))
                        ik+=1
                        for p, s in ii.soaps.iteritems():
                            slog.write("%d %d   " % p)
                            for sj in s:
                                slog.write("%8.4e " %(sj))                        
                            slog.write("\n")
            else:
               fii = None
            sii,senvii = structk(si, si, alchem, periodic, mode=kmode, fout=fii, peps=permanenteps, gamma=reggamma, csi=kcsi)        
             
            if fl_envsim:
              simenv[icount*nenv:icount*nenv+nenv,icount*nenv:icount*nenv+nenv]=senvii
            nrm[icount]=sii

            icount +=1    
        iframe +=1; 
    np.savetxt(prefix+".norm.dat",nrm)  
    print >> sys.stderr, "Computing kernel matrix"
    # must fix the normalization of the similarity matrix!
#    sys.stderr.write("Computing kernel normalization           \n")
#    nrm = np.zeros(nf,float)
#    for iframe in range (0, nf):           
#        if verbose:
#            fii = open(prefix+".environ-"+str(iframe)+"-"+str(iframe)+".dat", "w")
#        else:
#            fii = None
#        sii = structk(sl[iframe], sl[iframe], alchem, periodic, mode=kmode, fout=fii, peps=permanenteps, gamma=reggamma)        
#        nrm[iframe]=sii        

    
    if (ref_xyz !=""): # If ref landmarks are given and rectangular matrix is the only desired output             
        print >> sys.stderr, "Computing SOAPs"
        # sets alchemical matrix
        alchem = alchemy(mu=mu)
        if (lowmem):
          sl_ref = structurelist(basedir="./tmprefstructures/")
        else:
          sl_ref=[]
        iframe = 0
        if verbose:
            qlogref=quippy.AtomsWriter("log_ref.xyz")
            slogref=open("log_ref.soap", "w")

        # determines reference kit    
        if usekit:
            if kit == "auto":
                kit = {}
                iframe=0
                for at in alref:
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
        nf_ref = len(alref.n)
        nrm_ref = np.zeros(nf_ref,float)
        iframe=0
        for at in alref:
            if envij == None or iframe in envij: 
                sys.stderr.write("Frame %d                              \r" %(iframe) )
                if verbose: qlogref.write(at)
                # parses one of the structures, topping up atoms with isolated species if requested
                if restartflag: #if we are restarting from previously calculated soap files
                    if sl_ref.exists(iframe):
                        sys.stderr.write ("Reading SOAP for reference frame %d       \r " %(iframe))
                        si = sl_ref[iframe]
                    else: 
                        print >> sys.stderr, "\n Could not Find file for reference frame: ", iframe ,"\n" # At the moment the way it is implemented if it does not exist and recalculate
                        return                                                                 #  it will store the missing frame as first frame file.
                else:
                    si = structure(alchem)
                    si.parse(at, coff, nd, ld, gs, centerweight, nocenter, noatom, kit = kit)
                    if kmode == "fastavg" and not verbose: 
                        si.env = []
                    sl_ref.append(si)
                if verbose:
                    slog.write("# Frame %d \n" % (iframe))
                    for sp, el in si.env.iteritems():
                        ik=0
                        for ii in el:
                            slogref.write("# Species %d Environment %d \n" % (sp, ik))
                            ik+=1
                            for p, s in ii.soaps.iteritems():
                                slogref.write("%d %d   " % p)
                                for sj in s:
                                    slogref.write("%8.4e " %(sj))
                                slogref.write("\n")
                sii,senvii = structk(si, si, alchem, periodic, mode=kmode, fout=None, peps = permanenteps, gamma=reggamma, csi=kcsi)        
                if fl_envsim:
                      simenv[iframe*nenv:iframe*nenv+nenv,iframe*nenv:iframe*nenv+nenv]=senvii
                nrm_ref[iframe]=sii        
                iframe +=1;

        print >> sys.stderr, "Computing kernel matrix"
        # must fix the normalization of the similarity matrix!
     #   sys.stderr.write("Computing kernel normalization           \n")
#        for iframe in range (0, nf_ref):           

        sim = np.zeros((nf,nf_ref))
        sys.stderr.write("Computing Similarity Matrix           \n")
        if (partialsim): 
          pfkernel=open(prefix+"_rect.k.partial","w")
          pfkernel.write("# OOS Kernel matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s " % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
          if (nonorm):pfkernel.write( " Un-normalized kernels " )        
          if (usekit):pfkernel.write( " Using reference kit: %s " % (str(kit)) )
          if (alchemyrules!="none"):pfkernel.write( " Using alchemy rules: %s " % (alchemyrules) )
          if (kmode=="rematch"): pfkernel.write( " Regularized parameter: %f " % (reggamma) )
          pfkernel.write("\n")
        
        if (nprocs<=1):
         for iframe in range(nf):
           sys.stderr.write("Matrix row %d                           \r" % (iframe))
           sli=sl[iframe]
           for jframe in range(nf_ref):
             sij,senvij = structk(sli, sl_ref[jframe], alchem, periodic, mode=kmode, fout=None, peps = permanenteps, gamma=reggamma, csi=kcsi)
             if not nonorm: sij/=np.sqrt(nrm[iframe]*nrm_ref[jframe])
             sim[iframe][jframe]=sij
         if(partialsim):
              for x in sim[iframe,:]:
                pfkernel.write("%20.12e " % (x))
              pfkernel.write("\n")
              flush(pfkernel)
        else:    
            def dochunk(psim, iproc, nprocs):
                for iframe in range (iproc,nf,nprocs):
                    sys.stderr.write("Matrix row %d                           \r" % (iframe))           
                    sli=sl[iframe]  
                    for jframe in range(nf_ref):
                        sij,senvij = structk(sli, sl_ref[jframe], alchem, periodic, mode=kmode, fout=None, peps = permanenteps, gamma=reggamma, csi=kcsi)
                        if not nonorm: sij/=np.sqrt(nrm[iframe]*nrm_ref[jframe])
                        psim[iframe*nf_ref+jframe]=sij
            psim = Array('d', nf_ref*nf, lock=False)
            proclist=[]
            for iproc in range (nprocs):    
                sp = Process(target=dochunk, name="doframe proc", kwargs={"nprocs":nprocs,"iproc":iproc, "psim": psim})
                proclist.append(sp)
                sp.start()
            for ip in proclist:
                   while ip.is_alive(): ip.join(0.01)
                   if ip.exitcode != 0 :
                     raise ValueError("Invalid exit status for one of the child processes!")
            for iframe in range(nf):
                for jframe in range(0,nf_ref):
                    sim[iframe,jframe]=psim[jframe+iframe*nf_ref]
            
        if(partialsim):pfkernel.close()
        fkernel = open(prefix+"_rect.k", "w")  
        fkernel.write("# OOS Kernel matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s " % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
        if (nonorm):fkernel.write( " Un-normalized kernels " )        
        if (usekit):fkernel.write( " Using reference kit: %s " % (str(kit)) )
        if (alchemyrules!="none"):fkernel.write( " Using alchemy rules: %s " % (alchemyrules) )
        if (kmode=="rematch"): fkernel.write( " Regularized parameter: %f " % (reggamma) )
        fkernel.write("\n")
        for iframe in range(0,nf):
            for x in sim[iframe][0:nf_ref]:
                fkernel.write("%20.12e " % (x))
            fkernel.write("\n")   
            
        if printsim:
            fsim = open(prefix+"_rect.sim", "w")  
            fsim.write("# OOS Distance matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s " % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
            if (usekit):fsim.write( " Using reference kit: %s " % (str(kit)) )
            if (alchemyrules!="none"):fsim.write( " Using alchemy rules: %s " % (alchemyrules) )
            if (kmode=="rematch"): fsim.write( " Regularized parameter: %f " % (reggamma) )
            fsim.write("\n")
            for iframe in range(0,nf):
                for jframe in range(0,nf_ref):
               # for x in sim[iframe][0:nf_ref]:
                    if nonorm:
                      fsim.write("%20.12e " % (np.sqrt(abs(nrm[iframe]+nrm_ref[jframe]-2*sim[iframe][jframe]))))
                    else:
                      fsim.write("%20.12e " % (np.sqrt(abs(2-2*sim[iframe][jframe]))))
                fsim.write("\n")   

#=============================================================================
            
   
    elif (nlandmark>0): # we have just one input but we compute landmarks
        print >> sys.stderr, "##### FARTHEST POINT SAMPLING ######"
        print >> sys.stderr, "Selecting",nlandmark,"Frames from",nf, "Frames"
        print >> sys.stderr, "####################################"
        landlist=open(prefix+".landmarks","w")
        landxyz=quippy.AtomsWriter(prefix+".landmarks.xyz")   
        if (partialsim): 
          pfkernel=open(prefix+"oos.k.partial","w")
          pfkernel.write("# Kernel matrix for OOS from %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
          if (nonorm):pfkernel.write( " Un-normalized kernels " )        
          if (usekit):pfkernel.write( " Using reference kit: %s " % (str(kit)) )
          if (alchemyrules!="none"):pfkernel.write( " Using alchemy rules: %s " % (alchemyrules) )
          if (kmode=="rematch"): pfkernel.write( " Regularized parameter: %f " % (reggamma) )
          pfkernel.write("\n")
 
        m = nlandmark
        sim = np.zeros((m,m))
        sim_rect=np.zeros((m,nf))
        dist_list = np.zeros(nf, float)
        landmarks=[] 
        restartland=False
        if (restartflag and os.path.isfile("restart.k") and os.path.isfile("restart.landmarks")):restartland=True
        if (restartland):
            try:
                k_in=np.loadtxt("restart.k")
            except:
                k_in=np.genfromtxt("restart.k",skip_footer=1)
                print >> sys.stderr,"Incomplete last row. Ommiting last landmark"
            nlandmark_in=len(k_in)
            landmarks_in=np.loadtxt("restart.landmarks",dtype=int)
            if(len(k_in[0]) != nf):
              print >> sys.stderr,"Inconsistent frame numbers"
              return
            print >> sys.stderr,"Read Previously found ", nlandmark_in, " landmarks"
            iframe=landmarks_in[0]
            iland=0
            landmarks.append(iframe)
            for jframe in range(nf):            
                sim_rect[iland][jframe]=k_in[iland][jframe]
                if nonorm:
                  dist_list[jframe]=np.sqrt(abs(nrm[iframe]+nrm[jframe]-2*k_in[iland][jframe]))
                else:
                  dist_list[jframe] = np.sqrt(abs(2.0-2.0*k_in[iland][jframe])) # ??? use kernel metric
            landlist.write("# Landmark list\n")
            landlist.write("%d\n" % (landmarks[0]))
            landxyz.write(al[landmarks[0]])        
            flush(landlist)
            if(partialsim):
                for x in sim_rect[iland,:]:
                    pfkernel.write("%20.12e " % (x))
                pfkernel.write("\n")
                flush(pfkernel)
            
            for iland in range(1,nlandmark_in):
                maxd=0.0
                maxj=-1
                for jframe in range(nf):
                    if(dist_list[jframe]>maxd):
                        maxd=dist_list[jframe]
                        maxj=jframe
                landmarks.append(maxj)
                if (landmarks[iland]!=landmarks_in[iland]):
                    print >> sys.stderr,"ERROR !"
                    return
                landxyz.write(al[maxj])
                landlist.write("%d\n" % (maxj))
                flush(landlist)
                sys.stderr.write("Landmark %5d    maxd %f                          \r" % (iland, maxd))
                iframe = maxj
                for jframe in range(nf):                
                    sim_rect[iland][jframe]=k_in[iland][jframe]
                    if nonorm:
                      dij= np.sqrt(abs(nrm[iframe]+nrm[jframe]-2.0*k_in[iland][jframe])) # use kernel metric
                    else:
                      dij = np.sqrt(abs(2.0-2.0*k_in[iland][jframe]))
                    if(dij<dist_list[jframe]): dist_list[jframe]=dij
                if(partialsim):
                  for x in sim_rect[iland,:]:
                     pfkernel.write("%20.12e " % (x))
                  pfkernel.write("\n")
                  flush(pfkernel)
            nlandstart=nlandmark_in
        else:        
#            iframe=0       
            iframe=randint(0,nf-1)  # picks a random frame
            iland=0
            landmarks.append(iframe)
            sli=sl[iframe]
            for jframe in range(nf):            
                sij,senvij = structk(sli, sl[jframe], alchem, periodic, mode=kmode, fout=None,peps = permanenteps, gamma=reggamma, csi=kcsi)
                if not nonorm: 
                       sij/=np.sqrt(nrm[iframe]*nrm[jframe])
                       dist_list[jframe] = np.sqrt(abs(2.0-2.0*sij)) # use kernel metric
                else:
                       dist_list[jframe]= np.sqrt(abs(nrm[iframe]+nrm[jframe]-2.0*sij)) # use kernel metric
                sim_rect[iland][jframe]=sij
            #for x in sim_rect[iland][:]:
            #    fsim.write("%8.4e " %(x))
            #fsim.write("\n")
            landlist.write("# Landmark list\n")
            landlist.write("%d\n" % (landmarks[0]))
            landxyz.write(al[landmarks[0]])        
            flush(landlist)
                        
            if(partialsim):
                for x in sim_rect[iland,:]:
                    pfkernel.write("%20.12e " % (x))
                pfkernel.write("\n")
                flush(pfkernel)
            nlandstart=1
            
        for iland in range(nlandstart,m):
            maxd=0.0
            maxj=-1
            for jframe in range(nf):
                if(dist_list[jframe]>maxd):
                    maxd=dist_list[jframe]
                    maxj=jframe
            landmarks.append(maxj)
            landxyz.write(al[maxj])
            landlist.write("%d\n" % (maxj))
            flush(landlist)
                        
            
            sys.stderr.write("Landmark %5d    maxd %f                          \r" % (iland, maxd))
            iframe = maxj
            sli = sl[iframe]
            if (nprocs<=1):
              for jframe in range(nf):                
                sij,senvij = structk(sli, sl[jframe], alchem, periodic, mode=kmode, fout=None, peps = permanenteps, gamma=reggamma, csi=kcsi)
                # normalize the kernel
                if not nonorm: 
                       sij/=np.sqrt(nrm[iframe]*nrm[jframe])
                       dij = np.sqrt(abs(2-2*sij)) # use kernel metric
                else:
                       dij= np.sqrt(abs(nrm[iframe]+nrm[jframe]-2.0*sij)) # use kernel metric
                sim_rect[iland][jframe]=sij
              #  if not nonorm: sij/=np.sqrt(nrm[iframe]*nrm[jframe])
              #  sim_rect[iland][jframe]=sij
              #  dij = np.sqrt(max(0,2-2*sij))
                if(dij<dist_list[jframe]): dist_list[jframe]=dij
              if(partialsim):
                  for x in sim_rect[iland,:]:
                     pfkernel.write("%20.12e " % (x))
                  pfkernel.write("\n")
                  flush(pfkernel)
            else:
		      # multiple processors
              def docol(pdist, psim, iframe, nf, nproc, iproc):
                  for jframe in range(iproc, nf, nproc):                                
                     sij,senvij = structk(sli, sl[jframe], alchem, periodic, mode=kmode, fout=None, peps = permanenteps, gamma=reggamma, csi=kcsi)
                     if not nonorm: 
                            sij/=np.sqrt(nrm[iframe]*nrm[jframe])
                            dij = np.sqrt(abs(2-2*sij)) # use kernel metric
                     else:
                            dij= np.sqrt(abs(nrm[iframe]+nrm[jframe]-2.0*sij)) # use kernel metric
                     psim[jframe]=sij
                   #  if not nonorm: sij/=np.sqrt(nrm[iframe]*nrm[jframe])                
                   #  psim[jframe]=sij
                   #  dij= np.sqrt(max(0,2-2*sij))
                     if (dij < pdist[jframe]): pdist[jframe]=dij
               #   print iframe,jframe
               
              proclist = []   
              pdist = Array('d', nf, lock=False)
              psim = Array('d', nf, lock=False)
              pdist[:] = dist_list[:]

              jframe_split_list=[]
              for iproc in range(nprocs): 
                 while(len(proclist)>=nprocs):
                    for ip in proclist:
                        if not ip.is_alive(): proclist.remove(ip)            
                        time.sleep(0.01)
                 sp = Process(target=docol, name="docol proc", kwargs={"iproc":iproc,"pdist":pdist,"psim":psim,"iframe":iframe,"nproc":nprocs,"nf":nf})  
                 proclist.append(sp)
                 sp.start()
                 sys.stderr.write("Landmark %d, Process %d/%d          \r" % (iland, iproc, len(proclist)))
             
                 # waits for all threads to finish
              for ip in proclist:
                 while ip.is_alive(): ip.join(0.1)  
                 if ip.exitcode != 0 :
                     raise ValueError("Invalid exit status for one of the child processes!")
         
                # copies from the shared memory array to Sim.
              for jframe in range(nf):
                 dist_list[jframe]=pdist[jframe]
                 sim_rect[iland][jframe]=psim[jframe]    
              if(partialsim):
                for x in sim_rect[iland,:]:
                   pfkernel.write("%20.12e " % (x))
                pfkernel.write("\n")
                flush(pfkernel)
              #========================================================================        
              
        if(partialsim):pfkernel.close()
        for iland in range(0,m):
            for jland in range(0,m):
                sim[iland,jland] = sim_rect[iland, landmarks[jland] ]
        
        fkernel = open(prefix+".landmarks.k", "w")  
        fkernel.write("# Kernel matrix for landmarks from  %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s " % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
        if (nonorm):fkernel.write( " Un-normalized kernels " )        
        if (usekit):fkernel.write( " Using reference kit: %s " % (str(kit)) )
        if (alchemyrules!="none"):fkernel.write( " Using alchemy rules: %s " % (alchemyrules) )
        if (kmode=="rematch"): fkernel.write( " Regularized parameter: %f " % (reggamma) )
        fkernel.write("\n")
        for iframe in range(0,m):
            for x in sim[iframe][0:m]:
                fkernel.write("%20.12e " % (x))
            fkernel.write("\n")   
        fkernel.close()
        
        fkernel = open(prefix+".oos.k", "w")  
        fkernel.write("# Kernel matrix for OOS from %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
        if (nonorm):fkernel.write( " Un-normalized kernels " )        
        if (usekit):fkernel.write( " Using reference kit: %s " % (str(kit)) )
        if (alchemyrules!="none"):fkernel.write( " Using alchemy rules: %s " % (alchemyrules) )
        if (kmode=="rematch"): fkernel.write( " Regularized parameter: %f " % (reggamma) )
        fkernel.write("\n")
        for jframe in range(0,nf):
            for x in sim_rect[:,jframe]:
                fkernel.write("%20.12e " % (x))
            fkernel.write("\n")   
        fkernel.close()            
            
        if printsim:
            fsim = open(prefix+".landmarks.sim", "w")  
            fsim.write("# Distance matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)))
            if (usekit):fsim.write( " Using reference kit: %s " % (str(kit)) )
            if (alchemyrules!="none"):fsim.write( " Using alchemy rules: %s " % (alchemyrules) )
            if (kmode=="rematch"): fsim.write( " Regularized parameter: %f " % (reggamma) )
            fsim.write("\n")
            for iframe in range(0,m):
                for jframe in range(0,m):
               # for x in sim[iframe][0:m]:
                    fsim.write("%16.8e " % (np.sqrt(abs(sim[iframe][iframe]+sim[jframe][jframe]-2*sim[iframe][jframe]))))
                fsim.write("\n")   
            fsim.close()
            
            fsim = open(prefix+".oos.sim", "w")  
            fsim.write("# OOS Distance matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s" % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)))
            if (usekit):fsim.write( " Using reference kit: %s " % (str(kit)) )
            if (alchemyrules!="none"):fsim.write( " Using alchemy rules: %s " % (alchemyrules) )
            if (kmode=="rematch"): fsim.write( " Regularized parameter: %f " % (reggamma) )
            fsim.write("\n")
            for jframe in range(0,nf):
                for iframe in range(0,m):
                    if nonorm:
               # for x in sim_rect[:,iframe]:
                       fsim.write("%16.8e " % (np.sqrt(abs(nrm[jframe]+nrm[landmarks[iframe]]-2*sim_rect[iframe][jframe]))))
                    else:
                       fsim.write("%16.8e " % (np.sqrt(abs(2-2*sim_rect[iframe][jframe]))))
                       
                fsim.write("\n")   
            fsim.close()
#===============================================================================================================================================


            
    else:  # standard case (one input, compute everything
        sim=np.zeros((nf,nf))
        
        if (partialsim): 
          pfkernel=open(prefix+".k.partial","w")
          pfkernel.write("# Kernel matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s " % (filename, coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
          if (nonorm):pfkernel.write( " Un-normalized kernels " )        
          if (usekit):pfkernel.write( " Using reference kit: %s " % (str(kit)) )
          if (alchemyrules!="none"):pfkernel.write( " Using alchemy rules: %s " % (alchemyrules) )
          if (kmode=="rematch"): pfkernel.write( " Regularized parameter: %f " % (reggamma) )
          pfkernel.write("\n")
        if (nprocs<=1):
            # no multiprocess
            for iframe in range (0, nf):
                if nonorm: sim[iframe,iframe]=nrm[iframe]
                else: sim[iframe,iframe]=1.0
                
                sli = sl[iframe]
                for jframe in range(0,iframe):
                    if verbose:
                        fij = open(prefix+".environ-"+str(iframe)+"-"+str(jframe)+".dat", "w")
                    else: fij = None
                    # if periodic: sys.stderr.write("comparing %3d, atoms cell with  %3d atoms cell: lcm: %3d \r" % (sl[iframe].nenv, sl[jframe].nenv, lcm(sl[iframe].nenv,sl[jframe].nenv))) 
                    sij,senvij = structk(sli, sl[jframe], alchem, periodic, mode=kmode, fout=fij, peps = permanenteps, gamma=reggamma, csi=kcsi)
                    if not nonorm: sij/=np.sqrt(nrm[iframe]*nrm[jframe])
                    sim[iframe][jframe]=sim[jframe][iframe]=sij
                    if fl_envsim:
                      simenv[iframe*nenv:iframe*nenv+nenv,jframe*nenv:jframe*nenv+nenv]=senvij
                      simenv[jframe*nenv:jframe*nenv+nenv,iframe*nenv:iframe*nenv+nenv]=senvij.T
                sys.stderr.write("Matrix row %d                           \r" % (iframe))
                if(partialsim):
                  for x in sim[iframe,0:iframe]:
                    pfkernel.write("%20.12e " % (x))
                  pfkernel.write("\n")
                  flush(pfkernel)
        else:      
            # multiple processors
            psim = Array('d', nf*nf, lock=False)   
            def dochunk(psim, iproc, nprocs):
                for iframe in range(iproc, nf, nprocs):
                    sli=sl[iframe] 
                    sys.stderr.write("Matrix row %d %d    \r" % (iproc, iframe))
                    for jframe in range(0,iframe):
                        sij,senvij = structk(sli, sl[jframe], alchem, periodic, mode=kmode, peps = permanenteps, gamma=reggamma, csi=kcsi)
                        if not nonorm: sij/=np.sqrt(nrm[iframe]*nrm[jframe])
                        psim[jframe+iframe*nf]=sij
            proclist = []   
            for iproc in range(nprocs):
                sp = Process(target=dochunk, name="dochunk proc", kwargs={"nprocs":nprocs,"iproc":iproc, "psim": psim})  
                proclist.append(sp)
                sp.start()
            for ip in proclist:
                while ip.is_alive(): ip.join(0.01)  
                if ip.exitcode != 0 :
                    raise ValueError("Invalid exit status for one of the child processes!")
                       
            for iframe in range(nf):
                if nonorm: sim[iframe,iframe]=nrm[iframe]
                else: sim[iframe,iframe]=1.0 
                for jframe in range(0,iframe):
                    sim[iframe,jframe]=sim[jframe,iframe]=psim[jframe+iframe*nf]
            #~ for iframe in range (nf):   
              #~ sli=sl[iframe] 
              #~ def dorow(irow,nf,nprocs,iproc, psim):
                #~ for jframe in range(iproc,nf,nprocs):
                    #~ sij,senvij = structk(sli, sl[jframe], alchem, periodic, mode=kmode, peps = permanenteps, gamma=reggamma, csi=kcsi)
                    #~ if not nonorm: sij/=np.sqrt(nrm[irow]*nrm[jframe])  
                    #~ psim[jframe]=sij
              #~ proclist = []   
              #~ psim = Array('d', nf, lock=False)      
              #~ if nonorm: sim[iframe,iframe] = nrm[iframe]
              #~ else: sim[iframe,iframe] = 1
              
              #~ for iproc in range (nprocs):
                #~ while(len(proclist)>=nprocs):
                    #~ for ip in proclist:
                        #~ if not ip.is_alive(): proclist.remove(ip)            
                        #~ time.sleep(0.01)
                #~ sp = Process(target=dorow, name="doframe proc", kwargs={"irow":iframe,"nf":iframe, "nprocs":nprocs,"iproc":iproc, "psim": psim})  
                #~ proclist.append(sp)
                #~ sp.start()
                #~ sys.stderr.write("Matrix row %d, %d active processes     \r" % (iframe, len(proclist)))
              #~ for ip in proclist:
                   #~ while ip.is_alive(): ip.join(0.01)  
                   #~ if ip.exitcode != 0 :
                     #~ raise ValueError("Invalid exit status for one of the child processes!")
              #~ for jframe in range(0,iframe):
                    #~ sim[iframe,jframe]=sim[jframe,iframe]=psim[jframe]
              #~ if(partialsim):
                  #~ for x in sim[iframe,0:iframe]:
                    #~ pfkernel.write("%20.12e " % (x))
                  #~ pfkernel.write("\n")
                  #~ flush(pfkernel)
            
            # waits for all threads to finish
            #for ip in proclist:
            #    while ip.is_alive(): ip.join(0.1)  
         
            # copies from the shared memory array to Sim.
            #for iframe in range (0, nf):      
             #   for jframe in range(0,iframe):
            #        sim[iframe,jframe]=sim[jframe,iframe]=psim[iframe*nf+jframe]
        if(partialsim):pfkernel.close()
        fkernel = open(prefix+".k", "w")  
        fkernel.write("# Kernel matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  SOAP-csi: %f   Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s " % (filename, coff, nd, ld, gs, mu, centerweight, kcsi, periodic, kmode, str(noatom), str(nocenter)) )
        if (nonorm):fkernel.write( " Un-normalized kernels " )        
        if (usekit):fkernel.write( " Using reference kit: %s " % (str(kit)) )
        if (alchemyrules!="none"):fkernel.write( " Using alchemy rules: %s " % (alchemyrules) )
        if (kmode=="rematch"): fkernel.write( " Regularized parameter: %f " % (reggamma) )
        fkernel.write("\n")
        for iframe in range(0,nf):
            for x in sim[iframe][0:nf]:
                fkernel.write("%16.8e " % (x))
            fkernel.write("\n")   
            
        if printsim:
            fsim = open(prefix+".sim", "w")  
            fsim.write("# Distance matrix for %s. Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  SOAP-csi: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s " % (filename, coff, nd, ld, gs, mu, centerweight, kcsi, periodic, kmode, str(noatom), str(nocenter)) )
            if (usekit):fsim.write( " Using reference kit: %s " % (str(kit)) )
            if (alchemyrules!="none"):fsim.write( " Using alchemy rules: %s " % (alchemyrules) )
            if (kmode=="rematch"): fsim.write( " Regularized parameter: %f " % (reggamma) )
            fsim.write("\n")
            for iframe in range(0,nf):
                for jframe in range(0,nf):
               # for x in sim[iframe][0:nf]:
                    fsim.write("%16.8e " % (np.sqrt(abs(sim[iframe][iframe]+sim[jframe][jframe]-2*sim[iframe][jframe]))))
                fsim.write("\n") 

        if fl_envsim:
            atomicmap = {1:"H",2:"He",6:"C",7:"N",8:"O",9:"F"}
            fsimenv = open(prefix+".env."+atomicmap[envsim]+"_"+str(nenv)+".sim", "w")  
            fsimenv.write("# Environment Distance matrix for %s. species: %s Cutoff: %f  Nmax: %d  Lmax: %d  Atoms-sigma: %f  Mu: %f  Central-weight: %f  Periodic: %s  Kernel: %s  Ignored_Z: %s  Ignored_Centers_Z: %s " % (filename, atomicmap[envsim], coff, nd, ld, gs, mu, centerweight, periodic, kmode, str(noatom), str(nocenter)) )
            if (usekit):fsimenv.write( " Using reference kit: %s " % (str(kit)) )
            if (alchemyrules!="none"):fsimenv.write( " Using alchemy rules: %s " % (alchemyrules) )
            if (kmode=="rematch"): fsimenv.write( " Regularized parameter: %f " % (reggamma) )
            fsimenv.write("\n")
            for iframe in range(0,nf*nenv):
                for jframe in range(0,nf*nenv):
              #  for x in simenv[iframe][0:nf*nenv]:
                    fsimenv.write("%16.8e " % (np.sqrt(abs(simenv[iframe][iframe]+simenv[jframe][jframe]-2*simenv[iframe][jframe])))) # output distance matrix
                fsimenv.write("\n")  
            fsimenv.close()
    sys.stderr.write("\n ============= Glosim Ended Successfully ============== \n") 
    print >>sys.stderr, "          TIME:  ", ctime() ;
    end_time = datetime.now()
    print>>sys.stderr,('          Duration: {}'.format(end_time - start_time))

if __name__ == '__main__':
      parser = argparse.ArgumentParser(description="""Computes the similarity matrix between a set of atomic structures 
                           based on SOAP descriptors and an optimal assignment of local environments.""")
                           
      parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
      parser.add_argument("--periodic", action="store_true", help="Matches structures with different atom numbers by replicating the environments")
      parser.add_argument("--exclude", type=str, default="", help="Comma-separated list of atom Z to be removed from the input structures (e.g. --exclude 96,101)")
      parser.add_argument("--nocenter", type=str, default="", help="Comma-separated list of atom Z to be ignored as environment centers (e.g. --nocenter 1,2,4)")
      parser.add_argument("--envsim", type=int, default=0, help="Select the only species (ex: H->1, C->6,...) to perform the glosim algorithm on, i.e. coresponds to set all the other species in nocenter mode")
      parser.add_argument("--verbose",  action="store_true", help="Writes out diagnostics for the optimal match assignment of each pair of environments")   
      parser.add_argument("-n", type=int, default='8', help="Number of radial functions for the descriptor")
      parser.add_argument("-l", type=int, default='6', help="Maximum number of angular functions for the descriptor")
      parser.add_argument("-c", type=float, default='5.0', help="Radial cutoff")
      parser.add_argument("-g", type=float, default='0.5', help="Atom Gaussian sigma")
      parser.add_argument("-cw", type=float, default='1.0', help="Center atom weight")
      parser.add_argument("--mu", type=float, default='0.0', help="Extra penalty for comparing to missing atoms")
      parser.add_argument("--usekit", action="store_true", help="Computes the least commond denominator of all structures and uses that as a reference state")      
      parser.add_argument("--gamma", type=float, default="1.0", help="Regularization for entropy-smoothed best-match kernel")
      parser.add_argument("--kcsi", type=float, default="1.0", help="Exponent for the atomic SOAP kernel")
      parser.add_argument("--kit", type=str, default="auto", help="Dictionary-style kit specification (e.g. --kit '{4:1,6:10}'")
      parser.add_argument("--alchemy_rules", type=str, default="none", help='Dictionary-style rule specification in quote (e.g. --alchemy_rules "{(6,7):1,(6,8):1}"')
      parser.add_argument("--kernel", type=str, default="match", help="Global kernel mode (e.g. --kernel average / match / rematch / species / fastavg ")      
      parser.add_argument("--nonorm",  action="store_true", help="Does not normalize structural kernels")   
      parser.add_argument("--permanenteps", type=float, default="0.0", help="Tolerance level for approximate permanent (e.g. --permanenteps 1e-4")     
      parser.add_argument("--distance", action="store_true", help="Also prints out similarity (as kernel distance)")
      parser.add_argument("--np", type=int, default='1', help="Use multiple processes to compute the kernel matrix")
      parser.add_argument("--ij", type=str, default='', help="Compute and print diagnostics for the environment similarity between frames i,j (e.g. --ij 3,4)")
      parser.add_argument("--nlandmarks", type=int,default='0',help="Use farthest point sampling method to select n landmarks. This will also generate the OOS matrix for rest of the frames")     
      parser.add_argument("--refxyz", type=str, default='', help="ref xyz file if you want to compute the rectangular matrix contaning distances from ref configurations")
      parser.add_argument("--prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
      parser.add_argument("--livek",  action="store_true", help="Writes out diagnostics for the optimal match assignment of each pair of environments")   
      parser.add_argument("--lowmem",  action="store_true", help="Writes out diagnostics for the optimal match assignment of each pair of environments")         
      parser.add_argument("--restart",  action="store_true", help="Writes out diagnostics for the optimal match assignment of each pair of environments")   
      
           
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
                  
      main(args.filename, nd=args.n, ld=args.l, coff=args.c, gs=args.g, mu=args.mu, centerweight=args.cw, periodic=args.periodic, usekit=args.usekit, kit=args.kit,alchemyrules=args.alchemy_rules, kmode=args.kernel, nonorm=args.nonorm, permanenteps=args.permanenteps, reggamma=args.gamma, noatom=noatom, nocenter=nocenter, envsim=args.envsim, nprocs=args.np, verbose=args.verbose, envij=envij, prefix=args.prefix, nlandmark=args.nlandmarks, printsim=args.distance,ref_xyz=args.refxyz,partialsim=args.livek,lowmem=args.lowmem,restartflag=args.restart, kcsi=args.kcsi)
