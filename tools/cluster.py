#!/usr/bin/python
import argparse
import numpy as np
import sys
import numpy as np
import scipy.cluster.hierarchy as sc
import itertools
try:
  from matplotlib import pyplot as plt
except:
  print "matplotlib is not available. You will not be able to plot"

from collections import Counter

def main(distmatrixfile,nclust,mode='average',proplist='',plot=False,calc_sd=False):
   if proplist!='': prop=np.loadtxt(proplist)
   sim=np.loadtxt(distmatrixfile)
   Z=sc.linkage(sim,mode)
   n=len(sim)
#   acceleration = np.diff(Z[:,2], 2)  # 2nd derivative of the distances
#   nl=len(acceleration)
#   for i in range(nl-1,nl-15,-1):
#      if (acceleration[i]*acceleration[i-1])<0 : elbow=i
   print "mean+std, cutoffdist", np.sqrt(np.var(Z[:,2]))+np.mean(Z[:,2]),(Z[n-nclust,2])
   maxdist=(Z[n-nclust,2])*0.75
   cdist=Z[:,2] 
   cdist.sort()
   np.savetxt('dist.dat',cdist)
#   print "ncluster",n-elbow
   clist=sc.fcluster(Z,nclust,criterion='maxclust')
   c_count=Counter(clist)
   print "Number of clusters", len(c_count)
   print "nconfig     meand    variance   rep config"
   rep_ind=[]
   for iclust in range(1,len(c_count)+1):  #calculate mean dissimilary and pick representative structure for each cluster
      indices = [i for i, x in enumerate(clist) if x == iclust] #indices for cluster i
      sumd=0.0
      icount=0
      #calculate mean dissimilarity in each group
      for iconf in range(len(indices)):
         ind1=indices[iconf]
         for jconf in range(iconf):
           ind2=indices[jconf]
           sumd+=sim[ind1][ind2]
           icount+=1
      meand=sumd/icount
      
      # pick the configuration with min variance in the group
      icount=0
      minvar=9999
      var=0.0
      for iconf in range(len(indices)):
        ivar=0.0
        ind1=indices[iconf]
        for jconf in range(len(indices)):
          ind2=indices[jconf]
          ivar+=(sim[ind1][ind2]-meand)**2
        ivar=ivar/(len(indices)-1)
        var+=ivar  
        icount+=1      
        if(ivar<minvar):  
          minvar=ivar
          iselect=ind1
      var=var/(icount)  
      rep_ind.append(iselect)
      print len(indices), meand , np.sqrt(var) , iselect
#   print rep_ind
   filename=mode+'-cluster.index'
   f=open(filename,"w")
   f.write("groupid representative \n ")
   for i in range(len(sim)):
      iselect=0
      if i in rep_ind: iselect=2
      f.write("%d   %d \n " %(clist[i],  iselect)) 
   f.close()
   if plot: plotdendro(Z,nclust)
   if (calc_sd):
     filename=mode+'-sd.dat'
     f=open(filename,"w")
     f.write("dist_sd ")
     if proplist!='':f.write("prop_sd ")
     f.write("\n")
     sim_sd=dissimilarity_sd(Z,sim) 
     if proplist!='': psd=prop_sd(Z,prop)
     for i in range(len(Z)):
         f.write("%f" %(sim_sd[i]))
         if proplist!='':f.write("   %f" %(psd[i]))
         f.write("\n")

def plotdendro(Z,ncluster):
  plt.figure(figsize=(25, 10))
  plt.title('Hierarchical Clustering Dendrogram')
  plt.xlabel('sample index')
  plt.ylabel('distance')
  sc.dendrogram(Z,truncate_mode='lastp', p=ncluster,leaf_rotation=90.,leaf_font_size=8.,show_contracted=True)
  plt.show()
      
       

def dissimilarity_sd(Z,sim):
  n=len(sim)
  clusterlist=[]
  ncluster=0
  sdlist=[]
  for i in range(len(Z)):
    id1=int(Z[i,0])
    id2=int(Z[i,1])
    if((id1 < n) and (id2<n)):  # when two configurations are merged note their index
       clusterlist.append([id1,id2])
       ncluster+=1
    else:
      cl=[]
      icount=0
      if id1>=n:  # this means merging is happening with previously formed cluster
        icluster=int(id1)-n
        for x in clusterlist[icluster]: #we already have the list for the old cluster
          cl.append(x)
      else:cl.append(id1)
      if id2>=n: # same logic as before
        icluster=int(id2)-n
        for x in clusterlist[icluster]:
          cl.append(x)
      else:cl.append(id2)
      clusterlist.append(cl) # append the index of the members at this stage of clustering 
#   calculate mean dissimilarity of the cluster
    sumd=0.0
    icount=0
    for iconf in range(len(clusterlist[i])):
        ind1=clusterlist[i][iconf]
        for jconf in range(iconf):
          ind2=clusterlist[i][jconf]
          sumd+=sim[ind1][ind2]
          icount+=1
    meand=sumd/icount
#   calculate variance and sd
    var=0.0
    icount=0
    minvar=9999
    for iconf in range(len(clusterlist[i])):
        ind1=clusterlist[i][iconf]
        ivar=0.0
        for jconf in range(len(clusterlist[i])):
          ind2=clusterlist[i][jconf]
          ivar+=(sim[ind1][ind2]-meand)**2
        ivar=ivar/(len(clusterlist[i])-1)
        var+=ivar
        icount+=1
        if(ivar<minvar):
            minvar=ivar
            iselect=ind1
    var=var/(icount)
    sd=np.sqrt(var)
    sdlist.append(sd)
  return sdlist
#    print len(clusterlist[i]),meand,var,sd,iselect,Z[i,2]
#  print "clusters:", nl-elbow+2


def prop_sd(Z,prop):
  n=len(prop)
  clusterlist=[]
  ncluster=0
  sdlist=[]
  for i in range(len(Z)):
    id1=int(Z[i,0])
    id2=int(Z[i,1])
    if((id1 < n) and (id2<n)):  # when two configurations are merged note their index
       clusterlist.append([id1,id2])
       ncluster+=1
    else:
      cl=[]
      icount=0
      if id1>=n:  # this means merging is happening with previously formed cluster
        icluster=int(id1)-n
        for x in clusterlist[icluster]: #we already have the list for the old cluster
          cl.append(x)
      else:cl.append(id1)
      if id2>=n: # same logic as before
        icluster=int(id2)-n
        for x in clusterlist[icluster]:
          cl.append(x)
      else:cl.append(id2)
      clusterlist.append(cl) # append the index of the members at this stage of clustering 
#   calculate mean dissimilarity of the cluster
    sumd=0.0
    icount=0
#   calculate variance and sd
    sd=np.std(prop[clusterlist[i]])
    sdlist.append(sd)
  return sdlist
#    print len(clusterlist[i]),meand,var,sd,iselect,Z[i,2]
#  print "clusters:", nl-elbow+2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes KRR and analytics based on a kernel matrix and a property vector.""")

    parser.add_argument("sim", nargs=1, help="Kernel matrix")
    parser.add_argument("--mode", type=str, default="average", help="Train point selection (e.g. --mode all / random / fps / cur")
    parser.add_argument("--nclust", type=int, default='10', help="Number of clusters")
    parser.add_argument("--prop", type=str, default='', help="property file")
    parser.add_argument("--plot",  action="store_true", help="Plot the dendrogram")
    parser.add_argument("--calc_sd",  action="store_true", help="calculate standard div of the dist and prop for all level of clustering")

    args = parser.parse_args()
    main(args.sim[0],args.nclust,mode=args.mode,proplist=args.prop,plot=args.plot,calc_sd=args.calc_sd)

