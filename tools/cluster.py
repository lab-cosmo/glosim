#!/usr/bin/python
import argparse
import numpy as np
import sys
import numpy as np
import scipy.cluster.hierarchy as sc
import itertools
from os.path import basename
try:
  from matplotlib import pyplot as plt
except:
  print "matplotlib is not available. You will not be able to plot"

from collections import Counter

def main(distmatrixfile,nclust,mode='average',proplist='',plot=False,calc_sd=False,rect_matrixfile=''):
   project=False
   sim=np.loadtxt(distmatrixfile)
   if rect_matrixfile != '' :
      rect_matrix=np.loadtxt(rect_matrixfile)
      if (len(sim) != len(rect_matrix[0])):
         print "Inconsistent dimesion of rect matrix file"
         return
      project=True
   if proplist!='': prop=np.loadtxt(proplist)
   Z=sc.linkage(sim,mode)
   n=len(sim)
   cdist=Z[:,2] 
#   np.savetxt('dist.dat',cdist)
   if nclust<1: 
     nclust=estimate_ncluster(cdist)
     print "Estimated ncluster:",nclust
   print "mean+std, cutoffdist", np.sqrt(np.var(Z[:,2]))+np.mean(Z[:,2]),(Z[n-nclust,2])
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
         for jconf in range(len(indices)):
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
        ivar=ivar/max(1,(len(indices)-1))
        var+=ivar  
        icount+=1      
        if(ivar<minvar):  
          minvar=ivar
          iselect=ind1
      var=var/max(1,icount)  
      rep_ind.append(iselect)
      print len(indices), meand , np.sqrt(var) , iselect
#   print rep_ind
   filename=basename(distmatrixfile)+'-cluster.index'
   f=open(filename,"w")
   f.write("groupid representative \n ")
   for i in range(len(sim)):
      iselect=0
      if i in rep_ind: iselect=2
      f.write("%d   %d \n " %(clist[i],  iselect)) 
   f.close()
   if(project):
     project_groupid,project_rep=project_config(clist,rect_matrix,rep_ind)
     filename=basename(rect_matrixfile)+'-cluster.index'
     f=open(filename,"w")
     f.write("groupid representative \n ")
     for i in range(len(project_groupid)):
        iselect=0
        if i in project_rep: iselect=2
        f.write("%d   %d \n " %(project_groupid[i],  iselect)) 
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
      
def project_config(clusterlist,rect_matrix,rep_ind):
  nland=len(rect_matrix[0])
  if nland != len(clusterlist) : 
     print "Dimension Mismatch for rect matrix" 
     stop 
  n=len(rect_matrix)
  groupid=[]
  for i in range(n):
    mind=10
    for j  in range(nland): # find which cluster it belongs 
        d=rect_matrix[i][j]
        if d <mind : 
            mind=d #find min distance config from config from  all clusters 
            icluster_select=clusterlist[j]
    groupid.append(icluster_select)
  project_rep=[]
  for iconfig in rep_ind: 
    mind=np.min(rect_matrix[:,iconfig])
    if (mind <1E-9):
      iselect=np.argmin(rect_matrix[:,iconfig])               
      project_rep.append(iselect)
  return(groupid,project_rep)

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

def estimate_ncluster(dist):
  n=len(dist)
  b=[n-1,dist[n-1]-dist[0]]
  b=np.array(b)
  b=b/np.linalg.norm(b)
  dmax=0.0
  for i in range(n):
    p=[n-1-i,dist[n-1]-dist[i]]
    d=np.linalg.norm(p-np.dot(p,b)*b)
    if d>dmax :
       elbow=i
       dmax=d
  return int((n-elbow)*0.5)


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
    parser.add_argument("--mode", type=str, default="average", help="Linkage mode (e.g. --mode average/single/complete/median/centroid")
    parser.add_argument("--nclust", type=int, default='0', help="Number of clusters")
    parser.add_argument("--prop", type=str, default='', help="property file")
    parser.add_argument("--plot",  action="store_true", help="Plot the dendrogram")
    parser.add_argument("--calc_sd",  action="store_true", help="calculate standard div of the dist and prop for all level of clustering")
    parser.add_argument("--project",  type=str,default='', help="Project configurations using Rect Dist Matrix file")

    args = parser.parse_args()
    main(args.sim[0],args.nclust,mode=args.mode,proplist=args.prop,plot=args.plot,calc_sd=args.calc_sd,rect_matrixfile=args.project)

