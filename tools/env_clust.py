import env_corr
import cluster
import numpy as np 
import quippy
import sys
import argparse
from scipy.cluster import hierarchy as sc
import matplotlib.pyplot as plt

def main(fnamexyz, cutoffdist, fnamesim, zenv, nenv, dcut, selectgroupxyz2print=[], dcutisgood=False, isploting=False ):

	pyplot = isploting
	plotdendo = True

	# Generate the file name for the new dist matrices that are without dummy atoms
	nsim = len(nenv)
	fname = []
	fnamecorr = []
	fnamemathematica = []
	cnt = 0
	for fn in fnamesim:
		target = '_'+str(nenv[cnt])
		fname.append(fn[:fn.find(target)]+'.sim')
		fnamecorr.append(fn[:fn.find(target)]+'.corrmt')
		fnamemathematica.append(fn[:fn.find(target)]+'.mathematica-cluster.dat')
		cnt+=1
		print (fn[:fn.find(target)])
	
	print >> sys.stderr, "Reading input file", fnamexyz
	mols = quippy.AtomsList(fnamexyz);
	print >> sys.stderr, len(mols.n) , " Configurations Read"
	
	distmatrix = []
	clist = []
	nbclst = []
	clusterlist = []
	nspiecies = []
	Z = []
	dendo = []

	for it in range(nsim):
		print '############'
		print "Read distance matrix "+ fnamesim[it] + " with dummy atoms"
		olddistmatrix = np.loadtxt(fnamesim[it])
		
		fsim = open(fnamesim[it],'r')
		head = fsim.readline()
		head = head.strip('#');head = head.strip('\n')
		# print '##########' , head
		fsim.close()

		# Removes dummy atoms rows/columns from distance matrix
		newdistmatrix1, dummylist = env_corr.rmdummyfromsim(fnamexyz,olddistmatrix,zenv[it],nenv[it])
		print len(newdistmatrix1),' Real ',zenv[it],' atom '
		print 'Removes : ', len(dummylist),' rows/colomns from distmatrix'
		print newdistmatrix1.shape
		
		# Write the distmatrix without dummy atoms
		np.savetxt(fname[it],newdistmatrix1, header=head )

		# get list of the cluster groups idx (same order as in dist mat) 
		clist1,Z1 = env_corr.clusterdistmat(fname[it],newdistmatrix1,0.,mode='average',plot=plotdendo)
		dendo1 = sc.dendrogram(Z1)
		# Write mathematica cluster structure
		cluster.mathematica_cluster(Z1, newdistmatrix1,fnamemathematica[it])

		# get nb of cluster groups
		nbclst1 = len(np.unique(clist1))
		# print a==nbclst1
		distmatrix.append(newdistmatrix1)
		clist.append(clist1)
		nbclst.append(nbclst1)
		Z.append(Z1)
		dendo.append(dendo1)

		# Link the cluster groups with atom's frame and respective position in it
		if dcutisgood:
			clusterlist1, nspiecies1 = env_corr.linkgroup2atmidx(mols,clist1,zenv[it],nenv[it])
			clusterlist.append(clusterlist1)
			# count nb of species atom in each frame
			nspiecies.append(nspiecies1)
	
	# Compute the correlation between the species of 2 cluster 
	# look for the groups of the atoms (from cluster2) in the environment of an
	# atom from a group from cluster1
	if dcutisgood:
		corrmtx = []
		for it in range(1,nsim):	
			if len(selectgroupxyz2print)==0:
				selectgroupxyz2print = []
				for i in range(1,nsim+1):
					selectgroupxyz2print.append([]) 
			
			corrmtx1, corr1,idx1, idx2  = env_corr.getcorrofcluster(mols,cutoffdist,fnamexyz,clusterlist[0],clusterlist[it],nbclst[0],nbclst[it],zenv[0],zenv[it],nspiecies[0],nspiecies[it],selectgroupxyz2print[it-1])
			
			# idx1 = dendo[0]['leaves']
			# idx2 = dendo[it]['leaves']
			print len(idx1),len(idx2)
			print corrmtx1.shape
			# corrmtx1 = corrmtx1[idx1,:]
			# corrmtx1 = corrmtx1[:,idx2]

			corrmtx.append(corrmtx1)
			head1 = 'From '+fnamexyz+'  Correlation matrix between environment of atom '+str(zenv[0])+' and '+str(zenv[it])
			print '############'
			print head1
			head2 = 'From '+fnamexyz+'  Dendogram ordering of atom '+str(zenv[0])
			head3 = 'From '+fnamexyz+'  Dendogram ordering of atom '+str(zenv[it])

			np.savetxt(fnamecorr[it], corrmtx1, header=head1 )
			np.savetxt(fnamecorr[it]+str(zenv[0]), idx1, header=head2 )
			np.savetxt(fnamecorr[it]+str(zenv[it]), idx2, header=head3 )

			# print corrmtx1

	if pyplot:
		plt.show()

# take input argument like aa,bb,aa and output a list
# separator must be ','
def unpackstrlistinput(arg):
	tmpsim = ''
	arglist = []
	for char in arg:
		if char == ',':
			arglist.append(tmpsim)
			tmpsim = ''
			continue
		tmpsim += char
	arglist.append(tmpsim)
	return arglist

def unpackfloatlistinput(arg):
	tmpsim = ''
	arglist = []
	for char in arg:
		if char == ',':
			arglist.append(float(tmpsim))
			tmpsim = ''
			continue
		tmpsim += char
	arglist.append(float(tmpsim))
	return arglist

def unpackintlistinput(arg):
	tmpsim = ''
	arglist = []
	for char in arg:
		if char == ',':
			arglist.append(int(tmpsim))
			tmpsim = ''
			continue
		tmpsim += char
	arglist.append(int(tmpsim))
	return arglist


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="""Computes the correlation matrix from the hierarchical clustering between environments given as environmental similarity/distance matrices  produced by glosim from option --envsim.""")

	parser.add_argument("xyz", nargs=1, help="xyz data file")
	parser.add_argument("sim",type=list,  help="Distance matrices of given species environement. The species of the first file is taken as reference. Must be a list of simfilename with coma separator, i.e. sim = a.sim,b.sim")
    # parser.add_argument("--mode", type=str, default="average", help="Linkage mode (e.g. --mode average/single/complete/median/centroid")
	parser.add_argument("--cutoffdist", type=float, default=2., help="Cutoff distance used in glosim (in Angstrom)")
	parser.add_argument("--dcut", type=list, default='', help="List of distance cutoff to cut the dendrogram. Must be listed in same order as sim")
	parser.add_argument("--zenv", type=list, default='', help="List of the atomic number of the atoms of the similarity matrices listed. Must be listed in same order as sim")
	parser.add_argument("--nenv", type=list, default='', help="List of the number of dummy atoms of the similarity matrices listed. Must be listed in same order as sim")
	parser.add_argument("--select", type=list, default='', help="List of the group idx to be printed to an xyz file. ex: with 3 distmatrix, input : 1,2,4,6 ;  which will select group 1 from 1st species and group 2 of 2nd species and  group 4 from 1st species and group 6 of 3rd species")
	parser.add_argument("--plot",  action="store_true", help="Plot the dendrograms")
	parser.add_argument("--dcutok",  action="store_true", help="If the dcut are fine then do all the analisis")
    
	args = parser.parse_args()

	# Convert list of char into proper list using ',' as the delemiter
	fnamesim = unpackstrlistinput(args.sim)
	zenv = unpackintlistinput(args.zenv)
	nenv = unpackintlistinput(args.nenv)
	dcut = unpackfloatlistinput(args.dcut)
	
	# Convert list of char into proper list using ',' as the delemiter and set default value
	if len(args.select) > 0:
		simlist = unpackintlistinput(args.select)
		selectgroupxyz2print = zip(*[iter(simlist)]*2)
	else:
		selectgroupxyz2print = []

	main(args.xyz[0], args.cutoffdist, fnamesim, zenv, nenv, dcut, selectgroupxyz2print, dcutisgood=args.dcutok, isploting =args.plot)
