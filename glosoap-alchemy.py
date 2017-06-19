

import quippy as qp
import numpy  as np
import argparse



def atomicno_to_sym(atno):
  pdict={1: 'H', 2: 'He', 3: 'Li', 4: 'Be', 5: 'B', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 10: 'Ne', 11: 'Na', 12: 'Mg', 13: 'Al', 14: 'Si', 15: 'P', 16: 'S', 17: 'Cl', 18: 'Ar', 19: 'K', 20: 'Ca', 21: 'Sc', 22: 'Ti', 23: 'V', 24: 'Cr', 25: 'Mn', 26: 'Fe', 27: 'Co', 28: 'Ni', 29: 'Cu', 30: 'Zn', 31: 'Ga', 32: 'Ge', 33: 'As', 34: 'Se', 35: 'Br', 36: 'Kr', 37: 'Rb', 38: 'Sr', 39: 'Y', 40: 'Zr', 41: 'Nb', 42: 'Mo', 43: 'Tc', 44: 'Ru', 45: 'Rh', 46: 'Pd', 47: 'Ag', 48: 'Cd', 49: 'In', 50: 'Sn', 51: 'Sb', 52: 'Te', 53: 'I', 54: 'Xe', 55: 'Cs', 56: 'Ba', 57: 'La', 58: 'Ce', 59: 'Pr', 60: 'Nd', 61: 'Pm', 62: 'Sm', 63: 'Eu', 64: 'Gd', 65: 'Tb', 66: 'Dy', 67: 'Ho', 68: 'Er', 69: 'Tm', 70: 'Yb', 71: 'Lu', 72: 'Hf', 73: 'Ta', 74: 'W', 75: 'Re', 76: 'Os', 77: 'Ir', 78: 'Pt', 79: 'Au', 80: 'Hg', 81: 'Tl', 82: 'Pb', 83: 'Bi', 84: 'Po', 85: 'At', 86: 'Rn', 87: 'Fr', 88: 'Ra', 89: 'Ac', 90: 'Th', 91: 'Pa', 92: 'U', 93: 'Np', 94: 'Pu', 95: 'Am', 96: 'Cm', 97: 'Bk', 98: 'Cf', 99: 'Es', 100: 'Fm', 101: 'Md', 102: 'No', 103: 'Lr', 104: 'Rf', 105: 'Ha', 106: 'Sg', 107: 'Ns', 108: 'Hs', 109: 'Mt', 110: 'Unn', 111: 'Unu'}
  return pdict[atno]

def get_spkitMax(atoms):
    '''
    Get the set of species their maximum number across atoms.

    :param atoms: list of quippy Atoms object
    :return: Dictionary with species as key and return its
                largest number of occurrence
    '''
    spkitMax = {}

    for at in atoms:
        atspecies = {}
        for z in at.z:
            if z in atspecies:
                atspecies[z] += 1
            else:
                atspecies[z] = 1

        for (z, nz) in atspecies.iteritems():
            if z in spkitMax:
                if nz > spkitMax[z]: spkitMax[z] = nz
            else:
                spkitMax[z] = nz

    return spkitMax

def get_spkit(atom):
    '''
    Get the set of species their number across atom.

    :param atom: One quippy Atoms object
    :return:
    '''
    spkit = {}
    for z in atom.z:
        if z in spkit:
            spkit[z]+=1
        else:
            spkit[z] = 1
    return spkit


def get_soap(atom, spkit, spkitMax, centerweight=1., gaussian_width=0.5,
             cutoff=5.0, cutoff_transition_width=0.5, nmax=8, lmax=6):
    '''
    Get the soap vectors (power spectra) for each atomic environments in atom.

    :param atom: A quippy Atoms object
    :param spkit: Dictionary with specie as key and number of corresponding atom as item.
                    Returned by get_spkit(atom).
    :param spkitMax: Dictionary with species as key and return its largest number of occurrence.
                        Returned by get_spkitMax(atoms) .
    :param centerweight: Center atom weight
    :param gaussian_width: Atom Gaussian std
    :param cutoff: Cutoff radius for each atomic environment in the unit of cell and positions.
    :param cutoff_transition_width: Steepness of the smooth environmental cutoff radius. Smaller -> steeper
    :param nmax: Number of radial basis functions.
    :param lmax: Number of Spherical harmonics.
    :return: Soap vectors of atom. Dictionary (keys:atomic number of the central atom,
                items: list of power spectra for each central atom
                        with corresponding atomic number )
    '''
    zsp = spkitMax.keys()
    zsp.sort()
    lspecies = 'n_species=' + str(len(zsp)) + ' species_Z={ '
    for z in zsp:
        lspecies = lspecies + str(z) + ' '
    lspecies = lspecies + '}'

    atom.set_cutoff(cutoff)
    atom.calc_connect()

    soap = {}
    for (z, nz) in spkit.iteritems():
        soapstr = "soap central_reference_all_species=F central_weight=" + str(centerweight)+\
                  "  covariance_sigma0=0.0 atom_sigma=" + str(gaussian_width) +\
                  " cutoff=" + str(cutoff) + \
                  " cutoff_transition_width=" + str(cutoff_transition_width) + \
                  " n_max=" + str(nmax) + " l_max=" + str(lmax) + ' ' + lspecies +\
                  ' Z=' + str(z)

        desc = qp.descriptors.Descriptor(soapstr)

        sps = desc.calc(atom)["descriptor"]
        soap[z] = sps

    return soap


def Soap2AlchemySoap(rawsoap, spkit, nmax, lmax):
    '''
    Convert the soap vector of an environment from quippy descriptor to soap vectors
     with chemical channels.

    :param rawsoap: numpy array dim:(N,) containing the soap vector of one environment
    :param spkit: Dictionary with specie as key and number of corresponding atom as item.
                    Returned by get_spkit(atom).
    :param nmax: Number of radial basis functions.
    :param lmax: Number of Spherical harmonics.
    :return: Dictionary  (keys: species tuples (sp1,sp2),
                            items: soap vector, numpy array dim:(nmax ** 2 * (lmax + 1),) )
    '''
    # spkit keys are the center species in the full frame
    zspecies = sorted(spkit.keys())
    nspecies = len(spkit.keys())

    alchemySoap = {}
    ipair = {}
    # initialize the alchemical soap
    for s1 in xrange(nspecies):
        for s2 in xrange(
                nspecies):  # range(s1+1): we actually need to store also the reverse pairs if we want to go alchemical
            alchemySoap[(zspecies[s2], zspecies[s1])] = np.zeros(nmax ** 2 * (lmax + 1), float)
            ipair[(zspecies[s2], zspecies[s1])] = 0

    isoap = 0
    isqrttwo = 1.0 / np.sqrt(2.0)

    # selpair and revpair are modified and in turn modify soaps because they are all pointing at the same memory block
    for s1 in xrange(nspecies):
        for n1 in xrange(nmax):
            for s2 in xrange(s1 + 1):
                selpair = alchemySoap[(zspecies[s2], zspecies[s1])]
                # we need to reconstruct the spectrum for the inverse species order, that also swaps n1 and n2.
                # This is again only needed to enable alchemical combination of e.g. alpha-beta and beta-alpha. Shit happens
                revpair = alchemySoap[(zspecies[s1], zspecies[s2])]
                isel = ipair[(zspecies[s2], zspecies[s1])]
                for n2 in xrange(nmax if s2 < s1 else n1 + 1):
                    for l in xrange(lmax + 1):
                        # print s1, s2, n1, n2, isel, l+(self.lmax+1)*(n2+self.nmax*n1), l+(self.lmax+1)*(n1+self.nmax*n2)
                        # selpair[isel] = rawsoap[isoap]
                        if (s1 != s2):
                            selpair[isel] = rawsoap[
                                                isoap] * isqrttwo  # undo the normalization since we will actually sum over all pairs in all directions!
                            revpair[l + (lmax + 1) * (n1 + nmax * n2)] = selpair[isel]
                        else:
                            # diagonal species (s1=s2) have only half of the elements.
                            # this is tricky. we need to duplicate diagonal blocks "repairing" these to be full.
                            # this is necessary to enable alchemical similarity matching, where we need to combine
                            # alpha-alpha and alpha-beta environment fingerprints
                            selpair[l + (lmax + 1) * (n2 + nmax * n1)] = rawsoap[isoap] * (1 if n1 == n2 else isqrttwo)
                            selpair[l + (lmax + 1) * (n1 + nmax * n2)] = rawsoap[isoap] * (1 if n1 == n2 else isqrttwo)
                        # selpair[l + (lmax + 1) * (n2 + nmax * n1)] = selpair[l + (lmax + 1) * (n1 + nmax * n2)]  \
                        #                                                                                                   =  rawsoap[isoap] * (1 if n1 == n2 else isqrttwo)
                        isoap += 1
                        isel += 1
                ipair[(zspecies[s2], zspecies[s1])] = isel

    return alchemySoap


def get_Soaps(atoms,chem_channels=False, centerweight=1.0, gaussian_width=0.5, cutoff=3.5,
                     cutoff_transition_width=0.5 , nmax=8, lmax=6):
    '''
    Compute the SOAP vectors for each atomic environment in atoms and
    reorder them into chemical channels.

    :param atoms: list of quippy Atoms object
    :param centerweight: Center atom weight
    :param gaussian_width: Atom Gaussian std
    :param cutoff: Cutoff radius for each atomic environment in the unit of cell and positions.
    :param cutoff_transition_width: Steepness of the smooth environmental cutoff radius. Smaller -> steeper
    :param nmax: Number of radial basis functions.
    :param lmax: Number of Spherical harmonics.
    :return: Nested List/Dictionary: list->atoms,
                dict->(keys:atomic number,
                items:list of atomic environment), list->atomic environment,
                dict->(keys:chemical channel, (sp1,sp2) sp* is atomic number
                      inside the atomic environment),
                       items: SOAP vector, flat numpy array)
    '''

    Soaps = []
    # get the set of species their maximum number across atoms
    spkitMax = get_spkitMax(atoms)

    for atom in atoms:

        # to avoid side effect due to pointers
        atm = atom.copy()
        # get the set of species their number across atom
        spkit = get_spkit(atm)
        # get the soap vectors (power spectra) for each atomic environments in atm
        rawsoaps = get_soap(atm, spkit, spkitMax, centerweight, gaussian_width,
                            cutoff, cutoff_transition_width, nmax, lmax)

        # chemical channel separation for each central atom species
        # and each atomic environment
        if chem_channels:
            alchemySoap = {}
            for (z, soap) in rawsoaps.iteritems():
                Nenv, Npowerspectrum = soap.shape
                lsp = []
                # loop over the local environments of specie z
                for it in xrange(Nenv):
                    # soap[it] is (1,Npowerspectrum) so need to transpose it
                    #  convert the soap vector of an environment from quippy descriptor to soap vectors
                    # with chemical channels.
                    lsp.append(Soap2AlchemySoap(soap[it].T, spkit, nmax, lmax))
                # gather list of environment over the atomic number
                alchemySoap[z] = lsp
            # gather soaps over the atom
            Soaps.append(alchemySoap)
        # out put rawSoap
        else:
            Soaps.append(rawsoaps)

    return Soaps


def get_AvgSoaps(atoms, centerweight=1.0, gaussian_width=0.5, cutoff=3.5,
                 cutoff_transition_width=0.5, nmax=8, lmax=6, chem_channels=False):
    '''
    Compute the average SOAP vectors for each atomic environment in atoms and
    reorder them into chemical channels.

    :param atoms: list of quippy Atoms object
    :param centerweight: Center atom weight
    :param gaussian_width: Atom Gaussian std
    :param cutoff: Cutoff radius for each atomic environment in the unit of cell and positions.
    :param cutoff_transition_width: Steepness of the smooth environmental cutoff radius. Smaller -> steeper
    :param nmax: Number of radial basis functions.
    :param lmax: Number of Spherical harmonics.
    :return: Nested List/Dictionary: list->atoms,
                dict->(keys:chemical channel, (sp1,sp2) sp* is atomic number
                      inside the atomic environment),
                       items: SOAP vector, flat numpy array)
    '''
    AvgSoaps = []
    # get the set of species their maximum number across atoms
    spkitMax = get_spkitMax(atoms)
    for atom in atoms:
        # to avoid side effect due to pointers
        atm = atom.copy()
        # get the set of species their number across atom
        spkit = get_spkit(atm)
        # get the soap vectors (power spectra) for each atomic environments in atm
        rawsoaps = get_soap(atm, spkit, spkitMax, centerweight, gaussian_width,
                            cutoff, cutoff_transition_width, nmax, lmax)
        # compute the average soap over an atomic environment
        avgrawsoap = np.concatenate(rawsoaps.values(), axis=0).sum(axis=0)

        # chemical channel separation for each each atomic environment
        if chem_channels:
            AvgSoaps.append(Soap2AlchemySoap(avgrawsoap, spkit, nmax, lmax))
        # output average rawSoaps
        else:
            AvgSoaps.append(avgrawsoap)
    return AvgSoaps

def dumpAlchemySoapstxt(alchemySoaps,fout):
    '''
    Print in text format the alchemySoaps using the same format as in glosim --verbose

    :param alchemySoaps: Nested List/Dictionary: list->atoms,
                dict->(keys:atomic number,
                items:list of atomic environment), list->atomic environment,
                dict->(keys:chemical channel, (sp1,sp2) sp* is atomic number
                      inside the atomic environment),
                       items: SOAP vector, flat numpy array)
    :param fout: Writable python io object
    :return: None
    '''
    for iframe, alchemySoap in enumerate(alchemySoaps):
        fout.write("# Frame %d \n" % (iframe))

        for zatom, soapEnvList in alchemySoap.iteritems():
            for ienv, soapEnv in enumerate(soapEnvList):
                fout.write("# Species %d Environment %d \n" % (zatom, ienv))
                for (sp1, sp2), soap in soapEnv.iteritems():
                    fout.write("%d %d   " % (sp1, sp2))
                    for sj in soap:
                        fout.write("%8.4e " % (sj))
                    fout.write("\n")

def dumpAlchemySoapspickle(alchemySoaps, fout):
    '''
    Dump alchemySoaps in pickle binary format. Read with pck.load(filename)

    :param alchemySoaps: Nested List/Dictionary: list->atoms,
                dict->(keys:atomic number,
                items:list of atomic environment), list->atomic environment,
                dict->(keys:chemical channel, (sp1,sp2) sp* is atomic number
                      inside the atomic environment),
                       items: SOAP vector, flat numpy array)
    :param fout: Writable python io object
    :return: None
    '''
    import cPickle as pck
    pck.dump(alchemySoaps,fout,protocol=pck.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""Computes the SOAP vectors of a list of atomic frame 
            and differenciate the chemical channels. Ready for alchemical kernel.""")

    parser.add_argument("filename", nargs=1, help="Name of the LibAtom formatted xyz input file")
    parser.add_argument("-n", type=int, default='8', help="Number of radial functions for the descriptor")
    parser.add_argument("-l", type=int, default='6', help="Maximum number of angular functions for the descriptor")
    parser.add_argument("-c", type=float, default='5.0', help="Radial cutoff")
    parser.add_argument("-cotw", type=float, default='0.5', help="Cutoff transition width")
    parser.add_argument("-g", type=float, default='0.5', help="Atom Gaussian sigma")
    parser.add_argument("-cw", type=float, default='1.0', help="Center atom weight")
    parser.add_argument("-prefix", type=str, default='', help="Prefix for output files (defaults to input file name)")
    parser.add_argument("-first", type=int, default='0', help="Index of first frame to be read in")
    parser.add_argument("-last", type=int, default='0', help="Index of last frame to be read in")
    parser.add_argument("-outformat", type=str, default='pickle', help="Choose how to dump the alchemySoaps, e.g. pickle (default) or text (same as from glosim --verbose)")

    args = parser.parse_args()

    filename = args.filename[0]
    prefix = args.prefix
    centerweight = args.cw
    gaussian_width = args.g
    cutoff = args.c
    cutoff_transition_width = args.cotw
    nmax = args.n
    lmax = args.l
    first = args.first if args.first>0 else None
    last = args.last if args.last>0 else None

    if args.outformat in ['text','pickle']:
        outformat = args.outformat
    else:
        raise Exception('outformat is not recognised')



    if prefix=="": prefix=filename
    if prefix.endswith('.xyz'): prefix=prefix[:-4]
    prefix += "-n"+str(nmax)+"-l"+str(lmax)+"-c"+str(cutoff)+\
             "-g"+str(gaussian_width)+ "-cw"+str(centerweight)+ \
             "-cotw" +str(cutoff_transition_width)

    print  "using output prefix =", prefix
    # Reads input file using quippy
    print "Reading input file", filename

    # Reads the file and create a list of quippy Atoms object
    atoms = qp.AtomsList(filename, start=first, stop=last)

    alchemySoaps = get_Soaps(atoms, centerweight=centerweight, gaussian_width=gaussian_width, cutoff=cutoff,
                     cutoff_transition_width=cutoff_transition_width, nmax=nmax, lmax=lmax,chem_channels=True)


    if outformat == 'text':
        with open(prefix + "-soap.dat", "w") as fout:
            dumpAlchemySoapstxt(alchemySoaps, fout)
    elif outformat == 'pickle':
        with open(prefix + "-soap.pck", "w") as fout:
            dumpAlchemySoapspickle(alchemySoaps, fout)


