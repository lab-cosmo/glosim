#!/bin/bash
sed 's/.*Lattice="\([^"]*\)".*/# CELL(GENH): \1  Traj: libatoms/' | awk '!/^ *[A-Z]/{print $0} /^ *[A-Z]/{print $1, $2, $3, $4}'
