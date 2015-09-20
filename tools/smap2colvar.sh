#!/bin/bash

awk 'BEGIN{ print "#! FIELDS  index  cv1    cv2"; n=0; } !/#/{n++; printf "% 4d  % 10.5e  % 10.5e\n", n, $1, $2}' | paste - $1 
