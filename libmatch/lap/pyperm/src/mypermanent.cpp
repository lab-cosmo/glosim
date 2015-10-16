/* Functions to compute the permanent, given a numpy array */
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>
#include <vector>
#include <stdlib.h> 
#include <algorithm>
#include <random>
#include <cmath>
// Array access macros.
#define SM(x0, x1) (*(npy_double*)((PyArray_DATA(submatrix) + \
                    (x0) * PyArray_STRIDES(submatrix)[0] +  \
                    (x1) * PyArray_STRIDES(submatrix)[1])))
#define SM_shape(x0) (int) PyArray_DIM(submatrix, x0)


double fact(int n)
{
   double fn=1.0;
   for (int i=2; i<=n; ++i) fn*=double(i);
   return fn;
}
// Forward function declaration 
static PyObject *mypermanent(PyObject *self, PyObject *args);

// Method list
static PyMethodDef methods[] = {
  { "mypermanent", mypermanent, METH_VARARGS, "Computes the permanent of a numpy using the approx method available"},
  { NULL, NULL, 0, NULL } // Sentinel
};

// Module initialization
PyMODINIT_FUNC initmypermanent(void) {
  (void) Py_InitModule("mypermanent", methods);
  import_array();
}

static npy_double perm(PyArrayObject *submatrix)
{
 //   int n=mtx.size();
    int n = (int) PyArray_DIM(submatrix, 0);
    std::vector<int> idx(n);
    double eps=1e-3;
    for (int i=0; i<n; ++i) idx[i]=i;
    double pi, prm=0, prm2=0, fn=fact(n), ti;
    int i=0, istride=0, pstride=n*100; 
    while (true)
    {
        // combines shuffles and cyclic permutations (which are way cheaper!)
        if (i%n==0) std::random_shuffle(idx.begin(),idx.end());
        else { for (int i=0; i<n; ++i) idx[i]=(idx[i]+1)%n; }
        
        //for (int j=0; j<n; ++j) std::cerr<<idx[j]<<" ";        std::cerr<<"\n";
        
        // computes the product of elements for the selected permutation
        pi = SM(0, idx[0]);;
        for (int j=1; j<n; ++j)
            pi *= SM(j, idx[j]);
        
        // accumulates mean and mean square
        prm += pi;
        prm2 += pi*pi;
        ++i;
        if (i==pstride)  // check if we are converged
        {
            ++istride; i=0; ti=double(istride)*double(pstride);
            double err=sqrt((prm2-prm*prm/ti)/ti/(ti-1) ) / (prm/ti);
            //std::cerr <<istride<< " "<<std::setprecision(10)<<fn*prm/ti<< " "<<err<< "\n";
            if (err<eps) break;
        }
    }
    return prm/ti*fn;
}
// This is a wrapper which chooses the optimal permanent function
static PyObject *mypermanent(PyObject *self, PyObject *args) {
  // Parse the input
  PyArrayObject *submatrix;
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &submatrix)) {return NULL;}

  // Compute the permanent
  npy_double p = perm(submatrix);
  return PyFloat_FromDouble(p);
}


