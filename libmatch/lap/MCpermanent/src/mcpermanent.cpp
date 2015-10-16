// (c) Sandip De 
// 16th Oct 2015
// Lausanne 
// C++ implementation of python module to compute permanent of a matrix by random montecarlo 
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
#define SM(x0, x1) (*(npy_double*)((PyArray_DATA(matrix) + \
                    (x0) * PyArray_STRIDES(matrix)[0] +  \
                    (x1) * PyArray_STRIDES(matrix)[1])))
#define SM_shape(x0) (int) PyArray_DIM(matrix, x0)


double fact(int n)
{
   double fn=1.0;
   for (int i=2; i<=n; ++i) fn*=double(i);
   return fn;
}
// Forward function declaration 
static PyObject *mcpermanent(PyObject *self, PyObject *args);

// Method list
static PyMethodDef methods[] = {
  { "mcpermanent", mcpermanent, METH_VARARGS, "Computes the permanent of a numpy matrix by random montecarlo method upto given accuracy"},
  { NULL, NULL, 0, NULL } // Sentinel
};

// Module initialization
PyMODINIT_FUNC initmcpermanent(void) {
  (void) Py_InitModule("mcpermanent", methods);
  import_array();
}

static npy_double perm(PyArrayObject *matrix,PyFloatObject *eps)
{
 //   int n=mtx.size();
    int n = (int) PyArray_DIM(matrix, 0);
    std::vector<int> idx(n);
//    double eps=1e-3;
    double eps1=PyFloat_AS_DOUBLE(eps);
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
            if (err< eps1) break;
        }
    }
    return prm/ti*fn;
}
// This is a wrapper which chooses the optimal permanent function
static PyObject *mcpermanent(PyObject *self, PyObject *args) {
  // Parse the input
 
  PyArrayObject *matrix;
  PyFloatObject *eps;
  if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &matrix, &PyFloat_Type, &eps)) {return NULL;}
  // Compute the permanent
  npy_double p = perm(matrix,eps);
  return PyFloat_FromDouble(p);
}


