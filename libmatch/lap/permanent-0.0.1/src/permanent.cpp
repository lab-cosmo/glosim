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
#include <iostream>
// Array access macros.

#define SM(x0, x1) (*(npy_double*) (( (char*) PyArray_DATA(matrix) + \
                    (x0) * PyArray_STRIDES(matrix)[0] +  \
                    (x1) * PyArray_STRIDES(matrix)[1])))
#define SM_shape(x0) (int) PyArray_DIM(matrix, x0)

template <typename T>
class Matrix
{
        std::vector<T> inner_;
        unsigned int dimx_, dimy_;

public:
        unsigned int size() const { return dimx_; } 

        Matrix (unsigned int dimx, unsigned int dimy)
                : dimx_ (dimx), dimy_ (dimy)
        {
                inner_.resize (dimx_*dimy_);
        }

        inline T operator()(unsigned int x, unsigned int y) const
        {
                if (x >= dimx_ || y>= dimy_)
                        throw 0; // ouch
                return inner_[dimx_*y + x];
        }
        
        inline T& operator()(unsigned int x, unsigned int y)
        {
                if (x >= dimx_ || y>= dimy_)
                        throw 0; // ouch
                return inner_[dimx_*y + x];
        }
        
};

// Forward function declaration 
static PyObject *permanent_mc(PyObject *self, PyObject *args);
// Forward function declaration 
static PyObject *permanent_ryser(PyObject *self, PyObject *args);
// Forward function declaration 
static PyObject *rematch(PyObject *self, PyObject *args);



// Method list
static PyMethodDef methods[] = {
  { "permanent_mc", permanent_mc, METH_VARARGS, "Computes the permanent of a numpy matrix by random montecarlo method upto given accuracy"},
  { "permanent_ryser", permanent_ryser, METH_VARARGS, "Computes the permanent of a numpy matrix by Ryser algorithm"},
  { "rematch", rematch, METH_VARARGS, "Computes the permanent of a numpy matrix by Ryser algorithm"},
  { NULL, NULL, 0, NULL } // Sentinel
};

// Module initialization
PyMODINIT_FUNC initpermanent(void) {
  (void) Py_InitModule("permanent", methods);
  import_array();
}

double fact(int n)
{
   double fn=1.0;  for (int i=2; i<=n; ++i) fn*=double(i);
   return fn;
}

static npy_double _mcperm(PyArrayObject *matrix, PyFloatObject *eps, PyIntObject *ntry, PyIntObject *seed)
{
 //   int n=mtx.size();
    int n = (int) PyArray_DIM(matrix, 0);
    std::vector<int> idx(n);
//    double eps=1e-3;
    double eps1=PyFloat_AS_DOUBLE(eps);
    int ntry1=PyInt_AS_LONG(ntry);
    int seed1=PyInt_AS_LONG(seed);
    for (int i=0; i<n; ++i) idx[i]=i;
    double pi, prm=0, prm2=0, fn=fact(n), ti=0;
    int i=0, istride=0, pstride=n*100; 
    if (seed1>0) std::srand(seed1);
    //std::cerr<<eps<<" "<<ntry1<<"  "<<seed1<<"\n";
    
    while (true)
    {
        // combines shuffles and cyclic permutations (which are way cheaper!)
        if (i%n==0) std::random_shuffle(idx.begin(),idx.end());
        else { for (int i=0; i<n; ++i) idx[i]=(idx[i]+1)%n; }
        
        //if (i%10000==0) { for (int j=0; j<n; ++j) std::cerr<<idx[j]<<" ";        std::cerr<<"\n"; }
        
        // computes the product of elements for the selected permutation
        pi = SM(0, idx[0]);;
        for (int j=1; j<n; ++j)
            pi *= SM(j, idx[j]);
        
        // accumulates mean and mean square
        prm += pi;
        prm2 += pi*pi;
        ++i;
        if (ntry1>0 && i >=ntry1) { ti=i; break; }
        if (ntry1==0 && i==pstride)  // check if we are converged
        {
            ++istride; i=0; ti=double(istride)*double(pstride);
            double err=sqrt((prm2-prm*prm/ti)/ti/(ti-1) ) / (prm/ti);
            //std::cerr <<istride<< " "<<fn*prm/ti<< " "<<err<< "\n";
            if (err< eps1) break;
        }
    }
    //std::cerr <<i<< " "<<fn*prm/ti<< "\n";
    return prm/ti*fn;
}

// sinkhorn regularized best match 
// NB this assumes that the input matrix is a kernel matrix with entries \in [0,1], 
// NB this also works on rectangular matrices
static npy_double _shmatch(PyArrayObject* matrix, PyFloatObject *gamma, PyFloatObject *eps)
{
    int nx = (int) PyArray_DIM(matrix, 0);
    int ny = (int) PyArray_DIM(matrix, 1);
    std::vector<double> u(nx), ou(nx), v(ny);
    double ax = 1.0/nx, ay=1.0/ny;
    Matrix<double> Kg(nx,ny);
    for (int i=0; i<nx; ++i) u[i]=1.0;
    for (int i=0; i<ny; ++i) v[i]=1.0;
    double lambda=1.0/PyFloat_AS_DOUBLE(gamma), terr=PyFloat_AS_DOUBLE(eps)*PyFloat_AS_DOUBLE(eps), derr;
    
    for (int i=0; i<nx; ++i) for (int j=0; j<ny; ++j) Kg(i,j)=std::exp(-(1-SM(i,j))*lambda);
    
    do 
    {
        // u<-1.0/Kg.v
        for (int i=0; i<nx; ++i) { ou[i]=u[i]; u[i]=0.0; }            
        for (int i=0; i<nx; ++i) for (int j=0; j<ny; ++j) u[i]+=Kg(i,j)*v[j];
        // at this point we can compute how far off unity we are
        derr = 0.0;
        for (int i=0; i<nx; ++i) derr+=(ax-ou[i]*u[i])*(ax-ou[i]*u[i]);        
        for (int i=0; i<nx; ++i) u[i]=ax/u[i];
        
        // v<-1.0/Kg.u
        for (int i=0; i<ny; ++i) v[i]=0.0; 
        for (int i=0; i<ny; ++i) for (int j=0; j<nx; ++j) v[i]+=Kg(j,i)*u[j];
        for (int i=0; i<ny; ++i) v[i]=ay/v[i];
        //std::cerr<<derr<<"\n";
                
    } while (derr>terr);
    
    double rval=0, rrow; 
    for (int i=0; i<nx; ++i) 
    {
       rrow=0;
       for (int j=0; j<ny; ++j) rrow+=Kg(i,j)*SM(i,j)*v[j];
       rval+=u[i]*rrow;
    }   
    //std::cerr<<"regmatch "<< rval/n <<"\n";
    return rval;
}

// Count the number of set bits in a binary string
inline int countbits(unsigned int n) 
{
    int q=n;
    q = (q & 0x5555555555555555) + ((q & 0xAAAAAAAAAAAAAAAA) >> 1);
    q = (q & 0x3333333333333333) + ((q & 0xCCCCCCCCCCCCCCCC) >> 2);
    q = (q & 0x0F0F0F0F0F0F0F0F) + ((q & 0xF0F0F0F0F0F0F0F0) >> 4);
    q = (q & 0x00FF00FF00FF00FF) + ((q & 0xFF00FF00FF00FF00) >> 8);
    q = (q & 0x0000FFFF0000FFFF) + ((q & 0xFFFF0000FFFF0000) >> 16);
    q = (q & 0x00000000FFFFFFFF) + ((q & 0xFFFFFFFF00000000) >> 32); // This last & isn't strictly qecessary.
    return q;
}

inline int bitparity (unsigned int n) { return 1 - (countbits(n) & 1)*2; }

// Ryser's algorithm 
// Adapted from a complex-argument version by Pete Shadbolt
static npy_double _ryperm(PyArrayObject *matrix) {
    unsigned int n = (unsigned int) PyArray_DIM(matrix, 0);    
    npy_double sumz, prod;
    npy_double perm = 0;
    unsigned long two2n = 1 << n; 
    unsigned long i, y, z;
    for (i=0; i<two2n; ++i) {
        prod = 1.0;
        for (y=0; y<n; ++y) {               
            sumz = 0;
            for (z=0; z<n; ++z) { 
                if ((i & (1 << z)) != 0) { sumz += SM(z, y); }
            }
            prod*=sumz;
        }
        perm += prod * bitparity(i);
    }
    if (n%2 == 1) { perm*=-1; }
    return perm;
}

// Computes the permanent using a monte carlo scheme
static PyObject *permanent_mc(PyObject *self, PyObject *args) {
  // Parse the input 
  PyArrayObject *matrix;
  PyFloatObject *eps;
  PyIntObject *ntry;
  PyIntObject *seed;
  if (!PyArg_ParseTuple(args, "O!O!O!O!", &PyArray_Type, &matrix, &PyFloat_Type, &eps, &PyInt_Type, &ntry, &PyInt_Type, &seed)) {return NULL;}

  // Compute the permanent
  npy_double p = _mcperm(matrix,eps,ntry,seed);
  return PyFloat_FromDouble(p);
}

// Exact permanent based on Ryser algorithm
static PyObject *permanent_ryser(PyObject *self, PyObject *args) {
  // Parse the input 
  PyArrayObject *matrix;  
  if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &matrix)) {return NULL;}

  // Compute the permanent
  npy_double p = _ryperm(matrix);
  return PyFloat_FromDouble(p);
}

// Computes regularised best-match usin Sinkhorn algorithm 
static PyObject *rematch(PyObject *self, PyObject *args) {
  // Parse the input 
  PyArrayObject *matrix;
  PyFloatObject *eps;
  PyFloatObject *gamma;
  if (!PyArg_ParseTuple(args, "O!O!O!", &PyArray_Type, &matrix, &PyFloat_Type, &gamma, &PyFloat_Type, &eps)) {return NULL;}

  // Compute the permanent
  npy_double p = _shmatch(matrix,gamma,eps);
  return PyFloat_FromDouble(p);
}
