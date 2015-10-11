#include <iostream>
#include <iomanip> 
#include <vector>
#include <sstream>
#include <string>
#include <algorithm>
#include <random>
#include <cmath>
#include <stdlib.h> 
#include <time.h> 

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

double fact(int n)
{
   double fn=1.0;
   for (int i=2; i<=n; ++i) fn*=double(i);
   return fn;
}

double perm(const Matrix<double>& mtx, double eps=1e-4)
{
    int n=mtx.size(); std::vector<int> idx(n);
    
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
        pi = mtx(0, idx[0]);;
        for (int j=1; j<n; ++j)
            pi *= mtx(j, idx[j]);
        
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
    return prm/ti*fact(n);
}

/*  experimental (correlated permutations) */
std::mt19937 generator;
std::uniform_real_distribution<double> distribution(0,1);
auto rndu=std::bind(distribution, generator);

void shuffle(std::vector<int>& idx)
{
    int swp, n=idx.size();
    for (int i=n-1; i>0; --i)
    {
        int j = rndu()*(i+1);
        swp=idx[i];
        idx[i]=idx[j];
        idx[j]=swp;
    }    
}

void shuffle2(std::vector<int>& idx, Matrix<long>& cshuf)
{
    int swp, n=idx.size();
    static std::vector<double> cf;
    double pf;
    if (cf.size()!=n) cf.resize(n);
    int cmin, j=0;
    for (int i=n-1; i>0; --i)
    {
        cmin = cshuf(i,idx[0]);
        for (int k=1; k<=i; ++k) if (cshuf(i,idx[k])<cmin) cmin=cshuf(i,idx[k]);
        cf[0]=1.0/(1.0+cshuf(i,idx[0])-cmin);
        for (int k=1; k<=i; ++k) cf[k]=cf[k-1]+1.0/(1.0+cshuf(i,idx[k])-cmin);
        
        pf=rndu()*cf[i];
        j=i; for (int k=0; k<i; ++k) if (pf<cf[k]) { j=k; break; }        
        swp=idx[i];
        idx[i]=idx[j];
        idx[j]=swp;
    }    
    for (int k=0; k<n; ++k) cshuf(k, idx[k])++;
    //std::cerr<<" ====== cshuf ====== \n";  for (int i=0; i<n; ++i) { for (int j=0; j<n; ++j) std::cerr<<cshuf(i,j)<<" "; std::cerr<<"\n";}
}

double perm2(const Matrix<double>& mtx)
{
    int n=mtx.size(); 
    Matrix<long> cshuf(n,n); 
    
    std::vector<int> idx(n);
    for (int i=0; i<n; ++i) idx[i]=i;
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j) cshuf(i,j)=0;
    
    double pi, prm=0, prm2=0, fn=fact(n);
    int i=0, pstride=n*2000;     
    while (true)
    {
        if (i%n==0) shuffle2(idx, cshuf);
        else { for (int i=0; i<n; ++i) idx[i]=(idx[i]+1)%n; }
        
        //for (int j=0; j<n; ++j) std::cerr<<idx[j]<<" ";        std::cerr<<"\n";
        pi = 1.;
        for (int j=0; j<n; ++j)
            pi *= mtx(j, idx[j]);
            
        prm += pi;
        prm2 += pi*pi;
        i+=1;
        if (i%pstride==0) 
        {
            double err=sqrt((prm2-prm*prm/i)/i/(i-1) ) / (prm/i);
            std::cout <<i<< " "<<std::setprecision(10)<<fn*prm/i<< " "<<err<< "\n";
            if (err<1e-4) break;
        }
    }
    return prm/i*fact(n);
}

/* end experimental */

int main(int argc, char *argv[])
{
    int n;
    std::istringstream iargi(argv[1]);
    iargi>>n;

    srand(1234);
    Matrix<double> mtx(n,n);
   
    for (int i=0; i<n; ++i) for (int j=0; j<n; ++j)
        std::cin >> mtx(i,j); 
    
    double pp = perm(mtx,1e-4);
    std::cout << "PERMANENT "<<std::setprecision(10)<<pp<<"\n";
    
}



