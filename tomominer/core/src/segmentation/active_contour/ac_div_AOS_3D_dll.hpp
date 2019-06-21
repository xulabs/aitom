// code adapted from 
// http://www.mathworks.com/matlabcentral/fileexchange/24998-2d3d-image-segmentation-toolbox

#ifndef __ZY_ac_div_AOS_3D_dll_hpp__
#define __ZY_ac_div_AOS_3D_dll_hpp__

#include <exception>

#define DEBUG

#define PRINT_MACRO(str, p, N) \
    printf("%s", str); \
    for(int ii = 0; ii<N; ii++) printf("%f, ", p[ii]); \
    printf("\n"); 

void ac_tridiagonal_Thomas_decomposition(double* alpha, double* beta, double* gamma, 
										 double* l, double* m, double* r, unsigned long N);

void ac_tridiagonal_Thomas_solution(double* l, double* m, double* r, double* d, 
									double* y, unsigned long N);

template< class T>
class C3DImage
{
    public:
        unsigned int m_dims[3]; // [row, column, plane]
    private:
        T*  m_pV; 
        int m_plane_size;
    public:
        C3DImage(T* pV, const unsigned int* dims) : m_pV(pV)
        {
            for(int i = 0; i< 3; i++) m_dims[i] = dims[i]; 
            m_plane_size = m_dims[0]*m_dims[1];
        }
        
        void GetColon1(double* p, int idx1, int idx2) // (:,idx1,idx2)
        {
            int len = m_dims[0]; 
            T* pV = &(m_pV[idx1*m_dims[0]+idx2*m_plane_size]);
            for(int i = 0; i<len; i++)
            {
                p[i] = (double)(*pV);
                pV ++; 
            }
        }
        
        void SetColon1(double* p, int idx1, int idx2)
        {
            int len = m_dims[0]; 
            T* pV = &(m_pV[idx1*m_dims[0]+idx2*m_plane_size]);
            for(int i = 0; i<len; i++)
            {
                *pV = p[i];
                pV ++; 
            }
        }
        
        void AddColon1(double* p, int idx1, int idx2)
        {
            int len = m_dims[0]; 
            T* pV = &(m_pV[idx1*m_dims[0]+idx2*m_plane_size]);
            for(int i = 0; i<len; i++)
            {
                *pV = *pV + p[i];
                pV ++; 
            }
        }
        
        void GetColon2(double* p, int idx1, int idx2) // (idx1,:,idx2)
        {
            int len = m_dims[1];
            T* pV = &(m_pV[idx1+idx2*m_plane_size]); 
            for(int i = 0; i<len; i++)
            {
                p[i] = (double)(*pV); 
                pV += m_dims[0]; 
            }
        }
        
        void SetColon2(double* p, int idx1, int idx2)
        {
            int len = m_dims[1];
            T* pV = &(m_pV[idx1+idx2*m_plane_size]); 
            for(int i = 0; i<len; i++)
            {
                *pV = p[i]; 
                pV += m_dims[0]; 
            }
        }
        
        void AddColon2(double* p, int idx1, int idx2)
        {
            int len = m_dims[1];
            T* pV = &(m_pV[idx1+idx2*m_plane_size]); 
            for(int i = 0; i<len; i++)
            {
                *pV = *pV + p[i]; 
                pV += m_dims[0]; 
            }
        }
        
        void GetColon3(double* p, int idx1, int idx2) // (idx1,idx2,:)
        {
            int len = m_dims[2];
            T* pV = &(m_pV[idx1+idx2*m_dims[0]]); 
            for(int i = 0; i<len; i++)
            {
                p[i] = (double)(*pV); 
                pV += m_plane_size; 
            }
        }
        
        void SetColon3(double* p, int idx1, int idx2)
        {
            int len = m_dims[2];
            T* pV = &(m_pV[idx1+idx2*m_dims[0]]); 
            for(int i = 0; i<len; i++)
            {
                *pV = p[i];
                pV += m_plane_size; 
            }
        }
        
        void AddColon3(double* p, int idx1, int idx2)
        {
            int len = m_dims[2];
            T* pV = &(m_pV[idx1+idx2*m_dims[0]]); 
            for(int i = 0; i<len; i++)
            {
                *pV = *pV + p[i];
                pV += m_plane_size; 
            }
        }
};

void div_AOS_1D(double* p, double delta_t, int len, // input
    // because this functinon is called repeatedly by outside, so it's better 
    // to let the caller manages the memory of the following variables
    double* alpha, double *beta, double *gamma, // inputs
    double* L, double *M, double *R) // outputs
{
    double s = 9*delta_t; 
    
    gamma[0] = -s*(p[0] + p[1])/2;
    beta[0] = gamma[0]; alpha[0] = 3 - gamma[0];
    for(int i = 1; i < len-1; i++)
    {
        gamma[i] = -s*(p[i] + p[i+1])/2;
        beta[i] = gamma[i];
        alpha[i] = 3 - (gamma[i-1] + beta[i]); 
    }
    alpha[len-1] = 3 - beta[len-2]; 
    
//     PRINT_MACRO(" alpha : ", alpha, len);
//     PRINT_MACRO(" beta  : ", beta, len-1);
//     PRINT_MACRO(" gamma : ", gamma, len-1);
    
    ac_tridiagonal_Thomas_decomposition(alpha, beta, gamma, 
        L, M, R, len);       
}

#define NEW_MACRO(x)          \
    L = new double[x-1];      \
    M = new double[x];        \
    R = new double[x-1];      \
    alpha = new double[x];    \
    beta =  new double[x-1];  \
    gamma = new double[x-1];  \
    p = new double[x];        \
    d = new double[x];        \
    out = new double[x]; 

#define DELETE_MACRO    \
    delete []alpha;     \
    delete []beta;      \
    delete []gamma;     \
    delete []L;         \
    delete []M;         \
    delete []R;         \
    delete []p;         \
    delete []d;         \
    delete []out;

template< class T > 
void div_AOS_3D(C3DImage<double>& phi,  T& g, double delta_t,
    C3DImage<double>& phi_n)
{
    int j, k; 
    double *p, *d, *out, *L, *M, *R; 
    double *alpha, *beta, *gamma; 
    
    // Along the plane direction. 
    NEW_MACRO(g.m_dims[2]);
    for(j = 0; j < g.m_dims[0]; j++)
    {
        for(k = 0; k < g.m_dims[1]; k++)
        {
            g.GetColon3(p, j, k); 
            div_AOS_1D(p, delta_t, g.m_dims[2],alpha, beta, gamma, L, M, R);  
//             PRINT_MACRO(" L : ", L, g.m_dims[2]-1);
//             PRINT_MACRO(" M : ", M, g.m_dims[2]);
//             PRINT_MACRO(" R : ", R, g.m_dims[2]-1);
            phi.GetColon3(d, j, k); 
            ac_tridiagonal_Thomas_solution(L, M, R, d, out, g.m_dims[2]);
            phi_n.SetColon3(out, j, k);
        }
    }
    DELETE_MACRO;
    
    // Along the column direction. 
    NEW_MACRO(g.m_dims[1]);
    for(j = 0; j < g.m_dims[0]; j++)
    {
        for(k = 0; k < g.m_dims[2]; k++)
        {
            g.GetColon2(p, j, k); 
            div_AOS_1D(p, delta_t, g.m_dims[1],alpha, beta, gamma, L, M, R);  
            phi.GetColon2(d, j, k); 
            ac_tridiagonal_Thomas_solution(L, M, R, d, out, g.m_dims[1]);
            phi_n.AddColon2(out, j, k);
        }
    }
    DELETE_MACRO;
    
    // Along the row direction. 
    NEW_MACRO(g.m_dims[0]);
    for(j = 0; j < g.m_dims[1]; j++)
    {
        for(k = 0; k < g.m_dims[2]; k++)
        {
            g.GetColon1(p, j, k); 
            div_AOS_1D(p, delta_t, g.m_dims[0],alpha, beta, gamma, L, M, R);  
            phi.GetColon1(d, j, k); 
            ac_tridiagonal_Thomas_solution(L, M, R, d, out, g.m_dims[0]);
            phi_n.AddColon1(out, j, k);
        }
    }
    DELETE_MACRO;
}



void ac_tridiagonal_Thomas_decomposition(double* alpha, double* beta, double* gamma, 
										 double* l, double* m, double* r, unsigned long N)
{
	m[0] = alpha[0];
	for(unsigned long int i=0; i<N-1; i++)
	{
		r[i] = beta[i];
                l[i] = gamma[i]/m[i];
		m[i+1] = alpha[i+1] - l[i]*beta[i];
	}
}

void ac_tridiagonal_Thomas_solution(double* l, double* m, double* r, double* d, 
									double* y, unsigned long N)
{
	unsigned long i,idx;
	double *yy = new double[N];
    
	// forward
	yy[0] = d[0];
	for( i = 1; i<N; ++i)
		yy[i] = d[i] - l[i-1]*yy[i-1];

	// backward
	y[N-1] = yy[N-1]/m[N-1];
	for( i = N-1; i > 0; i--)
	{
		idx = i-1;

		y[idx] = (yy[idx] - r[idx]*y[idx+1])/m[idx];
        }

	delete [] yy;
}

#endif //__ZY_ac_div_AOS_3D_dll_hpp__

