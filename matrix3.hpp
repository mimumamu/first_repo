/**
 *  3-dimensional vector & matrix class
 */
#ifndef FLOW_MATRIX3_MATRIX3_HPP
#define FLOW_MATRIX3_MATRIX3_HPP

#if !defined(__CUDACC__)
    #define __host__
    #define __device__ 
#endif

#include <iostream>
#include <cmath>

namespace flow {

//[i] row
//[j] col

//      matrix3_t      vector3_t
//
//      j=0 j=1 j=2
// i=0  [O] [O] [O]     i=0  [O]
// i=1  [O] [O] [O]     i=1  [O]
// i=2  [O] [O] [O]     i=2  [O]
template<typename T>
class vector3_t
{
    T val_[3];
public:
    __host__ __device__ vector3_t();
    __host__ __device__ vector3_t(T f0,T f1,T f2);
    __host__ __device__ vector3_t(const vector3_t& other);
    __host__ __device__ ~vector3_t();
    __host__ __device__ T  operator()(int i) const;
    __host__ __device__ T& operator()(int i);
    __host__ __device__ T  operator[](int i) const;
    __host__ __device__ T& operator[](int i);
    __host__ __device__ vector3_t& operator= (const vector3_t &A);
    __host__ __device__ vector3_t& operator+=(const vector3_t &A);
    __host__ __device__ vector3_t& operator-=(const vector3_t &A);
    __host__ __device__ vector3_t& operator*=(T k);
    __host__ __device__ vector3_t& operator/=(T k);
    __host__ __device__ vector3_t  operator/(T k) const;
    __host__ __device__ vector3_t  operator+() const;
    __host__ __device__ vector3_t  operator-() const;
    __host__ __device__ T          dot  (const vector3_t& A) const;
    __host__ __device__ vector3_t  cross(const vector3_t& A) const;
    __host__ __device__ T          norm() const;
    __host__ __device__ vector3_t& normalize();
    __host__ __device__ vector3_t  normalized() const;
};
template<typename T> __host__ __device__ vector3_t<T> operator+(const vector3_t<T>& a, const vector3_t<T>& b);
template<typename T> __host__ __device__ vector3_t<T> operator-(const vector3_t<T>& a, const vector3_t<T>& b);
template<typename T> __host__ __device__ vector3_t<T> operator*(const vector3_t<T>& a, T k);
template<typename T> __host__ __device__ vector3_t<T> operator*(T k, const vector3_t<T>& a);

template<typename T>
class matrix3_t
{
    T val_[9];
    __host__ __device__ static inline int _idx(int i, int j) { return 3*i+j; }

public:
    __host__ __device__ matrix3_t();
    __host__ __device__ matrix3_t(T f00,T f01,T f02,
                                  T f10,T f11,T f12,
                                  T f20,T f21,T f22);
    __host__ __device__ matrix3_t(const matrix3_t& other);
    __host__ __device__  ~matrix3_t();
    __host__ __device__ T  operator()(int i, int j)const;
    __host__ __device__ T& operator()(int i, int j);
    __host__ __device__ matrix3_t& operator= (const matrix3_t& A);
    __host__ __device__ matrix3_t& operator+=(const matrix3_t& A);
    __host__ __device__ matrix3_t& operator-=(const matrix3_t& A);
    __host__ __device__ matrix3_t& operator*=(const matrix3_t& A);
    __host__ __device__ matrix3_t& operator*=(T k);
};
template<typename T> __host__ __device__ matrix3_t<T> operator+(const matrix3_t<T>& a, const matrix3_t<T>& b);
template<typename T> __host__ __device__ matrix3_t<T> operator-(const matrix3_t<T>& a, const matrix3_t<T>& b);
template<typename T> __host__ __device__ matrix3_t<T> operator*(const matrix3_t<T>& a, const matrix3_t<T>& b);
template<typename T> __host__ __device__ vector3_t<T> operator*(const matrix3_t<T>& A, const vector3_t<T>& b);
template<typename T> __host__ __device__ matrix3_t<T> operator*(const matrix3_t<T>& A, T k);
template<typename T> __host__ __device__ matrix3_t<T> operator*(T k, const matrix3_t<T>& A);

// functions
template<typename T> __host__ __device__ T            dot  (const vector3_t<T>& A,const vector3_t<T>& B);
template<typename T> __host__ __device__ vector3_t<T> cross(const vector3_t<T>& A,const vector3_t<T>& B);
template<typename T> __host__ __device__ T det(const matrix3_t<T>& A);
template<typename T> __host__ __device__ matrix3_t<T> transpose (const matrix3_t<T>& A);
template<typename T> __host__ __device__ matrix3_t<T> inv(const matrix3_t<T>& A);
template<typename T> __host__ __device__ matrix3_t<T> rotX(T theta_x);
template<typename T> __host__ __device__ matrix3_t<T> rotY(T theta_y);
template<typename T> __host__ __device__ matrix3_t<T> rotZ(T theta_z);
template<typename T> __host__ __device__ matrix3_t<T> rotZ_rotY_rotX(T theta_x,T theta_y,T theta_z);
// (rotZ_rotY_rotX) == (rotZ * rotY * rotX)

template<typename T> inline std::ostream& operator<<(std::ostream& os, const vector3_t<T>& vec);
template<typename T> inline std::ostream& operator<<(std::ostream& os, const matrix3_t<T>& mat);

// typedefs
typedef vector3_t<float>  vector3f;
typedef vector3_t<double> vector3d;

typedef matrix3_t<float>  matrix3f;
typedef matrix3_t<double> matrix3d;


//=======================================================================================
// DEFINE  
//=======================================================================================

// vector::ctor/dtor ----------
template<typename T>
inline __host__ __device__ 
vector3_t<T>::vector3_t() 
{}

template<typename T>
inline __host__ __device__
vector3_t<T>::vector3_t(T f0,T f1,T f2)
{
    val_[0] = f0; val_[1] = f1; val_[2] = f2;
}

template<typename T>
inline __host__ __device__
vector3_t<T>::vector3_t(const vector3_t<T>& A)
{
    for(int i=0;i<3;++i) (*this)(i) = A(i);
}

template<typename T>
inline __host__ __device__
vector3_t<T>::~vector3_t() 
{}

// vector::access ----------
template<typename T>
inline __host__ __device__
T  vector3_t<T>::operator()(int i) const
{
    return val_[i]; 
}

template<typename T>
inline __host__ __device__
T& vector3_t<T>::operator()(int i)
{
    return val_[i]; 
}

template<typename T>
inline __host__ __device__
T  vector3_t<T>::operator[](int i) const
{
    return val_[i]; 
}

template<typename T>
inline __host__ __device__
T& vector3_t<T>::operator[](int i)
{
    return val_[i]; 
}

// vector::assign ----------
template<typename T>
inline __host__ __device__
vector3_t<T>& vector3_t<T>::operator=(const vector3_t<T>& A)
{
    for(int i=0;i<3;++i) (*this)(i) = A(i);
    return (*this);
}

// vector::operator + ----------
template<typename T>
inline __host__ __device__
vector3_t<T>& vector3_t<T>::operator+=(const vector3_t<T>& A)
{
    for(int i=0;i<3;++i) (*this)(i) += A(i);
    return *this;
}

template<typename T>
inline __host__ __device__
vector3_t<T> operator+(const vector3_t<T>& a, const vector3_t<T>& b)
{
    return vector3_t<T>(a) += b;
}

template<typename t>
inline __host__ __device__
vector3_t<t> vector3_t<t>::operator+() const
{
    return *this;
}

// vector::operator - ----------
template<typename T>
inline __host__ __device__
vector3_t<T>& vector3_t<T>::operator-=(const vector3_t<T>& A)
{
    for(int i=0;i<3;++i) (*this)(i) -= A(i);
    return *this;
}

template<typename T>
inline __host__ __device__
vector3_t<T> operator-(const vector3_t<T>& a, const vector3_t<T>& b)
{
    return vector3_t<T>(a) -= b;
}

template<typename t>
inline __host__ __device__
vector3_t<t> vector3_t<t>::operator-() const
{
    return vector3_t<t>(-(*this)(0), -(*this)(1), -(*this)(2));
}

// vector::operator * ----------
template<typename T>
inline __host__ __device__
vector3_t<T>& vector3_t<T>::operator*=(T k)
{
    for(int i=0;i<3;++i) (*this)(i) *= k;
    return *this;
}

template<typename T> 
inline __host__ __device__ 
vector3_t<T> operator*(const vector3_t<T>& a, T k)
{
    return vector3_t<T>(a) *= k;
}

template<typename T> 
inline __host__ __device__ 
vector3_t<T> operator*(T k, const vector3_t<T>& a)
{
    return a * k;
}

// vector::operator / ----------
template<typename T>
inline __host__ __device__
vector3_t<T>& vector3_t<T>::operator/=(T k)
{
    for(int i=0;i<3;++i) (*this)(i) /= k;
    return *this;
}

template<typename T>
inline __host__ __device__
vector3_t<T> vector3_t<T>::operator/(T k) const
{
    return (*this) /= k;
}

// vector::functions ----------
template<typename T>
inline __host__ __device__
T vector3_t<T>::dot(const vector3_t<T>& A) const
{
    return A(0)*(*this)(0)
         + A(1)*(*this)(1)
         + A(2)*(*this)(2);
}

template<typename T>
inline __host__ __device__
vector3_t<T> vector3_t<T>::cross(const vector3_t<T>& A) const
{
	return vector3_t<T>(
        (*this)(1)*A(2) - (*this)(2)*A(1),
        (*this)(2)*A(0) - (*this)(0)*A(2),
        (*this)(0)*A(1) - (*this)(1)*A(0) );
	/*
    return vector3_t<T>(
        A(1)*(*this)(2) - A(2)*(*this)(1),
        A(2)*(*this)(0) - A(0)*(*this)(2),
        A(0)*(*this)(1) - A(1)*(*this)(0) );
	*/
}

template<typename T>
inline __host__ __device__
T vector3_t<T>::norm() const
{
    return sqrt(val_[0]*val_[0] + val_[1]*val_[1] + val_[2]*val_[2]);
}

template<typename T>
inline __host__ __device__
vector3_t<T>& vector3_t<T>::normalize()
{
    return *this /= norm();
}

template<typename T>
inline __host__ __device__
vector3_t<T> vector3_t<T>::normalized() const
{
    return vector3_t<T>(*this) /= norm();
}
// matrix_t --------------------------------------

// matrix::ctor/dtor ----------
template<typename T>
inline __host__ __device__
matrix3_t<T>::matrix3_t()
{}

template<typename T>
inline __host__ __device__
matrix3_t<T>::matrix3_t(T f00,T f01,T f02,
                        T f10,T f11,T f12,
                        T f20,T f21,T f22)
{
    val_[_idx(0,0)]=f00; val_[_idx(0,1)]=f01; val_[_idx(0,2)]=f02;
    val_[_idx(1,0)]=f10; val_[_idx(1,1)]=f11; val_[_idx(1,2)]=f12;
    val_[_idx(2,0)]=f20; val_[_idx(2,1)]=f21; val_[_idx(2,2)]=f22;
}

template<typename T>
inline __host__ __device__
matrix3_t<T>::matrix3_t(const matrix3_t<T>& A)
{
    for(int i=0;i<3;++i)
    for(int j=0;j<3;++j)
        (*this)(i,j) = A(i,j);
}

template<typename T>
inline __host__ __device__
matrix3_t<T>::~matrix3_t()
{}

// matrix::access ----------
template<typename T>
inline __host__ __device__
T  matrix3_t<T>::operator()(int i,int j) const
{
    return val_[_idx(i,j)]; 
}

template<typename T>
inline __host__ __device__
T& matrix3_t<T>::operator()(int i,int j)
{
    return val_[_idx(i,j)]; 
}

// matrix::assign ----------
template<typename T>
inline __host__ __device__
matrix3_t<T>& matrix3_t<T>::operator=(const matrix3_t<T>& A)
{
    for(int i=0;i<3;++i)
    for(int j=0;j<3;++j)
        (*this)(i,j) = A(i,j);
    return (*this);
}

// matrix::operator + ----------
template<typename T>
inline __host__ __device__
matrix3_t<T>& matrix3_t<T>::operator+=(const matrix3_t<T>& A)
{
    for(int i=0;i<3*3;++i) val_[i] += A.val_[i];
    return *this;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> operator+(const matrix3_t<T>& a, const matrix3_t<T>& b)
{
    return matrix3_t<T>(a) += b;
}

// matrix::operator - ----------
template<typename T>
inline __host__ __device__
matrix3_t<T>& matrix3_t<T>::operator-=(const matrix3_t<T>& A)
{
    for(int i=0;i<3*3;++i) val_[i] -= A.val_[i];
    return *this;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> operator-(const matrix3_t<T>& a, const matrix3_t<T>& b)
{
    return matrix3_t<T>(a) -= b;
}

// matrix::operator * ----------
template<typename T>
inline __host__ __device__
matrix3_t<T>& matrix3_t<T>::operator*=(T k)
{
    for(int i=0;i<3*3;++i) val_[i] *= k;
    return *this;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> operator*(const matrix3_t<T>& A, T k)
{
    return matrix3_t<T>(A) *= k;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> operator*(T k, const matrix3_t<T>& A)
{
    return A * k;
}

template<typename T>
inline __host__ __device__
matrix3_t<T>& matrix3_t<T>::operator*=(const matrix3_t<T>& A)
{
    matrix3_t<T> tmp = (*this);
    for(int j=0;j<3;++j){
        (*this)(0,j) = tmp(0,0)*A(0,j) + tmp(0,1)*A(1,j) + tmp(0,2)*A(2,j);
        (*this)(1,j) = tmp(1,0)*A(0,j) + tmp(1,1)*A(1,j) + tmp(1,2)*A(2,j);
        (*this)(2,j) = tmp(2,0)*A(0,j) + tmp(2,1)*A(1,j) + tmp(2,2)*A(2,j);
    }
    return *this;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> operator*(const matrix3_t<T>& A, const matrix3_t<T>& B)
{
    matrix3_t<T> answer;
    for(int j=0;j<3;++j){
        answer(0,j) = A(0,0)*B(0,j) + A(0,1)*B(1,j) + A(0,2)*B(2,j);
        answer(1,j) = A(1,0)*B(0,j) + A(1,1)*B(1,j) + A(1,2)*B(2,j);
        answer(2,j) = A(2,0)*B(0,j) + A(2,1)*B(1,j) + A(2,2)*B(2,j);
    }
    return answer;
}

template<typename T>
inline __host__ __device__
vector3_t<T> operator*(const matrix3_t<T>& A, const vector3_t<T>& b)
{
    vector3_t<T> answer;
    answer(0) = A(0,0)*b(0) + A(0,1)*b(1) + A(0,2)*b(2);
    answer(1) = A(1,0)*b(0) + A(1,1)*b(1) + A(1,2)*b(2);
    answer(2) = A(2,0)*b(0) + A(2,1)*b(1) + A(2,2)*b(2);
    return answer;
}

// functions ----------

template<typename T>
inline __host__ __device__
T dot(const vector3_t<T>& A,const vector3_t<T>& B)
{
    return A.dot(B);
}

template<typename T>
inline __host__ __device__
vector3_t<T> cross(const vector3_t<T>& A,const vector3_t<T>& B)
{
    return A.cross(B);
    
}

template<typename T>
inline __host__ __device__
T det(const matrix3_t<T>& A){
    T answer = 0.0;
    answer += A(0,0)*( A(1,1)*A(2,2)-A(1,2)*A(2,1) );
    answer -= A(0,1)*( A(1,0)*A(2,2)-A(1,2)*A(2,0) );
    answer += A(0,2)*( A(1,0)*A(2,1)-A(1,1)*A(2,0) );
    return answer;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> transpose(const matrix3_t<T>& A)
{
    matrix3_t<T> answer;
    for(int i=0;i<3;++i)
    for(int j=0;j<3;++j)
        answer(j,i) = A(i,j);
    return answer;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> inv(const matrix3_t<T>& A){
    T _det = det(A);

    T inv_det = 1.0/_det;
    matrix3_t<T> answer;
    answer(0,0) = +inv_det * ( A(1,1)*A(2,2) - A(1,2)*A(2,1) );
    answer(0,1) = -inv_det * ( A(0,1)*A(2,2) - A(0,2)*A(2,1) );
    answer(0,2) = +inv_det * ( A(0,1)*A(1,2) - A(0,2)*A(1,1) );
    answer(1,0) = -inv_det * ( A(1,0)*A(2,2) - A(1,2)*A(2,0) );
    answer(1,1) = +inv_det * ( A(0,0)*A(2,2) - A(0,2)*A(2,0) );
    answer(1,2) = -inv_det * ( A(0,0)*A(1,2) - A(0,2)*A(1,0) );
    answer(2,0) = +inv_det * ( A(1,0)*A(2,1) - A(1,1)*A(2,0) );
    answer(2,1) = -inv_det * ( A(0,0)*A(2,1) - A(0,1)*A(2,0) );
    answer(2,2) = +inv_det * ( A(0,0)*A(1,1) - A(0,1)*A(1,0) );
    return answer;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> rotX(T theta_x){
    matrix3_t<T> answer( 1.0 , 0.0          , 0.0           ,
                         0.0 , +cos(theta_x), -sin(theta_x) ,
                         0.0 , +sin(theta_x), +cos(theta_x) );
    return answer;
}
template<typename T>
inline __host__ __device__
matrix3_t<T> rotY(T theta_y){
    matrix3_t<T> answer( +cos(theta_y) , 0.0 , +sin(theta_y) ,
                         0.0           , 1.0 , 0.0           ,
                         -sin(theta_y) , 0.0 , +cos(theta_y) );
    return answer;
}

template<typename T>
inline __host__ __device__
matrix3_t<T> rotZ(T theta_z){
    matrix3_t<T> answer( +cos(theta_z) , -sin(theta_z) , 0.0 ,
                         +sin(theta_z) , +cos(theta_z) , 0.0 ,
                         0.0           , 0.0           , 1.0 );
    return answer;
}
template<typename T>
inline __host__ __device__
matrix3_t<T> rotZ_rotY_rotX(T theta_x,T theta_y,T theta_z){
    const T sx = sin(theta_x);
    const T sy = sin(theta_y);
    const T sz = sin(theta_z);
    const T cx = cos(theta_x);
    const T cy = cos(theta_y);
    const T cz = cos(theta_z);
    matrix3_t<T> answer( +cy*cz, -cx*sz+sx*sy*cz, +sx*sz+cx*sy*cz,
                         +cy*sz, +cx*cz+sx*sy*sz, -sx*cz+cx*sy*sz,
                         -sy   , +sx*cy         , +cx*cy         );
    return answer;
}

template<typename T> 
inline std::ostream& operator<<(std::ostream& os, const vector3_t<T>& vec)
{
    os << vec(0) << '\n' << vec(1) << '\n' << vec(2);
    return os;
}
template<typename T> 
inline std::ostream& operator<<(std::ostream& os, const matrix3_t<T>& mat)
{
    os << mat(0, 0) << ' ' << mat(0, 1) << ' ' << mat(0, 2) << '\n'
       << mat(1, 0) << ' ' << mat(1, 1) << ' ' << mat(1, 2) << '\n'
       << mat(2, 0) << ' ' << mat(2, 1) << ' ' << mat(2, 2);
    return os;
}

} // namespace flow
#endif // FLOW_MATRIX3_MATRIX3_HPP
