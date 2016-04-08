#include <iostream>
#include <vector>
#include <string>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_rng.h>
#include <ctime>
#include <cstdlib>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_permutation.h>
#include <gsl/gsl_linalg.h>
#include <math.h>
#include <fstream>

#define MNumArray 200
#define NumLattice 8
#define NumDim 4
#define BETA 1.719
#define U0 0.797


using namespace std;

// Returns a random number in [-1, 1)
double unirand(void){
    double u = ((double) rand() / (RAND_MAX) * 2.0 - 1.0);
    //printf ("%.5f\n", u);
    return u;
}

// Print a matrix
void prtmatrix(const gsl_matrix_complex *matrix){
    printf("matrix:\n");
    int i, j;
    int row = matrix->size1, colume = matrix->size2;
    gsl_complex matrixelem;
    for (i = 0; i < row; i++){
        for (j = 0; j < colume; j++){
            matrixelem = gsl_matrix_complex_get (matrix, i, j);
            printf ("(%d ,%d) = %g + i %g\n", i, j, GSL_REAL (matrixelem), GSL_IMAG (matrixelem));
        }
    }
}

// Print a complex number
void prtcplx(const gsl_complex cplx){
    printf("complex number:\n");
    printf ("%g + i %g\n", GSL_REAL (cplx), GSL_IMAG (cplx));
}

// Print a vector
void prtvector(const gsl_vector_complex *vector){
    printf("vector:\n");
    int i;
    gsl_complex vectorelem;
    int vz = vector->size;
    for (i = 0; i < vz; i++){
        vectorelem = gsl_vector_complex_get (vector, i);
        printf ("(%d) = %g + i %g\n", i, GSL_REAL (vectorelem), GSL_IMAG (vectorelem));
    }
}

// Find index number of matrix array from given coordinates and direction of U matrix
int get_index(const int mu, const int sit_index[4]){
    int indx = 0;
    if ((mu < NumDim) && (sit_index[0] < NumLattice) && (sit_index[1] < NumLattice) && (sit_index[2] < NumLattice) && (sit_index[3] < NumLattice)){
        indx = mu;
        indx = sit_index[0] * NumDim + indx;
        indx = sit_index[1] * NumLattice * NumDim + indx;
        indx = sit_index[2] * NumLattice * NumLattice * NumDim + indx;
        indx = sit_index[3] * NumLattice * NumLattice * NumLattice * NumDim + indx;
        //cout << indx << endl;
        return indx;
    }
    else{
        cout << "error: out of Largest Dimension or Lattice site" << endl;
        cout << sit_index[0]<<sit_index[1]<<sit_index[2]<<sit_index[3]<<endl;
        exit (EXIT_FAILURE);
    }
}

// This function calculate two matrixes multiplication
// The resault is stored in matrix1
void MxtMx(gsl_matrix_complex *matrix1, const gsl_matrix_complex *matrix2){
    gsl_matrix_complex * ZEROMX = gsl_matrix_complex_calloc (3, 3);
    gsl_blas_zgemm (CblasNoTrans, CblasNoTrans, gsl_complex_rect(1.0, 0.0), matrix1, matrix2, gsl_complex_rect (0.0, 0.0), ZEROMX);
    gsl_matrix_complex_memcpy (matrix1, ZEROMX);
    //prtmatrix(matrix1);
    gsl_matrix_complex_free(ZEROMX);
}

void MxtMx_store(const gsl_matrix_complex *matrix1, gsl_matrix_complex *matrix2){
    gsl_matrix_complex * ZEROMX = gsl_matrix_complex_calloc (3, 3);
    gsl_blas_zgemm (CblasNoTrans, CblasNoTrans, gsl_complex_rect(1.0, 0.0), matrix1, matrix2, gsl_complex_rect (0.0, 0.0), ZEROMX);
    gsl_matrix_complex_memcpy (matrix2, ZEROMX);
    //prtmatrix(matrix1);
    gsl_matrix_complex_free(ZEROMX);
}

// conjugate transpose of matrix
// result stores in matrix
// original matrix will be changed
void conjtrans(gsl_matrix_complex *matrix){
    gsl_matrix_complex * ZEROMX = gsl_matrix_complex_calloc (3, 3);
    gsl_matrix_complex * IDNTMX = gsl_matrix_complex_alloc (3, 3);
    gsl_matrix_complex_set_identity (IDNTMX);

    gsl_blas_zgemm (CblasConjTrans, CblasNoTrans, gsl_complex_rect(1.0, 0.0), matrix, IDNTMX, gsl_complex_rect (0.0, 0.0), ZEROMX);
    gsl_matrix_complex_memcpy (matrix, ZEROMX);
    //prtmatrix(matrix);
    
    gsl_matrix_complex_free(ZEROMX);
    gsl_matrix_complex_free(IDNTMX);
}

/*void cross_product(const gsl_vector_complex *u, const gsl_vector_complex *v, gsl_vector_complex *product){
    gsl_complex p0 = gsl_complex_sub(gsl_complex_mul (gsl_vector_complex_get(u, 1), gsl_vector_complex_get(v, 2)),
                                gsl_complex_mul (gsl_vector_complex_get(u, 2), gsl_vector_complex_get(v, 1)));

    gsl_complex p1 = gsl_complex_sub(gsl_complex_mul (gsl_vector_complex_get(u, 2), gsl_vector_complex_get(v, 0)),
                                gsl_complex_mul (gsl_vector_complex_get(u, 0), gsl_vector_complex_get(v, 2)));

    gsl_complex p2 = gsl_complex_sub(gsl_complex_mul (gsl_vector_complex_get(u, 0), gsl_vector_complex_get(v, 1)),
                                gsl_complex_mul (gsl_vector_complex_get(u, 1), gsl_vector_complex_get(v, 0)));

    gsl_vector_complex_set(product, 0, p0);
    gsl_vector_complex_set(product, 1, p1);
    gsl_vector_complex_set(product, 2, p2);
}*/

// copy vector elements to a column of a matrix

void writeColtoMx(const gsl_vector_complex *v, gsl_matrix_complex *matrix, const int col){
    int i;
    int len = v->size;
    for (i = 0; i < len; i++){
        //cout << i <<endl;
        //prtcplx(gsl_vector_complex_get(v, i));
        gsl_matrix_complex_set(matrix, i, col, gsl_vector_complex_get(v, i));
    }
}

// This function is to add 1 unit in mu direction
// Return a new site
void site_add(int new_site[4], const int org_site[4], const int dir){
    for (int i = 0; i < 4; i++){
        new_site[i] = org_site[i];
    }
    new_site[dir] = (org_site[dir] + 1) % NumLattice;
}

void site_add2(int new_site[4], const int org_site[4], const int dir){
    for (int i = 0; i < 4; i++){
        new_site[i] = org_site[i];
    }
    new_site[dir] = (org_site[dir] + 2) % NumLattice;
}

void site_add_munu(int new_site[4], const int org_site[4], const int mu, const int nu){
    for (int i = 0; i < 4; i++){
        new_site[i] = org_site[i];
    }
    new_site[mu] = (org_site[mu] + 1) % NumLattice;
    new_site[nu] = (org_site[nu] + 1) % NumLattice;
}

// This function is to subtract 1 unit in mu direction
// Return a new_site
void site_sub(int new_site[4], const int org_site[4], int dir){
    for (int i = 0; i < 4; i++){
        new_site[i] = org_site[i];
    }
    new_site[dir] = (org_site[dir] + NumLattice - 1) % NumLattice;
}

void site_sub2(int new_site[4], const int org_site[4], int dir){
    for (int i = 0; i < 4; i++){
        new_site[i] = org_site[i];
    }
    new_site[dir] = (org_site[dir] + NumLattice - 2) % NumLattice;
}

void site_sub_munu(int new_site[4], const int org_site[4], const int mu, const int nu){
    for (int i = 0; i < 4; i++){
        new_site[i] = org_site[i];
    }
    new_site[mu] = (org_site[mu] + NumLattice - 1) % NumLattice;
    new_site[nu] = (org_site[nu] + NumLattice - 1) % NumLattice;
}

// Get determinent of a matrix
gsl_complex Get_Det(const gsl_matrix_complex *matrix){
    gsl_matrix_complex* matrixbck = gsl_matrix_complex_alloc (matrix->size1, matrix->size2);
    gsl_matrix_complex_memcpy (matrixbck, matrix);
    gsl_permutation* permutation = gsl_permutation_alloc(matrix->size1);
    //gsl_permutation_init(permutation);
    int s;
    gsl_linalg_complex_LU_decomp (matrixbck, permutation, &s);
    gsl_complex det = gsl_linalg_complex_LU_det(matrixbck, s);

    gsl_matrix_complex_free(matrixbck);
    gsl_permutation_free (permutation);
    //prtcplx(det);
    return det;
}

// unitarize the matrix
// input matrix will be changed
void untrzMx(gsl_matrix_complex *matrix){
    gsl_vector_complex *col0 = gsl_vector_complex_alloc(3);
    gsl_vector_complex *col1 = gsl_vector_complex_alloc(3);
    gsl_vector_complex *col2 = gsl_vector_complex_alloc(3);
    gsl_vector_complex *col0bc = gsl_vector_complex_alloc(3);
    gsl_vector_complex *col1bc = gsl_vector_complex_alloc(3);
    gsl_vector_complex *col2bc = gsl_vector_complex_alloc(3);
    gsl_complex inner00, inner01, inner02, inner12, inner11;
    
    //cout <<"UUU"<<endl;
    gsl_complex det = Get_Det(matrix);
    
    gsl_matrix_complex_get_col (col0, matrix, 0);
    gsl_matrix_complex_get_col (col1, matrix, 1);
    gsl_matrix_complex_get_col (col2, matrix, 2);
    gsl_vector_complex_memcpy (col2bc, col2);

    gsl_vector_complex_memcpy (col0bc, col0);
    gsl_blas_zdotc (col0, col0, &inner00);
    gsl_blas_zdotc (col0, col1, &inner01);
    gsl_vector_complex_scale (col0bc, gsl_complex_div (inner01, inner00));
    gsl_vector_complex_sub (col1, col0bc);
    //cout<<"recol0 & 1"<<endl;
    //prtvector(col0);
    //prtvector(col1);

    gsl_vector_complex_memcpy (col0bc, col0);
    gsl_vector_complex_memcpy (col1bc, col1);
    gsl_blas_zdotc (col0, col0, &inner00);
    gsl_blas_zdotc (col0, col2, &inner02);
    gsl_vector_complex_scale (col0bc, gsl_complex_div (inner02, inner00));
    gsl_vector_complex_sub (col2, col0bc);
    gsl_blas_zdotc (col1, col1, &inner11);
    gsl_blas_zdotc (col1, col2, &inner12);
    gsl_vector_complex_scale (col1bc, gsl_complex_div (inner12, inner11));
    gsl_vector_complex_sub (col2, col1bc);

    //prtvector(col0);
    //prtcplx(gsl_complex_rect (1.0 / gsl_blas_dznrm2 (col0), 0));
    //prtvector(col1);
    //prtcplx(gsl_complex_rect (1.0 / gsl_blas_dznrm2 (col1), 0));
    //prtvector(col2);
    //prtcplx(gsl_complex_rect (1.0 / gsl_blas_dznrm2 (col2), 0));
    
    gsl_vector_complex_scale (col0, gsl_complex_rect (1.0 / gsl_blas_dznrm2 (col0), 0.0));
    gsl_vector_complex_scale (col1, gsl_complex_rect (1.0 / gsl_blas_dznrm2 (col1), 0.0));
    gsl_vector_complex_scale (col2, gsl_complex_rect (1.0 / gsl_blas_dznrm2 (col2), 0.0));
    
    /*gsl_blas_zdotc (col0, col0, &inner00);
    gsl_blas_zdotc (col0, col1, &inner01);
    gsl_blas_zdotc (col0, col2, &inner02);
    gsl_blas_zdotc (col1, col2, &inner12);
    prtcplx(inner00);
    prtcplx(inner01);
    prtcplx(inner02);
    prtcplx(inner12);*/
    
    writeColtoMx(col0, matrix, 0);
    writeColtoMx(col1, matrix, 1);
    writeColtoMx(col2, matrix, 2);
    
    det = Get_Det(matrix);
    
    gsl_complex scale_det = gsl_complex_polar (gsl_complex_abs (det), - (gsl_complex_arg (det)) / 3.0);

    //prtmatrix(matrix);
    gsl_matrix_complex_scale (matrix, scale_det);
    det = Get_Det(matrix);
    //prtcplx(det);
    
    
    gsl_vector_complex_free (col0);
    gsl_vector_complex_free (col1);
    gsl_vector_complex_free (col2);
    gsl_vector_complex_free (col0bc);
    gsl_vector_complex_free (col1bc);
    gsl_vector_complex_free (col2bc);
}

// Returns a pointer of a random n*n hermitian matrix
// the matrix elements (both REAL and IMG parts) are random numbers between −1 and 1
// the input matrix will be changed
void randH(gsl_matrix_complex *matrix){
    int i, j;
    gsl_complex temp;
    int r = matrix->size1;
    for (i = 0; i < r; i++){
        for (j = i; j < r; j++){
            if (i == j){
                temp = gsl_complex_rect (unirand(), 0);
                gsl_matrix_complex_set (matrix, i, j, temp);
            }
            else{
                temp = gsl_complex_rect (unirand(), unirand());
                gsl_matrix_complex_set (matrix, i, j, temp);
                temp = gsl_complex_conjugate (temp);
                gsl_matrix_complex_set (matrix, j, i, temp);
            }
        }
    }
    // prtmatrix(matrix);
}

// Returns a pointer of a M matrix
// M matrix is to update link variables
// M = 1 + i * \epsilon * (Hermitian Matrix)
void randM(double min, gsl_matrix_complex *matrixM){
    gsl_matrix_complex * herm = gsl_matrix_complex_alloc (3, 3);
    randH(herm);
    gsl_matrix_complex * ZEROMX = gsl_matrix_complex_calloc (3, 3);
    gsl_matrix_complex * IDNTMX = gsl_matrix_complex_alloc (3, 3);
    gsl_matrix_complex_set_identity (IDNTMX);
    gsl_matrix_complex_set_identity (matrixM);
    gsl_complex iTmin = gsl_complex_rect (0, min);
    
    gsl_blas_zgemm (CblasNoTrans, CblasNoTrans, iTmin, herm, IDNTMX, gsl_complex_rect (1.0, 0.0), matrixM);

    untrzMx(matrixM);

    gsl_matrix_complex_free(herm);
    gsl_matrix_complex_free(ZEROMX);
    gsl_matrix_complex_free(IDNTMX);
}

// Returns a pointer of updated unitary matrix by multiplying random M and unitarizing
// the pointer matrixU will be updated
void times_randM(gsl_matrix_complex * matrixM, gsl_matrix_complex * matrixU){
    //cout << "ddd"<<endl;
    //prtmatrix(matrixM);
    MxtMx(matrixU, matrixM);
    untrzMx(matrixU);
}

//This function is to generate a serious random M matrixes
//return a array of pointers of M matrixes
void init_Mmatrix_array(gsl_matrix_complex * array[], int NumArray){
    for (int i = 0; i < NumArray/2; i++){
        array[i] = gsl_matrix_complex_alloc (3, 3);
        randM(0.24, array[i]);
    }
    for (int i = NumArray/2; i < NumArray; i++){
        array[i] = gsl_matrix_complex_alloc (3, 3);
        gsl_matrix_complex_memcpy(array[i], array[(i-NumArray/2)]);
        conjtrans(array[i]);
    }
}

// This function is to initialize Umatrix
// return pointers of Identity matrixes
void init_matrixU(gsl_matrix_complex * matrixU[], int Num){
    for (int i = 0; i < Num; i++){
        matrixU[i] = gsl_matrix_complex_alloc (3, 3);
        gsl_matrix_complex_set_identity (matrixU[i]);
    }
}

// Return the real part of trace of a matrix
double get_tr_REAL(gsl_matrix_complex * matrix){
    gsl_complex trac = gsl_complex_rect (0.0, 0.0);
    int row = matrix->size1;
    for (int i = 0; i < row; i++){
        trac = gsl_complex_add (trac, gsl_matrix_complex_get (matrix, i, i));
    }
    return GSL_REAL (trac);
}

// Return the
double get_tr_IMG(gsl_matrix_complex * matrix){
    gsl_complex trac = gsl_complex_rect (0.0, 0.0);
    int row = matrix->size1;
    for (int i = 0; i < row; i++){
        trac = gsl_complex_add (trac, gsl_matrix_complex_get (matrix, i, i));
    }
    return GSL_IMAG (trac);
}

// check if coordinates are out of the Lattice
bool check_inLattice(const int coor[4]){
    if ((coor[0] >= 0) && (coor[1] >= 0) && (coor[2] >= 0) && (coor[3] >= 0) && (coor[0] < NumLattice) && (coor[1] < NumLattice) && (coor[2] < NumLattice) && (coor[3] < NumLattice)){
        return true;
    }
    else{
        cout<<"fuck"<<endl;;
        cout<<coor[0]<<coor[1]<<coor[2]<<coor[3]<<endl;
        return false;
    }
}

// plaquette of one wilson loop
// the loop is in coordinate site_coor[4], mu and nu direction
double plqt(int mu, int nu, const int site_coor[4], gsl_matrix_complex * matrixU[]){
    gsl_matrix_complex * p = gsl_matrix_complex_alloc (3, 3);
    gsl_matrix_complex * U1 = gsl_matrix_complex_alloc (3, 3);
    gsl_matrix_complex * U2 = gsl_matrix_complex_alloc (3, 3);
    int site_addmu[4];
    int site_addnu[4];
    double p_num;

    site_add(site_addmu, site_coor, mu);
    site_add(site_addnu, site_coor, nu);

    
    if (check_inLattice(site_coor)&&check_inLattice(site_addmu)&&check_inLattice(site_addnu)){
        gsl_matrix_complex_set_identity (p);
        //site_add(site_addmu, site_coor, mu);
        //site_add(site_addnu, site_coor, nu);

        MxtMx_store(matrixU[get_index(mu, site_coor)], p);
        MxtMx_store(matrixU[get_index(nu, site_addmu)], p);

        gsl_matrix_complex_memcpy (U1, matrixU[get_index(mu, site_addnu)]);
        conjtrans(U1);
        MxtMx_store(U1, p);

        gsl_matrix_complex_memcpy (U2, matrixU[get_index(nu, site_coor)]);
        conjtrans(U2);
        MxtMx_store(U2, p);

        p_num = 1.0 / 3.0 * get_tr_REAL(p);

        if (isnan(p_num)){
            //cout <<"start"<<endl;
            //cout <<mu<<" "<<site_coor[0]<<site_coor[1]<<site_coor[2]<<site_coor[3]<<endl;
            //prtmatrix(matrixU[get_index(mu, site_coor)]);
            //cout <<nu<<" "<<site_addmu[0]<<site_addmu[1]<<site_addmu[2]<<site_addmu[3]<<endl;
            //prtmatrix(matrixU[get_index(nu, site_addmu)]);
            //prtmatrix(U1);
            //prtmatrix(U2);
            exit(1);
        }
        
        gsl_matrix_complex_free(p);
        gsl_matrix_complex_free(U2);
        gsl_matrix_complex_free(U1);

        //cout<<p_num<<endl;
        return p_num;
    }
    else {
        cout<<"fuck"<<endl;
        cout<<site_coor[0]<<site_coor[1]<<site_coor[2]<<site_coor[3]<<endl;
        cout<<site_addmu[0]<<site_addmu[1]<<site_addmu[2]<<site_addmu[3]<<endl;
        cout<<site_addnu[0]<<site_addnu[1]<<site_addnu[2]<<site_addnu[3]<<endl;
        return 0;
    }
}

double plqt_21(int mu, int nu, const int site_coor[4], gsl_matrix_complex * matrixU[]){
    gsl_matrix_complex * p = gsl_matrix_complex_alloc (3, 3);
    gsl_matrix_complex * U1 = gsl_matrix_complex_alloc (3, 3);
    gsl_matrix_complex * U2 = gsl_matrix_complex_alloc (3, 3);
    gsl_matrix_complex * U3 = gsl_matrix_complex_alloc (3, 3);
    int site_addmu[4];
    int site_addnu[4];
    int site_addmu2[4];
    int site_addmunu[4];
    double p_num;

    site_add(site_addmu, site_coor, mu);
    site_add(site_addnu, site_coor, nu);
    site_add2(site_addmu2, site_coor, mu);
    site_add_munu(site_addmunu, site_coor, mu, nu);
    
    if (check_inLattice(site_coor)&&check_inLattice(site_addmu)&&check_inLattice(site_addnu)){
        gsl_matrix_complex_set_identity (p);
        //site_add(site_addmu, site_coor, mu);
        //site_add(site_addnu, site_coor, nu);

        if (((mu == 0)&(site_coor[0]==0)&(site_coor[1]==0)&(site_coor[2]==0)&(site_coor[3]==0))or((mu == 0)&(site_addmu[0]==0)&(site_addmu[1]==0)&(site_addmu[2]==0)&(site_addmu[3]==0))or((nu == 0)&(site_addmu2[0]==0)&(site_addmu2[1]==0)&(site_addmu2[2]==0)&(site_addmu2[3]==0))or((mu == 0)&(site_addmunu[0]==0)&(site_addmunu[1]==0)&(site_addmunu[2]==0)&(site_addmunu[3]==0))or((mu == 0)&(site_addnu[0]==0)&(site_addnu[1]==0)&(site_addnu[2]==0)&(site_addnu[3]==0))or((nu == 0)&(site_coor[0]==0)&(site_coor[1]==0)&(site_coor[2]==0)&(site_coor[3]==0))){
            //cout<<"1haha"<<endl;
        }
        
        MxtMx_store(matrixU[get_index(mu, site_coor)], p);
        MxtMx_store(matrixU[get_index(mu, site_addmu)], p);
        MxtMx_store(matrixU[get_index(nu, site_addmu2)], p);

        //prtmatrix(p);
        
        gsl_matrix_complex_memcpy (U1, matrixU[get_index(mu, site_addmunu)]);
        conjtrans(U1);
        MxtMx_store(U1, p);

        //cout<<"U1"<<endl;
        //prtmatrix(p);
        
        gsl_matrix_complex_memcpy (U2, matrixU[get_index(mu, site_addnu)]);
        conjtrans(U2);
        MxtMx_store(U2, p);

        //cout<<"U2"<<endl;
        //prtmatrix(p);
        
        gsl_matrix_complex_memcpy (U3, matrixU[get_index(nu, site_coor)]);
        conjtrans(U3);
        MxtMx_store(U3, p);

        //cout<<"U3"<<endl;
        //prtmatrix(p);
        
        p_num = 1.0 / 3.0 * get_tr_REAL(p);

        if (isnan(p_num)){
            //cout <<"start"<<endl;
            //cout <<mu<<" "<<site_coor[0]<<site_coor[1]<<site_coor[2]<<site_coor[3]<<endl;
            //prtmatrix(matrixU[get_index(mu, site_coor)]);
            //cout <<nu<<" "<<site_addmu[0]<<site_addmu[1]<<site_addmu[2]<<site_addmu[3]<<endl;
            //prtmatrix(matrixU[get_index(nu, site_addmu)]);
            //prtmatrix(U1);
            //prtmatrix(U2);
            exit(1);
        }
        
        gsl_matrix_complex_free(p);
        gsl_matrix_complex_free(U2);
        gsl_matrix_complex_free(U1);
        
        return p_num;
    }
    else {
        cout<<"fuck"<<endl;
        cout<<site_coor[0]<<site_coor[1]<<site_coor[2]<<site_coor[3]<<endl;
        cout<<site_addmu[0]<<site_addmu[1]<<site_addmu[2]<<site_addmu[3]<<endl;
        cout<<site_addnu[0]<<site_addnu[1]<<site_addnu[2]<<site_addnu[3]<<endl;
        return 0;
    }
}

// calculate the average wilson loop of the whole lattice in single configuration
double Measure_Avg_11(gsl_matrix_complex * matrixU[]){
    double loop = 0.0;
    int coor[4];
    int countnum = 0;
    for (int i = 0; i < (NumLattice); i++){
        for (int j = 0; j < (NumLattice); j++){
            for (int m = 0; m < (NumLattice); m++){
                for (int n = 0; n < (NumLattice); n++){
                    coor[0] = i;
                    coor[1] = j;
                    coor[2] = m;
                    coor[3] = n;
                    for (int mu = 0; mu < 4; mu++){
                        for(int nu = 0; nu < mu; nu++){
                            //cout << plqt(mu, nu, coor, matrixU) << endl;
                            //cout << mu << " "<< coor[0]<<coor[1]<<coor[2]<<coor[3] <<endl;
                            //cout << get_index(mu, coor) << endl;
                            loop = loop + plqt(mu, nu, coor, matrixU);
                            /*if (plqt(mu, nu, coor, matrixU) != 0){
                                countnum++;
                            }else{
                                cout <<mu<<nu<<coor[0]<<coor[1]<<coor[2]<<coor[3]<<endl;
                                }*/
                        }
                    }
                }
            }
        }
    }
    loop = (loop / (NumLattice) / (NumLattice) / (NumLattice) / (NumLattice) / 6.0) ;
    //cout <<countnum<<endl;
    //loop = loop / countnum;
    return loop;
}

double Measure_Avg_21(gsl_matrix_complex * matrixU[]){
    double loop = 0.0;
    int coor[4];
    int countnum = 0;
    for (int i = 0; i < (NumLattice); i++){
        for (int j = 0; j < (NumLattice); j++){
            for (int m = 0; m < (NumLattice); m++){
                for (int n = 0; n < (NumLattice); n++){
                    coor[0] = i;
                    coor[1] = j;
                    coor[2] = m;
                    coor[3] = n;
                    for (int mu = 0; mu < 4; mu++){
                        for(int nu = 0; nu < mu; nu++){
                            //cout << plqt(mu, nu, coor, matrixU) << endl;
                            //cout << mu << " "<< coor[0]<<coor[1]<<coor[2]<<coor[3] <<endl;
                            //cout << get_index(mu, coor) << endl;
                            loop = loop + plqt_21(mu, nu, coor, matrixU);
                            //cout << loop << endl;
                            if (plqt(mu, nu, coor, matrixU) != 0){
                                countnum++;
                            }else{
                                cout <<mu<<nu<<coor[0]<<coor[1]<<coor[2]<<coor[3]<<endl;
                            }
                        }
                    }
                }
            }
        }
    }
    loop = (loop / (NumLattice) / (NumLattice) / (NumLattice) / (NumLattice) / 6.0) ;
    //cout << (NumLattice) * (NumLattice) * (NumLattice) * (NumLattice) * 6.0 << endl;
    //cout <<"count"<<countnum<<endl;
    //loop = loop / countnum;
    return loop;
}

double Measure_order2(gsl_matrix_complex * matrixU[]){
    double loop = 0.0;
    int coor[4];
    int countnum = 0;
    for (int i = 0; i < (NumLattice); i++){
        for (int j = 0; j < (NumLattice); j++){
            for (int m = 0; m < (NumLattice); m++){
                for (int n = 0; n < (NumLattice); n++){
                    coor[0] = i;
                    coor[1] = j;
                    coor[2] = m;
                    coor[3] = n;
                    for (int mu = 0; mu < 4; mu++){
                        for(int nu = 0; nu < mu; nu++){
                            loop = loop - 1.0/12.0/pow(U0, 6.0)* plqt_21(mu, nu, coor, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(nu, mu, coor, matrixU);
                            loop = loop + 5.0/3.0/pow(U0, 4.0)*plqt(mu, nu, coor, matrixU);
                            countnum++;
                        }
                    }
                }
            }
        }
    }
    return loop;
}

// Update one unitary matrix after comparing between old and new action
// Return pointers of unitary matrixes array
void update_matrixU_order1(const int mu, const int site_coor[4], gsl_matrix_complex * matrixU[], gsl_matrix_complex * Mmatrix_array[], int * acc_N, int * acc_n){
    gsl_matrix_complex * old_temp_matrixU = gsl_matrix_complex_alloc (3, 3);
    double old_s = 0;
    double new_s = 0;
    double old_s_all = 0;
    double new_s_all = 0;
    double delta_s;
    double delta_s_all;
    int new_site[4];

    // calculate old action
    for (int i = 0; i < mu; i++){
        old_s = old_s + plqt(mu, i, site_coor, matrixU);
    }
    for (int i = 0; i < mu; i++){
        site_sub(new_site, site_coor, i);
        old_s = old_s + plqt(mu, i, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        site_sub(new_site, site_coor, i);
        old_s = old_s + plqt(i, mu, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        old_s = old_s + plqt(i, mu, site_coor, matrixU);
    }
    
    old_s = 1 + old_s * (- 1) * BETA;

    //old_s_all = Measure_Avg_11(matrixU) * NumLattice * NumLattice * NumLattice * NumLattice * 6.0;
    //old_s_all = 1 + old_s_all * (- 1) * BETA;
    
    // update one link variables
    gsl_matrix_complex_memcpy(old_temp_matrixU, matrixU[get_index(mu, site_coor)]);
    int randm = (int) rand() % MNumArray;
    MxtMx_store(Mmatrix_array[randm], matrixU[get_index(mu, site_coor)]);
    untrzMx(matrixU[get_index(mu, site_coor)]);

    // calculates new action
    for (int i = 0; i < mu; i++){
        new_s = new_s + plqt(mu, i, site_coor, matrixU);
        //site_sub(new_site, site_coor, i);
        //old_s = old_s + plqt(mu, i, new_site, matrixU);
        //site_sub(new_site, site_coor, mu);
        //old_s = old_s + plqt(mu, i, new_site, matrixU);
    }
    for (int i = 0; i < mu; i++){
        site_sub(new_site, site_coor, i);
        new_s = new_s + plqt(mu, i, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        site_sub(new_site, site_coor, i);
        new_s = new_s + plqt(i, mu, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        new_s = new_s + plqt(i, mu, site_coor, matrixU);
    }

    new_s = 1 + new_s * (-1) * BETA;

    //new_s_all = Measure_Avg_11(matrixU) * NumLattice * NumLattice * NumLattice * NumLattice * 6.0;
    //new_s_all = 1 + new_s_all * (- 1) * BETA;
    
    // calculate delta_action
    delta_s = new_s - old_s;

    /*delta_s_all = new_s_all - old_s_all;
    double delta_delta = (double) fabs(delta_s - delta_s_all);
    if (delta_delta > 0.0001){
    cout<<delta_s<<" "<<delta_s_all<<endl;
    exit(1);
    }*/

    double randmmm = ((double)rand()/(double)RAND_MAX);
    //cout<< randmmm<<endl;
    if ((delta_s>0) && (exp(-delta_s) < randmmm)){
        gsl_matrix_complex_memcpy(matrixU[get_index(mu, site_coor)], old_temp_matrixU);
        //cout << *acc_n << endl;
        //int nn = *acc_n + 1;
        //*acc_n = nn;
    }
    //int nnn = *acc_N + 1;
    //*acc_N = nnn;
}


void update_matrixU_order2_bac(const int mu, const int site_coor[4], gsl_matrix_complex * matrixU[], gsl_matrix_complex * Mmatrix_array[], int * acc_N, int * acc_n){
    gsl_matrix_complex * old_temp_matrixU = gsl_matrix_complex_alloc (3, 3);
    double old_s = 0;
    double new_s = 0;
    double old_s_all = 0;
    double new_s_all = 0;
    double delta_s;
    double delta_s_all;
    int new_site[4];

    // calculate old action
    // 1*1 loop
    for (int i = 0; i < mu; i++){
        old_s = old_s + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, site_coor, matrixU);
    }
    for (int i = 0; i < mu; i++){
        site_sub(new_site, site_coor, i);
        old_s = old_s + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        site_sub(new_site, site_coor, i);
        old_s = old_s + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        old_s = old_s + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, site_coor, matrixU);
    }
    
    // 2*1 loop
    for (int i = 0; i < mu; i++){
        if (i != mu){
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
        site_sub(new_site, site_coor, mu);
        old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
        site_sub2(new_site, site_coor, i);
        old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    
    for (int i = 0; i < mu; i++){
        if (i != mu){
        site_sub_munu(new_site, site_coor, mu, i);
        old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
        site_sub(new_site, site_coor, i);
        old_s = old_s + 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            old_s = old_s + 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }

    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, mu, i);
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }
    
    old_s = old_s * (- 1) * BETA;
    //cout<<"print xxx"<<endl;
    //old_s_all = Measure_Avg_21(matrixU) * NumLattice * NumLattice * NumLattice * NumLattice * 6.0;
    //cout<<"print www"<<endl;
    //old_s_all = 1 + old_s_all * (- 1) * BETA;
    
    // update one link variables
    gsl_matrix_complex_memcpy(old_temp_matrixU, matrixU[get_index(mu, site_coor)]);
    int randm = (int) rand() % MNumArray;
    MxtMx_store(Mmatrix_array[randm], matrixU[get_index(mu, site_coor)]);
    untrzMx(matrixU[get_index(mu, site_coor)]);

    //cout<<mu<<site_coor[0]<<site_coor[1]<<site_coor[2]<<site_coor[3]<<endl;
    
    // calculates new action
    // 1*1 loop
    for (int i = 0; i < mu; i++){
        new_s = new_s + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, site_coor, matrixU);
    }
    for (int i = 0; i < mu; i++){
        site_sub(new_site, site_coor, i);
        new_s = new_s + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        site_sub(new_site, site_coor, i);
        new_s = new_s + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        new_s = new_s + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, site_coor, matrixU);
    }
    
    // 2*1 loop
    for (int i = 0; i < mu; i++){
        if (i != mu){
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, mu, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            new_s = new_s + 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            new_s = new_s + 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }

    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, mu, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu + 1); i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }














/*
    for (int i = 0; i < 4; i++){
        if (i != mu){
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, mu, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }
*/ 
    new_s = new_s * (-1) * BETA;

    //cout<<"printininini"<<endl;
    
    //new_s_all = Measure_Avg_21(matrixU) * NumLattice * NumLattice * NumLattice * NumLattice * 6.0;
    //cout<<"piewhfiowh"<<endl;
    
    //new_s_all = 1 + new_s_all * (- 1) * BETA;
    
    // calculate delta_action
    delta_s = new_s - old_s;

    //delta_s_all = new_s_all - old_s_all;
    //double delta_delta = (double) fabs(delta_s - delta_s_all);
    //if (delta_delta > 0.0001){
    //cout<<delta_s<<" "<<delta_s_all<<endl;
    //exit(1);
    //}

    double randmmm = ((double)rand()/(double)RAND_MAX);
    //cout<< randmmm<<endl;
    if ((delta_s>0) && (exp(-delta_s) < randmmm)){
        gsl_matrix_complex_memcpy(matrixU[get_index(mu, site_coor)], old_temp_matrixU);
        //cout << *acc_n << endl;
        //int nn = *acc_n + 1;
        //*acc_n = nn;
    }
    //int nnn = *acc_N + 1;
    //*acc_N = nnn;
}

void update_matrixU_order2(const int mu, const int site_coor[4], gsl_matrix_complex * matrixU[], gsl_matrix_complex * Mmatrix_array[], int * acc_N, int * acc_n){
    gsl_matrix_complex * old_temp_matrixU = gsl_matrix_complex_alloc (3, 3);
    double old_s = 0;
    double new_s = 0;
    double old_s_test = 0;
    double new_s_test = 0;
    double old_s_all = 0;
    double new_s_all = 0;
    double delta_s;
    double delta_s_all;
    int new_site[4];

    // calculate old action
    // 1*1 loop
    for (int i = 0; i < mu; i++){
        old_s = old_s + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, site_coor, matrixU);
    }
    for (int i = 0; i < mu; i++){
        site_sub(new_site, site_coor, i);
        old_s = old_s + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        site_sub(new_site, site_coor, i);
        old_s = old_s + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        old_s = old_s + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, site_coor, matrixU);
    }
    
    // 2*1 loop
    for (int i = 0; i < 4; i++){
        if (i != mu){
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, mu, i);
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            old_s = old_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }
    old_s = old_s * (- 1) * BETA;

/*
// test
// 1*1 loop
    for (int i = 0; i < mu; i++){
        old_s_test = old_s_test + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, site_coor, matrixU);
    }
    for (int i = 0; i < mu; i++){
        site_sub(new_site, site_coor, i);
        old_s_test = old_s_test + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        site_sub(new_site, site_coor, i);
        old_s_test = old_s_test + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        old_s_test = old_s_test + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, site_coor, matrixU);
    }

//test
// 2*1 loop
    for (int i = 0; i < mu; i++){
        if (i != mu){
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, mu, i);
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }

    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, i, mu);
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            old_s_test = old_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }

    //test
*/

    //old_s_all = Measure_order2(matrixU)* (- 1) * BETA;
    
    //old_s_test = old_s_test * (- 1) * BETA;
    
    // update one link variables
    gsl_matrix_complex_memcpy(old_temp_matrixU, matrixU[get_index(mu, site_coor)]);
    int randm = (int) rand() % MNumArray;
    MxtMx_store(Mmatrix_array[randm], matrixU[get_index(mu, site_coor)]);
    untrzMx(matrixU[get_index(mu, site_coor)]);

    // calculates new action
    // 1*1 loop
    for (int i = 0; i < mu; i++){
        new_s = new_s + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, site_coor, matrixU);
    }
    for (int i = 0; i < mu; i++){
        site_sub(new_site, site_coor, i);
        new_s = new_s + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        site_sub(new_site, site_coor, i);
        new_s = new_s + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        new_s = new_s + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, site_coor, matrixU);
    }
    
    // 2*1 loop
    for (int i = 0; i < 4; i++){
        if (i != mu){
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, mu, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < 4; i++){
        if (i != mu){
            new_s = new_s - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }

    new_s = new_s * (-1) * BETA;
    
/*
// test
// 1*1 loop
    for (int i = 0; i < mu; i++){
        new_s_test = new_s_test + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, site_coor, matrixU);
    }
    for (int i = 0; i < mu; i++){
        site_sub(new_site, site_coor, i);
        new_s_test = new_s_test + (5.0/3.0)/pow(U0, 4.0)*plqt(mu, i, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        site_sub(new_site, site_coor, i);
        new_s_test = new_s_test + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, new_site, matrixU);
    }
    for (int i = (mu + 1); i < 4; i++){
        new_s_test = new_s_test + (5.0/3.0)/pow(U0, 4.0)*plqt(i, mu, site_coor, matrixU);
    }

//test
// 2*1 loop
    for (int i = 0; i < mu; i++){
        if (i != mu){
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, mu, i);
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU);
        }
    }
    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }

    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, mu);
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = 0; i < mu; i++){
        if (i != mu){
            site_sub2(new_site, site_coor, i);
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            site_sub_munu(new_site, site_coor, i, mu);
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, new_site, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, new_site, matrixU);
        }
    }
    for (int i = (mu+1); i < 4; i++){
        if (i != mu){
            site_sub(new_site, site_coor, i);
            new_s_test = new_s_test - 1.0/12.0/pow(U0, 6.0)*plqt_21(i, mu, site_coor, matrixU) - 1.0/12.0/pow(U0, 6.0)*plqt_21(mu, i, site_coor, matrixU);
        }
    }
*/

    //new_s_test = new_s_test * (- 1) * BETA;
    //new_s_all = Measure_order2(matrixU)* (- 1) * BETA;


    // Accept or Not
    delta_s = new_s - old_s;
    //delta_s_all = new_s_all - old_s_all;
/*
    double delta_delta = (double) fabs(delta_s - delta_s_all);
    if (delta_delta > 0.0001){
        cout<<delta_s<<" "<<delta_s_all<<endl;
        //exit(1);
    }
*/
    double randmmm = ((double)rand()/(double)RAND_MAX);
    if ((delta_s>0) && (exp(-delta_s) < randmmm)){
        gsl_matrix_complex_memcpy(matrixU[get_index(mu, site_coor)], old_temp_matrixU);
    }
}

// update matrixU for all site
void update_allsite_order1(gsl_matrix_complex * matrixU[], gsl_matrix_complex * Mmatrix_array[], int * acc_N, int * acc_n){
    int coor[4];
    for (int i = 0; i < NumLattice; i++){
        for (int j = 0; j < NumLattice; j++){
            for (int m = 0; m < NumLattice; m++){
                for (int n = 0; n < NumLattice; n++){
                    coor[0] = i;
                    coor[1] = j;
                    coor[2] = m;
                    coor[3] = n;
                    for (int mu = 0; mu < 4; mu++){
                        update_matrixU_order1(mu, coor, matrixU, Mmatrix_array, acc_N, acc_n);
                    }
                }
            }
        }
    }
}

void update_allsite_order2(gsl_matrix_complex * matrixU[], gsl_matrix_complex * Mmatrix_array[], int * acc_N, int * acc_n){
    int coor[4];
    for (int i = 0; i < NumLattice; i++){
        for (int j = 0; j < NumLattice; j++){
            for (int m = 0; m < NumLattice; m++){
                for (int n = 0; n < NumLattice; n++){
                    coor[0] = i;
                    coor[1] = j;
                    coor[2] = m;
                    coor[3] = n;
                    for (int mu = 0; mu < 4; mu++){
                        update_matrixU_order2(mu, coor, matrixU, Mmatrix_array, acc_N, acc_n);
                    }
                }
            }
        }
    }
}

// free array of matrixes
void free_MxArray(gsl_matrix_complex * MxArr[], const int elenum){
    int i;
    for (i = 0; i < elenum; i++){
        gsl_matrix_complex_free(MxArr[i]);
    }
}

//
void trac_REAL_array(gsl_matrix_complex * matrix_array[], double realtrac[], int NumArr){
    int i = 0;
    for (i =0; i<NumArr; i++){
        realtrac[i] = get_tr_REAL(matrix_array[i]);
    }
}
void trac_IMG_array(gsl_matrix_complex * matrix_array[], double imgtrac[], int NumArr){
    int i = 0;
    for (i =0; i<NumArr; i++){
        imgtrac[i] = get_tr_IMG(matrix_array[i]);
    }
}

int main(){
    srand(time(NULL));
    int site_coor[4] = {0, 0, 0, 0};
    gsl_matrix_complex * Mmatrix_array[MNumArray];
    double realtrac[MNumArray];
    double imgtrac[MNumArray];
    gsl_matrix_complex * matrixU[4 * NumLattice * NumLattice * NumLattice * NumLattice];
    int * acc_N;
    int * acc_n;
    
    //initialization
    init_Mmatrix_array(Mmatrix_array, MNumArray);
    init_matrixU(matrixU, NumDim * NumLattice * NumLattice * NumLattice * NumLattice);

    trac_REAL_array(Mmatrix_array, realtrac, MNumArray);
    trac_IMG_array(Mmatrix_array, imgtrac, MNumArray);

    // open file
    ofstream order1_11;
    order1_11.open("order1_11");
    ofstream order1_21;
    order1_21.open("order1_21");
    ofstream order2_11;
    order2_11.open("order2_11");
    ofstream order2_21;
    order2_21.open("order2_21");

    
    for(int ii = 0; ii < 5001; ii++){
        if (ii%10 == 0){
            cout << "iteration_order1: " << ii << endl;
            //cout << Measure_Avg_11(matrixU) << endl;
            //cout << Measure_Avg_21(matrixU) << endl;
            order1_11 << Measure_Avg_11(matrixU) <<endl;
            order1_21 << Measure_Avg_21(matrixU) <<endl;
        }
        update_allsite_order1(matrixU, Mmatrix_array, acc_N, acc_n);
    }

    init_matrixU(matrixU, NumDim * NumLattice * NumLattice * NumLattice * NumLattice);
    
    for(int ii = 0; ii < 5001; ii++){
        if (ii%10 == 0){
            cout << "iteration_order2: " << ii << endl;
            //cout << Measure_Avg_11(matrixU) << endl;
            //cout << Measure_Avg_21(matrixU) << endl;
            order2_11 << Measure_Avg_11(matrixU) <<endl;
            order2_21 << Measure_Avg_21(matrixU) <<endl;
        }
        update_allsite_order2(matrixU, Mmatrix_array, acc_N, acc_n);
    }
    
    cout << "Hello world!" << endl;

    // close files
    order1_11.close();
    order1_21.close();
    order2_11.close();
    order2_21.close();

    
    free_MxArray(Mmatrix_array, MNumArray);
    free_MxArray(matrixU, NumDim * NumLattice * NumLattice * NumLattice * NumLattice);
    return 0;
}