#ifndef MF_SPARSE_MATRIX_H
#define MF_SPARSE_MATRIX_H

#define NUM_OF_THREADS 2

int symetric_matrix_vector_scale_diag(int* Uorder, double* Ur, int* Vorder, double* Vr, double diag, double* inU, double* inV, double* outU, double* outV, int** Unum, int** Vnum, int nU, int nV);

int symetric_matrix_matrix_scale_diag(int* Uorder, double* Ur, int* Vorder, double* Vr, double diag, double** inU, double** inV, double** outU, double** outV, int** Unum, int** Vnum, int nU, int nV, int p);

double symetric_matrix_scale_diag_max_eig(int* Uorder, double* Ur, int* Vorder, double* Vr, double diag, double* inU, double* inV, double* outU, double* outV, int** Unum, int** Vnum, int nU, int nV, double M, int maxIter, double epsilon, int* trueIter, double* trueEpsilon);

#endif
