#include <stdio.h>
#include <math.h>
#include<omp.h>

#include <mkl_cblas.h>

#include "MF_sparse_matrix.h"

int symetric_matrix_vector_scale_diag(int* Uorder, double* Ur, int* Vorder, double* Vr, double diag, double* inU, double* inV, double* outU, double* outV, int** Unum, int** Vnum, int nU, int nV)
{
	cblas_dcopy(nU + nV, inU, 1, outU, 1);
	cblas_dscal(nU + nV, diag, outU, 1);
#pragma omp parallel
	{
		int ThreadID = omp_get_thread_num();
		for (int i = ThreadID; i < nU; i += NUM_OF_THREADS)
		{
			for (int j = 0; j < Unum[i][1]; j++)
				outU[i] += (Ur + Unum[i][0])[j] * inV[(Uorder + Unum[i][0])[j]];
		}
		for (int i = ThreadID; i < nV; i += NUM_OF_THREADS)
		{
			for (int j = 0; j < Vnum[i][1]; j++)
				outV[i] += (Vr + Vnum[i][0])[j] * inU[(Vorder + Vnum[i][0])[j]];
		}
	}

	return 0;
}

int symetric_matrix_matrix_scale_diag(int* Uorder, double* Ur, int* Vorder, double* Vr, double diag, double** inU, double** inV, double** outU, double** outV, int** Unum, int** Vnum, int nU, int nV, int p)
{
	cblas_dcopy((nU + nV) * p, inU[0], 1, outU[0], 1);
	cblas_dscal((nU + nV) * p, diag, outU[0], 1);
#pragma omp parallel
	{
		int ThreadID = omp_get_thread_num();
		for (int i = ThreadID; i < nU; i += NUM_OF_THREADS)
		{
			for (int j = 0; j < Unum[i][1]; j++)
			{
				for (int k = 0; k < p; k++)
					outU[i][k] += (Ur + Unum[i][0])[j] * inV[(Uorder + Unum[i][0])[j]][k];
			}
		}
		for (int i = ThreadID; i < nV; i += NUM_OF_THREADS)
		{
			for (int j = 0; j < Vnum[i][1]; j++)
			{
				for (int k = 0; k < p; k++)
					outV[i][k] += (Vr + Vnum[i][0])[j] * inU[(Vorder + Vnum[i][0])[j]][k];
			}
		}
	}

	return 0;
}

double symetric_matrix_scale_diag_max_eig(int* Uorder, double* Ur, int* Vorder, double* Vr, double diag, double* inU, double* inV, double* outU, double* outV, int** Unum, int** Vnum, int nU, int nV, double M, int maxIter, double epsilon, int* trueIter, double* trueEpsilon)
{
	double oeig, eig = 0.0;
	double tmp = cblas_ddot(nU + nV, inU, 1, inU, 1);
	double* tpU, *tpV;
	double* pU = outU;
	double* pV = outV;

	diag += M;

	(*trueIter) = 0;
	(*trueEpsilon) = 100.0;
	oeig = 100.0;
	while ((*trueIter < maxIter) && (*trueEpsilon > epsilon))
	{
		symetric_matrix_vector_scale_diag(Uorder, Ur, Vorder, Vr, diag, inU, inV, outU, outV, Unum, Vnum, nU, nV);
		eig = cblas_ddot(nU + nV, inU, 1, outU, 1) / tmp;
		tmp = cblas_ddot(nU + nV, outU, 1, outU, 1);
		cblas_dscal(nU + nV, M / tmp, outU, 1);
		tmp = cblas_ddot(nU + nV, outU, 1, outU, 1);
		(*trueEpsilon) = fabs(oeig - eig);
		(*trueIter)++;
		tpU = outU;
		tpV = outV;
		outU = inU;
		outV = inV;
		inU = tpU;
		inV = tpV;
		oeig = eig;
		//        printf("Iter: %d, eig: %g, err: %g\n", *trueIter, eig - M, *trueEpsilon);
	}

	if (pU != tpU)
		cblas_dcopy(nU + nV, tpU, 1, pU, 1);
	cblas_dscal(nU + nV, 1.0 / sqrt(tmp), pU, 1);

	return eig - M;
}

