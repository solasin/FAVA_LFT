#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <mkl_cblas.h>
#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#include<algorithm>

#include "MF_sparse_matrix.h"
#include "OUFA.h"
using namespace std;
int Add_Dimension(VSLStreamStatePtr stream, OUFA_INFO* info, OUFA_TMP* tmp){
	double* vec_tmp;
	double* det_u = tmp->eigV;
	double* det_v = tmp->eigV + info->nU;
	vec_tmp = (double *)malloc(sizeof(double)*info->N);
	vdRngGaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, info->N, vec_tmp, 0.0, 0.01);
	double dou_tmp = sqrt(cblas_ddot(info->num_tr, tmp->Ue, 1, tmp->Ue, 1)) / tmp->avr;
	int lter = 0;
	double err = 0;
	tmp->minEig = -symetric_matrix_scale_diag_max_eig(tmp->Uorder, tmp->Ue, tmp->Vorder, tmp->Ve, 0.0, vec_tmp, vec_tmp + info->nU, tmp->eigV, tmp->eigV + info->nU, tmp->Unum, tmp->Vnum, info->nU, info->nV, dou_tmp*tmp->M, 100000, 1e-6, &lter, &err);
#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < info->num_tr; i += NUM_OF_THREADS)
			tmp->Ve[i] = det_u[info->trIdx[i][0]] * det_v[info->trIdx[i][1]];
	}
	double beta = (-tmp->minEig) / (2 * cblas_ddot(info->num_tr, tmp->Ve, 1, tmp->Ve, 1));
	double sup = info->N*info->sigma[tmp->p - 1] * info->sigma[tmp->p - 1];
	cblas_dcopy(info->N, tmp->eigV, 1, tmp->Y[0] + tmp->p - 1, info->p);
	//prevent from over flowing
	if (beta < 0)
		beta = 0;
	if (beta < sup)
		cblas_dscal(info->N, sqrt(beta), tmp->Y[0] + tmp->p - 1, info->p);
	else
		cblas_dscal(info->N, sqrt(sup), tmp->Y[0] + tmp->p - 1, info->p);

	free(vec_tmp);
	return 0;
}

double Cal_Cost(OUFA_INFO* info, OUFA_TMP* tmp){
	double ret = 0;
#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < info->num_tr; i += NUM_OF_THREADS){
			double dou_tmp = cblas_ddot(tmp->p, tmp->Y[info->trIdx[i][0]], 1, (tmp->Y + info->nU)[info->trIdx[i][1]], 1);
			tmp->Ue[i] = tmp->Ur[i] - dou_tmp;
			tmp->Ve[tmp->UmapV[i]] = tmp->Ue[i];
		}
	}
	ret = 0.5*cblas_ddot(info->num_tr, tmp->Ue, 1, tmp->Ue, 1);
	return (ret);
}

int Cal_Gradient(OUFA_INFO* info, OUFA_TMP* tmp){
	symetric_matrix_matrix_scale_diag(tmp->Uorder, tmp->Ue, tmp->Vorder, tmp->Ve, 0.0, info->U, info->V, tmp->dY, tmp->dY + info->nU, tmp->Unum, tmp->Vnum, info->nU, info->nV, info->p);
	cblas_dscal(info->N*info->p, -1.0, tmp->dY[0], 1);
	return 0;
}

int Modi_Y(OUFA_INFO* info, OUFA_TMP* tmp){
	double E_Mo[NUM_OF_THREADS * 4];
	cblas_dscal(NUM_OF_THREADS * 4, 0.0, E_Mo, 1);
#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < info->num_tr; i += NUM_OF_THREADS){
			int now_u = info->trIdx[i][0];
			int now_v = info->trIdx[i][1];
			double M_dydy, M_yidyj, M_yjdyi, M_ryy;
			M_dydy = cblas_ddot(info->p, tmp->dY[now_u], 1, (tmp->dY + info->nU)[now_v], 1);
			M_yidyj = cblas_ddot(info->p, tmp->Y[now_u], 1, (tmp->dY + info->nU)[now_v], 1);
			M_yjdyi = cblas_ddot(info->p, (tmp->Y + info->nU)[now_v], 1, tmp->dY[now_u], 1);
			M_ryy = tmp->Ue[i];

			E_Mo[4 * now_id] += 2 * (M_dydy*M_dydy);
			E_Mo[4 * now_id + 1] += (-3.0*M_yidyj*M_dydy);
			E_Mo[4 * now_id + 1] += (-3.0*M_yjdyi*M_dydy);
			E_Mo[4 * now_id + 2] += (M_yjdyi*M_yjdyi + M_yjdyi*M_yidyj + M_yidyj*M_yjdyi + M_yidyj*M_yidyj);
			E_Mo[4 * now_id + 2] += (-2.0*M_ryy*M_dydy);
			E_Mo[4 * now_id + 3] += (M_ryy*M_yjdyi + M_ryy*M_yidyj);
		}
	}

	double M_Fa, M_Fb, M_Fc, M_Fd;
	M_Fa = M_Fb = M_Fc = M_Fd = 0;
	for (int i = 0; i<NUM_OF_THREADS; i++){
		M_Fa += E_Mo[4 * i];
		M_Fb += E_Mo[4 * i + 1];
		M_Fc += E_Mo[4 * i + 2];
		M_Fd += E_Mo[4 * i + 3];
	}

	//test 0.35
	/*
	double lolo = M_Fa*0.35*0.35*0.35 + M_Fb*0.35*0.35 + M_Fc*0.35 + M_Fd;
	*/
	//test end

	//Binary_Search
	double li, ri;
	li = 0; ri = 50.0;
	while (ri - li > 1e-6){
		double mid = (li + ri) / 2.0;
		double val = M_Fa*mid*mid*mid + M_Fb*mid*mid + M_Fc*mid + M_Fd;
		if (val > 0)
			ri = mid;
		else
			li = mid;
	}
	li = (li + ri) / 2.0;
	cblas_dcopy(info->N*info->p, tmp->Y[0], 1, tmp->bY[0], 1);
	cblas_dscal(info->N*info->p, li, tmp->dY[0], 1);
	cblas_daxpy(info->N*info->p, -1.0, tmp->dY[0], 1, tmp->Y[0], 1);
	for (int i = 0; i < info->p; i++){
		double len = cblas_ddot(info->N, &tmp->Y[0][i], info->p, &tmp->Y[0][i], info->p);
		double sup = double(info->N)*info->sigma[i] * info->sigma[i];
		if (len>sup)
			cblas_dscal(info->N, sqrt(sup / len), &tmp->Y[0][i], info->p);
	}
	//double ret = sqrt(cblas_ddot(info->N*info->p, tmp->dY[0], 1, tmp->dY[0], 1)/double(info->N*info->p));
	return (0);
}

int Local_Opt(OUFA_INFO* info, OUFA_TMP* tmp){
	double Pre_err = -1.0;
	double Now_err = 0.0;
	tmp->bY = (double **)malloc(sizeof(double*)*info->N);
	tmp->bY[0] = (double *)malloc(sizeof(double)*info->N*info->p);
	for (int i = 0; i < info->N; i++)
		tmp->bY[i] = tmp->bY[0] + info->p*i;
	Cal_Cost(info, tmp);
	Cal_Gradient(info, tmp);
	//test gra
	/*
	cblas_daxpy(info->N*info->p, -0.35, tmp->dY[0], 1, tmp->Y[0], 1);
	double tmp1 = Cal_Cost(info, tmp);
	cblas_daxpy(info->N*info->p, -1e-5, tmp->dY[0], 1, tmp->Y[0], 1);
	double tmp2 = Cal_Cost(info, tmp);
	double di = (tmp2 - tmp1) / (1e-5);
	double rmm = tsRMSE(info, tmp);
	*/
	//test end
	while (1){
		//double rmm = tsRMSE(info, tmp);
		Modi_Y(info, tmp);
		Now_err = Cal_Cost(info, tmp);
		if (Pre_err - Now_err < 1e-6 && Pre_err>0)
			break;
		else
			Pre_err = Now_err;
		Cal_Gradient(info, tmp);
	}
	if (Pre_err - Now_err < 0)
		cblas_dcopy(info->N*info->p, tmp->bY[0], 1, tmp->Y[0], 1);

	free(tmp->bY[0]);
	free(tmp->bY);
	return 0;
}

int Global_Opt(OUFA_INFO* info, OUFA_TMP* tmp){
	VSLStreamStatePtr stream;
	vslNewStream(&stream, VSL_BRNG_MCG31, 1);
	for (int i = 0; i < info->p; i++){
		tmp->p++;
		Add_Dimension(stream, info, tmp);
		Cal_Cost(info, tmp);
	}
	Local_Opt(info, tmp);
	return 0;
}
