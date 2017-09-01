#include<cstdio>
#include<cstring>
#include<iostream>
#include<string>
#include <mkl_cblas.h>
#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#include<omp.h>
#include<algorithm>
#include"OUFA.h"
#include"Data_Init.h"
#include"MF_sparse_matrix.h"
#include<ctime>
#include<cmath>
using namespace std;
double glo_rmse;
double glo_cnt;
int tmp_cnt;
double tsRMSE(OUFA_TMP* src_tmp, OUFA_INFO* src_info, OUFA_INFO* dst_info){
	double ret = 0;
	int cnt = 0;
	/*for (int i = 0; i < info->num_ts; i++){
	double doti = cblas_ddot(info->p, tmp->Y[info->tsIdx[i][0]], 1, (tmp->Y + info->nU)[info->tsIdx[i][1]], 1);
	doti += tmp->Meanr;
	if (doti < 0.0) doti = 0.0;
	if (doti>1.0) doti = 1.0;
	double ch = tmp->MinR + (tmp->MaxR - tmp->MinR)*doti;
	ret = ret + (info->tsR[i] - ch)*(info->tsR[i] - ch);
	}
	ret = sqrt(ret / double(info->num_ts));*/
	for (int i = 0; i < dst_info->num_tr; i++){
		int uid = dst_info->U_secmap[dst_info->trIdx[i][0]];
		uid = lower_bound(src_info->U_secmap, src_info->U_secmap + src_info->nU, uid) - src_info->U_secmap;
		int vid = dst_info->V_secmap[dst_info->trIdx[i][1]];
		vid = lower_bound(src_info->V_secmap, src_info->V_secmap + src_info->nV, vid) - src_info->V_secmap;
		if (src_info->U_secmap[uid] != dst_info->U_secmap[dst_info->trIdx[i][0]] || src_info->V_secmap[vid] != dst_info->V_secmap[dst_info->trIdx[i][1]])
			continue;
		cnt++;
		double doti = cblas_ddot(src_info->p, &(src_info->U[uid][0]), 1, &(src_info->V[vid][0]), 1);
		doti += src_tmp->Meanr;
		if (doti < 0.0) doti = 0.0;
		if (doti>1.0) doti = 1.0;
		double ch = src_tmp->MinR + (src_tmp->MaxR - src_tmp->MinR)*doti;
		ret = ret + (dst_info->trR[i] - ch)*(dst_info->trR[i] - ch);
	}
	ret = sqrt(ret / double(cnt));
	//debug
	glo_rmse = glo_rmse*glo_rmse*glo_cnt + ret*ret*double(cnt);
	glo_cnt = glo_cnt + double(cnt);
	glo_rmse = sqrt(glo_rmse / glo_cnt);
	//debug end
	return (ret);
}
double Cal_Cost_Inc(OUFA_INFO* info, OUFA_TMP* tmp){
	double ret = 0;
#pragma omp parallel
	{
		int now = omp_get_thread_num();
		for (int i = now; i < info->nU; i += NUM_OF_THREADS)
		for (int j = 0; j < info->p; j++){
			tmp->OP_T[i][j] = cblas_ddot(2 * info->p, &info->U[i][0], 1, &tmp->Y[0][j], info->p);
		}
		for (int i = now; i < info->nV; i += NUM_OF_THREADS)
		for (int j = 0; j < info->p; j++){
			tmp->OP_T[i + info->nU][j] = cblas_ddot(2 * info->p, &info->V[i][0], 1, &tmp->Y[0][j], info->p);
		}
	}
#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < info->num_tr; i+=NUM_OF_THREADS){
			int udx = info->trIdx[i][0];
			int idx = info->trIdx[i][1]+info->nU;
			double dou_tmp = cblas_ddot(info->p, &tmp->OP_T[udx][0], 1, &tmp->OP_T[idx][0], 1);
			tmp->Ue[i] = tmp->Ur[i] - dou_tmp;
			tmp->Ve[tmp->UmapV[i]] = tmp->Ue[i];
		}
	}
	ret = cblas_ddot(info->num_tr, tmp->Ue, 1, tmp->Ue, 1);
	return (ret);
}

int Cal_Gradient_Inc(OUFA_INFO* info, OUFA_TMP* tmp){
	symetric_matrix_matrix_scale_diag(tmp->Uorder, tmp->Ue, tmp->Vorder, tmp->Ve, 0.0, tmp->OP_T,tmp->OP_T+info->nU,tmp->dOP_T,tmp->dOP_T+info->nU,tmp->Unum,tmp->Vnum,info->nU,info->nV,info->p);
	cblas_dscal(info->N*info->p, -2.0, &tmp->dOP_T[0][0], 1);
	for (int i = 0; i < info->p * 2;i++)
	for (int j = 0; j < info->p; j++){
		double dou_tmp = cblas_ddot(info->nU, &info->U[0][i], info->p * 2, &tmp->dOP_T[0][j], info->p)+cblas_ddot(info->nV,&info->V[0][i],info->p*2,&tmp->dOP_T[info->nU][j],info->p);
		tmp->dY[i][j] = dou_tmp;
	}
	/*double len = cblas_ddot(info->p * 2 * info->p, &tmp->dY[0][0], 1, &tmp->dY[0][0], 1);
	cblas_dscal(info->p * 2 * info->p, 1.0 / len, &tmp->dY[0][0], 1);*/
	return 0;
}

double Find_Minx(double a, double b, double c, double d){
	double a_now, b_now, c_now;
	int cnt = 0;
	double ret[3];
	a_now = b*b - 3 * a*c;
	b_now = b*c - 9 * a*d;
	c_now = c*c - 3 * b*d;
	//x1==x2==x3
	if (a_now == b_now){
		ret[cnt] = -1 * b / (3.0*a);
		cnt++;
	}
	else{
		double det = b_now*b_now - 4 * a_now*c_now;
		if (det > 0){
			double y1 = a_now*b + 3 * a*((-1 * b_now - sqrt(b_now*b_now - 4 * a_now*c_now)) / 2.0);
			double y2 = a_now*b + 3 * a*((-1 * b_now + sqrt(b_now*b_now - 4 * a_now*c_now)) / 2.0);
			if (y1 >= 0)
				y1 = pow(double(y1), double(1.0 / 3.0));
			else
				y1 = -1.0*pow(double(-1.0*y1), double(1.0 / 3.0));
			if (y2 >= 0)
				y2 = pow(double(y2), double(1.0 / 3.0));
			else
				y2 = -1.0*pow(double(-1.0*y2), double(1.0 / 3.0));
			ret[cnt] = (-1.0*b - (y1 + y2)) / (3.0*a);
			cnt++;
		}
		if (det == 0){
			double k_now = b_now / a_now;
			ret[cnt] = -1.0*b / a + k_now;
			cnt++;
			ret[cnt] = (-1.0*k_now) / 2.0;
			cnt++;
		}
		if (det < 0){
			double t_now = (2 * a_now*b - 3 * a*b_now) / (2 * sqrt(pow(a_now, 3.0)));
			double ta = acos(t_now);
			ret[cnt] = (-1.0*b - 2 * sqrt(a_now)*cos(ta / 3.0)) / (3.0*a);
			cnt++;
			ret[cnt] = (-1.0*b + sqrt(a_now)*(cos(ta / 3.0) - sqrt(3.0)*sin(ta / 3.0))) / (3.0*a);
			cnt++;
			ret[cnt] = (-1.0*b + sqrt(a_now)*(cos(ta / 3.0) + sqrt(3.0)*sin(ta / 3.0))) / (3.0*a);
			cnt++;
		}
	}
	double ans = -1.0;
	for (int i = 0; i < cnt;i++)
	if (ret[i]>0){
		if (ans < 0 || ret[i] < ans)
			ans = ret[i];
	}
	return ans;
}


int Modi_Y_Inc(OUFA_INFO* info, OUFA_TMP* tmp){
#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < info->nU;i+=NUM_OF_THREADS)
		for (int j = 0; j < info->p; j++)
			tmp->dOP_T[i][j] = cblas_ddot(info->p * 2, &info->U[i][0], 1, &tmp->dY[0][j], info->p);
		for (int i = now_id; i < info->nV;i+=NUM_OF_THREADS)
		for (int j = 0; j < info->p; j++)
			tmp->dOP_T[info->nU+i][j] = cblas_ddot(info->p * 2, &info->V[i][0], 1, &tmp->dY[0][j], info->p);
	}
	/*
	double** tmp_matrix_l;
	double** tmp_matrix_r;
	tmp_matrix_l = (double **)malloc(sizeof(double*)*info->p);
	tmp_matrix_r = (double **)malloc(sizeof(double*)*info->p);
	tmp_matrix_l[0] = (double*)malloc(sizeof(double)*info->p*info->p);
	tmp_matrix_r[0] = (double*)malloc(sizeof(double)*info->p*info->p);
	for (int i = 1; i < info->p; i++){
		tmp_matrix_l[i] = tmp_matrix_l[0] + i*info->p;
		tmp_matrix_r[i] = tmp_matrix_r[0] + i*info->p;
	}
	double M_Fa, M_Fb, M_Fc, M_Fd;
	M_Fa = M_Fb = M_Fc = M_Fd = 0;
	for (int i = 0; i < info->p; i++){
		for (int j = 0; j < info->p; j++){
			tmp_matrix_l[i][j] = cblas_ddot(info->N, &tmp->dOP_T[0][i], info->p, &tmp->dOP_T[0][j], info->p);
			printf("%lf\t", tmp_matrix_l[i][j]);
		}
		printf("\n");
			
	}
	
	for (int i = 0; i < info->p; i++)
		M_Fa += cblas_ddot(info->p, &tmp_matrix_l[i][0], 1, &tmp_matrix_l[0][i], info->p);

	for (int i = 0; i < info->p;i++)
	for (int j = 0; j < info->p; j++)
		tmp_matrix_r[i][j] = cblas_ddot(info->N, &tmp->OP_T[0][i], info->p, &tmp->dOP_T[0][j], info->p);
	for (int i = 0; i < info->p; i++)
		M_Fb += (-3.0*cblas_ddot(info->p, &tmp_matrix_l[i][0], 1, &tmp_matrix_r[0][i], info->p));

	for (int i = 0; i < info->p; i++)
		M_Fc += cblas_ddot(info->p, &tmp_matrix_r[i][0], 1, &tmp_matrix_r[0][i], info->p);

	for (int i = 0; i < info->p;i++)
	for (int j = 0; j < info->p; j++)
		tmp_matrix_r[i][j] = cblas_ddot(info->N, &tmp->OP_T[0][i], info->p, &tmp->OP_T[0][j], info->p);

	for (int i = 0; i < info->p; i++)
		M_Fc += cblas_ddot(info->p, &tmp_matrix_l[i][0], 1, &tmp_matrix_r[0][i], info->p);

	for (int i = 0; i < info->p; i++)
		M_Fd += cblas_ddot(info->p*2, &tmp->dY[0][i], info->p, &tmp->dY[0][i], info->p);
	M_Fd *= (-0.5);

	free(tmp_matrix_l[0]);
	free(tmp_matrix_r[0]);
	free(tmp_matrix_l);
	free(tmp_matrix_r);

	symetric_matrix_matrix_scale_diag(tmp->Uorder, tmp->Ue, tmp->Vorder, tmp->Ve, 0.0, tmp->dOP_T, tmp->dOP_T + info->nU, tmp->OP_T, tmp->OP_T + info->nU, tmp->Unum, tmp->Vnum, info->nU, info->nV, info->p);
	cblas_dscal(info->N*info->p, -1.0, &tmp->OP_T[0][0], 1);

	for (int i = 0; i < info->p; i++)
		M_Fc += cblas_ddot(info->N, &tmp->OP_T[0][i], info->p, &tmp->dOP_T[0][i], info->p);
	//debug
	double now_gra = 0.1*0.1*0.1*M_Fa + 0.1*0.1*M_Fb + 0.1*M_Fc + M_Fd;
	//debug end
	*/
	double E_Mo[NUM_OF_THREADS * 4];
	cblas_dscal(NUM_OF_THREADS * 4, 0.0, E_Mo, 1);
#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < info->num_tr; i += NUM_OF_THREADS){
			int now_u = info->trIdx[i][0];
			int now_v = info->trIdx[i][1];
			double M_dydy, M_yidyj, M_yjdyi, M_ryy;
			M_dydy = cblas_ddot(info->p, tmp->dOP_T[now_u], 1, (tmp->dOP_T + info->nU)[now_v], 1);
			M_yidyj = cblas_ddot(info->p, tmp->OP_T[now_u], 1, (tmp->dOP_T + info->nU)[now_v], 1);
			M_yjdyi = cblas_ddot(info->p, (tmp->OP_T + info->nU)[now_v], 1, tmp->dOP_T[now_u], 1);
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
	//double tmp_gra = 0.1*0.1*0.1*M_Fa + 0.1*0.1*M_Fb + 0.1*M_Fc + M_Fd;
	double ste = Find_Minx(M_Fa, M_Fb, M_Fc, M_Fd);
	cblas_dcopy(info->p*info->p * 2, tmp->Y[0], 1, tmp->bY[0], 1);
	cblas_dscal(info->p*info->p * 2, ste, tmp->dY[0], 1);
	cblas_daxpy(info->p*info->p * 2, -1.0, tmp->dY[0], 1, tmp->Y[0], 1);
#pragma omp parallel
	{
		int now = omp_get_thread_num();
		for (int i = now; i < info->nU; i += NUM_OF_THREADS)
		for (int j = 0; j < info->p; j++){
			tmp->OP_T[i][j] = cblas_ddot(2 * info->p, &info->U[i][0], 1, &tmp->Y[0][j], info->p);
		}
		for (int i = now; i < info->nV; i += NUM_OF_THREADS)
		for (int j = 0; j < info->p; j++){
			tmp->OP_T[i + info->nU][j] = cblas_ddot(2 * info->p, &info->V[i][0], 1, &tmp->Y[0][j], info->p);
		}
	}
	for (int i = 0; i < info->p; i++){
		double len = cblas_ddot(info->N, &tmp->OP_T[0][i], info->p, &tmp->OP_T[0][i], info->p);
		double sup = double(info->N)*info->sigma[i] * info->sigma[i];
		if (len>sup)
			cblas_dscal(info->p * 2, sqrt(sup / len), &tmp->Y[0][i], info->p);
	}
	return (0);
}

int Local_Opt_Inc(OUFA_INFO* info, OUFA_TMP* tmp){
	double Pre_err = -1.0;
	double Now_err = 0.0;
	tmp->bY = (double**)malloc(sizeof(double*)*info->p * 2);
	tmp->bY[0] = (double*)malloc(sizeof(double)*info->p*info->p * 2);
	for (int i = 0; i < info->p * 2; i++)
		tmp->bY[i] = tmp->bY[0] + i*info->p;
	Cal_Cost_Inc(info, tmp);
	Cal_Gradient_Inc(info, tmp);
	//debug
	
	/*cblas_daxpy(info->p*info->p * 2, -1.0*0.1, &tmp->dY[0][0], 1, &tmp->Y[0][0], 1);
	double rm_now =Cal_Cost_Inc(info, tmp)*0.5;
	cblas_daxpy(info->p*info->p * 2, -1.0*(1e-5), &tmp->dY[0][0], 1, &tmp->Y[0][0], 1);
	double rm_nex=Cal_Cost_Inc(info, tmp)*0.5;
	double pr = (rm_nex - rm_now) / (1e-5);
	printf("%lf\n\n", pr);
	cblas_daxpy(info->p*info->p * 2, 1.0*(0.1+1e-5), &tmp->dY[0][0], 1, &tmp->Y[0][0], 1);
	Cal_Cost_Inc(info, tmp);*/
	
	//debug end
	int cnt = 0;
	while (1){
		cnt++;
		Modi_Y_Inc(info, tmp);
		Now_err = Cal_Cost_Inc(info, tmp);
		//modify 1e-6
		if (Pre_err - Now_err<1.0 && Pre_err>0)
			break;
		else
			Pre_err = Now_err;
		Cal_Gradient_Inc(info, tmp);
	}
	//printf("It = %d\n", cnt);
	if (Pre_err - Now_err < 0)
		cblas_dcopy(info->p*info->p*2, tmp->bY[0], 1, tmp->Y[0], 1);
	free(tmp->bY[0]);  //check
	free(tmp->bY);	//check
	return 0;
}

int Global_Opt_Inc(OUFA_INFO* info, OUFA_TMP* tmp){
	cblas_dscal(info->p*info->p * 2, 0.0, &tmp->Y[0][0], 1);
	for (int i = 0; i < info->p; i++)
		tmp->Y[i][i] = 1;
	//modify
	/*for (int i = info->p; i < info->p * 2; i++)
		tmp->Y[i][i - info->p] = 1.0;*/
	tmp->OP_T = (double**)malloc(sizeof(double*)*info->N);
	tmp->OP_T[0] = (double*)malloc(sizeof(double)*info->N*info->p);
#pragma omp parallel
	{
		int now = omp_get_thread_num();
		for (int i = now; i < info->N; i += NUM_OF_THREADS)
			tmp->OP_T[i] = tmp->OP_T[0] + i*(info->p);
	}
	tmp->dOP_T = (double**)malloc(sizeof(double*)*info->N);
	tmp->dOP_T[0] = (double*)malloc(sizeof(double)*info->N*info->p);
#pragma omp parallel
	{
		int now = omp_get_thread_num();
		for (int i = now; i < info->N; i += NUM_OF_THREADS)
			tmp->dOP_T[i] = tmp->dOP_T[0] + i*(info->p);
	}
	Local_Opt_Inc(info, tmp);
	free(tmp->OP_T[0]);
	free(tmp->OP_T);
	free(tmp->dOP_T[0]);
	free(tmp->dOP_T);
	return 0;
}
int Poi_Release(OUFA_INFO* dst_info, OUFA_TMP* dst_tmp, OUFA_INFO* src_info, OUFA_TMP* src_tmp, OUFA_INFO* ano_info, OUFA_TMP* ano_tmp){

	free(ano_tmp->Y[0]);
	free(ano_tmp->Y);
	free(ano_tmp->dY[0]);
	free(ano_tmp->dY);
	free(ano_info->U_secmap);
	free(ano_info->V_secmap);
	free(ano_info->sigma);
	free(ano_info->trIdx[0]);
	free(ano_info->trIdx);
	free(ano_info->trR);
	free(ano_tmp->eigV);
	free(ano_tmp->Uorder);
	free(ano_tmp->Unum);

	free(dst_tmp->Y[0]);
	free(dst_tmp->Y);
	free(dst_tmp->dY[0]);
	free(dst_tmp->dY);
	free(dst_info->U_secmap);
	free(dst_info->V_secmap);
	free(dst_info->sigma);
	free(dst_info->trIdx[0]);
	free(dst_info->trIdx);
	free(dst_info->trR);
	free(dst_tmp->eigV);
	free(dst_tmp->Uorder);
	free(dst_tmp->Unum);
	double** Now_Y;
	Now_Y = (double **)malloc(sizeof(double*)*src_info->N);
	Now_Y[0] = (double*)malloc(sizeof(double)*src_info->N*src_info->p);
	for (int i = 1; i < src_info->N; i++)
		Now_Y[i] = Now_Y[0] + i*src_info->p;
#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < src_info->N; i += NUM_OF_THREADS)
		for (int j = 0; j < src_info->p; j++)
			Now_Y[i][j] = cblas_ddot(src_info->p * 2, &src_info->U[i][0], 1, &src_tmp->Y[0][j], src_info->p);
	}
	dst_info->N = src_info->N;
	dst_info->nU = src_info->nU;
	dst_info->nV = src_info->nV;
	dst_info->num_tr = src_info->num_tr;
	dst_info->U = (double **)malloc(sizeof(double *)*dst_info->N);
	dst_info->V = dst_info->U + dst_info->nU;
	dst_info->trIdx = (int **)malloc(sizeof(int*)*dst_info->num_tr);
	dst_info->trIdx[0] = (int *)malloc(sizeof(int)*dst_info->num_tr * 2);
	for (int i = 1; i < dst_info->num_tr; i++)
		dst_info->trIdx[i] = dst_info->trIdx[0] + 2 * i;
	dst_info->trR = (double *)malloc(sizeof(double)*dst_info->num_tr * 2);
	dst_info->ptrR = dst_info->trR + dst_info->num_tr;
	dst_tmp->eigV = (double*)malloc(sizeof(double)*(dst_info->N + dst_info->num_tr * 3));
	dst_tmp->Ur = dst_tmp->eigV + dst_info->N;
	dst_tmp->Ue = dst_tmp->Ur + dst_info->num_tr;
	dst_tmp->Ve = dst_tmp->Ue + dst_info->num_tr;
	dst_tmp->Uorder = (int*)malloc(sizeof(int)*(dst_info->num_tr * 3 + dst_info->N * 2));
	dst_tmp->Vorder = dst_tmp->Uorder + dst_info->num_tr;
	dst_tmp->UmapV = dst_tmp->Vorder + dst_info->num_tr;
	dst_tmp->Unum = (int **)malloc(sizeof(int *)*dst_info->N);
	dst_tmp->Vnum = dst_tmp->Unum + dst_info->nU;
	for (int i = 0; i < dst_info->N; i++)
		dst_tmp->Unum[i] = dst_tmp->UmapV + dst_info->num_tr + 2 * i;

	dst_tmp->Y = dst_info->U;
	dst_tmp->dY = (double **)malloc(sizeof(double *)*dst_info->N);

	

#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < dst_info->num_tr; i += NUM_OF_THREADS){
			dst_info->trIdx[i][0] = src_info->trIdx[i][0];
			dst_info->trIdx[i][1] = src_info->trIdx[i][1];
			dst_info->trR[i] = src_info->trR[i];
			dst_tmp->Uorder[i] = src_tmp->Uorder[i];
			dst_tmp->Vorder[i] = src_tmp->Vorder[i];
			dst_tmp->UmapV[i] = src_tmp->UmapV[i];

		}
	}
	dst_tmp->MaxR = src_tmp->MaxR;
	dst_tmp->MinR = src_tmp->MinR;
	dst_tmp->avr = src_tmp->avr;
	dst_info->U_secmap = (int *)malloc(sizeof(int)*dst_info->nU);
	dst_info->V_secmap = (int *)malloc(sizeof(int)*dst_info->nV);
#pragma omp parallel
	{
		int now_id = omp_get_thread_num();
		for (int i = now_id; i < dst_info->nU; i += NUM_OF_THREADS){
			dst_tmp->Unum[i][0] = src_tmp->Unum[i][0];
			dst_tmp->Unum[i][1] = src_tmp->Unum[i][1];
			dst_info->U_secmap[i] = src_info->U_secmap[i];
		}
		for (int i = now_id; i < dst_info->nV; i += NUM_OF_THREADS){
			dst_tmp->Vnum[i][0] = src_tmp->Vnum[i][0];
			dst_tmp->Vnum[i][1] = src_tmp->Vnum[i][1];
			dst_info->V_secmap[i] = src_info->V_secmap[i];
		}
	}
	dst_info->sigma = (double *)malloc(sizeof(double)*dst_info->p);
	cblas_dcopy(dst_info->p, &src_info->sigma[0], 1, &dst_info->sigma[0], 1);
	dst_tmp->p = dst_info->p;

	dst_tmp->Y[0] = (double*)malloc(sizeof(double)*dst_info->N*dst_info->p);
	for (int i = 0; i < dst_info->N; i++)
		dst_tmp->Y[i] = dst_tmp->Y[0] + i*dst_info->p;
	cblas_dcopy(dst_info->N*dst_info->p, &Now_Y[0][0], 1, &dst_tmp->Y[0][0], 1);

	dst_tmp->dY[0] = (double*)malloc(sizeof(double)*dst_info->N*dst_info->p);
	for (int i = 0; i < dst_info->N; i++)
		dst_tmp->dY[i] = dst_tmp->dY[0] + i*dst_info->p;
	dst_tmp->Meanr = src_tmp->Meanr;
	dst_tmp->M = src_tmp->M;

	free(src_info->sigma);
	free(src_tmp->Y[0]);
	free(src_tmp->Y);
	free(src_tmp->dY[0]);
	free(src_tmp->dY);
	free(src_info->U[0]);
	free(src_info->U);
	free(src_info->U_secmap);
	free(src_info->V_secmap);
	free(src_info->trIdx[0]);
	free(src_info->trIdx);
	free(src_info->trR);
	free(src_tmp->eigV);
	free(src_tmp->Uorder);
	free(src_tmp->Unum);
	free(Now_Y[0]);
	free(Now_Y);
	/*dst_info = src_info;
	dst_tmp = src_tmp;*/
	/*
	dst_info->U = Now_Y;
	//dst_info->U[0]=Now_Y->
	dst_info->V = dst_info->U + dst_info->nU;
	dst_tmp->Y = dst_info->U;
	dst_tmp->dY = (double**)malloc(sizeof(double*)*dst_info->N);
	dst_tmp->dY[0] = (double *)malloc(sizeof(double)*dst_info->N*dst_info->p);
	for (int i = 1; i < dst_info->N; i++)
		dst_tmp->dY[i] = dst_tmp->dY[0] + i*dst_info->p;

	src_info = NULL;
	src_tmp = NULL;*/
	return (0);
}
int main(){
	char outp_data_path[100];
	glo_rmse = 0;
	glo_cnt = 0;
	int dep = 199;
	omp_set_num_threads(NUM_OF_THREADS);
	OUFA_INFO fin_info[2],com_info;
	OUFA_TMP fin_tmp[2], com_tmp;
	int now = 0;
	int nex = 1;
	Data_Input(&fin_info[nex], &fin_tmp[nex],0);
	Global_Opt(&fin_info[nex], &fin_tmp[nex]);
	for (int cas = 1; cas <= dep; cas++){
		printf("%d\t", cas);
		now = 1 ^ now;
		nex = 1 ^ nex;

		Data_Input(&fin_info[nex], &fin_tmp[nex], cas);

		double rmse = tsRMSE(&fin_tmp[now], &fin_info[now], &fin_info[nex]);
        printf("%.8lf\t%.8lf\t", glo_rmse, rmse);

		Global_Opt(&fin_info[nex], &fin_tmp[nex]);

		Data_Combine(&fin_info[now], &fin_tmp[now], &fin_info[nex], &fin_tmp[nex], &com_info, &com_tmp);

		Global_Opt_Inc(&com_info, &com_tmp);

		Poi_Release(&fin_info[nex], &fin_tmp[nex], &com_info, &com_tmp, &fin_info[now], &fin_tmp[now]);
	}
	return 0;
}
