#include<cstdio>
#include<cstring>
#include<iostream>
#include<string>
#include <mkl_cblas.h>
#include <mkl_vml_functions.h>
#include <mkl_vsl.h>
#include<omp.h>
#include<algorithm>
#include<ctime>
#include<cmath>
#include"OUFA.h"
#include"Data_Init.h"

using namespace std;
int Cmp_Usr(COM_REC rec1, COM_REC rec2){
	if (rec1.usr != rec2.usr)
		return (rec1.usr < rec2.usr);
	return (rec1.itm < rec2.itm);
}
int Cmp_Itm(COM_REC rec1, COM_REC rec2){
	if (rec1.itm != rec2.itm)
		return (rec1.itm < rec2.itm);
	return (rec1.usr < rec2.usr);
}
int Data_Input(OUFA_INFO *info, OUFA_TMP *tmp, int sta){
	char path[100];
	sprintf(path, "%sdata.statistics.%d", DATA_PATH ,sta);
	freopen(path, "r", stdin);
	int nU, nV;
	scanf("%d%d", &(info->nU), &(info->nV));
	info->N = info->nU + info->nV;
	scanf("%d", &(info->num_tr));
	fclose(stdin);

	info->U = (double **)malloc(sizeof(double *)*info->N);
	info->V = info->U + info->nU;

	info->trIdx = (int **)malloc(sizeof(int*)*info->num_tr);
	info->trIdx[0] = (int *)malloc(sizeof(int)*info->num_tr * 2);

	for (int i = 1; i < info->num_tr ; i++)
		info->trIdx[i] = info->trIdx[0] + 2 * i;

	info->trR = (double *)malloc(sizeof(double)*info->num_tr * 2);
	info->ptrR = info->trR + info->num_tr;

	tmp->eigV = (double*)malloc(sizeof(double)*(info->N + info->num_tr * 3));
	tmp->Ur = tmp->eigV + info->N;
	tmp->Ue = tmp->Ur + info->num_tr;
	tmp->Ve = tmp->Ue + info->num_tr;

	tmp->Uorder = (int*)malloc(sizeof(int)*(info->num_tr * 3 + info->N * 2));
	tmp->Vorder = tmp->Uorder + info->num_tr;
	tmp->UmapV = tmp->Vorder + info->num_tr;
	tmp->Unum = (int **)malloc(sizeof(int *)*info->N);
	tmp->Vnum = tmp->Unum + info->nU;
	for (int i = 0; i < info->N; i++)
		tmp->Unum[i] = tmp->UmapV + info->num_tr + 2 * i;

	tmp->Y = info->U;
	tmp->dY = (double **)malloc(sizeof(double *)*info->N);

	tmp->MaxR = 0.0;
	tmp->MinR = 100.0;
	tmp->avr = 0.0;

	sprintf(path, "%sUorder_data.%d", DATA_PATH, sta);
	freopen(path, "r", stdin);
	for (int i = 0; i < info->num_tr; i++){
		scanf("%d%d%lf", &(info->trIdx[i][0]), &(info->trIdx[i][1]), &(info->trR[i]));
		info->trIdx[i][0]--;
		info->trIdx[i][1]--;
		tmp->Ur[i] = info->trR[i];
		tmp->Uorder[i] = info->trIdx[i][1];
		tmp->avr += info->trR[i] * info->trR[i];
		if (info->trR[i]>tmp->MaxR)
			tmp->MaxR = info->trR[i];
		if (info->trR[i] < tmp->MinR)
			tmp->MinR = info->trR[i];
	}
	tmp->avr = sqrt(tmp->avr);
	fclose(stdin);

	sprintf(path, "%sVorder_data.%d", DATA_PATH,sta);
	freopen(path, "r", stdin);
	for (int i = 0; i < info->num_tr; i++){
		int tp;
		double tpp;
		scanf("%d%d%lf", &tp, &(tmp->Vorder[i]), &tpp);
		tmp->Vorder[i]--;
	}
	fclose(stdin);

	sprintf(path, "%sUorder_Map_Vorder.%d", DATA_PATH, sta);
	freopen(path, "r", stdin);
	for (int i = 0; i < info->num_tr; i++){
		int tp, tpp;
		scanf("%d%d%d", &tp, &tpp, &(tmp->UmapV[i]));
		tmp->UmapV[i]--;
	}
	fclose(stdin);

	sprintf(path, "%sUser_Num.%d", DATA_PATH ,sta);
	freopen(path, "r", stdin);
	for (int i = 0; i < info->nU; i++){
		scanf("%d%d", &(tmp->Unum[i][0]), &(tmp->Unum[i][1]));
		tmp->Unum[i][0]--;
	}
	fclose(stdin);

	sprintf(path, "%sItem_Num.%d", DATA_PATH, sta);
	freopen(path, "r", stdin);
	for (int i = 0; i < info->nV; i++){
		scanf("%d%d", &(tmp->Vnum[i][0]), &(tmp->Vnum[i][1]));
		tmp->Vnum[i][0]--;
	}
	fclose(stdin);

	sprintf(path, "%sSec_User_Map.%d", DATA_PATH, sta);
	freopen(path, "r", stdin);
	info->U_secmap = (int *)malloc(sizeof(int)*info->nU);
	for (int i = 0; i < info->nU; i++)
		scanf("%d", &(info->U_secmap[i]));
	fclose(stdin);

	sprintf(path, "%sSec_Item_Map.%d", DATA_PATH,sta);
	freopen(path, "r", stdin);
	info->V_secmap = (int *)malloc(sizeof(int)*info->nV);
	for (int i = 0; i < info->nV; i++)
		scanf("%d", &(info->V_secmap[i]));
	fclose(stdin);

	sprintf(path, "%ssigma.train", DATA_PATH);
	freopen(path, "r", stdin);
	scanf("%d", &info->p);
	info->sigma = (double *)malloc(sizeof(double)*info->p);
	for (int i = 0; i < info->p; i++){
        scanf("%lf", &info->sigma[i]);
    }
	fclose(stdin);

	tmp->p = 0;

	tmp->Y[0] = (double*)malloc(sizeof(double)*info->N*info->p);
	for (int i = 0; i < info->N; i++)
		tmp->Y[i] = tmp->Y[0] + i*info->p;

	tmp->dY[0] = (double*)malloc(sizeof(double)*info->N*info->p);
	for (int i = 0; i < info->N; i++)
		tmp->dY[i] = tmp->dY[0] + i*info->p;

	tmp->Meanr = 0.0;
	for (int i = 0; i < info->num_tr; i++){
		tmp->Ur[i] = (tmp->Ur[i] - tmp->MinR) / (tmp->MaxR - tmp->MinR);
		tmp->Meanr += tmp->Ur[i];
	}
	tmp->Meanr /= double(info->num_tr);

	for (int i = 0; i < info->num_tr; i++){
		tmp->Ur[i] -= tmp->Meanr;
		tmp->Ue[i] = tmp->Ur[i];
		tmp->Ve[tmp->UmapV[i]] = tmp->Ue[i];
	}

	tmp->M = info->N / 20.0;
	return 0;

}
int Data_Combine(OUFA_INFO* pre_info, OUFA_TMP* pre_tmp, OUFA_INFO* now_info, OUFA_TMP* now_tmp, OUFA_INFO* com_info, OUFA_TMP* com_tmp){
	com_info->p = now_info->p;
	com_info->U_secmap = (int *)malloc(sizeof(int)*(pre_info->nU + now_info->nU));
	com_info->V_secmap = (int *)malloc(sizeof(int)*(pre_info->nV + now_info->nV));
	com_info->nU = 0;
	com_info->nV = 0;
	int pre_poi = 0;
	int now_poi = 0;
	// combine U_secmap
	while (pre_poi < pre_info->nU || now_poi < now_info->nU){
		if (pre_poi == pre_info->nU){
			com_info->U_secmap[com_info->nU] = now_info->U_secmap[now_poi];
			com_info->nU++;
			now_poi++;
			continue;
		}
		if (now_poi == now_info->nU){
			com_info->U_secmap[com_info->nU] = pre_info->U_secmap[pre_poi];
			com_info->nU++;
			pre_poi++;
			continue;
		}
		if (pre_info->U_secmap[pre_poi] < now_info->U_secmap[now_poi]){
			com_info->U_secmap[com_info->nU] = pre_info->U_secmap[pre_poi];
			com_info->nU++;
			pre_poi++;
			continue;
		}
		if (pre_info->U_secmap[pre_poi] > now_info->U_secmap[now_poi]){
			com_info->U_secmap[com_info->nU] = now_info->U_secmap[now_poi];
			com_info->nU++;
			now_poi++;
			continue;
		}
		if (pre_info->U_secmap[pre_poi] == now_info->U_secmap[now_poi]){
			com_info->U_secmap[com_info->nU] = now_info->U_secmap[now_poi];
			com_info->nU++;
			now_poi++;
			pre_poi++;
			continue;
		}
	}
	pre_poi = 0;
	now_poi = 0;
	// combine V_secmap
	while (pre_poi < pre_info->nV || now_poi < now_info->nV){
		if (pre_poi == pre_info->nV){
			com_info->V_secmap[com_info->nV] = now_info->V_secmap[now_poi];
			now_poi++;
			com_info->nV++;
			continue;
		}
		if (now_poi == now_info->nV){
			com_info->V_secmap[com_info->nV] = pre_info->V_secmap[pre_poi];
			pre_poi++;
			com_info->nV++;
			continue;
		}
		if (pre_info->V_secmap[pre_poi] < now_info->V_secmap[now_poi]){
			com_info->V_secmap[com_info->nV] = pre_info->V_secmap[pre_poi];
			pre_poi++;
			com_info->nV++;
			continue;
		}
		if (pre_info->V_secmap[pre_poi]>now_info->V_secmap[now_poi]){
			com_info->V_secmap[com_info->nV] = now_info->V_secmap[now_poi];
			now_poi++;
			com_info->nV++;
			continue;
		}
		if (pre_info->V_secmap[pre_poi] == now_info->V_secmap[now_poi]){
			com_info->V_secmap[com_info->nV] = now_info->V_secmap[now_poi];
			now_poi++;
			pre_poi++;
			com_info->nV++;
			continue;
		}
	}
	//debug
	/*printf("U List:\n");
	for (int i = 0; i < com_info->nU; i++)
		printf("%d ", com_info->U_secmap[i]);
	printf("\nV List:\n");
	for (int i = 0; i < com_info->nV; i++)
		printf("%d ", com_info->V_secmap[i]);
	printf("\n");*/
	//debug end
	com_info->N = com_info->nU + com_info->nV;
	com_info->U = (double **)malloc(sizeof(double *)*com_info->N);
	com_info->V = com_info->U + com_info->nU;
	com_info->U[0] = (double *)malloc(sizeof(double)*com_info->N*com_info->p * 2);
	cblas_dscal(com_info->N*com_info->p * 2, 0.0, com_info->U[0], 1);

	for (int i = 1; i < com_info->nU; i++)
		com_info->U[i] = com_info->U[0] + i*(2 * com_info->p);
	for (int i = 0; i < com_info->nV; i++)
		com_info->V[i] = com_info->U[0] + (i + com_info->nU)*(2 * com_info->p);

	//这部分记得加并行化
	for (int i = 0; i < pre_info->nU; i++){
		int uid = lower_bound(com_info->U_secmap, com_info->U_secmap + com_info->nU, pre_info->U_secmap[i]) - com_info->U_secmap;
		cblas_dcopy(pre_info->p, &pre_info->U[i][0], 1, &com_info->U[uid][0], 1);
	}
	for (int i = 0; i < pre_info->nV; i++){
		int vid = lower_bound(com_info->V_secmap, com_info->V_secmap + com_info->nV, pre_info->V_secmap[i]) - com_info->V_secmap;
		cblas_dcopy(pre_info->p, &pre_info->V[i][0], 1, &com_info->V[vid][0], 1);
	}
	for (int i = 0; i < now_info->nU; i++){
		int uid = lower_bound(com_info->U_secmap, com_info->U_secmap + com_info->nU, now_info->U_secmap[i]) - com_info->U_secmap;
		cblas_dcopy(pre_info->p, &now_info->U[i][0], 1, &com_info->U[uid][com_info->p], 1);
	}
	for (int i = 0; i < now_info->nV; i++){
		int vid = lower_bound(com_info->V_secmap, com_info->V_secmap + com_info->nV, now_info->V_secmap[i]) - com_info->V_secmap;
		cblas_dcopy(pre_info->p, &now_info->V[i][0], 1, &com_info->V[vid][com_info->p], 1);
	}

	com_info->sigma = (double *)malloc(sizeof(double)*com_info->p);
	cblas_dcopy(com_info->p, now_info->sigma, 1, com_info->sigma, 1);
	COM_REC* com_rating;
	com_info->num_tr = 0;
	srand(unsigned(time(0)));
	com_rating = (COM_REC*)malloc(sizeof(COM_REC)*(pre_info->num_tr + now_info->num_tr));

	int sam_num = 0;
	for (int i = 0; i < pre_info->num_tr; i++){
		int simp = rand();
		if ((simp % 6) ==0)
			continue;
		sam_num++;
		int uid = pre_info->U_secmap[pre_info->trIdx[i][0]];
		uid = lower_bound(com_info->U_secmap, com_info->U_secmap + com_info->nU, uid) - com_info->U_secmap;
		int vid = pre_info->V_secmap[pre_info->trIdx[i][1]];
		vid = lower_bound(com_info->V_secmap, com_info->V_secmap + com_info->nV, vid) - com_info->V_secmap;
		com_rating[com_info->num_tr].usr = uid;
		com_rating[com_info->num_tr].itm = vid;
		com_rating[com_info->num_tr].rating = pre_info->trR[i];
		com_rating[com_info->num_tr].vorder_id = -1;
		com_info->num_tr++;
	}
	for (int i = 0; i < now_info->num_tr; i++){
		int simp = rand();
		if ((simp % 6)==-1)
			continue;
		sam_num++;
		int uid = now_info->U_secmap[now_info->trIdx[i][0]];
		uid = lower_bound(com_info->U_secmap, com_info->U_secmap + com_info->nU, uid) - com_info->U_secmap;
		int vid = now_info->V_secmap[now_info->trIdx[i][1]];
		vid = lower_bound(com_info->V_secmap, com_info->V_secmap + com_info->nV, vid) - com_info->V_secmap;
		com_rating[com_info->num_tr].usr = uid;
		com_rating[com_info->num_tr].itm = vid;
		com_rating[com_info->num_tr].rating = now_info->trR[i];
		com_rating[com_info->num_tr].vorder_id = -1;
		com_info->num_tr++;
	}
	
	printf("%d\t%d\t", sam_num,sam_num);

	com_info->trIdx = (int **)malloc(sizeof(int *)*com_info->num_tr);
	com_info->trIdx[0] = (int*)malloc(sizeof(int)* 2 * com_info->num_tr);
	for (int i = 1; i < com_info->num_tr; i++)
		com_info->trIdx[i] = com_info->trIdx[0] + 2 * i;
	com_info->trR = (double *)malloc(sizeof(double)*com_info->num_tr);

	//init OUFA com_tmp
	com_tmp->eigV = (double*)malloc(sizeof(double)*(com_info->N + com_info->num_tr * 3));
	com_tmp->Ur = com_tmp->eigV + com_info->N;
	com_tmp->Ue = com_tmp->Ur + com_info->num_tr;
	com_tmp->Ve = com_tmp->Ue + com_info->num_tr;

	com_tmp->Uorder = (int*)malloc(sizeof(int)*(com_info->num_tr * 3 + com_info->N * 2));
	com_tmp->Vorder = com_tmp->Uorder + com_info->num_tr;
	com_tmp->UmapV = com_tmp->Vorder + com_info->num_tr;
	com_tmp->Unum = (int **)malloc(sizeof(int *)*com_info->N);
	com_tmp->Vnum = com_tmp->Unum + com_info->nU;
	for (int i = 0; i < com_info->N; i++)
		com_tmp->Unum[i] = com_tmp->UmapV + com_info->num_tr + 2 * i;

	memset(com_tmp->Unum[0], 0, sizeof(int)* 2 * com_info->N);

	com_tmp->MaxR = 0.0;
	com_tmp->MinR = 100.0;
	com_tmp->avr = 0.0;

	sort(com_rating, com_rating + com_info->num_tr, Cmp_Itm);


	for (int i = 0; i < com_info->num_tr; i++){
		com_tmp->Vorder[i] = com_rating[i].usr;
		com_rating[i].vorder_id = i;
	}

	int itm_now = -1;
	for (int i = 0; i < com_info->num_tr; i++){
		if (com_rating[i].itm != itm_now){
			itm_now = com_rating[i].itm;
			com_tmp->Vnum[itm_now][0] = i;
			com_tmp->Vnum[itm_now][1] = 1;
		}
		else
			com_tmp->Vnum[itm_now][1]++;
	}


	sort(com_rating, com_rating + com_info->num_tr, Cmp_Usr);
	for (int i = 0; i < com_info->num_tr; i++){
		com_info->trIdx[i][0] = com_rating[i].usr;
		com_info->trIdx[i][1] = com_rating[i].itm;
		com_info->trR[i] = com_rating[i].rating;
		com_tmp->Ur[i] = com_info->trR[i];
		com_tmp->Uorder[i] = com_info->trIdx[i][1];
		com_tmp->avr += (com_info->trR[i] * com_info->trR[i]);
		if (com_info->trR[i]>com_tmp->MaxR)
			com_tmp->MaxR = com_info->trR[i];
		if (com_info->trR[i] < com_tmp->MinR)
			com_tmp->MinR = com_info->trR[i];
	}
	com_tmp->avr = sqrt(com_tmp->avr);

	for (int i = 0; i < com_info->num_tr; i++)
		com_tmp->UmapV[i] = com_rating[i].vorder_id;

	int usr_now = -1;
	for (int i = 0; i < com_info->num_tr; i++){
		if (com_rating[i].usr != usr_now){
			usr_now = com_rating[i].usr;
			com_tmp->Unum[usr_now][0] = i;
			com_tmp->Unum[usr_now][1] = 1;
		}
		else
			com_tmp->Unum[usr_now][1]++;
	}

	com_tmp->p = com_info->p;

	com_tmp->Y = (double **)malloc(sizeof(double *)* 2 * com_info->p);
	com_tmp->Y[0] = (double *)malloc(sizeof(double)* 2 * com_info->p*com_info->p);
	for (int i = 1; i < 2 * com_info->p; i++)
		com_tmp->Y[i] = com_tmp->Y[0] + i*com_info->p;

	com_tmp->dY = (double**)malloc(sizeof(double *)* 2 * com_info->p);
	com_tmp->dY[0] = (double *)malloc(sizeof(double)* 2 * com_info->p*com_info->p);
	for (int i = 1; i < 2 * com_info->p; i++)
		com_tmp->dY[i] = com_tmp->dY[0] + i*com_info->p;

	com_tmp->Meanr = 0.0;
	for (int i = 0; i < com_info->num_tr; i++){
		com_tmp->Ur[i] = (com_tmp->Ur[i] - com_tmp->MinR) / (com_tmp->MaxR - com_tmp->MinR);
		com_tmp->Meanr += com_tmp->Ur[i];
	}
	com_tmp->Meanr /= double(com_info->num_tr);

	for (int i = 0; i < com_info->num_tr; i++){
		com_tmp->Ur[i] -= com_tmp->Meanr;
		com_tmp->Ue[i] = com_tmp->Ur[i];
		com_tmp->Ve[com_tmp->UmapV[i]] = com_tmp->Ue[i];
	}

	com_tmp->M = com_info->N / 20.0;
	free(com_rating);
	return 0;
	//释放inc_info inc_tmp com_info com_tmp
}
