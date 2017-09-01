#ifndef OUFA_H
#define OUFA_H

struct OUFA_info
{
	int nU;
	int nV;
	int N;

	int p;

	double* sigma;

	double** U;
	double** V;

	int* U_secmap;
	int* V_secmap;

	int num_tr;
	int** trIdx;
	double* trR;
	double* ptrR;

};

struct OUFA_tmp
{
	double MaxR;
	double MinR;

	double Meanr;
	double avr;

	double M;
	double minEig;
	double* eigV;
	int p;

	double epsilon;

	int* Uorder;
	int* Vorder;
	double* Ur;
	double* Ue;
	double* Ve;

	int* UmapV;

	int** Unum;
	int** Vnum;

	double** OP_T;
	double** dOP_T;
	double** Y;
	double** dY;
	double** bY;
};

typedef struct OUFA_info OUFA_INFO;
typedef struct OUFA_tmp OUFA_TMP;

int Global_Opt(OUFA_INFO* info, OUFA_TMP* tmp);
//double tsRMSE(OUFA_TMP* src_tmp,OUFA_INFO* src_info, OUFA_INFO* dst_info);

#endif