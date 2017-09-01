#ifndef DATA_INIT_H
#define DATA_INIT_H

#include"OUFA.h"
#define DATA_PATH "/home/xunpenghuang/LFT_Experiment/data/MovieLens10M/incremental_departs_LFT/"
struct COM_Rec{
	int usr, itm;
	double rating;
	int vorder_id;
};
typedef struct COM_Rec COM_REC;
int Cmp_Usr(COM_REC rec1, COM_REC rec2);
int Cmp_Itm(COM_REC rec1, COM_REC rec2);
int Data_Input(OUFA_INFO *info, OUFA_TMP *tmp, int sta);
int Data_Combine(OUFA_INFO* pre_info, OUFA_TMP* pre_tmp, OUFA_INFO* now_info, OUFA_TMP* now_tmp, OUFA_INFO* com_info, OUFA_TMP* com_tmp);
#endif