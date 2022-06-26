/*
 *
 * Copyright (C) 2022, Numenta, Inc. All rights reserved.
 *
 * The information and source code contained herein is the
 * exclusive property of Numenta Inc.  No part of this software
 * may be used, reproduced, stored or distributed in any form,
 * without explicit written authorization from Numenta Inc.
 *
 * */


const int N = 64;
// test input vectors for length 64
const float x_test[64] = {0.000000,
	1.000000,
	2.000000,
	3.000000,
	4.000000,
	5.000000,
	6.000000,
	7.000000,
	8.000000,
	9.000000,
	10.000000,
	11.000000,
	12.000000,
	13.000000,
	14.000000,
	15.000000,
	16.000000,
	17.000000,
	18.000000,
	19.000000,
	20.000000,
	21.000000,
	22.000000,
	23.000000,
	24.000000,
	25.000000,
	26.000000,
	27.000000,
	28.000000,
	29.000000,
	30.000000,
	31.000000,
	32.000000,
	33.000000,
	34.000000,
	35.000000,
	36.000000,
	37.000000,
	38.000000,
	39.000000,
	40.000000,
	41.000000,
	42.000000,
	43.000000,
	44.000000,
	45.000000,
	46.000000,
	47.000000,
	48.000000,
	49.000000,
	50.000000,
	51.000000,
	52.000000,
	53.000000,
	54.000000,
	55.000000,
	56.000000,
	57.000000,
	58.000000,
	59.000000,
	60.000000,
	61.000000,
	62.000000,
	63.000000,
};

const float v_test[64] = {0.000000,
	0.833333,
	1.666667,
	2.500000,
	3.333333,
	4.166667,
	5.000000,
	5.833333,
	6.666667,
	7.500000,
	8.333333,
	9.166666,
	10.000000,
	10.833333,
	11.666666,
	12.500000,
	13.333333,
	14.166666,
	15.000000,
	15.833333,
	16.666666,
	17.500000,
	18.333332,
	19.166666,
	20.000000,
	20.833332,
	21.666666,
	22.500000,
	23.333332,
	24.166666,
	25.000000,
	25.833332,
	26.666666,
	27.500000,
	28.333332,
	29.166666,
	30.000000,
	30.833332,
	31.666666,
	32.500000,
	33.333332,
	34.166664,
	35.000000,
	35.833332,
	36.666664,
	37.500000,
	38.333332,
	39.166664,
	40.000000,
	40.833332,
	41.666664,
	42.500000,
	43.333332,
	44.166664,
	45.000000,
	45.833332,
	46.666664,
	47.500000,
	48.333332,
	49.166664,
	50.000000,
	50.833332,
	51.666664,
	52.500000,
};

const float gamma_test[64] = {2.300000,
	3.069231,
	3.838462,
	4.607692,
	5.376923,
	6.146154,
	6.915385,
	7.684616,
	8.453846,
	9.223077,
	9.992308,
	10.761539,
	11.530769,
	12.300000,
	13.069231,
	13.838462,
	14.607693,
	15.376923,
	16.146154,
	16.915384,
	17.684616,
	18.453846,
	19.223078,
	19.992308,
	20.761538,
	21.530769,
	22.299999,
	23.069231,
	23.838461,
	24.607693,
	25.376923,
	26.146154,
	26.915384,
	27.684616,
	28.453846,
	29.223078,
	29.992308,
	30.761538,
	31.530769,
	32.299999,
	33.069229,
	33.838463,
	34.607693,
	35.376923,
	36.146152,
	36.915386,
	37.684616,
	38.453846,
	39.223076,
	39.992310,
	40.761539,
	41.530769,
	42.299999,
	43.069229,
	43.838463,
	44.607693,
	45.376923,
	46.146152,
	46.915386,
	47.684616,
	48.453846,
	49.223076,
	49.992310,
	50.761539,
};

const float beta_test[64] = {0.250000,
	0.964286,
	1.678571,
	2.392857,
	3.107143,
	3.821429,
	4.535714,
	5.250000,
	5.964286,
	6.678571,
	7.392857,
	8.107142,
	8.821428,
	9.535714,
	10.250000,
	10.964286,
	11.678572,
	12.392858,
	13.107142,
	13.821428,
	14.535714,
	15.250000,
	15.964286,
	16.678572,
	17.392857,
	18.107143,
	18.821428,
	19.535715,
	20.250000,
	20.964285,
	21.678572,
	22.392857,
	23.107143,
	23.821428,
	24.535715,
	25.250000,
	25.964285,
	26.678572,
	27.392857,
	28.107143,
	28.821428,
	29.535715,
	30.250000,
	30.964285,
	31.678572,
	32.392857,
	33.107143,
	33.821430,
	34.535713,
	35.250000,
	35.964287,
	36.678570,
	37.392857,
	38.107143,
	38.821430,
	39.535713,
	40.250000,
	40.964287,
	41.678570,
	42.392857,
	43.107143,
	43.821430,
	44.535713,
	45.250000,
};

const float b_test[64] = {0.000000,
	0.909091,
	1.818182,
	2.727273,
	3.636364,
	4.545455,
	5.454545,
	6.363636,
	7.272727,
	8.181818,
	9.090909,
	10.000000,
	10.909091,
	11.818182,
	12.727273,
	13.636364,
	14.545455,
	15.454546,
	16.363636,
	17.272728,
	18.181818,
	19.090910,
	20.000000,
	20.909092,
	21.818182,
	22.727274,
	23.636364,
	24.545456,
	25.454546,
	26.363638,
	27.272728,
	28.181820,
	29.090910,
	30.000000,
	30.909092,
	31.818182,
	32.727272,
	33.636364,
	34.545456,
	35.454548,
	36.363636,
	37.272728,
	38.181820,
	39.090912,
	40.000000,
	40.909092,
	41.818184,
	42.727276,
	43.636364,
	44.545456,
	45.454548,
	46.363636,
	47.272728,
	48.181820,
	49.090912,
	50.000000,
	50.909092,
	51.818184,
	52.727276,
	53.636364,
	54.545456,
	55.454548,
	56.363640,
	57.272728,
};

//Expected outut values for Y 
const float Y_test[64] = {-3.671951,
	-4.103208,
	-4.451182,
	-4.715874,
	-4.897285,
	-4.995415,
	-5.010262,
	-4.941828,
	-4.790110,
	-4.555111,
	-4.236831,
	-3.835269,
	-3.350424,
	-2.782299,
	-2.130890,
	-1.396200,
	-0.578229,
	0.323025,
	1.307559,
	2.375377,
	3.526473,
	4.760858,
	6.078518,
	7.479464,
	8.963692,
	10.531200,
	12.181986,
	13.916062,
	15.733416,
	17.634050,
	19.617970,
	21.685167,
	23.835649,
	26.069412,
	28.386457,
	30.786783,
	33.270393,
	35.837280,
	38.487453,
	41.220909,
	44.037640,
	46.937664,
	49.920967,
	52.987541,
	56.137405,
	59.370552,
	62.686981,
	66.086685,
	69.569687,
	73.135956,
	76.785515,
	80.518341,
	84.334457,
	88.233856,
	92.216553,
	96.282509,
	100.431763,
	104.664276,
	108.980103,
	113.379181,
	117.861557,
	122.427200,
	127.076157,
	131.808380,
};
