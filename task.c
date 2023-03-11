#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>


double matrixCalc(int size, double* matrixOld, double* matrixNew)
{
	double error = 0.0;
#pragma acc parallel loop independent collapse(2) vector vector_length(size) gang num_gangs(size) reduction(max:error) present(matrixOld[0:size*size], matrixNew[0:size*size])
	for (size_t i = 1; i < size - 1; i++)
	{
		for (size_t j = 1; j < size - 1; j++)
		{
			matrixNew[i * size + j] = 0.25 * (
				matrixOld[i * size + j - 1] +
				matrixOld[(i - 1) * size + j] +
				matrixOld[(i + 1) * size + j] +
				matrixOld[i * size + j + 1]);
			error = fmax(error, matrixNew[i * size + j] - matrixOld[i * size + j]);
		}
	}
	return error;
}

void matrixSwap(int totalSize)
{
#pragma acc data present(matrixOld[0:totalSize], matrixNew[0:totalSize])
	{
		double* temp = matrixOld;
		matrixOld = matrixNew;
		matrixNew = temp;
	}
}

int main(int argc, char** argv)
{
	int cornerUL = 10;
	int cornerUR = 20;
	int cornerBR = 30;
	int cornerBL = 20;
	char* eptr;
	const double maxError = strtod((argv[1]), &eptr);
	const int size = atoi(argv[2]);
	const int maxIteration = atoi(argv[3]);

	int totalSize = size * size;

	double* matrixOld = (double*)calloc(totalSize, sizeof(double));
	double* matrixNew = (double*)calloc(totalSize, sizeof(double));

	const double fraction = 10.0 / (size - 1);
	double errorNow = 1.0;
	int iterNow = 0;
#pragma acc enter data create(matrixOld[0:totalSize], matrixNew[0:totalSize])
#pragma acc parallel loop 
	for (int i = 0; i < size; i++)
	{
		matrixOld[i] = cornerUL + i * fraction;
		matrixOld[i * size] = cornerUL + i * fraction;
		matrixOld[size * i + size - 1] = cornerUR + i * fraction;
		matrixOld[size * (size - 1) + i] = cornerUR + i * fraction;

		matrixNew[i] = matrixOld[i];
		matrixNew[i * size] = matrixOld[i * size];
		matrixNew[size * i + size - 1] = matrixOld[size * i + size - 1];
		matrixNew[size * (size - 1) + i] = matrixOld[size * (size - 1) + i];
	}
    clock_t algBegin = clock();
	while (errorNow > maxError && iterNow < maxIteration)
	{
		iterNow++;
		errorNow = matrixCalc(size, matrixOld, matrixNew);
		matrixSwap(totalSize);
	}
#pragma acc update host(matrixOld[0:totalSize], matrixNew[0:totalSize])
#pragma acc exit data delete(matrixOld[0:totalSize], matrixNew[0:totalSize])
	  clock_t algEnd = clock();
	printf("iterations = %d, error = %lf, time = %lf\n", iterNow, errorNow, 1.0 * (algEnd - algBegin) / CLOCKS_PER_SEC);
	if (argc>4)
	{
	printf("\n");
	for (int i = 0; i < size; i++)
	{
		for (int j = 0; j < size; j++)
		{
			printf("%lf\t", matrixNew[i * size + j]);
		}
		printf("\n");
	}
	}
	return 0;
}
