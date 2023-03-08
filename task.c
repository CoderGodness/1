#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

	double* matrixOld = (double*)malloc(totalSize * sizeof(double));
	double* matrixNew = (double*)malloc(totalSize * sizeof(double));

	matrixOld[0] = cornerUL;
	matrixOld[size - 1] = cornerUR;
	matrixOld[totalSize - 1] = cornerBR;
	matrixOld[totalSize - size] = cornerBL;


	const double fraction = (cornerUR - cornerUL) / size;
	#pragma acc enter data copyin(matrixOld[0:totalSize]) create(matrixNew[0:totalSize])
	#pragma acc parallel loop 
	for (int i = 1; i < size - 1; i++)
	{
		matrixOld[i] = cornerUL + i * fraction;
		matrixOld[i * size] = cornerUL + i * fraction;
		matrixOld[(size - 1) * i] = cornerUR + i * fraction;
		matrixOld[size * (size - 1) + i] = cornerUR + i * fraction;
		
		matrixNew[i] = matrixOld[i];
		matrixNew[i * size] = matrixOld[i * size];
		matrixNew[(size - 1) * i] = matrixOld[(size - 1) * i];
		matrixNew[size * (size - 1) + i] = matrixOld[size * (size - 1) + i];
	}

	double errorNow = 1.0;
	int iterNow = 0;
	while (errorNow > maxError && iterNow < maxIteration)
	{
		iterNow++;
		#pragma acc parallel loop independent collapse(2) vector vector_length(size) gang num_gangs(size) reduction(max:errorNow)
		for (int i = 1; i < size - 1; i++)
		{
			for (int j = 1; j < size - 1; j++)
			{
				matrixNew[i * size + j] = 
					( matrixOld[i * size + j - 1]
					+ matrixOld[(i - 1) * size + j] 
					+ matrixOld[(i + 1) * size + j] 
					+ matrixOld[i * size + j + 1] ) / 4;
				errorNow = fmax(errorNow, matrixNew[i * size + j] - matrixOld[i * size + j]);
			}
		}
		double* temp = matrixOld;
		matrixOld = matrixNew;
		matrixNew = temp;
	}
	#pragma acc update host(errorNow)
	#pragma acc exit data delete(matrixOld, matrixNew)
	printf("iterations = %d, error = %d", errorNow, errorNow);
	free(matrixOld);
	free(matrixNew);
	return 0;
}
