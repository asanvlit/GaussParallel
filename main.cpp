#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int size, rank;

const double accuracy = 1.e-6;

// определяют номера строк матрицы, выбираемых в качестве ведущих, по итерациям прямого хода
// метода Гаусса – определяет далее порядок выполнения итераций для обратного хода
int* pParallelPivotPos;

// определяют номера итераций прямого хода метода Гаусса, на которых строки процесса использовались в
// качестве ведущих – нулевое значение элемента означает, что соответствующая строка должна обрабатываться при исключении
// неизвестных
int* pProcPivotIter;

int* pProcInd; // Number of the first row located on the processes
int* pProcNum; // Number of the linear system rows located on the processes

void fillUpMatrixWithRandom(double* pMatrix, double* pVector, int n) {
    srand(unsigned(clock()));

    for (int i = 0; i < n; i++) {
        pVector[i] = rand() % 20;
        for (int j = 0; j < n; j++) {
            pMatrix[i * n + j] = rand() % 20;
        }
    }
}

void initProcess(double*& pMatrix, double*& pVector, double*& pResult, double*& pProcRows, double*& pProcVector, double*& pProcResult, int& n, int& pBlockSize) {
    int remRowsNumb;

    if (rank == 0) {
        n = -1;
        while (n < size) {
            printf_s("\nEnter the size of the matrix: ");
            scanf_s("%d", &n);
            if (n < size) {
                printf("Invalid matrix size: n must be greater than the number of processes \n");
            }
        }
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    remRowsNumb = n; // количество оставшихся строк
    for (int i = 0; i < rank; i++) {
        remRowsNumb = remRowsNumb - remRowsNumb / (size - i);
    }
    pBlockSize = remRowsNumb / (size - rank); // количество строк для обработки для данного процесса

    pProcRows = new double[pBlockSize * n];
    pProcVector = new double[pBlockSize];
    pProcResult = new double[pBlockSize];

    pParallelPivotPos = new int[n];
    pProcPivotIter = new int[pBlockSize];

    pProcInd = new int[size];
    pProcNum = new int[size];

    for (int i = 0; i < pBlockSize; i++) {
        pProcPivotIter[i] = -1;
    }

    if (rank == 0) {
        pMatrix = new double[n * n]; pVector = new double[n]; pResult = new double[n];
        fillUpMatrixWithRandom(pMatrix, pVector, n);
    }
}

// Распределение данных между процессами
void dataDistribution(double* pMatrix, double* pProcRows, double* pVector, double* pProcVector, int n, int blockSize) {
    int* pSendNum;	     // Количество элементов, отправленных на процесс
    int* pSendInd;	     // Номер первого элемента, отпарвленного на процесс
    int remRowsNumb = n; // Количество строк, которые еще не были распределены

    pSendInd = new int[size];
    pSendNum = new int[size];

    // Определение позиций матрицы для текущего процесса:
    blockSize = (n / size);
    pSendNum[0] = blockSize * n;
    pSendInd[0] = 0;
    for (int i = 1; i < size; i++) {
        remRowsNumb -= blockSize;
        blockSize = remRowsNumb / (size - i);
        pSendNum[i] = blockSize * n;
        pSendInd[i] = pSendInd[i - 1] + pSendNum[i - 1];
    }

    // Рассылка строк
    MPI_Scatterv(pMatrix, pSendNum, pSendInd, MPI_DOUBLE, pProcRows, pSendNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Определение позиций строк матрицы для текущего процесса:
    remRowsNumb = n;
    pProcInd[0] = 0;
    pProcNum[0] = n / size;
    for (int i = 1; i < size; i++) {
        remRowsNumb -= pProcNum[i - 1];
        pProcNum[i] = remRowsNumb / (size - i);
        pProcInd[i] = pProcInd[i - 1] + pProcNum[i - 1];
    }

    MPI_Scatterv(pVector, pProcNum, pProcInd, MPI_DOUBLE, pProcVector, pProcNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] pSendNum;
    delete[] pSendInd;
}

void resultCollection(double* pProcResult, double* pResult) {
    //Gather the whole result vector on every processor
    MPI_Gatherv(pProcResult, pProcNum[rank], MPI_DOUBLE, pResult, pProcNum, pProcInd, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void printMatrix(double* pMatrix, int RowCount, int ColCount) {
    for (int i = 0; i < RowCount; i++) {
        for (int j = 0; j < ColCount; j++) {
            printf("%7.4f ", pMatrix[i * ColCount + j]);
        }
        printf("\n");
    }
}

void printVector(double* pVector, int n) {
    for (int i = 0; i < n; i++) {
        printf("%7.4f ", pVector[i]);
    }
}

void printResultVector(double* pResult, int n) {
    for (int i = 0; i < n; i++)
        printf("%7.4f ", pResult[pParallelPivotPos[i]]);
}

void parallelEliminateColumns(double* pProcRows, double* pProcVector, double* pPivotRow, int n, int pBlockSize, int iter) {
    double multiplier;

    // вычитание ведущей строки из строк процесса, которые еще не использовались в качестве ведущих
    for (int i = 0; i < pBlockSize; i++) {
        if (pProcPivotIter[i] == -1) {
            multiplier = pProcRows[i * n + iter] / pPivotRow[iter];

            for (int j = iter; j < n; j++) {
                pProcRows[i * n + j] -= pPivotRow[j] * multiplier;
            }

            pProcVector[i] -= pPivotRow[n] * multiplier;
        }
    }
}

// Прямой ход
void parallelGaussianElimination(double* pProcRows, double* pProcVector, int n, int RowNum) {
    int	PivotPos; // Позиция ведущего элемента

    // Вспомогательная структура для выбора ведущей строки
    struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;

    // Ведущая строка и соответсвующий элемент вектора (после = )
    double* pPivotRow = new double[n + 1];

    // Итерации
    for (int i = 0; i < n; i++) {
        // Поиск ведущей строки
        double MaxValue = 0; // Значение ведущего элемента
        for (int j = 0; j < RowNum; j++) {
            if ((pProcPivotIter[j] == -1) && (MaxValue < fabs(pProcRows[j * n + i]))) {
                MaxValue = fabs(pProcRows[j * n + i]);
                PivotPos = j;
            }
        }
        ProcPivot.MaxValue = MaxValue;
        ProcPivot.ProcRank = rank;

        // Поиск "ведущего" процесса - процесса с максимальным значением maxValue
        // Функция MPI_ALLREDUCE отличается от MPI_REDUCE тем, что результат появляется в буфере приема у всех членов группы.
        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // Рассылка ведущей строки
        if (rank == Pivot.ProcRank) {
            pProcPivotIter[PivotPos] = i; // номер итерации
            pParallelPivotPos[i] = pProcInd[rank] + PivotPos;
        }
        MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank,MPI_COMM_WORLD);

        if (rank == Pivot.ProcRank) {
            // Заполнение ведущей строки
            for (int j = 0; j < n; j++) {
                pPivotRow[j] = pProcRows[PivotPos * n + j];
            }
            pPivotRow[n] = pProcVector[PivotPos];
        }
        MPI_Bcast(pPivotRow, n + 1, MPI_DOUBLE, Pivot.ProcRank, MPI_COMM_WORLD);

        parallelEliminateColumns(pProcRows, pProcVector, pPivotRow, n, RowNum, i);
    }
}

// Function for finding the pivot row of the back substitution
void FindBackPivotRow(int RowIndex, int n, int& IterProcRank, int& IterPivotPos) {
    for (int i = 0; i < size - 1; i++) {
        if ((pProcInd[i] <= RowIndex) && (RowIndex < pProcInd[i + 1])) IterProcRank = i;
    }
    if (RowIndex >= pProcInd[size - 1]) IterProcRank = size - 1;
    IterPivotPos = RowIndex - pProcInd[IterProcRank];
}

// Обратный ход:
void ParallelBackSubstitution(double* pProcRows, double* pProcVector, double* pProcResult, int n, int RowNum) {
    int iterProcRank;	// Номер процесса с данной ведущей строкой
    int iterPivotPos;	// Позиция ведущей строки в процессе
    double iterResult;	// Вычисленное значение неизвестной
    double val;

    // Итерации
    for (int i = n - 1; i >= 0; i--) {
        // Определение ранга процесса, который содержит ведущую строку
        FindBackPivotRow(pParallelPivotPos[i], n, iterProcRank, iterPivotPos);

        // Вычисление неизвестной
        if (rank == iterProcRank) {
            iterResult = pProcVector[iterPivotPos] / pProcRows[iterPivotPos * n + i]; pProcResult[iterPivotPos] = iterResult;
        }
        // Рассылка значения вычисленной неизвестной
        MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterProcRank, MPI_COMM_WORLD);

        // Обновление значений вектора
        for (int j = 0; j < RowNum; j++) {
            if (pProcPivotIter[j] < i) {
                val = pProcRows[j * n + i] * iterResult;
                pProcVector[j] = pProcVector[j] - val;
            }
        }
    }
}

void printInitialData(double* pMatrix, double* pVector, double* pProcRows, double* pProcVector, int n, int RowNum) {
    if (rank == 0) {
        printf("Initial Matrix: \n");
        printMatrix(pMatrix, n, n);
        printf("Initial Vector: \n");
        printVector(pVector, n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void gauss(double* pProcRows, double* pProcVector, double* pProcResult, int n, int RowNum) {
    parallelGaussianElimination(pProcRows, pProcVector, n, RowNum);
    ParallelBackSubstitution(pProcRows, pProcVector, pProcResult, n, RowNum);
}

// Освобождение памяти
void processTermination(double* pMatrix, double* pVector, double* pResult, double* pProcRows, double* pProcVector, double* pProcResult) {
    if (rank == 0) {
        delete[] pMatrix;
        delete[] pVector;
        delete[] pResult;
    }

    delete[] pProcRows;
    delete[] pProcVector;
    delete[] pProcResult;

    delete[] pParallelPivotPos;
    delete[] pProcPivotIter;

    delete[] pProcInd;
    delete[] pProcNum;
}

// Проверка результата
void testResult(double* pMatrix, double* pVector, double* pResult, int n) {
    // Буфер для хранения вектора, являющегося результатом умножения матрицы линейной системы на вектор неизвестных
    double* pRightPartVector;

    // Флаг, который показывает, идентичны ли векторы обеих частей или нет
    int isEqual = 0;

    if (rank == 0) {
        pRightPartVector = new double[n];
        for (int i = 0; i < n; i++) {
            pRightPartVector[i] = 0;
            for (int j = 0; j < n; j++) {
                pRightPartVector[i] += pMatrix[i * n + j] * pResult[pParallelPivotPos[j]];
            }
        }

        for (int i = 0; i < n; i++) {
            if (fabs(pRightPartVector[i] - pVector[i]) > accuracy) {
                isEqual = 1;
            }
        }
        if (isEqual == 1)
            printf("The result of the parallel Gauss algorithm is NOT correct." "Check your code.");
        else
            printf("The result of the parallel Gauss algorithm is correct.");
        delete[] pRightPartVector;
    }
}

int main(int argc, char* argv[]) {
    double* pMatrix;
    double* pVector;	// Right parts of the linear system
    double* pResult;
    double* pProcRows;	// Rows of the matrix A
    double* pProcVector;	// Block of the vector b
    double* pProcResult;	// Block of the vector x
    int	n;
    int	rowNum;	// Number of the matrix rows
    double start, finish, duration;

    setvbuf(stdout, 0, _IONBF, 0);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    initProcess(pMatrix, pVector, pResult, pProcRows, pProcVector, pProcResult, n, rowNum);

    start = MPI_Wtime();
    dataDistribution(pMatrix, pProcRows, pVector, pProcVector, n, rowNum);
    gauss(pProcRows, pProcVector, pProcResult, n, rowNum);
    printInitialData(pMatrix, pVector, pProcRows, pProcVector, n, rowNum);
    resultCollection(pProcResult, pResult);
    finish = MPI_Wtime(); duration = finish - start;

    if (rank == 0) {
        printf("\n Result Vector: \n");
        printResultVector(pResult, n);
    }
//    testResult(pMatrix, pVector, pResult, n);

    if (rank == 0) {
        printf("\n Time of execution: %f\n", duration);
    }

    processTermination(pMatrix, pVector, pResult, pProcRows, pProcVector, pProcResult);

    MPI_Finalize();
}
