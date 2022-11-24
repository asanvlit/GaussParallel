#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <mpi.h>

int size, rank;

const double accuracy = 1.e-6;

// хранит номера строк матрицы, выбираемых в качестве ведущих, по итерациям прямого хода
// метода Гаусса – определяет далее порядок выполнения итераций для обратного хода
int* pParallelPivotPos;

// хранит номера итераций, на которых определенная строка использовалась в качестве ведущей (если -1, то должна обрабатываться при обратном ходе)
int* pPivotIter;

int* pRowStartIndex; // Номер строки, с которой начинается блок для данного процесса
int* pRowNumber;     // Количество строк для данного процесса

void printMatrix(double* m, int rowCount, int colCount) {
    for (int i = 0; i < rowCount; i++) {
        for (int j = 0; j < colCount; j++) {
            printf("%7.4f ", m[i * colCount + j]);
        }
        printf("\n");
    }
}

void printVector(double* v, int n) {
    for (int i = 0; i < n; i++) {
        printf("%7.4f ", v[i]);
    }
}

void printResultVector(double* resultV, int n) {
    for (int i = 0; i < n; i++) {
        printf("%7.4f ", resultV[pParallelPivotPos[i]]);
    }
}

void fillUpMatrixWithRandom(double* pMatrix, double* pVector, int n) {
    srand(unsigned(clock()));

    for (int i = 0; i < n; i++) {
        pVector[i] = rand() % 20;
        for (int j = 0; j < n; j++) {
            pMatrix[i * n + j] = rand() % 20;
        }
    }
}

void initProcess(double*& a, double*& v, double*& result, double*& pA, double*& pV, double*& pResult, int& n, int& pBlockSize) {
    if (rank == 0) {
        n = -1;
        while (n < size) {
            printf_s("\nEnter the size of the matrix: ");
            scanf_s("%d", &n);

            if (n < size) {
                printf("Invalid matrix size: n must be greater than the number of processes \n");
            }
        }

        a = new double[n * n];
        v = new double[n];
        result = new double[n];
        fillUpMatrixWithRandom(a, v, n);
    }
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int remRowsNumb = n; // количество оставшихся строк
    for (int i = 0; i < rank; i++) {
        remRowsNumb = remRowsNumb - remRowsNumb / (size - i);
    }
    pBlockSize = remRowsNumb / (size - rank); // количество строк для обработки для данного процесса

    pA = new double[pBlockSize * n];
    pV = new double[pBlockSize];
    pResult = new double[pBlockSize];

    pParallelPivotPos = new int[n];
    pPivotIter = new int[pBlockSize];

    pRowStartIndex = new int[size];
    pRowNumber = new int[size];

    for (int i = 0; i < pBlockSize; i++) {
        pPivotIter[i] = -1; // пока никакие строки в качестве ведущих не использовались
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
    pRowStartIndex[0] = 0;
    pRowNumber[0] = n / size;
    for (int i = 1; i < size; i++) {
        remRowsNumb -= pRowNumber[i - 1];
        pRowNumber[i] = remRowsNumb / (size - i);
        pRowStartIndex[i] = pRowStartIndex[i - 1] + pRowNumber[i - 1];
    }

    MPI_Scatterv(pVector, pRowNumber, pRowStartIndex, MPI_DOUBLE, pProcVector, pRowNumber[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] pSendNum;
    delete[] pSendInd;
}

void buildResult(double* pProcResult, double* pResult) {
    MPI_Gatherv(pProcResult, pRowNumber[rank], MPI_DOUBLE, pResult, pRowNumber,
                pRowStartIndex, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void parallelEliminateColumns(double* pProcRows, double* pProcVector, double* pPivotRow, int n, int pBlockSize, int iter) {
    double multiplier;

    // вычитание ведущей строки из строк процесса, которые еще не использовались в качестве ведущих
    for (int i = 0; i < pBlockSize; i++) {
        if (pPivotIter[i] == -1) {
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
            if ((pPivotIter[j] == -1) && (MaxValue < fabs(pProcRows[j * n + i]))) {
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
            pPivotIter[PivotPos] = i; // номер итерации
            pParallelPivotPos[i] = pRowStartIndex[rank] + PivotPos;
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
void findBackPivotRow(int RowIndex, int n, int& IterProcRank, int& IterPivotPos) {
    for (int i = 0; i < size - 1; i++) {
        if ((pRowStartIndex[i] <= RowIndex) && (RowIndex < pRowStartIndex[i + 1])) IterProcRank = i;
    }
    if (RowIndex >= pRowStartIndex[size - 1]) IterProcRank = size - 1;
    IterPivotPos = RowIndex - pRowStartIndex[IterProcRank];
}

// Обратный ход:
void parallelBackSubstitution(double* pProcRows, double* pProcVector, double* pProcResult, int n, int RowNum) {
    int iterProcRank;	// Номер процесса с данной ведущей строкой
    int iterPivotPos;	// Позиция ведущей строки в процессе
    double iterResult;	// Вычисленное значение неизвестной
    double val;

    // Итерации
    for (int i = n - 1; i >= 0; i--) {
        // Определение ранга процесса, который содержит ведущую строку
        findBackPivotRow(pParallelPivotPos[i], n, iterProcRank, iterPivotPos);

        // Вычисление неизвестной
        if (rank == iterProcRank) {
            iterResult = pProcVector[iterPivotPos] / pProcRows[iterPivotPos * n + i]; pProcResult[iterPivotPos] = iterResult;
        }
        // Рассылка значения вычисленной неизвестной
        MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterProcRank, MPI_COMM_WORLD);

        // Обновление значений вектора
        for (int j = 0; j < RowNum; j++) {
            if (pPivotIter[j] < i) {
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

void printResultData(double* result, int n) {
    if (rank == 0) {
        printf("\n Result Vector: \n");
        printResultVector(result, n);
    }
}

void gauss(double* pProcRows, double* pProcVector, double* pProcResult, int n, int RowNum) {
    parallelGaussianElimination(pProcRows, pProcVector, n, RowNum);
    parallelBackSubstitution(pProcRows, pProcVector, pProcResult, n, RowNum);
}

// Проверка результата
void checkResult(double* pMatrix, double* pVector, double* pResult, int n) {
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

// Освобождение памяти
void freeMemory(double* pMatrix, double* pVector, double* pResult, double* pProcRows, double* pProcVector, double* pProcResult) {
    if (rank == 0) {
        delete[] pMatrix;
        delete[] pVector;
        delete[] pResult;
    }

    delete[] pProcRows;
    delete[] pProcVector;
    delete[] pProcResult;

    delete[] pParallelPivotPos;
    delete[] pPivotIter;

    delete[] pRowStartIndex;
    delete[] pRowNumber;
}

int main(int argc, char* argv[]) {
    double* a;            // Матрица
    double* v;            // Вектор
    double* result;       // Вычисленные неизвестные
    double* pA;           // Строки матрицы для данного процесса
    double* pV;	          // Кусок вектора для данного процесса
    double* pResult;  // Block of the v x
    int	n;
    int	rowNum;	// Number of the matrix rows
    double start, finish, duration;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    setvbuf(stdout, 0, _IONBF, 0);
    initProcess(a, v, result, pA, pV, pResult, n, rowNum);

    start = MPI_Wtime();
    dataDistribution(a, pA, v, pV, n, rowNum);
    gauss(pA, pV, pResult, n, rowNum);
    printInitialData(a, v, pA, pV, n, rowNum);
    buildResult(pResult, result);
    finish = MPI_Wtime();

    duration = finish - start;

    printResultData(result, n);
    checkResult(a, v, result, n);

    if (rank == 0) {
        printf("\n Time of execution: %f\n", duration);
    }

    freeMemory(a, v, result, pA, pV, pResult);

    MPI_Finalize();
}
