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

void printSystem(double* m, double* v, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%7.4f*x%i", m[i * n + j], j);
            if (j < n - 1) {
                printf(" + ");
            }
        }
        printf(" = %f\n", v[i]);
    }
}

void printResultVector(double* resultV, int n) {
    for (int i = 0; i < n; i++) {
        printf("%7.4f ", resultV[pParallelPivotPos[i]]);
    }
}

void fillUpSystemWithRandom(double* pMatrix, double* pVector, int n) {
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
        fillUpSystemWithRandom(a, v, n);
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

    for (int i = 0; i < pBlockSize; i++) {
        pPivotIter[i] = -1; // пока никакие строки в качестве ведущих не использовались
    }

    pRowStartIndex = new int[size];
    pRowNumber = new int[size];
}

// Распределение данных между процессами
void distributeSystem(double* a, double* pRows, double* v, double* pVector, int n, int blockSize) {
    int* pSendNum;	     // Количество элементов, отправленных на процесс
    int* pSendInd;	     // Номер первого элемента, отправленного на процесс
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
    MPI_Scatterv(a, pSendNum, pSendInd, MPI_DOUBLE, pRows,
                 pSendNum[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Определение позиций строк матрицы для текущего процесса:
    remRowsNumb = n;
    pRowStartIndex[0] = 0;
    pRowNumber[0] = n / size;
    for (int i = 1; i < size; i++) {
        remRowsNumb -= pRowNumber[i - 1];
        pRowNumber[i] = remRowsNumb / (size - i);
        pRowStartIndex[i] = pRowStartIndex[i - 1] + pRowNumber[i - 1];
    }

    MPI_Scatterv(v, pRowNumber, pRowStartIndex, MPI_DOUBLE, pVector,
                 pRowNumber[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

    delete[] pSendNum;
    delete[] pSendInd;
}

void buildResult(double* pProcResult, double* pResult) {
    MPI_Gatherv(pProcResult, pRowNumber[rank], MPI_DOUBLE, pResult, pRowNumber,
                pRowStartIndex, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

// Прямой ход
void forwardGaussStep(double* pRows, double* pVector, int n, int pBlockSize) {
    int	pivotPos; // Позиция ведущего элемента

    // Вспомогательная структура для выбора ведущей строки
    struct { double MaxValue; int ProcRank; } ProcPivot, Pivot;

    // Ведущая строка и соответсвующий элемент вектора
    double* pPivotRow = new double[n + 1];

    // Итерации
    for (int i = 0; i < n; i++) {
        // Поиск ведущей строки
        double maxValue = 0; // Значение ведущего элемента
        for (int j = 0; j < pBlockSize; j++) {
            if ((pPivotIter[j] == -1) && (maxValue < fabs(pRows[j * n + i]))) {
                maxValue = fabs(pRows[j * n + i]);
                pivotPos = j;
            }
        }
        ProcPivot.MaxValue = maxValue;
        ProcPivot.ProcRank = rank;

        // Поиск "ведущего" процесса - процесса с максимальным значением maxValue
        // Функция MPI_ALLREDUCE отличается от MPI_REDUCE тем, что результат появляется в буфере приема у всех членов группы.
        MPI_Allreduce(&ProcPivot, &Pivot, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);

        // Рассылка ведущей строки
        if (rank == Pivot.ProcRank) {
            pPivotIter[pivotPos] = i; // номер итерации
            pParallelPivotPos[i] = pRowStartIndex[rank] + pivotPos;
        }
        MPI_Bcast(&pParallelPivotPos[i], 1, MPI_INT, Pivot.ProcRank,MPI_COMM_WORLD);

        if (rank == Pivot.ProcRank) {
            // Заполнение ведущей строки
            for (int j = 0; j < n; j++) {
                pPivotRow[j] = pRows[pivotPos * n + j];
            }
            pPivotRow[n] = pVector[pivotPos];
        }
        MPI_Bcast(pPivotRow, n + 1, MPI_DOUBLE, Pivot.ProcRank, MPI_COMM_WORLD);

        double multiplier;
        // вычитание ведущей строки из строк процесса, которые еще не использовались в качестве ведущих
        for (int x = 0; x < pBlockSize; x++) {
            if (pPivotIter[x] == -1) {
                multiplier = pRows[x * n + i] / pPivotRow[i];

                for (int j = i; j < n; j++) {
                    pRows[x * n + j] -= pPivotRow[j] * multiplier;
                }

                pVector[x] -= pPivotRow[n] * multiplier;
            }
        }
    }
}

// Функция для нахождения ведущей строки обратной подстановки
void findBackPivotRow(int rowIndex, int& iterProcRank, int& iterPivotPos) {
    for (int i = 0; i < size - 1; i++) {
        if ((pRowStartIndex[i] <= rowIndex) && (rowIndex < pRowStartIndex[i + 1])) {
            iterProcRank = i;
        }
    }
    if (rowIndex >= pRowStartIndex[size - 1]) {
        iterProcRank = size - 1;
    }

    iterPivotPos = rowIndex - pRowStartIndex[iterProcRank];
}

// Обратный ход:
void backwardGaussStep(double* pRows, double* pVector, double* pResult, int n, int pBlockSize) {
    int iterProcRank;	// Номер процесса с данной ведущей строкой
    int iterPivotPos;	// Позиция ведущей строки в процессе
    double iterResult;	// Вычисленное значение неизвестной
    double value;

    // Итерации
    for (int i = n - 1; i >= 0; i--) {
        // Определение ранга процесса, который содержит ведущую строку
        findBackPivotRow(pParallelPivotPos[i], iterProcRank, iterPivotPos);

        // Вычисление неизвестной
        if (rank == iterProcRank) {
            iterResult = pVector[iterPivotPos] / pRows[iterPivotPos * n + i];
            pResult[iterPivotPos] = iterResult;
        }
        // Рассылка значения вычисленной неизвестной
        MPI_Bcast(&iterResult, 1, MPI_DOUBLE, iterProcRank, MPI_COMM_WORLD);

        // Обновление значений вектора
        for (int j = 0; j < pBlockSize; j++) {
            if (pPivotIter[j] < i) {
                value = pRows[j * n + i] * iterResult;
                pVector[j] = pVector[j] - value;
            }
        }
    }
}

void printInitialData(double* pMatrix, double* pVector, double* pProcRows, double* pProcVector, int n, int pBlockSize) {
    if (rank == 0) {
        printf("Initial data: \n");
        printSystem(pMatrix, pVector, n);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

void printResultData(double* result, int n) {
    if (rank == 0) {
        printf("\nResult Vector: \n");
        printResultVector(result, n);
    }
}

void gauss(double* pRows, double* pVector, double* pResult, int n, int rowNum) {
    forwardGaussStep(pRows, pVector, n, rowNum);
    backwardGaussStep(pRows, pVector, pResult, n, rowNum);
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
        if (isEqual == 1) {
            printf("\nThe result of the Gauss algorithm is NOT correct");
        }
        else {
            printf("\nThe result of the Gauss algorithm is correct");
        }
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
    int	n;
    int	pBlockSize;	  // Количество строк матрицы в блоке

    double* a;        // Матрица
    double* v;        // Вектор
    double* result;   // Вычисленные неизвестные
    double* pA;       // Строки матрицы для данного процесса
    double* pV;	      // Кусок вектора для данного процесса
    double* pResult;  // Block of the v x

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    setvbuf(stdout, 0, _IONBF, 0);
    initProcess(a, v, result, pA, pV, pResult, n, pBlockSize);

    double start = MPI_Wtime();
    distributeSystem(a, pA, v, pV, n, pBlockSize);
    gauss(pA, pV, pResult, n, pBlockSize);
    printInitialData(a, v, pA, pV, n, pBlockSize);
    buildResult(pResult, result);
    double finish = MPI_Wtime();

    double duration = finish - start;
    if (rank == 0) {
        printf("\nTime spent: %f\n", duration);
    }

    printResultData(result, n);
    checkResult(a, v, result, n);

    freeMemory(a, v, result, pA, pV, pResult);

    MPI_Finalize();
}

