package com.flag4j.operations.concurrency.algorithms.transpose;

import com.flag4j.Matrix;
import com.flag4j.operations.concurrency.DenseThreadAllocator;
import com.flag4j.operations.concurrency.UnaryThreadFactory;
import com.flag4j.util.ErrorMessages;

import java.util.List;


/**
 * A class for computing the matrix transpose concurrently.
 */
public final class ConcurrentTranspose {

    private ConcurrentTranspose() {
        // Hide default constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the transpose of matrix A using multiple threads.
     * @param A The matrix to compute the transpose of.
     * @return The transpose of this matrix.
     */
    public static Matrix T(Matrix A) {
        Matrix T = new Matrix(A.numCols(), A.numRows());
        UnaryThreadFactory<double[]> factory = DenseThreadAllocator.matrixTransposeThreadFactory;
        List<Thread> threadList;

        // Allocates the threads.
        threadList = DenseThreadAllocator.allocateThreads(T, A, factory);

        for(Thread thread : threadList) { // Join the threads together
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return T;
    }
}
