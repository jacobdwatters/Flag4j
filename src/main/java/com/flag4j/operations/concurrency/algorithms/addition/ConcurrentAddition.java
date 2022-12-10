package com.flag4j.operations.concurrency.algorithms.addition;

import com.flag4j.Matrix;
import com.flag4j.operations.concurrency.BinaryThreadFactory;
import com.flag4j.operations.concurrency.DenseThreadAllocator;
import com.flag4j.util.ErrorMessages;

import java.util.List;


/**
 * A class for computing matrix additions concurrently.
 */
public class ConcurrentAddition {

    private ConcurrentAddition() {
        // Hide default constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the transpose of matrix A using multiple threads.
     * @param A The matrix to compute the transpose of.
     * @return The transpose of this matrix.
     */
    public static Matrix add(Matrix A, Matrix B) {
        Matrix sum = new Matrix(A.numRows(), A.numCols());
        BinaryThreadFactory<double[]> factory = DenseThreadAllocator.matrixAddThreadFactory;
        List<Thread> threadList = DenseThreadAllocator.allocateThreads(sum, A, B, factory);

        for(Thread thread : threadList) { // Join the threads together
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return sum;
    }
}
