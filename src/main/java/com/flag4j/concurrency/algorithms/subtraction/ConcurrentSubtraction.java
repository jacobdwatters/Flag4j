package com.flag4j.concurrency.algorithms.subtraction;

import com.flag4j.Matrix;
import com.flag4j.concurrency.BinaryThreadFactory;
import com.flag4j.concurrency.ThreadAllocator;
import com.flag4j.util.ErrorMessages;

import java.util.List;

public class ConcurrentSubtraction {


    private ConcurrentSubtraction() {
        // Hide default constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the transpose of matrix A using multiple threads.
     * @param A The matrix to compute the transpose of.
     * @return The transpose of this matrix.
     */
    public static Matrix add(Matrix A, Matrix B) {
        Matrix difference = new Matrix(A.numRows(), A.numCols());
        BinaryThreadFactory factory = ThreadAllocator.matrixSubThreadFactory;
        List<Thread> threadList = ThreadAllocator.allocateThreads(difference, A, B, factory);

        for(Thread thread : threadList) { // Join the threads together
            try {
                thread.join();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }

        return difference;
    }
}
