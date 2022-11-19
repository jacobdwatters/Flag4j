package com.flag4j.concurrency;

import com.flag4j.Matrix;
import com.flag4j.TypedMatrix;
import com.flag4j.concurrency.algorithms.addition.MatrixAdditionWorker;
import com.flag4j.concurrency.algorithms.subtraction.MatrixSubtractionWorker;
import com.flag4j.concurrency.algorithms.transpose.MatrixTransposeWorker;
import com.flag4j.util.ErrorMessages;

import java.util.ArrayList;
import java.util.List;


/**
 * This class contains methods useful for allocating threads to work on matrices concurrently.
 */
public final class ThreadAllocator {

    private ThreadAllocator() {
        // Hide default constructor
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }

    public static BinaryThreadFactory<Matrix> matrixAddThreadFactory = (Matrix dest, Matrix src1, Matrix src2,
                                                                        int rowStart, int rowEnd,
                                                                        int colStart, int colEnd)
            -> new MatrixAdditionWorker(dest, src1, src2, rowStart, rowEnd, colStart, colEnd);

    public static BinaryThreadFactory<Matrix> matrixSubThreadFactory = (Matrix dest, Matrix src1, Matrix src2,
                                                                        int rowStart, int rowEnd,
                                                                        int colStart, int colEnd)
            -> new MatrixSubtractionWorker(dest, src1, src2, rowStart, rowEnd, colStart, colEnd);

    public static UnaryThreadFactory<Matrix> matrixTransposeThreadFactory = (Matrix dest, Matrix src,
                                                                             int rowStart, int rowEnd,
                                                                             int colStart, int colEnd)
            -> new MatrixTransposeWorker(dest, src, rowStart, rowEnd, colStart, colEnd);


    /**
     * Allocate and start threads for binary matrix operation.
     * @param dest Destination matrix for transpose.
     * @param src1 Source matrix for transpose.
     * @param src2 Source matrix for transpose.
     * @return A list of {@link MatrixAdditionWorker} threads which will concurrently transpose the matrix.
     */
    public static List<Thread> allocateThreads(TypedMatrix dest, TypedMatrix src1, TypedMatrix src2,
                                               BinaryThreadFactory factory) {
        List<Thread> threadList = new ArrayList<>();
        int rowEnd, colEnd;
        int[] blockSizes = getBlockSizes(src1.numRows(), src1.numCols());
        int rowBlockSize = blockSizes[0];
        int colBlockSize = blockSizes[1];

        for(int i=0; i<src1.numRows(); i += rowBlockSize) {
            for(int j=0; j<src1.numCols(); j += colBlockSize) {
                rowEnd = Math.min(i+rowBlockSize, src1.numRows());
                colEnd = Math.min(j+colBlockSize, src1.numCols());

                threadList.add(factory.makeThread(dest, src1, src2, i, rowEnd, j, colEnd));
                threadList.get(threadList.size()-1).start(); // Start the thread
            }
        }

        return threadList;
    }


    /**
     * Allocate and start threads for concurrent unary operation.
     * @param dest Destination matrix for operation.
     * @param src Source matrix for operation to be applied to.
     * @return A list of worker threads which will concurrently apply operation to the source matrix.
     */
    public static List<Thread> allocateThreads(TypedMatrix dest, TypedMatrix src,
                                               UnaryThreadFactory factory) {
        List<Thread> threadList = new ArrayList<>();
        int rowEnd, colEnd;
        int[] blockSizes = getBlockSizes(src.numRows(), src.numCols());
        int rowBlockSize = blockSizes[0];
        int colBlockSize = blockSizes[1];

        for(int i=0; i<src.numRows(); i += rowBlockSize) {
            for(int j=0; j<src.numCols(); j += colBlockSize) {
                rowEnd = Math.min(i+rowBlockSize, src.numRows());
                colEnd = Math.min(j+colBlockSize, src.numCols());

                threadList.add(factory.makeThread(dest, src, i, rowEnd, j, colEnd));
                threadList.get(threadList.size()-1).start(); // Start the thread
            }
        }

        return threadList;
    }


    /**
     * Computes block sizes for each thread.
     * @param numRows Number of rows in the matrix.
     * @param numCols Number of columns in the matrix.
     * @return The row and column block sizes.
     */
    private static int[] getBlockSizes(int numRows, int numCols) {
        int numThreads = Configurations.getNumThreads();
        int rowBlockSize;
        int colBlockSize;

        if(numRows < numThreads && numCols < numThreads) {
            rowBlockSize = 1;
            colBlockSize = 1;

        } else if(numRows >= numThreads && numCols >= numThreads) {
            rowBlockSize = (int) Math.floor( ((double) numRows) / Math.sqrt(numThreads) );
            colBlockSize = (int) Math.floor( ((double) numCols) / Math.sqrt(numThreads) );

        } else if(numRows >= numCols) {
            rowBlockSize = Math.max(1, 4*numRows/numThreads);
            colBlockSize = Math.max(1, numCols/4);
        } else {
            colBlockSize = Math.max(1, 4*numCols/numThreads);
            rowBlockSize = Math.max(1, numRows/4);
        }

        return new int[]{rowBlockSize, colBlockSize};
    }
}
