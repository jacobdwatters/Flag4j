package com.flag4j.concurrency;

import com.flag4j.TypedMatrix;


/**
 * A Thread factory for creating threads for binary, element-wise operations.
 * @param <M> The type of matrix.
 */
public interface BinaryThreadFactory<M extends TypedMatrix<?>> {


    /**
     * Creates a thread for performing a binary matrix operation concurrently.
     * @param dest Destination matrix for binary operation.
     * @param src1 First matrix in the operation.
     * @param src2 Second matrix in the operation.
     * @param rowStart Staring row for thread to work on.
     * @param rowEnd Ending row for thread to work on.
     * @param colStart Starting column for thread to work on.
     * @param colEnd Ending column for thread to work on.
     * @return The created thread.
     */
    Thread makeThread(M dest, M src1, M src2, int rowStart, int rowEnd, int colStart, int colEnd);
}
