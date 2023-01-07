/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

package com.flag4j.operations;

import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.operations.dense.real.RealDenseTranspose;
import com.flag4j.util.ErrorMessages;


/**
 * Provides a dispatch method for dynamically choosing the best matrix transpose algorithm.
 */
public final class MatrixTranspose {

    private MatrixTranspose() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Threshold for using standard transpose implementation.
     */
    private static final int STANDARD_THRESHOLD = 1500;
    /**
     * Threshold for number of elements in matrix to use concurrent implementation.
     */
    private static final int CONCURRENT_THRESHOLD = 125_000;


    /**
     * Dispatches a matrix transpose problem to the appropriate algorithm based in its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    public static Matrix dispatch(Matrix src) {

        double[] dest;

        Algorithm algorithm = chooseAlgorithm(src.shape);

        switch(algorithm) {
            case STANDARD:
                dest = RealDenseTranspose.standardMatrix(src.entries, src.numRows, src.numCols);
                break;
            case BLOCKED:
                dest = RealDenseTranspose.blockedMatrix(src.entries, src.numRows, src.numCols);
                break;
            case CONCURRENT_STANDARD:
                dest = RealDenseTranspose.standardMatrixConcurrent(src.entries, src.numRows, src.numCols);
                break;
            case CONCURRENT_BLOCKED:
                dest = RealDenseTranspose.blockedMatrixConcurrent(src.entries, src.numRows, src.numCols);
                break;
            default:
                dest = RealDenseTranspose.blockedMatrix(src.entries, src.numRows, src.numCols);
                break;
        }

        return new Matrix(src.numCols, src.numRows, dest);
    }


    /**
     * Chooses the appropriate matrix transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static Algorithm chooseAlgorithm(Shape shape) {
        Algorithm algorithm = Algorithm.BLOCKED;

        int numEntries = shape.totalEntries().intValue();

        if(numEntries < STANDARD_THRESHOLD) {
            // Use standard algorithm.
            algorithm = Algorithm.STANDARD;
        } else if(numEntries < CONCURRENT_THRESHOLD) {
            // Use blocked algorithm
            algorithm = Algorithm.BLOCKED;
        } else {
            // Use concurrent blocked implementation.
            algorithm = Algorithm.CONCURRENT_BLOCKED;
        }

        return algorithm;
    }


    /**
     * Simple enum class containing available algorithms for computing a matrix transpose.
     */
    private enum Algorithm {
        STANDARD, BLOCKED, CONCURRENT_STANDARD, CONCURRENT_BLOCKED
    }
}
