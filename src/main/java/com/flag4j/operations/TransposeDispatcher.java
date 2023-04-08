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

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.dense.complex.ComplexDenseTranspose;
import com.flag4j.operations.dense.real.RealDenseTranspose;
import com.flag4j.util.ErrorMessages;


/**
 * Provides a dispatch method for dynamically choosing the best matrix transpose algorithm.
 */
public final class TransposeDispatcher {

    /**
     * Threshold for using complex blocked algorithm.
     */
    private static final int COMPLEX_BLOCKED_THRESHOLD = 5_000;
    /**
     * Threshold for using blocked hermation algorithm
     */
    private static final int HERMATION_BLOCKED_THRESHOLD = 50_000;
    /**
     * Threshold for using standard transpose implementation.
     */
    private static final int STANDARD_THRESHOLD = 1500;
    /**
     * Threshold for number of elements in matrix to use concurrent implementation.
     */
    private static final int CONCURRENT_THRESHOLD = 125_000;


    private TransposeDispatcher() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


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
            default:
                dest = RealDenseTranspose.blockedMatrixConcurrent(src.entries, src.numRows, src.numCols);
                break;
        }

        return new Matrix(src.numCols, src.numRows, dest);
    }


    /**
     * Dispatches a matrix transpose problem to the appropriate algorithm based in its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    public static CMatrix dispatch(CMatrix src) {

        CNumber[] dest;

        Algorithm algorithm = chooseAlgorithm(src.shape);

        if(algorithm==Algorithm.BLOCKED) {
            dest = ComplexDenseTranspose.blockedMatrix(src.entries, src.numRows, src.numCols);
        } else {
            dest = ComplexDenseTranspose.blockedMatrixConcurrent(src.entries, src.numRows, src.numCols);
        }

        return new CMatrix(src.numCols, src.numRows, dest);
    }


    /**
     * Dispatches a matrix hermation transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Matrix to transpose.
     * @return The transpose of the source matrix.
     */
    public static CMatrix dispatchHermation(CMatrix src) {
        CNumber[] dest;

        Algorithm algorithm = chooseAlgorithmHermation(src.shape);

        if(algorithm==Algorithm.BLOCKED) {
            dest = ComplexDenseTranspose.blockedMatrixHerm(src.entries, src.numRows, src.numCols);
        } else {
            dest = ComplexDenseTranspose.blockedMatrixConcurrentHerm(src.entries, src.numRows, src.numCols);
        }

        return new CMatrix(src.numCols, src.numRows, dest);
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Tensor to transpose.
     * @param axis1 First axis in tensor transpose.
     * @param axis2 Second axis in tensor transpose.
     * @return
     */
    public static Tensor dispatchTensor(Tensor src, int axis1, int axis2) {
        double[] dest;
        Algorithm algorithm = chooseAlgorithmTensor(src.shape.get(axis1), src.shape.get(axis2));

        dest = algorithm == Algorithm.STANDARD ?
                RealDenseTranspose.standard(src.entries, src.shape, axis1, axis2):
                RealDenseTranspose.standardConcurrent(src.entries, src.shape, axis1, axis2);

        return new Tensor(src.shape.copy().swapAxes(axis1, axis2), dest);
    }


    /**
     * Dispatches a tensor transpose problem to the appropriate algorithm based on its shape and size.
     * @param src Tensor to transpose.
     * @param axis1 First axis in tensor transpose.
     * @param axis2 Second axis in tensor transpose.
     * @return
     */
    public static CTensor dispatchTensor(CTensor src, int axis1, int axis2) {
        CNumber[] dest;
        Algorithm algorithm = chooseAlgorithmTensor(src.shape.get(axis1), src.shape.get(axis2));

        dest = algorithm == Algorithm.STANDARD ?
                ComplexDenseTranspose.standard(src.entries, src.shape, axis1, axis2):
                ComplexDenseTranspose.standardConcurrent(src.entries, src.shape, axis1, axis2);

        return new CTensor(src.shape.copy().swapAxes(axis1, axis2), dest);
    }


    /**
     * Chooses the appropriate algorithm for computing a tensor transpose.
     * @param length1 Length of first axis in tensor transpose.
     * @param length2 Length of second axis in tensor transpose.
     * @return
     */
    private static Algorithm chooseAlgorithmTensor(int length1, int length2) {
        Algorithm algorithm;

        int numEntries = length1*length2; // Number of entries involved in transpose.

        if(numEntries < CONCURRENT_THRESHOLD) {
            // Use standard algorithm.
            algorithm = Algorithm.STANDARD;
        } else {
            // Use concurrent standard algorithm
            algorithm = Algorithm.CONCURRENT_STANDARD;
        }

        return algorithm;
    }



    /**
     * Chooses the appropriate matrix transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static Algorithm chooseAlgorithm(Shape shape) {
        Algorithm algorithm;

        int numEntries = shape.totalEntries().intValue();

        if(numEntries < COMPLEX_BLOCKED_THRESHOLD) {
            // Use blocked algorithm
            algorithm = Algorithm.BLOCKED;
        } else {
            // Use concurrent blocked implementation.
            algorithm = Algorithm.CONCURRENT_BLOCKED;
        }

        return algorithm;
    }


    /**
     * Chooses the appropriate matrix hermation transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static Algorithm chooseAlgorithmHermation(Shape shape) {
        Algorithm algorithm;

        int numEntries = shape.totalEntries().intValue();

        if(numEntries < HERMATION_BLOCKED_THRESHOLD) {
            // Use blocked algorithm
            algorithm = Algorithm.BLOCKED;
        } else {
            // Use concurrent blocked implementation.
            algorithm = Algorithm.CONCURRENT_BLOCKED;
        }

        return algorithm;
    }


    /**
     * Chooses the appropriate matrix transpose algorithm based on the shape of a matrix.
     * @param shape Shape of matrix to transpose.
     * @return The appropriate matrix transpose algorithm.
     */
    private static Algorithm chooseAlgorithmComplex(Shape shape) {
        Algorithm algorithm;

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
        /**
         * Standard transpose algorithm
         */
        STANDARD,
        /**
         * Blocked transpose algorithm
         */
        BLOCKED,
        /**
         * A concurrent implementation of the standard algorithm
         */
        CONCURRENT_STANDARD,
        /**
         * A concurrent implementation of the blocked algorithm
         */
        CONCURRENT_BLOCKED
    }
}
