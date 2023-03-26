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
import com.flag4j.operations.dense.real.RealDenseMatrixMultiplication;
import com.flag4j.util.Axis2D;

import java.util.HashMap;
import java.util.Map;

import static com.flag4j.util.ParameterChecks.assertMatMultShapes;


/**
 * Singleton class which stores a map of all viable real dense matrix multiply algorithms and uses that map to dispatch
 * a real dense matrix multiply problem to the appropriate algorithm.
 */
public final class RealDenseMatrixMultiplyDispatcher {

    /**
     * Singleton instance of this class.
     */
    private static RealDenseMatrixMultiplyDispatcher singletonInstance = null;
    /**
     * lookup table for all real dense matrix multiply algorithms.
     */
    private final Map<AlgorithmNames, RealDenseTensorBinaryOperation> algorithmMap;
    /**
     * Array of all real dense matrix multiply algorithms.
     */
    private final RealDenseTensorBinaryOperation[] algorithms = {
            RealDenseMatrixMultiplication::standard,
            RealDenseMatrixMultiplication::reordered,
            RealDenseMatrixMultiplication::blocked,
            RealDenseMatrixMultiplication::blockedReordered,
            RealDenseMatrixMultiplication::concurrentStandard,
            RealDenseMatrixMultiplication::concurrentReordered,
            RealDenseMatrixMultiplication::concurrentBlocked,
            RealDenseMatrixMultiplication::concurrentBlockedReordered,
            RealDenseMatrixMultiplication::standardVector,
            RealDenseMatrixMultiplication::blockedVector,
            RealDenseMatrixMultiplication::concurrentStandardVector,
            RealDenseMatrixMultiplication::concurrentBlockedVector
    };

    /**
     * Ration measuring squareness. the closer to one, the more square the matrix is.
     */
    private static final double SQUARENESS_RATIO = 0.75;
    /**
     * Threshold for small matrices which should be multiplied using the standard ikj algorithm.
     */
    private static final int SEQUENTIAL_SWAPPED_THRESHOLD = 40;
    /**
     * Threshold for matrices to use the concurrent ikj algorithm.
     */
    private static final int CONCURRENT_SWAPPED_THRESHOLD = 3072;


    /**
     * Creates an instance containing a map of all viable real dense matrix multiply algorithms.
     */
    private RealDenseMatrixMultiplyDispatcher() {
        AlgorithmNames[] names = AlgorithmNames.values();
        algorithmMap = new HashMap<>();

        for(int i=0; i<algorithms.length; i++) {
            algorithmMap.put(names[i], algorithms[i]);
        }
    }


    /**
     * Gets the singleton instance of this class. If this class has not been instanced, a new instance will be created.
     * @return The singleton instance of this class.
     */
    public static synchronized RealDenseMatrixMultiplyDispatcher getInstance()  {
        return (singletonInstance == null) ? new RealDenseMatrixMultiplyDispatcher() : singletonInstance;
    }


    /**
     * Dispatches a matrix multiply problem to the appropriate algorithm based on the size of the matrices.
     * @return The result of the matrix multiplication.
     */
    public static double[] dispatch(Matrix A, Matrix B) {
        assertMatMultShapes(A.shape, B.shape); // Ensure matrix shapes are conducive to matrix multiplication.

        RealDenseMatrixMultiplyDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithm(A.shape, B.shape);
        return dispatcher.algorithmMap.get(name).apply(A.entries, A.shape, B.entries, B.shape);
    }


    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    private static AlgorithmNames selectAlgorithm(Shape shape1, Shape shape2) {
        AlgorithmNames names;

        int rows1 = shape1.get(Axis2D.row());
        int cols1 = shape1.get(Axis2D.col());

        // Determine the matrix shape.
        int matrixShape;
        if (getRatio(shape1) >= SQUARENESS_RATIO) {
            matrixShape = 0; // Approximately Square
        } else if (rows1 > cols1) {
            matrixShape = 1; // More rows than columns
        } else {
            matrixShape = 2; // More columns than rows
        }

        // Determine the algorithm name based on the matrix shape and size.
        switch (matrixShape) {
            case 0: // Square
                if(rows1 < SEQUENTIAL_SWAPPED_THRESHOLD) {
                    names = AlgorithmNames.REORDERED;
                } else if (rows1 < CONCURRENT_SWAPPED_THRESHOLD) {
                    names = AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    names = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
                }
                break;

            case 1: // More rows than columns
                if(rows1 <= 100 && cols1 <= 5) {
                    names = AlgorithmNames.REORDERED;
                } else {
                    names = AlgorithmNames.CONCURRENT_REORDERED;
                }
                break;

            case 2: // More columns than rows
                if(cols1 <= 100) {
                    names = (rows1 <= 20) ? AlgorithmNames.REORDERED : AlgorithmNames.CONCURRENT_REORDERED;
                } else if(cols1 <= 500) {
                    names = (rows1 <= 10) ? AlgorithmNames.REORDERED : AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    if(rows1 <= 5) {
                        names = AlgorithmNames.REORDERED;
                    } else if(rows1 <= 50) {
                        names = AlgorithmNames.CONCURRENT_STANDARD;
                    } else {
                        names = AlgorithmNames.CONCURRENT_REORDERED;
                    }
                }
                break;
            default:
                names = AlgorithmNames.BLOCKED_REORDERED; // Default to blocked reordered algorithm.
        }

        return names;
    }


    /**
     * Computes the squareness ratio of a matrix. This is a value between 0 and 1 with 1 being perfectly
     * square and 0 being a row/column vector.
     * @param shape Shape of the matrix to compute the squareness ratio of.
     * @return The squareness ratio for the specified shape.
     */
    private static double getRatio(Shape shape) {
        int numRows = shape.get(Axis2D.row());
        int numCols = shape.get(Axis2D.col());

        double ratio = Math.abs(numRows-numCols);
        return 1-ratio/Math.max(numRows, numCols);
    }


    /**
     * Simple enum class containing all possible choices of real dense matrix multiply algorithms.
     */
    private enum AlgorithmNames {
        STANDARD, REORDERED, BLOCKED, BLOCKED_REORDERED,
        CONCURRENT_STANDARD, CONCURRENT_REORDERED, CONCURRENT_BLOCKED, CONCURRENT_BLOCKED_REORDERED,
        STANDARD_VECTOR, BLOCKED_VECTOR, CONCURRENT_STANDARD_VECTOR, CONCURRENT_BLOCKED_VECTOR
    }
}