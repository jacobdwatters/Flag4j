/*
 * MIT License
 *
 * Copyright (c) 2022 Jacob Watters
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
import com.flag4j.operations.dense.real.RealMatrixMultiplication;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ParameterChecks;


/**
 * Dispatches matrix multiplication to the appropriate algorithm based on the size of the matrices to be multiplied.
 */
public class MatrixMultiply {

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
     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication.
     * @return The result of the matrix multiplication.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication.
     */
    public static double[] dispatch(Matrix A, Matrix B) {
        ParameterChecks.assertMatMultShapes(A.shape, B.shape);

        Algorithm algorithm;
        double[] dest;

        if(B.numCols==1) {
            algorithm = chooseAlgorithmVector(A.shape);
        } else {
            algorithm = chooseAlgorithm(A.shape, B.shape);
        }



        switch(algorithm) {
            case STANDARD:
                dest = RealMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
                break;
            case REORDERED:
                dest = RealMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED:
                dest = RealMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED_REORDERED:
                dest = RealMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_STANDARD:
                dest = RealMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_REORDERED:
                dest = RealMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_BLOCKED:
                dest = RealMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                dest = RealMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            default:
                // Default to the concurrent reordered implementation just in case.
                dest = RealMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
        }

        return dest;
    }


    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    private static Algorithm chooseAlgorithm(Shape shape1, Shape shape2) {
        Algorithm algorithm;

        int rows1 = shape1.get(Axis2D.row());
        int cols1 = shape1.get(Axis2D.col());
        int cols2 = shape2.get(Axis2D.col());

        // TODO: Extract constants to final variables
        if(getRatio(shape1) >= SQUARENESS_RATIO) {
            // Then the first matrix is approximately square.
            if(cols2==1) {
                // Multiplying by a column vector.
                if(rows1<=100) {
                    algorithm = Algorithm.STANDARD;
                } else if(rows1<=300) {
                    algorithm = Algorithm.BLOCKED;
                } else if(rows1<=1024) {
                    algorithm = Algorithm.CONCURRENT_BLOCKED;
                } else {
                    algorithm = Algorithm.CONCURRENT_STANDARD;
                }

            } else {
                if(rows1<SEQUENTIAL_SWAPPED_THRESHOLD) {
                    algorithm = Algorithm.REORDERED;
                } else if(rows1<CONCURRENT_SWAPPED_THRESHOLD) {
                    algorithm = Algorithm.CONCURRENT_REORDERED;
                } else {
                /* For large matrices, use a concurrent, blocked algorithm with the j-k loops swapped for
                better cache performance on modern systems */
                    algorithm = Algorithm.CONCURRENT_BLOCKED_REORDERED;
                }
            }

        } else if(rows1>cols1) {
            // Then there are more rows than columns in the first matrix
            if(rows1<=100 && cols1<=5) {
                algorithm = Algorithm.REORDERED;
            } else {
                algorithm = Algorithm.CONCURRENT_REORDERED;
            }
        } else {
            // Then there are more columns than rows in the first matrix
            if(cols1<=100) {
                if(rows1<=20) {
                    algorithm = Algorithm.REORDERED;
                } else {
                    algorithm = Algorithm.CONCURRENT_REORDERED;
                }
            } else if(cols1<=500) {
                if(rows1<=10) {
                    algorithm = Algorithm.REORDERED;
                } else {
                    algorithm = Algorithm.CONCURRENT_REORDERED;
                }
            } else {
                if(rows1<=5) {
                    algorithm = Algorithm.REORDERED;
                } else if(rows1<=50){
                    algorithm = Algorithm.CONCURRENT_STANDARD;
                } else {
                    algorithm = Algorithm.CONCURRENT_REORDERED;
                }
            }
        }

        return algorithm;
    }


    public static Algorithm chooseAlgorithmVector(Shape shape) {
        Algorithm algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<=300) {
            algorithm = Algorithm.BLOCKED_VECTOR;
        } else if(rows<=2048) {
            algorithm = Algorithm.CONCURRENT_BLOCKED_VECTOR;
        } else {
            algorithm = Algorithm.CONCURRENT_STANDARD_VECTOR;
        }

        return algorithm;
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
     * Simple enum class containing all possible choices of matrix multiply algorithms.
     */
    private enum Algorithm {
        STANDARD, REORDERED, BLOCKED, BLOCKED_REORDERED,
        CONCURRENT_STANDARD, CONCURRENT_REORDERED, CONCURRENT_BLOCKED, CONCURRENT_BLOCKED_REORDERED,
        STANDARD_VECTOR, BLOCKED_VECTOR, CONCURRENT_STANDARD_VECTOR, CONCURRENT_BLOCKED_VECTOR
    }
}
