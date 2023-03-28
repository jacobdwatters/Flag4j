/*
 * MIT License
 *
 * Copyright (c) 2022-2023 Jacob Watters
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
import com.flag4j.operations.dense.complex.ComplexDenseMatrixMultiplication;
import com.flag4j.operations.dense.real.RealDenseMatrixMultiplication;
import com.flag4j.operations.dense.real_complex.RealComplexDenseMatrixMultiplication;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;

import static com.flag4j.util.ParameterChecks.assertMatMultShapes;

/**
 * Dispatches matrix multiplication to the appropriate algorithm based on the size of the matrices to be multiplied.
 */
public final class MatrixMultiplyDispatcher {

    private MatrixMultiplyDispatcher() {
        // Hide constructor of utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /*
        TODO: Move all dispatch methods to their own singleton classes like RealDenseMatrixMultiplyDispatcher.
     */


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
     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
     * @param A Matrix to multiply.
     * @param b Vector to multiply.
     * @return The result of the matrix-vector multiplication.
     */
    public static double[] dispatch(Matrix A, Vector b) {
        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
        assertMatMultShapes(A.shape, bMatShape);

        AlgorithmNames algorithm;
        double[] dest;

        algorithm = chooseAlgorithmRealVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                dest = RealDenseMatrixMultiplication.standardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case BLOCKED_VECTOR:
                dest = RealDenseMatrixMultiplication.blockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = RealDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            default:
                dest = RealDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
        }

        return dest;
    }


    /**
     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
     * @param A Matrix to multiply.
     * @param b Vector to multiply.
     * @return The result of the matrix-vector multiplication.
     */
    public static CNumber[] dispatch(Matrix A, CVector b) {
        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
        assertMatMultShapes(A.shape, bMatShape);

        AlgorithmNames algorithm;
        CNumber[] dest;

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case BLOCKED_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            default:
                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
        }

        return dest;
    }


    /**
     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
     * @param A Matrix to multiply.
     * @param b Vector to multiply.
     * @return The result of the matrix-vector multiplication.
     */
    public static CNumber[] dispatch(CMatrix A, Vector b) {
        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
        assertMatMultShapes(A.shape, bMatShape);

        AlgorithmNames algorithm;
        CNumber[] dest;

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case BLOCKED_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            default:
                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
        }

        return dest;
    }


    /**
     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
     * @param A Matrix to multiply.
     * @param b Vector to multiply.
     * @return The result of the matrix-vector multiplication.
     */
    public static CNumber[] dispatch(CMatrix A, CVector b) {
        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
        assertMatMultShapes(A.shape, bMatShape);

        AlgorithmNames algorithm;
        CNumber[] dest;

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                dest = ComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case BLOCKED_VECTOR:
                dest = ComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = ComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            default:
                dest = ComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication.
     * @return The result of the matrix multiplication.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication.
     */
    public static CNumber[] dispatch(CMatrix A, CMatrix B) {
        assertMatMultShapes(A.shape, B.shape);

        AlgorithmNames algorithm;
        CNumber[] dest;

        if(B.numCols==1) {
            algorithm = chooseAlgorithmComplexVector(A.shape);
        } else {
            algorithm = chooseAlgorithmComplex(A.shape, B.shape);
        }

        switch(algorithm) {
            case STANDARD:
                dest = ComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
                break;
            case REORDERED:
                dest = ComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED:
                dest = ComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED_REORDERED:
                dest = ComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_STANDARD:
                dest = ComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_REORDERED:
                dest = ComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_BLOCKED:
                dest = ComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                dest = ComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case STANDARD_VECTOR:
                dest = ComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED_VECTOR:
                dest = ComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = ComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, B.entries, B.shape);
                break;
            default:
                dest = ComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication.
     * @return The result of the matrix multiplication.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication.
     */
    public static CNumber[] dispatch(Matrix A, CMatrix B) {
        assertMatMultShapes(A.shape, B.shape);

        AlgorithmNames algorithm;
        CNumber[] dest;

        if(B.numCols==1) {
            algorithm = chooseAlgorithmRealComplexVector(A.shape);
        } else {
            algorithm = chooseAlgorithmRealComplex(A.shape, B.shape);
        }

        switch(algorithm) {
            case STANDARD:
                dest = RealComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
                break;
            case REORDERED:
                dest = RealComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED:
                dest = RealComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED_REORDERED:
                dest = RealComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_STANDARD:
                dest = RealComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_REORDERED:
                dest = RealComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_BLOCKED:
                dest = RealComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case STANDARD_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, B.entries, B.shape);
                break;
            default:
                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication.
     * @return The result of the matrix multiplication.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication.
     */
    public static CNumber[] dispatch(CMatrix A, Matrix B) {
        assertMatMultShapes(A.shape, B.shape);

        AlgorithmNames algorithm;
        CNumber[] dest;

        if(B.numCols==1) {
            algorithm = chooseAlgorithmRealComplexVector(A.shape);
        } else {
            algorithm = chooseAlgorithmRealComplex(A.shape, B.shape);
        }

        switch(algorithm) {
            case STANDARD:
                dest = RealComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
                break;
            case REORDERED:
                dest = RealComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED:
                dest = RealComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED_REORDERED:
                dest = RealComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_STANDARD:
                dest = RealComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_REORDERED:
                dest = RealComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_BLOCKED:
                dest = RealComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
                break;
            case STANDARD_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, B.entries, B.shape);
                break;
            case BLOCKED_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, B.entries, B.shape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = RealComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, B.entries, B.shape);
                break;
            default:
                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape);
                break;
        }

        return dest;
    }


    /**
     * Dynamically chooses matrix-vector multiply algorithm based on the shapes of the matrix to multiply.
     * @param shape The shape of the matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmNames chooseAlgorithmRealVector(Shape shape) {
        AlgorithmNames algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<=300) {
            algorithm = AlgorithmNames.BLOCKED_VECTOR;
        } else if(rows<=2048) {
            algorithm = AlgorithmNames.CONCURRENT_BLOCKED_VECTOR;
        } else {
            algorithm = AlgorithmNames.CONCURRENT_STANDARD_VECTOR;
        }

        return algorithm;
    }


    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmNames chooseAlgorithmComplex(Shape shape1, Shape shape2) {
        AlgorithmNames algorithm;

        int rows1 = shape1.get(Axis2D.row());
        int cols1 = shape1.get(Axis2D.col());

        // TODO: Extract constants to final variables
        if(getRatio(shape1) >= SQUARENESS_RATIO) {
            // Then the first matrix is approximately square.
            if(rows1<=30) {
                algorithm = AlgorithmNames.REORDERED;
            } else if(rows1<=250) {
                algorithm = AlgorithmNames.CONCURRENT_REORDERED;
            } else {
            /* For large matrices, use a concurrent, blocked algorithm with the j-k loops swapped for
            better cache performance on modern systems */
                algorithm = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
            }

        } else if(rows1>cols1) {
            // Then there are more rows than columns in the first matrix
            if(rows1<=100) {
                if(cols1<=4) algorithm = AlgorithmNames.REORDERED;
                else algorithm = AlgorithmNames.CONCURRENT_REORDERED;
            } else {
                if(cols1<=45) algorithm = AlgorithmNames.CONCURRENT_REORDERED;
                else algorithm = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
            }
        } else {
            // Then there are more columns than rows in the first matrix
            if(cols1<=100) {
                if(rows1<=20) {
                    algorithm = AlgorithmNames.REORDERED;
                } else {
                    algorithm = AlgorithmNames.CONCURRENT_REORDERED;
                }
            } else if(cols1<=500) {
                if(rows1<=10) {
                    algorithm = AlgorithmNames.REORDERED;
                } else if(rows1<=200) {
                    algorithm = AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
                }
            } else {
                if(rows1<=5) {
                    algorithm = AlgorithmNames.REORDERED;
                } else if(rows1<=15){
                    algorithm = AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
                }
            }
        }

        return algorithm;
    }


    /**
     * Dynamically chooses matrix-vector multiply algorithm based on the shapes of the matrix to multiply.
     * @param shape The shape of the matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmNames chooseAlgorithmComplexVector(Shape shape) {
        AlgorithmNames algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<=250) {
            algorithm = AlgorithmNames.STANDARD_VECTOR;
        } else if(rows<=1024) {
            algorithm = AlgorithmNames.CONCURRENT_BLOCKED_VECTOR;
        } else {
            algorithm = AlgorithmNames.CONCURRENT_STANDARD_VECTOR;
        }

        return algorithm;
    }


    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmNames chooseAlgorithmRealComplex(Shape shape1, Shape shape2) {
        AlgorithmNames algorithm;

        int rows1 = shape1.get(Axis2D.row());
        int cols1 = shape1.get(Axis2D.col());

        // TODO: Extract constants to final variables
        if(getRatio(shape1) >= SQUARENESS_RATIO) {
            // Then the first matrix is approximately square.
            if(rows1<=40) {
                algorithm = AlgorithmNames.REORDERED;
            } else if(rows1<=225) {
                algorithm = AlgorithmNames.CONCURRENT_REORDERED;
            } else {
            /* For large matrices, use a concurrent, blocked algorithm with the j-k loops swapped for
            better cache performance on modern systems */
                algorithm = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
            }

        } else if(rows1>cols1) {
            // Then there are more rows than columns in the first matrix
            if(rows1<=100) {
                if(cols1<=2) algorithm = AlgorithmNames.REORDERED;
                else algorithm = AlgorithmNames.CONCURRENT_REORDERED;
            } else {
                if(cols1<=45) algorithm = AlgorithmNames.CONCURRENT_REORDERED;
                else algorithm = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
            }
        } else {
            // Then there are more columns than rows in the first matrix
            if(cols1<=100) {
                if(rows1<=15) {
                    algorithm = AlgorithmNames.REORDERED;
                } else {
                    algorithm = AlgorithmNames.CONCURRENT_REORDERED;
                }
            } else if(cols1<=500) {
                if(rows1<=15) {
                    algorithm = AlgorithmNames.REORDERED;
                } else if(rows1<=100) {
                    algorithm = AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
                }
            } else {
                if(rows1<=2) {
                    algorithm = AlgorithmNames.REORDERED;
                } else if(rows1<=15){
                    algorithm = AlgorithmNames.BLOCKED_REORDERED;
                } else if(rows1<=150) {
                    algorithm = AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
                }
            }
        }

        return algorithm;
    }


    /**
     * Dynamically chooses matrix-vector multiply algorithm based on the shapes of the matrix to multiply.
     * @param shape The shape of the matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmNames chooseAlgorithmRealComplexVector(Shape shape) {
        AlgorithmNames algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<=600) {
            algorithm = AlgorithmNames.STANDARD_VECTOR;
        } else {
            algorithm = AlgorithmNames.CONCURRENT_BLOCKED_VECTOR;
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
    private enum AlgorithmNames {
        STANDARD, REORDERED, BLOCKED, BLOCKED_REORDERED,
        CONCURRENT_STANDARD, CONCURRENT_REORDERED, CONCURRENT_BLOCKED, CONCURRENT_BLOCKED_REORDERED,
        STANDARD_VECTOR, BLOCKED_VECTOR, CONCURRENT_STANDARD_VECTOR, CONCURRENT_BLOCKED_VECTOR
    }
}
