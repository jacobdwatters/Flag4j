/*
 * MIT License
 *
 * Copyright (c) 2024. Jacob Watters
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

package org.flag4j.operations.dense.field_ops;


// TODO: The selection algorithm should be redesigned. All threshold values should be specified in a config file.
// TODO: Investigate the performance of utilizing a selection cache which caches the implementation to be used for recent matrix sizes.
//  Should probably be implemented as a LRU cache (or similar) by extending LinkedHashMap

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.DenseFieldMatrixBase;
import org.flag4j.arrays.backend.DenseFieldTensorBinaryOperation;
import org.flag4j.arrays.backend.DenseFieldVectorBase;
import org.flag4j.util.ValidateParameters;

import java.util.HashMap;
import java.util.Map;

/**
 * Singleton class which stores a map of all viable dense {@link Field}
 * matrix multiply algorithms and uses that map to dispatch a dense {@link Field} matrix
 * multiply problem to the appropriate algorithm.
 */
public final class DenseFieldMatMultDispatcher {

    /**
     * Singleton instance of this class.
     */
    private static DenseFieldMatMultDispatcher singletonInstance = null;
    /**
     * lookup table for all real dense matrix multiply algorithms.
     */
    private final Map<AlgorithmNames, DenseFieldTensorBinaryOperation> algorithmMap;

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
    private DenseFieldMatMultDispatcher() {
        AlgorithmNames[] names = AlgorithmNames.values();
        algorithmMap = new HashMap<>();

        DenseFieldTensorBinaryOperation[] algorithms = {
                DenseFieldMatrixMultiplication::standard,
                DenseFieldMatrixMultiplication::reordered,
                DenseFieldMatrixMultiplication::blocked,
                DenseFieldMatrixMultiplication::blockedReordered,
                DenseFieldMatrixMultiplication::concurrentStandard,
                DenseFieldMatrixMultiplication::concurrentReordered,
                DenseFieldMatrixMultiplication::concurrentBlocked,
                DenseFieldMatrixMultiplication::concurrentBlockedReordered,
                DenseFieldMatrixMultiplication::standardVector,
                DenseFieldMatrixMultiplication::blockedVector,
                DenseFieldMatrixMultiplication::concurrentStandardVector,
                DenseFieldMatrixMultiplication::concurrentBlockedVector,
                DenseFieldMatrixMultTranspose::multTranspose,
                DenseFieldMatrixMultTranspose::multTransposeBlocked,
                DenseFieldMatrixMultTranspose::multTransposeConcurrent,
                DenseFieldMatrixMultTranspose::multTransposeBlockedConcurrent,
        };

        for(int i=0; i< algorithms.length; i++) {
            algorithmMap.put(names[i], algorithms[i]);
        }
    }


    /**
     * Gets the singleton instance of this class. If this class has not been instanced, a new instance will be created.
     * @return The singleton instance of this class.
     */
    public static synchronized DenseFieldMatMultDispatcher getInstance()  {
        singletonInstance = (singletonInstance == null) ? new DenseFieldMatMultDispatcher() : singletonInstance;
        return singletonInstance;
    }


    /**
     * Dispatches a matrix multiply problem to the appropriate algorithm based on the size of the matrices.
     * @param A First matrix in the multiplication.
     * @param B Second matrix in the multiplication.
     * @return The result of the matrix multiplication.
     */
    public static <T extends Field<T>> Field<T>[] dispatch(DenseFieldMatrixBase<?, ?, ?, ?, T> A, DenseFieldMatrixBase<?, ?, ?, ?, T> B) {
        ValidateParameters.ensureMatMultShapes(A.shape, B.shape); // Ensure matrix shapes are conducive to matrix multiplication.

        DenseFieldMatMultDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithm(A.shape, B.shape);
        return dispatcher.algorithmMap.get(name).apply(A.entries, A.shape, B.entries, B.shape);
    }


    /**
     * Dispatches a matrix-vector multiplication problem to the appropriate algorithm based on the size of the matrix and vector.
     * @param src1 Matrix in the matrix-vector multiplication problem.
     * @param src2 Vector in the matrix-vector multiplication problem.
     * @return The result of the matrix-vector multiplication.
     */
    public static <T extends Field<T>> Field<T>[] dispatch(DenseFieldMatrixBase<?, ?, ?, ?, T> src1,
                                                           DenseFieldVectorBase<?, ?, ?, T> src2) {
        Shape bMatShape = new Shape(src2.size, 1);
        ValidateParameters.ensureMatMultShapes(src1.shape, bMatShape);

        AlgorithmNames algorithm = selectAlgorithmVector(src1.shape);
        Field<T>[] dest;

        switch(algorithm) {
            case STANDARD_VECTOR:
                dest = DenseFieldMatrixMultiplication.standardVector(src1.entries, src1.shape, src2.entries, bMatShape);
                break;
            case BLOCKED_VECTOR:
                dest = DenseFieldMatrixMultiplication.blockedVector(src1.entries, src1.shape, src2.entries, bMatShape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = DenseFieldMatrixMultiplication.concurrentStandardVector(src1.entries, src1.shape, src2.entries, bMatShape);
                break;
            default:
                dest = DenseFieldMatrixMultiplication.concurrentBlockedVector(src1.entries, src1.shape, src2.entries, bMatShape);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix multiply problem to the appropriate algorithm based on the size of the matrices.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @return The result of the matrix multiplication.
     */
    public static <T extends Field<T>> Field<T>[] dispatch(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        ValidateParameters.ensureMatMultShapes(shape1, shape2); // Ensure matrix shapes are conducive to matrix multiplication.

        DenseFieldMatMultDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithm(shape1, shape2);
        return dispatcher.algorithmMap.get(name).apply(src1, shape1, src2, shape2);
    }


    /**
     * Dispatches a matrix multiply-transpose problem equivalent to {@code A.mult(B.T())} to the appropriate algorithm based
     * on the size of the matrices.
     * @param A First matrix in the multiplication.
     * @param B Second matrix in the multiplication and the matrix to transpose.
     * @return The matrix multiply-transpose result of {@code A} and {@code B}.
     */
    public static <T extends Field<T>> Field<T>[] dispatchTranspose(DenseFieldMatrixBase<?, ?, ?, ?, T> A, DenseFieldMatrixBase<?, ?, ?, ?, T> B) {
        ValidateParameters.ensureArrayLengthsEq(A.numCols, B.numCols);

        DenseFieldMatMultDispatcher dispatcher = getInstance();
        AlgorithmNames name = selectAlgorithmTranspose(A.shape);
        return dispatcher.algorithmMap.get(name).apply(A.entries, A.shape, B.entries, B.shape);
    }



    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    static AlgorithmNames selectAlgorithm(Shape shape1, Shape shape2) {
        AlgorithmNames name;

        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);

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
                    name = AlgorithmNames.REORDERED;
                } else if (rows1 < CONCURRENT_SWAPPED_THRESHOLD) {
                    name = AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    name = AlgorithmNames.CONCURRENT_BLOCKED_REORDERED;
                }
                break;

            case 1: // More rows than columns
                if(rows1 <= 100 && cols1 <= 5) {
                    name = AlgorithmNames.REORDERED;
                } else {
                    name = AlgorithmNames.CONCURRENT_REORDERED;
                }
                break;

            case 2: // More columns than rows
                if(cols1 <= 100) {
                    name = (rows1 <= 20) ? AlgorithmNames.REORDERED : AlgorithmNames.CONCURRENT_REORDERED;
                } else if(cols1 <= 500) {
                    name = (rows1 <= 10) ? AlgorithmNames.REORDERED : AlgorithmNames.CONCURRENT_REORDERED;
                } else {
                    if(rows1 <= 5) {
                        name = AlgorithmNames.REORDERED;
                    } else if(rows1 <= 50) {
                        name = AlgorithmNames.CONCURRENT_STANDARD;
                    } else {
                        name = AlgorithmNames.CONCURRENT_REORDERED;
                    }
                }
                break;
            default:
                name = AlgorithmNames.BLOCKED_REORDERED; // Default to blocked reordered algorithm.
        }

        return name;
    }


    /**
     * Selects the matrix multiplication-transpose algorithm to use based on the size of the first matrix.
     * @param shape Shape of the first matrix.
     * @return The algorithm to use to compute the matrix multiplication-transpose.
     */
    static AlgorithmNames selectAlgorithmTranspose(Shape shape) {
        AlgorithmNames name;
        int rows = shape.get(0);

        // TODO: This currently only works well if both matrices are square.

        if(rows < 40) {
            name = AlgorithmNames.MULT_T;
        } else if(rows < 55) {
            name = AlgorithmNames.MULT_T_BLOCKED;
        } else if(rows < 1200) {
            name = AlgorithmNames.MULT_T_CONCURRENT;
        } else {
            name = AlgorithmNames.MULT_T_BLOCKED_CONCURRENT;
        }

        return name;
    }


    /**
     * Dynamically chooses matrix-vector multiply algorithm based on the shapes of the matrix to multiply.
     * @param shape The shape of the matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmNames selectAlgorithmVector(Shape shape) {
        if(shape.get(0) <=600) return AlgorithmNames.STANDARD_VECTOR;
        else return AlgorithmNames.CONCURRENT_BLOCKED_VECTOR;
    }


    /**
     * Computes the squareness ratio of a matrix. This is a value between 0 and 1 with 1 being perfectly
     * square and 0 being a row/column vector.
     * @param shape Shape of the matrix to compute the squareness ratio of.
     * @return The squareness ratio for the specified shape.
     */
    private static double getRatio(Shape shape) {
        int numRows = shape.get(0);
        int numCols = shape.get(1);

        double ratio = Math.abs(numRows-numCols);
        return 1-ratio/Math.max(numRows, numCols);
    }


    /**
     * Simple enum class containing all possible choices of real dense matrix multiply algorithms.
     */
    protected enum AlgorithmNames {
        STANDARD, REORDERED, BLOCKED, BLOCKED_REORDERED,
        CONCURRENT_STANDARD, CONCURRENT_REORDERED, CONCURRENT_BLOCKED, CONCURRENT_BLOCKED_REORDERED,
        STANDARD_VECTOR, BLOCKED_VECTOR, CONCURRENT_STANDARD_VECTOR, CONCURRENT_BLOCKED_VECTOR, MULT_T, MULT_T_BLOCKED,
        MULT_T_CONCURRENT, MULT_T_BLOCKED_CONCURRENT
    }
}
