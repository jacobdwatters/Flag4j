/*
 * MIT License
 *
 * Copyright (c) 2022-2024. Jacob Watters
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

package org.flag4j.operations;

import org.flag4j.core.Shape;
import org.flag4j.core_temp.arrays.dense.FieldMatrix;
import org.flag4j.core_temp.arrays.dense.FieldVector;
import org.flag4j.core_temp.arrays.dense.Matrix;
import org.flag4j.core_temp.arrays.dense.Vector;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.operations.dense.field_ops.DenseFieldMatrixMultiplication;
import org.flag4j.operations.dense.real.RealDenseMatrixMultiplication;
import org.flag4j.util.Axis2D;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ParameterChecks;

/**
 * Dispatches matrix multiplication to the appropriate algorithm based on the size of the matrices to be multiplied.
 */
public final class MatrixMultiplyDispatcher {

    private MatrixMultiplyDispatcher() {
        // Hide constructor of utility class
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
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
        ParameterChecks.ensureMatMultShapes(A.shape, bMatShape);

        AlgorithmName algorithm;
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


//    /**
//     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
//     * @param A Matrix to multiply.
//     * @param b Vector to multiply.
//     * @return The result of the matrix-vector multiplication.
//     */
//    public static CNumber[] dispatch(Matrix A, CVector b) {
//        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
//        ParameterChecks.assertMatMultShapes(A.shape, bMatShape);
//
//        AlgorithmName algorithm;
//        CNumber[] dest;
//
//        algorithm = chooseAlgorithmRealComplexVector(A.shape);
//
//        switch(algorithm) {
//            case STANDARD_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, b.entries, bMatShape);
//                break;
//            case BLOCKED_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, b.entries, bMatShape);
//                break;
//            case CONCURRENT_STANDARD_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, b.entries, bMatShape);
//                break;
//            default:
//                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, b.entries, bMatShape);
//                break;
//        }
//
//        return dest;
//    }
//
//
//    /**
//     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
//     * @param A Matrix to multiply.
//     * @param b Vector to multiply.
//     * @return The result of the matrix-vector multiplication.
//     */
//    public static CNumber[] dispatch(CMatrix A, Vector b) {
//        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
//        ParameterChecks.assertMatMultShapes(A.shape, bMatShape);
//
//        AlgorithmName algorithm;
//        CNumber[] dest;
//
//        algorithm = chooseAlgorithmRealComplexVector(A.shape);
//
//        switch(algorithm) {
//            case STANDARD_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, b.entries, bMatShape);
//                break;
//            case BLOCKED_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, b.entries, bMatShape);
//                break;
//            case CONCURRENT_STANDARD_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, b.entries, bMatShape);
//                break;
//            default:
//                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, b.entries, bMatShape);
//                break;
//        }
//
//        return dest;
//    }


    /**
     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
     * @param A Matrix to multiply.
     * @param b Vector to multiply.
     * @return The result of the matrix-vector multiplication.
     */
    public static <T extends Field<T>> Field<T>[] dispatch(FieldMatrix<T> A, FieldVector<T> b) {
        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
        ParameterChecks.ensureMatMultShapes(A.shape, bMatShape);

        AlgorithmName algorithm;
        Field<T>[] dest;

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                dest = DenseFieldMatrixMultiplication.standardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case BLOCKED_VECTOR:
                dest = DenseFieldMatrixMultiplication.blockedVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = DenseFieldMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, b.entries, bMatShape);
                break;
            default:
                dest = DenseFieldMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, b.entries, bMatShape);
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
    public static <T extends Field<T>> Field<T>[] dispatch(FieldMatrix<T> A, FieldMatrix<T> B) {
        return dispatch(A.entries, A.shape, B.entries, B.shape);
    }


    /**
     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @return The result of the matrix multiplication between the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] dispatch(T[] src1, Shape shape1, T[] src2, Shape shape2) {
        ParameterChecks.ensureMatMultShapes(shape1, shape2);

        AlgorithmName algorithm;
        Field<T>[] dest;

        if(shape2.get(1)==1) {
            // Then B is a column vector.
            algorithm = chooseAlgorithmComplexVector(shape1);
        } else {
            algorithm = chooseAlgorithmComplex(shape1, shape2);
        }

        switch(algorithm) {
            case STANDARD:
                dest = DenseFieldMatrixMultiplication.standard(src1, shape1, src2, shape2);
                break;
            case REORDERED:
                dest = DenseFieldMatrixMultiplication.reordered(src1, shape1, src2, shape2);
                break;
            case BLOCKED:
                dest = DenseFieldMatrixMultiplication.blocked(src1, shape1, src2, shape2);
                break;
            case BLOCKED_REORDERED:
                dest = DenseFieldMatrixMultiplication.blockedReordered(src1, shape1, src2, shape2);
                break;
            case CONCURRENT_STANDARD:
                dest = DenseFieldMatrixMultiplication.concurrentStandard(src1, shape1, src2, shape2);
                break;
            case CONCURRENT_REORDERED:
                dest = DenseFieldMatrixMultiplication.concurrentReordered(src1, shape1, src2, shape2);
                break;
            case CONCURRENT_BLOCKED:
                dest = DenseFieldMatrixMultiplication.concurrentBlocked(src1, shape1, src2, shape2);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                dest = DenseFieldMatrixMultiplication.concurrentBlockedReordered(src1, shape1, src2, shape2);
                break;
            case STANDARD_VECTOR:
                dest = DenseFieldMatrixMultiplication.standardVector(src1, shape1, src2, shape2);
                break;
            case BLOCKED_VECTOR:
                dest = DenseFieldMatrixMultiplication.blockedVector(src1, shape1, src2, shape2);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = DenseFieldMatrixMultiplication.concurrentStandardVector(src1, shape1, src2, shape2);
                break;
            default:
                dest = DenseFieldMatrixMultiplication.concurrentBlockedVector(src1, shape1, src2, shape2);
                break;
        }

        return dest;
    }


//    /**
//     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
//     * @param A First matrix in matrix multiplication.
//     * @param B Second matrix in matrix multiplication.
//     * @return The result of the matrix multiplication.
//     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication.
//     */
//    public static CNumber[] dispatch(Matrix A, CMatrix B) {
//        ParameterChecks.assertMatMultShapes(A.shape, B.shape);
//
//        AlgorithmName algorithm;
//        CNumber[] dest;
//
//        if(B.numCols==1) {
//            algorithm = chooseAlgorithmRealComplexVector(A.shape);
//        } else {
//            algorithm = chooseAlgorithmRealComplex(A.shape, B.shape);
//        }
//
//        switch(algorithm) {
//            case STANDARD:
//                dest = RealComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case REORDERED:
//                dest = RealComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case BLOCKED:
//                dest = RealComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case BLOCKED_REORDERED:
//                dest = RealComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_STANDARD:
//                dest = RealComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_REORDERED:
//                dest = RealComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_BLOCKED:
//                dest = RealComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_BLOCKED_REORDERED:
//                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case STANDARD_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case BLOCKED_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_STANDARD_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, B.entries, B.shape);
//                break;
//            default:
//                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape);
//                break;
//        }
//
//        return dest;
//    }
//
//
//    /**
//     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
//     * @param A First matrix in matrix multiplication.
//     * @param B Second matrix in matrix multiplication.
//     * @return The result of the matrix multiplication.
//     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication.
//     */
//    public static CNumber[] dispatch(CMatrix A, Matrix B) {
//        ParameterChecks.assertMatMultShapes(A.shape, B.shape);
//
//        AlgorithmName algorithm;
//        CNumber[] dest;
//
//        if(B.numCols==1) {
//            algorithm = chooseAlgorithmRealComplexVector(A.shape);
//        } else {
//            algorithm = chooseAlgorithmRealComplex(A.shape, B.shape);
//        }
//
//        switch(algorithm) {
//            case STANDARD:
//                dest = RealComplexDenseMatrixMultiplication.standard(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case REORDERED:
//                dest = RealComplexDenseMatrixMultiplication.reordered(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case BLOCKED:
//                dest = RealComplexDenseMatrixMultiplication.blocked(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case BLOCKED_REORDERED:
//                dest = RealComplexDenseMatrixMultiplication.blockedReordered(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_STANDARD:
//                dest = RealComplexDenseMatrixMultiplication.concurrentStandard(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_REORDERED:
//                dest = RealComplexDenseMatrixMultiplication.concurrentReordered(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_BLOCKED:
//                dest = RealComplexDenseMatrixMultiplication.concurrentBlocked(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_BLOCKED_REORDERED:
//                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedReordered(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case STANDARD_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.standardVector(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case BLOCKED_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.blockedVector(A.entries, A.shape, B.entries, B.shape);
//                break;
//            case CONCURRENT_STANDARD_VECTOR:
//                dest = RealComplexDenseMatrixMultiplication.concurrentStandardVector(A.entries, A.shape, B.entries, B.shape);
//                break;
//            default:
//                dest = RealComplexDenseMatrixMultiplication.concurrentBlockedVector(A.entries, A.shape, B.entries, B.shape);
//                break;
//        }
//
//        return dest;
//    }
//
//
//    /**
//     * Dispatches a matrix multiplication-transpose problem to the appropriate algorithm based on the size.
//     * @param A First matrix in matrix multiplication.
//     * @param B Second matrix in matrix multiplication and the matrix to transpose.
//     * @return The result of the matrix multiplication-transpose.
//     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication-transpose.
//     */
//    public static CNumber[] dispatchTranspose(CMatrix A, Matrix B) {
//        ParameterChecks.assertEquals(A.numCols, B.numCols);
//        AlgorithmName algorithm = chooseAlgorithmRealComplexTranspose(A.shape);
//        CNumber[] dest;
//
//        switch(algorithm) {
//            case MULT_T:
//                dest = RealComplexDenseMatrixMultTranspose.multTranspose(
//                        A.entries, A.shape, B.entries, B.shape);
//                break;
//            case MULT_T_BLOCKED:
//                dest = RealComplexDenseMatrixMultTranspose.multTransposeBlocked(
//                        A.entries, A.shape, B.entries, B.shape);
//                break;
//            case MULT_T_CONCURRENT:
//                dest = RealComplexDenseMatrixMultTranspose.multTransposeConcurrent(
//                        A.entries, A.shape, B.entries, B.shape);
//                break;
//            default:
//                dest = RealComplexDenseMatrixMultTranspose.multTransposeBlockedConcurrent(
//                        A.entries, A.shape, B.entries, B.shape);
//        }
//
//        return dest;
//    }
//
//
//    /**
//     * Dispatches a matrix multiplication-transpose problem to the appropriate algorithm based on the size.
//     * @param A First matrix in matrix multiplication.
//     * @param B Second matrix in matrix multiplication and the matrix to transpose.
//     * @return The result of the matrix multiplication-transpose.
//     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication-transpose.
//     */
//    public static CNumber[] dispatchTranspose(Matrix A, CMatrix B) {
//        ParameterChecks.assertEquals(A.numCols, B.numCols);
//        AlgorithmName algorithm = chooseAlgorithmRealComplexTranspose(A.shape);
//        CNumber[] dest;
//
//        switch(algorithm) {
//            case MULT_T:
//                dest = RealComplexDenseMatrixMultTranspose.multTranspose(
//                        A.entries, A.shape, B.entries, B.shape);
//                break;
//            case MULT_T_BLOCKED:
//                dest = RealComplexDenseMatrixMultTranspose.multTransposeBlocked(
//                        A.entries, A.shape, B.entries, B.shape);
//                break;
//            case MULT_T_CONCURRENT:
//                dest = RealComplexDenseMatrixMultTranspose.multTransposeConcurrent(
//                        A.entries, A.shape, B.entries, B.shape);
//                break;
//            default:
//                dest = RealComplexDenseMatrixMultTranspose.multTransposeBlockedConcurrent(
//                        A.entries, A.shape, B.entries, B.shape);
//        }
//
//        return dest;
//    }
//
//
//    /**
//     * Dispatches a matrix multiplication-transpose problem to the appropriate algorithm based on the size.
//     * @param A First matrix in matrix multiplication.
//     * @param B Second matrix in matrix multiplication and the matrix to transpose.
//     * @return The result of the matrix multiplication-transpose.
//     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication-transpose.
//     */
//    public static CNumber[] dispatchTranspose(CMatrix A, CMatrix B) {
//        ParameterChecks.assertEquals(A.numCols, B.numCols);
//        AlgorithmName algorithm = chooseAlgorithmRealComplexTranspose(A.shape);
//        CNumber[] dest;
//
//        switch(algorithm) {
//            case MULT_T:
//                dest = ComplexDenseMatrixMultTranspose.multTranspose(
//                        A.entries, A.shape, B.entries, B.shape);
//                break;
//            case MULT_T_CONCURRENT:
//                dest = ComplexDenseMatrixMultTranspose.multTransposeConcurrent(
//                        A.entries, A.shape, B.entries, B.shape);
//                break;
//            default:
//                dest = ComplexDenseMatrixMultTranspose.multTransposeBlockedConcurrent(
//                        A.entries, A.shape, B.entries, B.shape);
//        }
//
//        return dest;
//    }


    /**
     * Dynamically chooses the matrix multiplication-transpose algorithm to used based on the shape of the first matrix.
     * @param shape Shape of the first matrix.
     */
    public static AlgorithmName chooseAlgorithmRealComplexTranspose(Shape shape) {
        AlgorithmName algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<50) {
            algorithm = AlgorithmName.MULT_T;
        } else if(rows<60) {
            algorithm = AlgorithmName.MULT_T_BLOCKED;
        } else if(rows<300) {
            algorithm = AlgorithmName.MULT_T_CONCURRENT;
        } else {
            algorithm = AlgorithmName.MULT_T_BLOCKED_CONCURRENT;
        }

        return algorithm;
    }


    /**
     * Dynamically chooses the matrix multiplication-transpose algorithm to used based on the shape of the first matrix.
     * @param shape Shape of the first matrix.
     */
    public static AlgorithmName chooseAlgorithmComplexTranspose(Shape shape) {
        AlgorithmName algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<25) {
            algorithm = AlgorithmName.MULT_T;
        } else if(rows<750) {
            algorithm = AlgorithmName.MULT_T_CONCURRENT;
        } else {
            algorithm = AlgorithmName.MULT_T_BLOCKED_CONCURRENT;
        }

        return algorithm;
    }


    /**
     * Dynamically chooses matrix-vector multiply algorithm based on the shapes of the matrix to multiply.
     * @param shape The shape of the matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmName chooseAlgorithmRealVector(Shape shape) {
        AlgorithmName algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<=300) {
            algorithm = AlgorithmName.BLOCKED_VECTOR;
        } else if(rows<=2048) {
            algorithm = AlgorithmName.CONCURRENT_BLOCKED_VECTOR;
        } else {
            algorithm = AlgorithmName.CONCURRENT_STANDARD_VECTOR;
        }

        return algorithm;
    }


    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmName chooseAlgorithmComplex(Shape shape1, Shape shape2) {
        AlgorithmName algorithm;

        int rows1 = shape1.get(Axis2D.row());
        int cols1 = shape1.get(Axis2D.col());

        // TODO: Extract constants to final variables
        if(getRatio(shape1) >= SQUARENESS_RATIO) {
            // Then the first matrix is approximately square.
            if(rows1<=30) {
                algorithm = AlgorithmName.REORDERED;
            } else if(rows1<=250) {
                algorithm = AlgorithmName.CONCURRENT_REORDERED;
            } else {
            /* For large matrices, use a concurrent, blocked algorithm with the j-k loops swapped for
            better cache performance on modern systems */
                algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
            }

        } else if(rows1>cols1) {
            // Then there are more rows than columns in the first matrix
            if(rows1<=100) {
                if(cols1<=4) algorithm = AlgorithmName.REORDERED;
                else algorithm = AlgorithmName.CONCURRENT_REORDERED;
            } else {
                if(cols1<=45) algorithm = AlgorithmName.CONCURRENT_REORDERED;
                else algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
            }
        } else {
            // Then there are more columns than rows in the first matrix
            if(cols1<=100) {
                if(rows1<=20) {
                    algorithm = AlgorithmName.REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                }
            } else if(cols1<=500) {
                if(rows1<=10) {
                    algorithm = AlgorithmName.REORDERED;
                } else if(rows1<=200) {
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
                }
            } else {
                if(rows1<=5) {
                    algorithm = AlgorithmName.REORDERED;
                } else if(rows1<=15){
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
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
    public static AlgorithmName chooseAlgorithmComplexVector(Shape shape) {
        AlgorithmName algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<=250) {
            algorithm = AlgorithmName.STANDARD_VECTOR;
        } else if(rows<=1024) {
            algorithm = AlgorithmName.CONCURRENT_BLOCKED_VECTOR;
        } else {
            algorithm = AlgorithmName.CONCURRENT_STANDARD_VECTOR;
        }

        return algorithm;
    }


    /**
     * Dynamically chooses matrix multiply algorithm based on the shapes of the two matrices to multiply.
     * @param shape1 The shape of the first matrix.
     * @param shape2 The shape fo the second matrix.
     * @return The algorithm to use in the matrix multiplication.
     */
    public static AlgorithmName chooseAlgorithmRealComplex(Shape shape1, Shape shape2) {
        AlgorithmName algorithm;

        int rows1 = shape1.get(Axis2D.row());
        int cols1 = shape1.get(Axis2D.col());

        // TODO: Extract constants to final variables
        if(getRatio(shape1) >= SQUARENESS_RATIO) {
            // Then the first matrix is approximately square.
            if(rows1<=40) {
                algorithm = AlgorithmName.REORDERED;
            } else if(rows1<=225) {
                algorithm = AlgorithmName.CONCURRENT_REORDERED;
            } else {
            /* For large matrices, use a concurrent, blocked algorithm with the j-k loops swapped for
            better cache performance on modern systems */
                algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
            }

        } else if(rows1>cols1) {
            // Then there are more rows than columns in the first matrix
            if(rows1<=100) {
                if(cols1<=2) algorithm = AlgorithmName.REORDERED;
                else algorithm = AlgorithmName.CONCURRENT_REORDERED;
            } else {
                if(cols1<=45) algorithm = AlgorithmName.CONCURRENT_REORDERED;
                else algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
            }
        } else {
            // Then there are more columns than rows in the first matrix
            if(cols1<=100) {
                if(rows1<=15) {
                    algorithm = AlgorithmName.REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                }
            } else if(cols1<=500) {
                if(rows1<=15) {
                    algorithm = AlgorithmName.REORDERED;
                } else if(rows1<=100) {
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
                }
            } else {
                if(rows1<=2) {
                    algorithm = AlgorithmName.REORDERED;
                } else if(rows1<=15){
                    algorithm = AlgorithmName.BLOCKED_REORDERED;
                } else if(rows1<=150) {
                    algorithm = AlgorithmName.CONCURRENT_REORDERED;
                } else {
                    algorithm = AlgorithmName.CONCURRENT_BLOCKED_REORDERED;
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
    public static AlgorithmName chooseAlgorithmRealComplexVector(Shape shape) {
        AlgorithmName algorithm;

        int rows = shape.get(Axis2D.row());

        if(rows<=600) {
            algorithm = AlgorithmName.STANDARD_VECTOR;
        } else {
            algorithm = AlgorithmName.CONCURRENT_BLOCKED_VECTOR;
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
    private enum AlgorithmName {
        STANDARD, REORDERED, BLOCKED, BLOCKED_REORDERED,
        CONCURRENT_STANDARD, CONCURRENT_REORDERED, CONCURRENT_BLOCKED, CONCURRENT_BLOCKED_REORDERED,
        STANDARD_VECTOR, BLOCKED_VECTOR, CONCURRENT_STANDARD_VECTOR, CONCURRENT_BLOCKED_VECTOR, MULT_T, MULT_T_BLOCKED,
        MULT_T_CONCURRENT, MULT_T_BLOCKED_CONCURRENT
    }
}
