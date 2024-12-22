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

package org.flag4j.linalg.ops;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.algebraic_structures.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.*;
import org.flag4j.linalg.ops.dense.real.RealDenseMatrixMultiplication;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseMatMult;
import org.flag4j.linalg.ops.dense.real_field_ops.RealFieldDenseMatMultTranspose;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringMatMult;
import org.flag4j.linalg.ops.dense.semiring_ops.DenseSemiringMatMultTranspose;
import org.flag4j.util.ValidateParameters;

/**
 * Dispatches matrix multiplication to the appropriate algorithm based on the size of the matrices to be multiplied.
 */
public final class MatrixMultiplyDispatcher {

    private MatrixMultiplyDispatcher() {
        // Hide constructor for utility class. of utility class
        
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
        ValidateParameters.ensureMatMultShapes(A.shape, bMatShape);

        AlgorithmName algorithm;
        double[] dest;

        algorithm = chooseAlgorithmRealVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                dest = RealDenseMatrixMultiplication.standardVector(A.data, A.shape, b.data, bMatShape);
                break;
            case BLOCKED_VECTOR:
                dest = RealDenseMatrixMultiplication.blockedVector(A.data, A.shape, b.data, bMatShape);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                dest = RealDenseMatrixMultiplication.concurrentStandardVector(A.data, A.shape, b.data, bMatShape);
                break;
            default:
                dest = RealDenseMatrixMultiplication.concurrentBlockedVector(A.data, A.shape, b.data, bMatShape);
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
    public static Complex128[] dispatch(Matrix A, CVector b) {
        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
        ValidateParameters.ensureMatMultShapes(A.shape, bMatShape);

        AlgorithmName algorithm;
        Complex128[] dest = new Complex128[A.numRows];

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                RealFieldDenseMatMult.standardVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            case BLOCKED_VECTOR:
                RealFieldDenseMatMult.blockedVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                RealFieldDenseMatMult.concurrentStandardVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            default:
                RealFieldDenseMatMult.concurrentBlockedVector(A.data, A.shape, b.data, bMatShape, dest);
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
    public static CVector dispatch(CMatrix A, Vector b) {
        ValidateParameters.ensureMatMultShapes(A.shape, b.shape);


        AlgorithmName algorithm;
        Complex128[] dest = new Complex128[A.numRows];

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                RealFieldDenseMatMult.standardVector(A.data, A.shape, b.data, b.shape, dest);
                break;
            case BLOCKED_VECTOR:
                RealFieldDenseMatMult.blockedVector(A.data, A.shape, b.data, b.shape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                RealFieldDenseMatMult.concurrentStandardVector(A.data, A.shape, b.data, b.shape, dest);
                break;
            default:
                RealFieldDenseMatMult.concurrentBlockedVector(A.data, A.shape, b.data, b.shape, dest);
                break;
        }

        return new CVector(dest);
    }


    /**
     * Dynamically chooses the appropriate matrix-vector multiplication algorithm based on the shapes of the matrix and vector.
     * @param A Matrix to multiply.
     * @param b Vector to multiply.
     * @return The result of the matrix-vector multiplication.
     */
    public static <T extends Field<T>> T[] dispatch(FieldMatrix<T> A, FieldVector<T> b) {
        Shape bMatShape = new Shape(b.totalEntries().intValue(), 1);
        ValidateParameters.ensureMatMultShapes(A.shape, bMatShape);
        AlgorithmName algorithm;
        T[] dest = A.makeEmptyDataArray(A.numRows);

        algorithm = chooseAlgorithmRealComplexVector(A.shape);

        switch(algorithm) {
            case STANDARD_VECTOR:
                DenseSemiringMatMult.standardVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            case BLOCKED_VECTOR:
                DenseSemiringMatMult.blockedVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                DenseSemiringMatMult.concurrentStandardVector(A.data, A.shape, b.data, bMatShape, dest);
                break;
            default:
                DenseSemiringMatMult.concurrentBlockedVector(A.data, A.shape, b.data, bMatShape, dest);
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
    public static <T extends Field<T>> T[] dispatch(FieldMatrix<T> A, FieldMatrix<T> B) {
        T[] dest = A.makeEmptyDataArray(A.numRows*B.numCols);
        dispatch(A.data, A.shape, B.data, B.shape, dest);
        return dest;
    }


    /**
     * Dispatches a matrix multiplication problem to the appropriate algorithm based on the size.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @return The result of the matrix multiplication between the two matrices.
     */
    public static <T extends Field<T>> T[] dispatch(T[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        ValidateParameters.ensureMatMultShapes(shape1, shape2);

        AlgorithmName algorithm;

        if(shape2.get(1)==1) {
            // Then B is a column vector.
            algorithm = chooseAlgorithmComplexVector(shape1);
        } else {
            algorithm = chooseAlgorithmComplex(shape1, shape2);
        }

        switch(algorithm) {
            case STANDARD:
                DenseSemiringMatMult.standard(src1, shape1, src2, shape2, dest);
                break;
            case REORDERED:
                DenseSemiringMatMult.reordered(src1, shape1, src2, shape2, dest);
                break;
            case BLOCKED:
                DenseSemiringMatMult.blocked(src1, shape1, src2, shape2, dest);
                break;
            case BLOCKED_REORDERED:
                DenseSemiringMatMult.blockedReordered(src1, shape1, src2, shape2, dest);
                break;
            case CONCURRENT_STANDARD:
                DenseSemiringMatMult.concurrentStandard(src1, shape1, src2, shape2, dest);
                break;
            case CONCURRENT_REORDERED:
                DenseSemiringMatMult.concurrentReordered(src1, shape1, src2, shape2, dest);
                break;
            case CONCURRENT_BLOCKED:
                DenseSemiringMatMult.concurrentBlocked(src1, shape1, src2, shape2, dest);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                DenseSemiringMatMult.concurrentBlockedReordered(src1, shape1, src2, shape2, dest);
                break;
            case STANDARD_VECTOR:
                DenseSemiringMatMult.standardVector(src1, shape1, src2, shape2, dest);
                break;
            case BLOCKED_VECTOR:
                DenseSemiringMatMult.blockedVector(src1, shape1, src2, shape2, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                DenseSemiringMatMult.concurrentStandardVector(src1, shape1, src2, shape2, dest);
                break;
            default:
                DenseSemiringMatMult.concurrentBlockedVector(src1, shape1, src2, shape2, dest);
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
    public static Complex128[] dispatch(Matrix A, CMatrix B) {
        ValidateParameters.ensureMatMultShapes(A.shape, B.shape);

        AlgorithmName algorithm;
        Complex128[] dest = new Complex128[A.numRows*B.numCols];

        if(B.numCols==1) {
            algorithm = chooseAlgorithmRealComplexVector(A.shape);
        } else {
            algorithm = chooseAlgorithmRealComplex(A.shape, B.shape);
        }

        switch(algorithm) {
            case STANDARD:
                RealFieldDenseMatMult.standard(A.data, A.shape, B.data, B.shape, dest);
                break;
            case REORDERED:
                RealFieldDenseMatMult.reordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED:
                RealFieldDenseMatMult.blocked(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED_REORDERED:
                RealFieldDenseMatMult.blockedReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_STANDARD:
                RealFieldDenseMatMult.concurrentStandard(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_REORDERED:
                RealFieldDenseMatMult.concurrentReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_BLOCKED:
                RealFieldDenseMatMult.concurrentBlocked(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                RealFieldDenseMatMult.concurrentBlockedReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case STANDARD_VECTOR:
                RealFieldDenseMatMult.standardVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED_VECTOR:
                RealFieldDenseMatMult.blockedVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                RealFieldDenseMatMult.concurrentStandardVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            default:
                RealFieldDenseMatMult.concurrentBlockedVector(A.data, A.shape, B.data, B.shape, dest);
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
    public static Complex128[] dispatch(CMatrix A, Matrix B) {
        ValidateParameters.ensureMatMultShapes(A.shape, B.shape);

        AlgorithmName algorithm;
        Complex128[] dest = new Complex128[A.numRows*B.numCols];

        if(B.numCols==1) {
            algorithm = chooseAlgorithmRealComplexVector(A.shape);
        } else {
            algorithm = chooseAlgorithmRealComplex(A.shape, B.shape);
        }

        switch(algorithm) {
            case STANDARD:
                RealFieldDenseMatMult.standard(A.data, A.shape, B.data, B.shape, dest);
                break;
            case REORDERED:
                RealFieldDenseMatMult.reordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED:
                RealFieldDenseMatMult.blocked(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED_REORDERED:
                RealFieldDenseMatMult.blockedReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_STANDARD:
                RealFieldDenseMatMult.concurrentStandard(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_REORDERED:
                RealFieldDenseMatMult.concurrentReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_BLOCKED:
                RealFieldDenseMatMult.concurrentBlocked(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_BLOCKED_REORDERED:
                RealFieldDenseMatMult.concurrentBlockedReordered(A.data, A.shape, B.data, B.shape, dest);
                break;
            case STANDARD_VECTOR:
                RealFieldDenseMatMult.standardVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            case BLOCKED_VECTOR:
                RealFieldDenseMatMult.blockedVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            case CONCURRENT_STANDARD_VECTOR:
                RealFieldDenseMatMult.concurrentStandardVector(A.data, A.shape, B.data, B.shape, dest);
                break;
            default:
                RealFieldDenseMatMult.concurrentBlockedVector(A.data, A.shape, B.data, B.shape, dest);
                break;
        }

        return dest;
    }


    /**
     * Dispatches a matrix multiplication-transpose problem to the appropriate algorithm based on the size. 
     * Computes {@code A.mult(B.T())} without explicitly transposing {@code B}.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication and the matrix to transpose.
     * @return The result of the matrix multiplication-transpose.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication-transpose.
     */
    public static Complex128[] dispatchTranspose(CMatrix A, Matrix B) {
        ValidateParameters.ensureEquals(A.numCols, B.numCols);
        AlgorithmName algorithm = chooseAlgorithmRealComplexTranspose(A.shape);
        Complex128[] dest = new Complex128[A.numRows*B.numRows];

        switch(algorithm) {
            case MULT_T:
                RealFieldDenseMatMultTranspose.multTranspose(
                        A.data, A.shape, B.data, B.shape, dest);
                break;
            case MULT_T_BLOCKED:
                RealFieldDenseMatMultTranspose.multTransposeBlocked(
                        A.data, A.shape, B.data, B.shape, dest);
                break;
            case MULT_T_CONCURRENT:
                RealFieldDenseMatMultTranspose.multTransposeConcurrent(
                        A.data, A.shape, B.data, B.shape, dest);
                break;
            default:
                RealFieldDenseMatMultTranspose.multTransposeBlockedConcurrent(
                        A.data, A.shape, B.data, B.shape, dest);
        }

        return dest;
    }


    /**
     * Dispatches a matrix multiplication-transpose problem to the appropriate algorithm based on the size.
     * Computes {@code A.mult(B.T())} but avoids explicit transposing of {@code B}.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication and the matrix to transpose.
     * @return The result of the matrix multiplication-transpose {@code A.mult(B.T())}.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication-transpose.
     */
    public static Complex128[] dispatchTranspose(Matrix A, CMatrix B) {
        ValidateParameters.ensureEquals(A.numCols, B.numCols);
        AlgorithmName algorithm = chooseAlgorithmRealComplexTranspose(A.shape);
        Complex128[] dest = new Complex128[A.numRows*B.numRows];

        switch(algorithm) {
            case MULT_T:
                RealFieldDenseMatMultTranspose.multTranspose(
                        A.data, A.shape, B.data, B.shape, dest);
                break;
            case MULT_T_BLOCKED:
                RealFieldDenseMatMultTranspose.multTransposeBlocked(
                        A.data, A.shape, B.data, B.shape, dest);
                break;
            case MULT_T_CONCURRENT:
                RealFieldDenseMatMultTranspose.multTransposeConcurrent(
                        A.data, A.shape, B.data, B.shape, dest);
                break;
            default:
                RealFieldDenseMatMultTranspose.multTransposeBlockedConcurrent(
                        A.data, A.shape, B.data, B.shape, dest);
        }

        return dest;
    }


    /**
     * Dispatches a matrix multiplication-transpose problem to the appropriate algorithm based on the size.
     * @param A First matrix in matrix multiplication.
     * @param B Second matrix in matrix multiplication and the matrix to transpose.
     * @return The result of the matrix multiplication-transpose.
     * @throws IllegalArgumentException If the shapes of the two matrices are not conducive to matrix multiplication-transpose.
     */
    public static Complex128[] dispatchTranspose(CMatrix A, CMatrix B) {
        ValidateParameters.ensureEquals(A.numCols, B.numCols);
        AlgorithmName algorithm = chooseAlgorithmRealComplexTranspose(A.shape);
        Complex128[] dest = new Complex128[A.numRows*A.numCols];

        switch(algorithm) {
            case MULT_T:
                DenseSemiringMatMultTranspose.multTranspose(
                        A.data, A.shape, B.data, B.shape, dest);
                break;
            case MULT_T_CONCURRENT:
                DenseSemiringMatMultTranspose.multTransposeConcurrent(
                        A.data, A.shape, B.data, B.shape, dest);
                break;
            default:
                DenseSemiringMatMultTranspose.multTransposeBlockedConcurrent(
                        A.data, A.shape, B.data, B.shape, dest);
        }

        return dest;
    }


    /**
     * Dynamically chooses the matrix multiplication-transpose algorithm to used based on the shape of the first matrix.
     * @param shape Shape of the first matrix.
     */
    public static AlgorithmName chooseAlgorithmRealComplexTranspose(Shape shape) {
        AlgorithmName algorithm;

        int rows = shape.get(0);

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

        int rows = shape.get(0);

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

        int rows = shape.get(0);

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

        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);

        // TODO: Extract constants to final variables.
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

        int rows = shape.get(0);

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

        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);

        // TODO: Extract constants to final variables.
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
        int rows = shape.get(0);

        if(rows<=600) return AlgorithmName.STANDARD_VECTOR;
        else return AlgorithmName.CONCURRENT_BLOCKED_VECTOR;
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

        double ratio = Math.abs(numRows-numCols) / Math.max(numRows, numCols);
        return 1 - ratio;
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
