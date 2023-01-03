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

package com.flag4j.operations.dense.real;

import com.flag4j.Shape;
import com.flag4j.operations.concurrency.Configurations;
import com.flag4j.operations.concurrency.ThreadManager;
import com.flag4j.util.Axis2D;
import com.flag4j.util.ErrorMessages;
import com.flag4j.util.ShapeArrayChecks;

/**
 * This class contains several low level methods for computing matrix-matrix multiplications. This includes transpose
 * multiplications. <br>
 * <b>WARNING:</b> These methods do not perform any sanity checks.
 */
public class RealMatrixMultiplication {

    private RealMatrixMultiplication() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.utilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between two real dense matrices using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] standard(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];

        for(int i=0; i<rows1; i++) {
            for(int j=0; j<cols2; j++) {
                for(int k=0; k<rows2; k++) {
                    dest[i*cols2 + j] += src1[i*cols1 + k]*src2[k*cols2 + j];
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two real dense matrices using the standard algorithm with j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] reordered(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];

        for(int i=0; i<rows1; i++) {
            for(int k=0; k<rows2; k++) {
                for(int j=0; j<cols2; j++) {
                    dest[i*cols2 + j] += src1[i*cols1 + k]*src2[k*cols2 + j];
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] blocked(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];
        int bsize = Configurations.getBlockSize();

        // Blocked matrix multiply
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int jj=0; jj<cols2; jj += bsize) {
                for(int kk=0; kk<rows2; kk += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        for(int j=jj; j<jj+bsize && j<cols2; j++) {
                            for(int k=kk; k<kk+bsize && k<rows2; k++) {
                                dest[i*cols2 + j] += src1[i*cols1 + k]*src2[k*cols2 + j];
                            }
                        }
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a blocked algorithm with the j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] blockedReordered(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];
        int bsize = Configurations.getBlockSize();

        // Blocked matrix multiply
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int kk=0; kk<rows2; kk += bsize) {
                for(int jj=0; jj<cols2; jj += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        for(int k=kk; k<kk+bsize && k<rows2; k++) {
                            for(int j=jj; j<jj+bsize && j<cols2; j++) {
                                dest[i*cols2 + j] += src1[i*cols1 + k]*src2[k*cols2 + j];
                            }
                        }
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentStandard(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];

        ThreadManager.concurrentLoop(0, rows1, (i) -> {
            for(int j=0; j<cols2; j++) {
                for(int k=0; k<rows2; k++) {
                    dest[i*cols2 + j] += src1[i*cols1 + k]*src2[k*cols2 + j];
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of the standard
     * matrix multiplication algorithm with j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentReordered(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];

        ThreadManager.concurrentLoop(0, rows1, (i) -> {
            for(int k=0; k<rows2; k++) {
                for(int j=0; j<cols2; j++) {
                    dest[i*cols2 + j] += src1[i*cols1 + k]*src2[k*cols2 + j];
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentBlocked(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];
        int bsize = Configurations.getBlockSize();

        ThreadManager.concurrentLoop(0, rows1, bsize, (ii) -> {
            // Blocked matrix multiply
            for(int jj=0; jj<cols2; jj += bsize) {
                for(int kk=0; kk<rows2; kk += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        for(int j=jj; j<jj+bsize && j<cols2; j++) {
                            for(int k=kk; k<kk+bsize && k<rows2; k++) {
                                dest[i*cols2 + j] += src1[i*cols1 + k]*src2[k*cols2 + j];
                            }
                        }
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of a blocked
     * algorithm with the j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentBlockedReordered(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];
        int cols2 = shape2.dims[Axis2D.col()];

        double[] dest = new double[rows1*cols2];
        int bsize = Configurations.getBlockSize();

        ThreadManager.concurrentLoop(0, rows1, bsize, (ii) -> {
            // Blocked matrix multiply
            for(int kk=0; kk<rows2; kk += bsize) {
                for(int jj=0; jj<cols2; jj += bsize) {
                    // Multiply the current blocks
                    for(int i=ii; i<ii+bsize && i<rows1; i++) {
                        for(int k=kk; k<kk+bsize && k<rows2; k++) {
                            for(int j=jj; j<jj+bsize && j<cols2; j++) {
                                dest[i*cols2 + j] += src1[i*cols1 + k]*src2[k*cols2 + j];
                            }
                        }
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the multiplication of a real dense matrix with a real dense vector using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] standardVector(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];

        double[] dest = new double[rows1];

        for(int i=0; i<rows1; i++) {
            for(int k=0; k<rows2; k++) {
                dest[i] += src1[i*cols1 + k]*src2[k];
            }
        }

        return dest;
    }


    /**
     * Computes the multiplication of a real dense matrix with a real dense vector using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] blockedVector(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];

        double[] dest = new double[rows1];
        int bsize = Configurations.getBlockSize();

        // Blocked matrix-vector multiply
        for(int ii=0; ii<rows1; ii += bsize) {
            for(int kk=0; kk<rows2; kk += bsize) {
                // Multiply the current blocks
                for(int i=ii; i<ii+bsize && i<rows1; i++) {
                    for(int k=kk; k<kk+bsize && k<rows2; k++) {
                        dest[i] += src1[i*cols1 + k]*src2[k];
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the multiplication of a real dense matrix with a real dense vector using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentStandardVector(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];

        double[] dest = new double[rows1];

        ThreadManager.concurrentLoop(0, rows1, (i) -> {
            for(int k=0; k<rows2; k++) {
                dest[i] += src1[i*cols1 + k]*src2[k];
            }
        });

        return dest;
    }


    /**
     * Computes the multiplication of a real dense matrix with a real dense vector using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static double[] concurrentBlockedVector(double[] src1, Shape shape1, double[] src2, Shape shape2) {
        int rows1 = shape1.dims[Axis2D.row()];
        int cols1 = shape1.dims[Axis2D.col()];
        int rows2 = shape2.dims[Axis2D.row()];

        double[] dest = new double[rows1];
        int bsize = Configurations.getBlockSize();

        ThreadManager.concurrentLoop(0, rows1, bsize, (ii) -> {
            // Blocked matrix-vector multiply
            for(int kk=0; kk<rows2; kk += bsize) {
                // Multiply the current blocks
                for(int i=ii; i<ii+bsize && i<rows1; i++) {
                    for(int k=kk; k<kk+bsize && k<rows2; k++) {
                        dest[i] += src1[i*cols1 + k]*src2[k];
                    }
                }
            }
        });

        return dest;
    }
}
