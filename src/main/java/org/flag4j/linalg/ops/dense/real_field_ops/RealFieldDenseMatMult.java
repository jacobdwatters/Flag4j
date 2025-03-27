/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.linalg.ops.dense.real_field_ops;


import org.flag4j.numbers.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;

import java.util.Arrays;

/**
 * <p>This class contains several low level methods for computing dense matrix-matrix multiplications between one field matrix and one
 * real matrix.
 *
 * <p>Warning: This class does not perform any sanity checks on the input.
 */
public final class RealFieldDenseMatMult {

    private RealFieldDenseMatMult() {
        // Hide default constructor.
        
    }


    /**
     * Computes the matrix multiplication between a real dense matrix with a dense field matrix using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void standard(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        int src1Index, src2Index, destIndex, src1IndexStart, destIndexStart, end;

        for(int i=0; i<rows1; i++) {
            src1IndexStart = i*rows2;
            destIndexStart = i*cols2;

            for(int j=0; j<cols2; j++) {
                src2Index = j;
                src1Index = src1IndexStart;
                destIndex = destIndexStart + j;
                end = src1Index + rows2;
                T sum = dest[destIndex];

                while(src1Index<end) {
                    sum = sum.add(src2[src2Index].mult(src1[src1Index++]));
                    src2Index += cols2;
                }

                dest[destIndex] = sum;
            }
        }
    }


    /**
     * Computes the matrix multiplication between a real dense matrix with a dense field matrix using the standard algorithm with j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void reordered(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        int src2Index, destIndex, src1Start, destIndexStart, end;
        int src1Index;

        for(int i=0; i<rows1; i++) {
            src1Start = i*cols1;
            destIndexStart = i*cols2;

            for(int k=0; k<cols1; k++) {
                src1Index = src1Start+k;
                src2Index = k*cols2;
                destIndex = destIndexStart;
                end = src2Index + cols2;

                while(src2Index<end) {
                    dest[destIndex] = dest[destIndex].add(src2[src2Index++].mult(src1[src1Index]));
                    destIndex++;
                }
            }
        }
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void blocked(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        int cols1 = shape1.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        final int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int src1Start, destStart, stopIndex;
        int destIndex, src1Index, src2Index;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);
            for(int jj = 0; jj<cols2; jj+=blockSize) {
                jBound = Math.min(jj + blockSize, cols2);
                for(int kk = 0; kk<cols1; kk+=blockSize) {
                    kBound = Math.min(kk + blockSize, cols1);

                    // Multiply current blocks.
                    for(int i=ii; i<iBound; i++) {
                        src1Start = i*cols1 + kk;
                        stopIndex = src1Start+(kBound-kk);
                        destStart = i*cols2;

                        for (int j=jj; j<jBound; j++) {
                            destIndex = destStart + j;
                            src1Index = src1Start;
                            src2Index = kk*cols2 + j;
                            T sum = dest[destIndex];

                            while(src1Index < stopIndex) {
                                sum = sum.add(src2[src2Index].mult(src1[src1Index++]));
                                src2Index += cols2;
                            }

                            dest[destIndex] = sum;
                        }
                    }
                }
            }
        }
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a blocked algorithm with the j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void blockedReordered(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        int cols1 = shape1.get(1);
        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        final int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int destStart, src1Start, stopIndex;
        int destIndex, src1Index, src2Index;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);

            for(int kk = 0; kk<cols1; kk+=blockSize) {
                kBound = Math.min(kk + blockSize, cols1);

                for(int jj = 0; jj<cols2; jj+=blockSize) {
                    jBound = Math.min(jj + blockSize, cols2);

                    // Multiply current blocks.
                    for(int i=ii; i<iBound; i++) {
                        destStart = i*cols2;
                        src1Start = i*cols1;
                        stopIndex = destStart+jBound;

                        for (int k=kk; k<kBound; k++) {
                            destIndex = destStart + jj;
                            src1Index = src1Start + k;
                            src2Index = k*cols2 + jj;

                            while(destIndex<stopIndex) {
                                dest[destIndex] = dest[destIndex].add(src2[src2Index].mult(src1[src1Index]));
                                destIndex++;
                                src2Index++;
                            }
                        }
                    }
                }
            }
        }
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void concurrentStandard(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*cols1;
                int destIndexStart = i*cols2;

                for(int j=0; j<cols2; j++) {
                    int src2Index = j;
                    int src1Index = src1IndexStart;
                    int destIndex = destIndexStart + j;
                    T sum = dest[destIndex];

                    for(int k=0; k<cols1; k++) {
                        sum = sum.add(src2[src2Index].mult(src1[src1Index++]));
                        src2Index += cols2;
                    }

                    dest[destIndex] = sum;
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a concurrent implementation of the standard
     * matrix multiplication algorithm with j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void concurrentReordered(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*rows2;
                int destIndexStart = i*cols2;

                for(int k=0; k<rows2; k++) {
                    int src2Index = k*cols2;
                    int destIndex = destIndexStart;
                    int end = src2Index + cols2;

                    while(src2Index<end) {
                        dest[destIndex] = dest[destIndex].add(src2[src2Index++].mult(src1[src1IndexStart + k]));
                        destIndex++;
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void concurrentBlocked(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii + blockSize, rows1);

                for(int jj = 0; jj<cols2; jj+=blockSize) {
                    int jBound = Math.min(jj + blockSize, cols2);

                    for(int kk = 0; kk<cols1; kk+=blockSize) {
                        int kBound = Math.min(kk + blockSize, cols1);

                        // Multiply current blocks.
                        for(int i=ii; i<iBound; i++) {
                            int src1Start = i*cols1 + kk;
                            int stopIndex = src1Start+(kBound-kk);
                            int destStart = i*cols2;

                            for (int j=jj; j<jBound; j++) {
                                int destIndex = destStart + j;
                                int src1Index = src1Start;
                                int src2Index = kk*cols2 + j;
                                T sum = dest[destIndex];

                                while(src1Index < stopIndex) {
                                    sum = sum.add(src2[src2Index].mult(src1[src1Index++]));
                                    src2Index+=cols2;
                                }

                                dest[destIndex] = sum;
                            }
                        }
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a concurrent implementation of a blocked
     * algorithm with the j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void concurrentBlockedReordered(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii + blockSize, rows1);

                for(int kk = 0; kk<cols1; kk+=blockSize) {
                    int kBound = Math.min(kk + blockSize, cols1);

                    for(int jj = 0; jj<cols2; jj+=blockSize) {
                        int jBound = Math.min(jj + blockSize, cols2);

                        // Multiply current blocks.
                        for(int i=ii; i<iBound; i++) {
                            int destStart = i*cols2;
                            int src1Start = i*cols1;
                            int stopIndex = destStart+jBound;

                            for (int k=kk; k<kBound; k++) {
                                int destIndex = destStart + jj;
                                int src1Index = src1Start + k;
                                int src2Index = k*cols2 + jj;

                                while(destIndex<stopIndex) {
                                    dest[destIndex] = dest[destIndex].add(src2[src2Index].mult(src1[src1Index]));
                                    destIndex++;
                                    src2Index++;
                                }
                            }
                        }
                    }
                }
            }
        });
    }


    /**
     * Computes the multiplication of a real dense matrix with a dense field vector using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix-vector multiplication in (modified). Must have size length
     * {@code shape1.get(0)]}.
     */
    public static <T extends Field<T>> void standardVector(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);
        int src1Index, src2Index;

        for(int i=0; i<rows1; i++) {
            src1Index = i*cols1;
            src2Index = 0;
            T sum = dest[i];

            while(src2Index<rows2)
                sum = sum.add(src2[src2Index++].mult(src1[src1Index++]));

            dest[i] = sum;
        }
    }


    /**
     * Computes the multiplication of a real dense matrix with a dense field vector using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix-vector multiplication in (modified). Must have size length
     * {@code shape1.get(0)]}.
     */
    public static <T extends Field<T>> void blockedVector(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();
        int iBound, kBound;
        int src1Index, src2Index;

        // Blocked matrix-vector multiply
        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii+blockSize, rows1);

            for(int kk=0; kk<rows2; kk+=blockSize) {
                kBound = Math.min(kk+blockSize, rows2);

                // Multiply the current blocks
                for(int i=ii; i<iBound; i++) {
                    src1Index = i*cols1 + kk;
                    src2Index = kk;
                    T sum = dest[i];

                    while(src2Index<kBound) {
                        sum = sum.add(src2[src2Index++].mult(src1[src1Index++]));
                    }

                    dest[i] = sum;
                }
            }
        }
    }


    /**
     * Computes the multiplication of a real dense matrix with a dense field vector using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix-vector multiplication in (modified). Must have size length
     * {@code shape1.get(0)]}.
     */
    public static <T extends Field<T>> void concurrentStandardVector(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1Index = i*cols1;
                int src2Index = 0;
                T sum = dest[i];

                while(src2Index<rows2)
                    sum = sum.add(src2[src2Index++].mult(src1[src1Index++]));

                dest[i] = sum;
            }
        });
    }


    /**
     * Computes the multiplication of a real dense matrix with a dense field vector using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix-vector multiplication in (modified). Must have size length
     * {@code shape1.get(0)]}.
     */
    public static <T extends Field<T>> void concurrentBlockedVector(double[] src1, Shape shape1, T[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Arrays.fill(dest, (src2.length > 0) ? src2[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii+blockSize, rows1);

                for(int kk=0; kk<rows2; kk+=blockSize) {
                    int kBound = Math.min(kk+blockSize, rows2);

                    // Multiply the current blocks
                    for(int i=ii; i<iBound; i++) {
                        int src1Index = i*cols1 + kk;
                        int src2Index = kk;
                        T sum = dest[i];

                        while(src2Index<kBound)
                            sum = sum.add(src2[src2Index++].mult(src1[src1Index++]));

                        dest[i] = sum;
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication between a real dense matrix with a dense field matrix using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void standard(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);

        int src1Index, src2Index, destIndex, src1IndexStart, destIndexStart, end;

        for(int i=0; i<rows1; i++) {
            src1IndexStart = i*rows2;
            destIndexStart = i*cols2;

            for(int j=0; j<cols2; j++) {
                src2Index = j;
                src1Index = src1IndexStart;
                destIndex = destIndexStart + j;
                end = src1Index + rows2;
                T sum = dest[destIndex];

                while(src1Index<end) {
                    sum = sum.add(src1[src1Index++].mult(src2[src2Index]));
                    src2Index += cols2;
                }

                dest[destIndex] = sum;
            }
        }
    }


    /**
     * Computes the matrix multiplication between a real dense matrix with a dense field matrix using the standard algorithm with j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void reordered(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);

        int src2Index, destIndex, src1Start, destIndexStart, end;
        int src1Index;

        for(int i=0; i<rows1; i++) {
            src1Start = i*cols1;
            destIndexStart = i*cols2;

            for(int k=0; k<cols1; k++) {
                src1Index = src1Start+k;
                src2Index = k*cols2;
                destIndex = destIndexStart;
                end = src2Index + cols2;

                while(src2Index<end) {
                    dest[destIndex] = dest[destIndex].add(src1[src1Index].mult(src2[src2Index++]));
                    destIndex++;
                }
            }
        }
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void blocked(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        int cols1 = shape1.get(1);
        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);

        final int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int src1Start, destStart, stopIndex;
        int destIndex, src1Index, src2Index;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);
            for(int jj = 0; jj<cols2; jj+=blockSize) {
                jBound = Math.min(jj + blockSize, cols2);
                for(int kk = 0; kk<cols1; kk+=blockSize) {
                    kBound = Math.min(kk + blockSize, cols1);

                    // Multiply current blocks.
                    for(int i=ii; i<iBound; i++) {
                        src1Start = i*cols1 + kk;
                        stopIndex = src1Start+(kBound-kk);
                        destStart = i*cols2;

                        for (int j=jj; j<jBound; j++) {
                            destIndex = destStart + j;
                            src1Index = src1Start;
                            src2Index = kk*cols2 + j;
                            T sum = dest[destIndex];

                            while(src1Index < stopIndex) {
                                sum = sum.add(src1[src1Index++].mult(src2[src2Index]));
                                src2Index+=cols2;
                            }

                            dest[destIndex] = sum;
                        }
                    }
                }
            }
        }
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a blocked algorithm with the j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void blockedReordered(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        int cols1 = shape1.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);

        final int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int destStart, src1Start, stopIndex;
        int destIndex, src1Index, src2Index;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);

            for(int kk = 0; kk<cols1; kk+=blockSize) {
                kBound = Math.min(kk + blockSize, cols1);

                for(int jj = 0; jj<cols2; jj+=blockSize) {
                    jBound = Math.min(jj + blockSize, cols2);

                    // Multiply current blocks.
                    for(int i=ii; i<iBound; i++) {
                        destStart = i*cols2;
                        src1Start = i*cols1;
                        stopIndex = destStart+jBound;

                        for (int k=kk; k<kBound; k++) {
                            destIndex = destStart + jj;
                            src1Index = src1Start + k;
                            src2Index = k*cols2 + jj;

                            while(destIndex<stopIndex) {
                                dest[destIndex] = dest[destIndex].add(src1[src1Index].mult(src2[src2Index]));
                                destIndex++;
                                src2Index++;
                            }
                        }
                    }
                }
            }
        }
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void concurrentStandard(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*cols1;
                int destIndexStart = i*cols2;

                for(int j=0; j<cols2; j++) {
                    int src2Index = j;
                    int src1Index = src1IndexStart;
                    int destIndex = destIndexStart + j;
                    T sum = dest[destIndex];

                    for(int k=0; k<cols1; k++) {
                        sum = sum.add(src1[src1Index++].mult(src2[src2Index]));
                        src2Index += cols2;
                    }

                    dest[destIndex] = sum;
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a concurrent implementation of the standard
     * matrix multiplication algorithm with j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void concurrentReordered(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*rows2;
                int destIndexStart = i*cols2;

                for(int k=0; k<rows2; k++) {
                    int src2Index = k*cols2;
                    int destIndex = destIndexStart;
                    int end = src2Index + cols2;

                    while(src2Index<end) {
                        dest[destIndex] = dest[destIndex].add(src1[src1IndexStart + k].mult(src2[src2Index++]));
                        destIndex++;
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void concurrentBlocked(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii + blockSize, rows1);

                for(int jj = 0; jj<cols2; jj+=blockSize) {
                    int jBound = Math.min(jj + blockSize, cols2);

                    for(int kk = 0; kk<cols1; kk+=blockSize) {
                        int kBound = Math.min(kk + blockSize, cols1);

                        // Multiply current blocks.
                        for(int i=ii; i<iBound; i++) {
                            int src1Start = i*cols1 + kk;
                            int stopIndex = src1Start+(kBound-kk);
                            int destStart = i*cols2;

                            for (int j=jj; j<jBound; j++) {
                                int destIndex = destStart + j;
                                int src1Index = src1Start;
                                int src2Index = kk*cols2 + j;
                                T sum = dest[destIndex];

                                while(src1Index < stopIndex) {
                                    sum = sum.add(src1[src1Index++].mult(src2[src2Index]));
                                    src2Index+=cols2;
                                }

                                dest[destIndex] = sum;
                            }
                        }
                    }
                }
            }
        });
    }


    /**
     * Computes the matrix multiplication of a real dense matrix with a dense field matrix using a concurrent implementation of a blocked
     * algorithm with the j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix multiplication in (modified). Must have size length
     * {@code shape1.get(0)*shape2.get(1)}.
     */
    public static <T extends Field<T>> void concurrentBlockedReordered(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii + blockSize, rows1);

                for(int kk = 0; kk<cols1; kk+=blockSize) {
                    int kBound = Math.min(kk + blockSize, cols1);

                    for(int jj = 0; jj<cols2; jj+=blockSize) {
                        int jBound = Math.min(jj + blockSize, cols2);

                        // Multiply current blocks.
                        for(int i=ii; i<iBound; i++) {
                            int destStart = i*cols2;
                            int src1Start = i*cols1;
                            int stopIndex = destStart+jBound;

                            for (int k=kk; k<kBound; k++) {
                                int destIndex = destStart + jj;
                                int src1Index = src1Start + k;
                                int src2Index = k*cols2 + jj;

                                while(destIndex<stopIndex) {
                                    dest[destIndex] = dest[destIndex].add(src1[src1Index].mult(src2[src2Index]));
                                    destIndex++;
                                    src2Index++;
                                }
                            }
                        }
                    }
                }
            }
        });
    }


    /**
     * Computes the multiplication of a real dense matrix with a dense field vector using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix-vector multiplication in (modified). Must have size length
     * {@code shape1.get(0)]}.
     */
    public static <T extends Field<T>> void standardVector(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);
        int src1Index, src2Index;

        for(int i=0; i<rows1; i++) {
            src1Index = i*cols1;
            src2Index = 0;
            T sum = dest[i];

            while(src2Index<rows2)
                sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));

            dest[i] = sum;
        }
    }


    /**
     * Computes the multiplication of a real dense matrix with a dense field vector using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix-vector multiplication in (modified). Must have size length
     * {@code shape1.get(0)]}.
     */
    public static <T extends Field<T>> void blockedVector(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();
        int iBound, kBound;
        int src1Index, src2Index;

        // Blocked matrix-vector multiply
        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii+blockSize, rows1);

            for(int kk=0; kk<rows2; kk+=blockSize) {
                kBound = Math.min(kk+blockSize, rows2);

                // Multiply the current blocks
                for(int i=ii; i<iBound; i++) {
                    src1Index = i*cols1 + kk;
                    src2Index = kk;
                    T sum = dest[i];

                    while(src2Index<kBound)
                        sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));

                    dest[i] = sum;
                }
            }
        }
    }


    /**
     * Computes the multiplication of a real dense matrix with a dense field vector using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix-vector multiplication in (modified). Must have size length
     * {@code shape1.get(0)]}.
     */
    public static <T extends Field<T>> void concurrentStandardVector(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1Index = i*cols1;
                int src2Index = 0;
                T sum = dest[i];

                while(src2Index<rows2)
                    sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));

                dest[i] = sum;
            }
        });
    }


    /**
     * Computes the multiplication of a real dense matrix with a dense field vector using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @param dest Array to store result of the matrix-vector multiplication in (modified). Must have size length
     * {@code shape1.get(0)]}.
     */
    public static <T extends Field<T>> void concurrentBlockedVector(T[] src1, Shape shape1, double[] src2, Shape shape2, T[] dest) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Arrays.fill(dest, (src2.length > 0) ? src1[0].getZero() : null);
        final int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii+blockSize, rows1);

                for(int kk=0; kk<rows2; kk+=blockSize) {
                    int kBound = Math.min(kk+blockSize, rows2);

                    // Multiply the current blocks
                    for(int i=ii; i<iBound; i++) {
                        int src1Index = i*cols1 + kk;
                        int src2Index = kk;
                        T sum = dest[i];

                        while(src2Index<kBound)
                            sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));

                        dest[i] = sum;
                    }
                }
            }
        });
    }
}
