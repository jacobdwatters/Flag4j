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


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * <p>This class contains several low-level methods for computing matrix-matrix multiplications between two
 * {@link Field} matrices. This includes transpose multiplications.</p>
 */
public final class DenseFieldMatrixMultiplication {

    private DenseFieldMatrixMultiplication() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the matrix multiplication between two dense matrices using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] standard(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        Field<T>[] dest = new Field[rows1*cols2];
        Arrays.fill(dest, src1[0].getZero());

        int src1Index, src2Index, destIndex, src1IndexStart, destIndexStart, end;

        for(int i=0; i<rows1; i++) {
            src1IndexStart = i*rows2;
            destIndexStart = i*cols2;

            for(int j=0; j<cols2; j++) {
                src2Index = j;
                src1Index = src1IndexStart;
                destIndex = destIndexStart + j;
                end = src1Index + rows2;
                Field<T> sum = dest[destIndex];

                while(src1Index<end) {
                    sum = sum.add(src1[src1Index++].mult((T) src2[src2Index]));
                    src2Index += cols2;
                }

                dest[destIndex] = sum;
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication between two dense matrices using the standard algorithm with j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] reordered(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Field<T>[] dest = new Field[rows1*cols2];
        Arrays.fill(dest, src1[0].getZero());

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
                    dest[destIndex] = dest[destIndex].add(src1[src1Index].mult((T) src2[src2Index++]));
                    destIndex++;
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two dense matrices using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] blocked(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        int cols1 = shape1.get(1);

        Field<T>[] dest = new Field[rows1 * cols2];
        Arrays.fill(dest, src1[0].getZero());

        int blockSize = Configurations.getBlockSize();
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
                            Field<T> sum = dest[destIndex];

                            while(src1Index < stopIndex) {
                                sum = sum.add(src1[src1Index++].mult((T) src2[src2Index]));
                                src2Index+=cols2;
                            }

                            dest[destIndex] = sum;
                        }
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two dense matrices using a blocked algorithm with the j-k loops
     * swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] blockedReordered(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        int cols1 = shape1.get(1);

        Field<T>[] dest = new Field[rows1*cols2];
        Arrays.fill(dest, src1[0].getZero());

        int blockSize = Configurations.getBlockSize();
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
                                dest[destIndex] = dest[destIndex].add(src1[src1Index].mult((T) src2[src2Index]));
                                destIndex++;
                                src2Index++;
                            }
                        }
                    }
                }
            }
        }

        return dest;
    }


    /**
     * Computes the matrix multiplication of two dense matrices using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] concurrentStandard(Field<T>[] src1, Shape shape1, Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Field<T>[] dest = new Field[rows1*cols2];
        Arrays.fill(dest, src1[0].getZero());

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*cols1;
                int destIndexStart = i*cols2;

                for(int j=0; j<cols2; j++) {
                    int src2Index = j;
                    int src1Index = src1IndexStart;
                    int destIndex = destIndexStart + j;
                    Field<T> sum = dest[destIndex];

                    for(int k=0; k<cols1; k++) {
                        sum = sum.add(src1[src1Index++].mult((T) src2[src2Index]));
                        src2Index += cols2;
                    }

                    dest[destIndex] = sum;
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two dense matrices using a concurrent implementation of the standard
     * matrix multiplication algorithm with j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] concurrentReordered(Field<T>[] src1, Shape shape1,Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        Field<T>[] dest = new Field[rows1*cols2];
        Arrays.fill(dest, src1[0].getZero());

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*rows2;
                int destIndexStart = i*cols2;

                for(int k=0; k<rows2; k++) {
                    int src2Index = k*cols2;
                    int destIndex = destIndexStart;
                    int end = src2Index + cols2;

                    while(src2Index<end) {
                        dest[destIndex] = dest[destIndex].add(src1[src1IndexStart + k].mult((T) src2[src2Index++]));
                        destIndex++;
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two dense matrices using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] concurrentBlocked(Field<T>[] src1, Shape shape1,Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Field<T>[] dest = new Field[rows1*cols2];
        Arrays.fill(dest, src1[0].getZero());
        int blockSize = Configurations.getBlockSize();

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
                                Field<T> sum = dest[destIndex];

                                while(src1Index < stopIndex) {
                                    sum = sum.add(src1[src1Index++].mult((T) src2[src2Index]));
                                    src2Index+=cols2;
                                }

                                dest[destIndex] = sum;
                            }
                        }
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the matrix multiplication of two dense matrices using a concurrent implementation of a blocked
     * algorithm with the j-k loops swapped.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] concurrentBlockedReordered(Field<T>[] src1, Shape shape1,Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        Field<T>[] dest = new Field[rows1*cols2];
        Arrays.fill(dest, src1[0].getZero());
        int blockSize = Configurations.getBlockSize();

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
                                    dest[destIndex] = dest[destIndex].add(src1[src1Index].mult((T) src2[src2Index]));
                                    destIndex++;
                                    src2Index++;
                                }
                            }
                        }
                    }
                }
            }
        });

        return dest;
    }


    /**
     * Computes the multiplication of a dense matrix with a dense vector using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] standardVector(Field<T>[] src1, Shape shape1,Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Field<T>[] dest = new Field[rows1];
        Arrays.fill(dest, src1[0].getZero());
        int src1Index, src2Index;

        for(int i=0; i<rows1; i++) {
            src1Index = i*cols1;
            src2Index = 0;
            Field<T> sum = dest[i];

            while(src2Index<rows2) {
                sum = sum.add(src1[src1Index++].mult((T) src2[src2Index++]));
            }

            dest[i] = sum;
        }

        return dest;
    }


    /**
     * Computes the multiplication of a dense matrix with a dense vector using a blocked algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] blockedVector(Field<T>[] src1, Shape shape1,Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Field<T>[] dest = new Field[rows1];
        Arrays.fill(dest, src1[0].getZero());
        int blockSize = Configurations.getBlockSize();
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
                    Field<T> sum = dest[i];

                    while(src2Index<kBound) {
                        sum = sum.add(src1[src1Index++].mult((T) src2[src2Index++]));
                    }

                    dest[i] = sum;
                }
            }
        }

        return dest;
    }


    /**
     * Computes the multiplication of a dense matrix with a dense vector using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] concurrentStandardVector(Field<T>[] src1, Shape shape1,Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Field<T>[] dest = new Field[rows1];
        Arrays.fill(dest, src1[0].getZero());

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1Index = i*cols1;
                int src2Index = 0;
                Field<T> sum = dest[i];

                while(src2Index<rows2) {
                    sum = sum.add(src1[src1Index++].mult((T) src2[src2Index++]));
                }

                dest[i] = sum;
            }
        });

        return dest;
    }


    /**
     * Computes the multiplication of a dense matrix with a dense vector using a concurrent implementation of a blocked
     * algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static <T extends Field<T>> Field<T>[] concurrentBlockedVector(Field<T>[] src1, Shape shape1,Field<T>[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        Field<T>[] dest = new Field[rows1];
        Arrays.fill(dest, src1[0].getZero());
        int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=startIdx; ii<endIdx; ii+=blockSize) {
                int iBound = Math.min(ii+blockSize, rows1);

                for(int kk=0; kk<rows2; kk+=blockSize) {
                    int kBound = Math.min(kk+blockSize, rows2);

                    // Multiply the current blocks
                    for(int i=ii; i<iBound; i++) {
                        int src1Index = i*cols1 + kk;
                        int src2Index = kk;
                        Field<T> sum = dest[i];

                        while(src2Index<kBound) {
                            sum = sum.add(src1[src1Index++].mult((T) src2[src2Index++]));
                        }

                        dest[i] = sum;
                    }
                }
            }
        });

        return dest;
    }
}
