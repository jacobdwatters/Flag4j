/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.operations.dense.complex;

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.concurrency.Configurations;
import org.flag4j.concurrency.ThreadManager;
import org.flag4j.core.Shape;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;


/**
 * This class contains several low level methods for computing complex matrix-matrix multiplications. This includes transpose
 * multiplications. <br>
 * <b>WARNING:</b> These methods do not perform any sanity checks.
 */
public final class ComplexDenseMatrixMultiplication {

    private ComplexDenseMatrixMultiplication() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Computes the matrix multiplication between two real dense matrices using the standard algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static CNumber[] standard(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        int src1Index, src2Index, destIndex, src1IndexStart, destIndexStart, end;

        for(int i=0; i<rows1; i++) {
            src1IndexStart = i*rows2;
            destIndexStart = i*cols2;

            for(int j=0; j<cols2; j++) {
                src2Index = j;
                src1Index = src1IndexStart;
                destIndex = destIndexStart + j;
                end = src1Index + rows2;

                while(src1Index<end) {
                    dest[destIndex] = dest[destIndex].add(src1[src1Index++].mult(src2[src2Index]));
                    src2Index += cols2;
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
    public static CNumber[] reordered(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

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
    public static CNumber[] blocked(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        int cols1 = shape1.get(1);

        CNumber[] dest = new CNumber[rows1 * cols2];
        ArrayUtils.fill(dest, 0);

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

                            while(src1Index < stopIndex) {
                                dest[destIndex] = dest[destIndex].add(src1[src1Index++].mult(src2[src2Index]));
                                src2Index+=cols2;
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
    public static CNumber[] blockedReordered(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols2 = shape2.get(1);
        int cols1 = shape1.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

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
                                dest[destIndex] = dest[destIndex].add(src1[src1Index].mult(src2[src2Index]));
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
     * Computes the matrix multiplication of two real dense matrices using a concurrent implementation of the standard
     * matrix multiplication algorithm.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape fo the second matrix.
     * @return The result of matrix multiplying the two matrices.
     */
    public static CNumber[] concurrentStandard(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentLoop(0, rows1, (i) -> {
            int src1IndexStart = i*cols1;
            int destIndexStart = i*cols2;

            for(int j=0; j<cols2; j++) {
                int src2Index = j;
                int src1Index = src1IndexStart;
                int destIndex = destIndexStart + j;

                for(int k=0; k<cols1; k++) {
                    dest[destIndex] = dest[destIndex].add(src1[src1Index++].mult(src2[src2Index]));
                    src2Index += cols2;
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
    public static CNumber[] concurrentReordered(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentLoop(0, rows1, (i) -> {
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
    public static CNumber[] concurrentBlocked(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);
        int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentLoop(0, rows1, blockSize, (ii) -> {
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

                            while(src1Index < stopIndex) {
                                dest[destIndex] = dest[destIndex].add(src1[src1Index++].mult(src2[src2Index]));
                                src2Index+=cols2;
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
    public static CNumber[] concurrentBlockedReordered(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*cols2];
        ArrayUtils.fill(dest, 0);
        int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentLoop(0, rows1, blockSize, (ii) -> {
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
    public static CNumber[] standardVector(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);
        int src1Index, src2Index;

        for(int i=0; i<rows1; i++) {
            src1Index = i*cols1;
            src2Index = 0;

            while(src2Index<rows2) {
                dest[i] = dest[i].add(src1[src1Index++].mult(src2[src2Index++]));
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
    public static CNumber[] blockedVector(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);
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

                    while(src2Index<kBound) {
                        dest[i] = dest[i].add(src1[src1Index++].mult(src2[src2Index++]));
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
    public static CNumber[] concurrentStandardVector(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);

        ThreadManager.concurrentLoop(0, rows1, (i) -> {
            int src1Index = i*cols1;
            int src2Index = 0;

            while(src2Index<rows2) {
                dest[i] = dest[i].add(src1[src1Index++].mult(src2[src2Index++]));
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
    public static CNumber[] concurrentBlockedVector(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int cols1 = shape1.get(1);
        int rows2 = shape2.get(0);

        CNumber[] dest = new CNumber[rows1];
        ArrayUtils.fill(dest, 0);
        int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentLoop(0, rows1, blockSize, (ii) -> {
            int iBound = Math.min(ii+blockSize, rows1);

            for(int kk=0; kk<rows2; kk+=blockSize) {
                int kBound = Math.min(kk+blockSize, rows2);

                // Multiply the current blocks
                for(int i=ii; i<iBound; i++) {
                    int src1Index = i*cols1 + kk;
                    int src2Index = kk;

                    while(src2Index<kBound) {
                        dest[i] = dest[i].add(src1[src1Index++].mult(src2[src2Index++]));
                    }
                }
            }
        });

        return dest;
    }
}
