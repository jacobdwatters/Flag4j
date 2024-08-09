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
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;


/**
 * This class contains several low level methods for computing matrix-matrix multiplications with a transpose for two
 * dense complex matrices. <br>
 * <b>WARNING:</b> These methods do not perform any sanity checks.
 */
public final class ComplexDenseMatrixMultTranspose {

    private ComplexDenseMatrixMultTranspose() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix. This method may be significantly faster than computing the
     * transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @return The result of multiplying the first matrix with the transpose of the second matrix.
     */
    public static CNumber[] multTranspose(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*rows2]; // Since second matrix is transposed, its columns will become rows.
        Arrays.fill(dest, CNumber.ZERO);

        int src1Index, src2Index, destIndex, src1IndexStart, destIndexStart, end;

        for(int i=0; i<rows1; i++) {
            src1IndexStart = i*cols2;
            destIndexStart = i*rows2;
            end = src1IndexStart + cols2;

            for(int j=0; j<rows2; j++) {
                src1Index = src1IndexStart;
                src2Index = j*cols2;
                destIndex = destIndexStart + j;
                CNumber sum = dest[destIndex];

                while(src1Index<end) {
                    sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));
                }

                dest[destIndex] = sum;
            }
        }

        return dest;
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix using a blocked algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @return The result of multiplying the first matrix with the transpose of the second matrix.
     */
    public static CNumber[] multTransposeBlocked(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*rows2];
        Arrays.fill(dest, CNumber.ZERO);

        int blockSize = Configurations.getBlockSize();
        int iBound, jBound, kBound;
        int src1Start, destStart, end;
        int destIndex, src1Index, src2Index;

        for(int ii=0; ii<rows1; ii+=blockSize) {
            iBound = Math.min(ii + blockSize, rows1);

            for(int jj = 0; jj<rows2; jj+=blockSize) {
                jBound = Math.min(jj + blockSize, rows2);

                for(int kk = 0; kk<cols2; kk+=blockSize) {
                    kBound = Math.min(kk + blockSize, cols2);

                    // Multiply the blocks
                    for(int i=ii; i<iBound; i++) {
                        destStart = i*rows2;
                        src1Start = i*cols2 + kk;
                        end = src1Start + kBound - kk;

                        for(int j=jj; j<jBound; j++) {
                            destIndex = destStart + j;
                            src1Index = src1Start;
                            src2Index = j*cols2 + kk;
                            CNumber sum = dest[destIndex];

                            while(src1Index<end) {
                                sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));
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
     * Multiplies a matrix to the transpose of a second matrix using a concurrent algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @return The result of multiplying the first matrix with the transpose of the second matrix.
     */
    public static CNumber[] multTransposeConcurrent(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*rows2]; // Since second matrix is transposed, its columns will become rows.
        Arrays.fill(dest, CNumber.ZERO);

        ThreadManager.concurrentOperation(rows1, (startIdx, endIdx) -> {
            for(int i=startIdx; i<endIdx; i++) {
                int src1IndexStart = i*cols2;
                int destIndexStart = i*rows2;
                int end = src1IndexStart + cols2;

                for(int j=0; j<rows2; j++) {
                    int src1Index = src1IndexStart;
                    int src2Index = j*cols2;
                    int destIndex = destIndexStart + j;
                    CNumber sum = dest[destIndex];

                    while(src1Index<end) {
                        sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));
                    }

                    dest[destIndex] = sum;
                }
            }
        });

        return dest;
    }


    /**
     * Multiplies a matrix to the transpose of a second matrix using a concurrent implementation of a blocked algorithm.
     * This method may be significantly faster than computing the transpose and multiplication in two steps.
     * @param src1 Entries of the first matrix.
     * @param shape1 Shape of the first matrix.
     * @param src2 Entries of the second matrix.
     * @param shape2 Shape of the second matrix.
     * @return The result of multiplying the first matrix with the transpose of the second matrix.
     */
    public static CNumber[] multTransposeBlockedConcurrent(CNumber[] src1, Shape shape1, CNumber[] src2, Shape shape2) {
        int rows1 = shape1.get(0);
        int rows2 = shape2.get(0);
        int cols2 = shape2.get(1);

        CNumber[] dest = new CNumber[rows1*rows2];
        Arrays.fill(dest, CNumber.ZERO);

        int blockSize = Configurations.getBlockSize();

        ThreadManager.concurrentBlockedOperation(rows1, blockSize, (startIdx, endIdx) -> {
            for(int ii=0; ii<rows2; ii+=blockSize) {
                int iBound = Math.min(ii + blockSize, rows1);

                for(int jj = 0; jj<rows2; jj+=blockSize) {
                    int jBound = Math.min(jj + blockSize, rows2);

                    for(int kk = 0; kk<cols2; kk+=blockSize) {
                        int kBound = Math.min(kk + blockSize, cols2);

                        // Multiply the blocks
                        for(int i=ii; i<iBound; i++) {
                            int destStart = i*rows2;
                            int src1Start = i*cols2 + kk;
                            int end = src1Start + kBound - kk;

                            for(int j=jj; j<jBound; j++) {
                                int destIndex = destStart + j;
                                int src1Index = src1Start;
                                int src2Index = j*cols2 + kk;
                                CNumber sum = dest[destIndex];

                                while(src1Index<end) {
                                    sum = sum.add(src1[src1Index++].mult(src2[src2Index++]));
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
}
