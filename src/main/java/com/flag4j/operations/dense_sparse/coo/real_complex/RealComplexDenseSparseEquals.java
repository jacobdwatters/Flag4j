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

package com.flag4j.operations.dense_sparse.coo.real_complex;

import com.flag4j.*;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.common.complex.ComplexProperties;
import com.flag4j.operations.common.real.RealProperties;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class contains methods for checking the equality of real dense/sparse and complex dense/sparse tensors.
 */
public class RealComplexDenseSparseEquals {

    private RealComplexDenseSparseEquals() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if a real dense vector is equal to a complex sparse vector equals.
     * @param src1 Entries of dense vector.
     * @param src2 Non-zero Entries of sparse vector.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param sparseSize Size of the sparse vector.
     * @return True if the two vectors are equal. Returns false otherwise.
     */
    public static boolean vectorEquals(double[] src1, CNumber[] src2, int[] indices, int sparseSize) {
        boolean equal = true;

        if(src1.length==sparseSize) {
            int index;
            double[] src1Copy = Arrays.copyOf(src1, src1.length);

            for(int i=0; i<src2.length; i++) {
                index = indices[i];

                if(!src2[i].equals(src1[index])) {
                    equal=false;
                    break;

                } else {
                    src1Copy[index] = 0;
                }
            }

            if(equal) {
                // Now, if this vector is equal to the sparse vector, there should only be zeros left in the entriesStack
                equal = RealProperties.isZeros(src1Copy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a complex dense vector is equal to a real sparse vector.
     * @param src1 Entries of dense vector.
     * @param src2 Non-zero Entries of sparse vector.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param sparseSize Size of the sparse vector.
     * @return True if the two vectors are equal. Returns false otherwise.
     */
    public static boolean vectorEquals(CNumber[] src1, double[] src2, int[] indices, int sparseSize) {
        boolean equal = true;

        if(src1.length == sparseSize) {
            int index;
            CNumber[] src1Copy = new CNumber[src1.length];
            ArrayUtils.copy2CNumber(src1, src1Copy);

            for(int i=0; i<indices.length; i++) {
                index = indices[i];

                if(!src1[index].equals(src2[i])) {
                    equal=false;
                    break;

                } else {
                    src1Copy[index] = new CNumber();
                }
            }

            if(equal) {
                // Now, if this vector is equal to the sparse vector, there should only be zeros left in the entriesStack
                equal = ComplexProperties.isZeros(src1Copy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a real dense matrix is equal to a sparse complex matrix.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(Matrix A, CooCMatrix B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            double[] entriesCopy = Arrays.copyOf(A.entries, A.entries.length);

            int rowIndex, colIndex;
            int entriesIndex;

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nonZeroEntries(); i++) {
                rowIndex = B.rowIndices[i];
                colIndex = B.colIndices[i];
                entriesIndex = A.shape.entriesIndex(rowIndex, colIndex);

                if(entriesCopy[entriesIndex] != B.entries[i].re || B.entries[i].im != 0) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.entriesIndex(rowIndex, colIndex)] = 0;
            }

            if(equal) {
                // Now, if this matrix is equal to the sparse matrix, there should only be zeros left in the entriesStack
                equal = RealProperties.isZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a dense complex matrix is equal to a real sparse matrix.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(CMatrix A, CooMatrix B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            CNumber[] entriesCopy = Arrays.copyOf(A.entries, A.entries.length);

            int rowIndex, colIndex;
            int entriesIndex;

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nonZeroEntries(); i++) {
                rowIndex = B.rowIndices[i];
                colIndex = B.colIndices[i];
                entriesIndex = A.shape.entriesIndex(rowIndex, colIndex);

                if(!entriesCopy[entriesIndex].equals(B.entries[i])) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.entriesIndex(rowIndex, colIndex)] = new CNumber();
            }

            if(equal) {
                // Now, if this matrix is equal to the sparse matrix, there should only be zeros left in the entriesStack
                equal = ComplexProperties.isZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a real dense tensor is equal to a complex sparse tensor.
     * @param A Real dense tensor.
     * @param B Complex sparse tensor.
     * @return True if the two tensors are element-wise equivalent.
     */
    public static boolean tensorEquals(Tensor A, CooCTensor B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            double[] entriesCopy = Arrays.copyOf(A.entries, A.entries.length);
            int entriesIndex;

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nonZeroEntries(); i++) {
                entriesIndex = A.shape.entriesIndex(B.indices[i]);

                if(entriesCopy[entriesIndex] != B.entries[i].re || B.entries[i].im != 0) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.entriesIndex(B.indices[i])] = 0;
            }

            if(equal) {
                // Now, if this tensor is equal to the sparse tensor, there should only be zeros left in the entriesStack
                equal = RealProperties.isZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a complex dense tensor is equal to a real sparse tensor.
     * @param A Complex dense tensor.
     * @param B Real sparse tensor.
     * @return True if the two tensors are element-wise equivalent.
     */
    public static boolean tensorEquals(CTensor A, CooTensor B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            CNumber[] entriesCopy = new CNumber[A.entries.length];
            ArrayUtils.copy2CNumber(entriesCopy, A.entries);
            int entriesIndex;

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nonZeroEntries(); i++) {
                entriesIndex = A.shape.entriesIndex(B.indices[i]);

                if(entriesCopy[entriesIndex].re != B.entries[i] || entriesCopy[entriesIndex].im != 0) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.entriesIndex(B.indices[i])] = new CNumber();
            }

            if(equal) {
                // Now, if this tensor is equal to the sparse tensor, there should only be zeros left in the entriesStack
                equal = ComplexProperties.isZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }
}
