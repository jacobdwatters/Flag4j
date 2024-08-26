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

package org.flag4j.operations_old.dense_sparse.coo.complex;

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CTensorOld;
import org.flag4j.arrays_old.sparse.CooCMatrixOld;
import org.flag4j.arrays_old.sparse.CooCTensorOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.operations_old.common.complex.ComplexProperties;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class provides methods for checking the equality of a complex dense tensor with a complex sparse tensor.
 */
public final class ComplexDenseSparseEquals {

    private ComplexDenseSparseEquals() {
        // Hide constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if two real dense matrices are equal.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(CMatrixOld A, CooCMatrixOld B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            CNumber[] entriesCopy = Arrays.copyOf(A.entries, A.entries.length);

            int rowIndex, colIndex;
            
            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nonZeroEntries(); i++) {
                rowIndex = B.rowIndices[i];
                colIndex = B.colIndices[i];
                int idx = rowIndex*A.numCols + colIndex;

                if(!entriesCopy[idx].equals(B.entries[i])) {
                    equal = false;
                    break;
                }

                entriesCopy[idx] = CNumber.ZERO;
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
     * Checks if a complex dense vector is equal to a complex sparse vector.
     * @param src1 Entries of dense vector.
     * @param src2 Non-zero Entries of sparse vector.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param sparseSize Size of the sparse vector.
     * @return True if the two vectors are equal. Returns false otherwise.
     */
    public static boolean vectorEquals(CNumber[] src1, CNumber[] src2, int[] indices, int sparseSize) {
        boolean equal = true;

        if(src1.length==sparseSize) {
            int index;
            CNumber[] src1Copy = new CNumber[src1.length];
            System.arraycopy(src1, 0, src1Copy, 0, src1.length);

            for(int i=0; i<src2.length; i++) {
                index = indices[i];

                if(!src1[index].equals(src2[i])) {
                    equal = false;
                    break;

                } else {
                    src1Copy[index] = CNumber.ZERO;
                }
            }

            if(equal) {
                // Now, if this vector is equal to the sparse vector, there should only be zeros left in the copy
                equal = ComplexProperties.isZeros(src1Copy);
            }

        } else {
            equal = false;
        }

        return equal;
    }


    /**
     * Checks if a complex dense tensor is equal to a complex sparse tensor.
     * @param A Complex dense tensor.
     * @param B Complex sparse tensor.
     * @return True if the two tensors are element-wise equivalent.
     */
    public static boolean tensorEquals(CTensorOld A, CooCTensorOld B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            CNumber[] entriesCopy = new CNumber[A.entries.length];
            System.arraycopy(A.entries, 0, entriesCopy, 0, A.entries.length);
            int entriesIndex;

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nonZeroEntries(); i++) {
                entriesIndex = A.shape.entriesIndex(B.indices[i]);

                if(!entriesCopy[entriesIndex].equals(B.entries[i])) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.entriesIndex(B.indices[i])] = CNumber.ZERO;
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
