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

package org.flag4j.linalg.operations.dense_sparse.coo.field_ops;

import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.*;
import org.flag4j.linalg.operations.common.field_ops.FieldProperties;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class provides methods for checking the equality of a dense field tensor with a sparse field tensor.
 */
public final class DenseCooFieldEquals {

    private DenseCooFieldEquals() {
        // Hide constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if two real dense matrices are equal.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static <T extends Field<T>> boolean matrixEquals(
            DenseFieldMatrixBase<?, ?, ?, ?, T> A,
            CooFieldMatrixBase<?, ?, ?, ?, T> B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            Field<T>[] entriesCopy = Arrays.copyOf(A.entries, A.entries.length);

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nnz; i++) {
                int rowIndex = B.rowIndices[i];
                int colIndex = B.colIndices[i];
                int idx = rowIndex*A.numCols + colIndex;

                if(!entriesCopy[idx].equals(B.entries[i])) {
                    equal = false;
                    break;
                }

                entriesCopy[idx] = A.getZeroElement();
            }

            if(equal) {
                // Now, if this matrix is equal to the sparse matrix, there should only be zeros left in the entriesStack
                equal = FieldProperties.isZeros(entriesCopy);
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
    public static <T extends Field<T>> boolean vectorEquals(
            DenseFieldVectorBase<?, ?, ?, T> src1, CooFieldVectorBase<?, ?, ?, ?, T> src2) {
        boolean equal = true;
        final T ZERO = (src1.size > 0) ? src1.entries[0].getZero() : null;

        if(src1.size == src2.size) {
            int index;
            Field<T>[] src1Copy = new Field[src1.size];
            System.arraycopy(src1.entries, 0, src1Copy, 0, src1.size);

            for(int i=0; i<src2.size; i++) {
                index = src2.indices[i];

                if(!src1.entries[index].equals(src2.entries[i])) {
                    equal = false;
                    break;
                } else {
                    src1Copy[index] = ZERO;
                }
            }

            if(equal) {
                // Now, if this vector is equal to the sparse vector, there should only be zeros left in the copy
                equal = FieldProperties.isZeros(src1Copy);
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
    public static <T extends Field<T>> boolean tensorEquals(
            DenseFieldTensorBase<?, ?, T> A, CooFieldTensorBase<?, ?, T> B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            Complex128[] entriesCopy = new Complex128[A.entries.length];
            System.arraycopy(A.entries, 0, entriesCopy, 0, A.entries.length);
            int entriesIndex;

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nnz; i++) {
                entriesIndex = A.shape.entriesIndex(B.indices[i]);

                if(!entriesCopy[entriesIndex].equals(B.entries[i])) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.entriesIndex(B.indices[i])] = Complex128.ZERO;
            }

            if(equal) {
                // Now, if this tensor is equal to the sparse tensor, there should only be zeros left in the entriesStack
                equal = FieldProperties.isZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }
}
