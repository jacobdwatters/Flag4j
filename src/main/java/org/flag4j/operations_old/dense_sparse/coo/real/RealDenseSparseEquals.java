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

package org.flag4j.operations_old.dense_sparse.coo.real;

import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.arrays_old.dense.TensorOld;
import org.flag4j.arrays_old.sparse.CooMatrixOld;
import org.flag4j.arrays_old.sparse.CooTensorOld;
import org.flag4j.operations_old.common.real.RealProperties;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class contains methods for checking the equality of a real dense and real sparse tensors.
 */
public class RealDenseSparseEquals {

    private RealDenseSparseEquals() {
        // Hide default constructor.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if a real dense vector is equal to a sparse vector equals.
     * @param src1 Entries of dense vector.
     * @param src2 Non-zero Entries of sparse vector.
     * @param indices Indices of non-zero entries in the sparse vector.
     * @param sparseSize Size of the sparse vector.
     * @return True if the two matrices are equal. Returns false otherwise.
     */
    public static boolean vectorEquals(double[] src1, double[] src2, int[] indices, int sparseSize) {
        boolean equal = true;

        if(src1.length==sparseSize) {
            int index;
            double[] src1Copy = Arrays.copyOf(src1, src1.length);

            for(int i=0; i<src2.length; i++) {
                index = indices[i];

                if(src1[index]!=src2[i]) {
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
     * Checks if a real dense matrix is equal to a real sparse matrix.
     * @param A Real dense matrix.
     * @param B Real sparse matrix.
     * @return True if the two matrices are element-wise equivalent (as if both were dense).
     */
    public static boolean matrixEquals(MatrixOld A, CooMatrixOld B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            double[] entriesCopy = Arrays.copyOf(A.entries, A.entries.length);

            int rowIndex, colIndex;
            int entriesIndex;

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nonZeroEntries(); i++) {
                rowIndex = B.rowIndices[i];
                colIndex = B.colIndices[i];
                int idx = rowIndex*A.numCols + colIndex;

                if(entriesCopy[idx] != B.entries[i]) {
                    equal = false;
                    break;
                }

                entriesCopy[idx] = 0;
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
     * Checks if a real dense tensor is equal to a real sparse tensor.
     * @param A Real dense tensor.
     * @param B Real sparse tensor.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean tensorEquals(TensorOld A, CooTensorOld B) {
        boolean equal = true;

        if(A.shape.equals(B.shape)) {
            double[] entriesCopy = Arrays.copyOf(A.entries, A.entries.length);

            int entriesIndex;

            // Remove all nonZero entries from the entries of this matrix.
            for(int i=0; i<B.nonZeroEntries(); i++) {
                entriesIndex = A.shape.entriesIndex(B.indices[i]);

                if(entriesCopy[entriesIndex] != B.entries[i]) {
                    equal = false;
                    break;
                }

                entriesCopy[A.shape.entriesIndex(B.indices[i])] = 0;
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
}
