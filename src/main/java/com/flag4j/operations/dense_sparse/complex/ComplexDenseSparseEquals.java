/*
 * MIT License
 *
 * Copyright (c) 2023 Jacob Watters
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

package com.flag4j.operations.dense_sparse.complex;

import com.flag4j.CMatrix;
import com.flag4j.SparseCMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.concurrency.util.ArrayUtils;
import com.flag4j.operations.concurrency.util.ErrorMessages;

import java.util.Arrays;

/**
 * This class provides methods for checking the equality of a complex dense tensor with a complex sparse tensor.
 */
public class ComplexDenseSparseEquals {

    private ComplexDenseSparseEquals() {
        // Hide constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if two real dense matrices are equal.
     * @param A First matrix.
     * @param B Second matrix.
     * @return True if the two matrices are element-wise equivalent.
     */
    public static boolean matrixEquals(CMatrix A, SparseCMatrix B) {
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
                equal = ArrayUtils.isZeros(entriesCopy);
            }

        } else {
            equal = false;
        }

        return equal;
    }
}
