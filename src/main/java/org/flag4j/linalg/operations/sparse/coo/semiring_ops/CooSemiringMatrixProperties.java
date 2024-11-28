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

package org.flag4j.linalg.operations.sparse.coo.semiring_ops;

import org.flag4j.algebraic_structures.semirings.Semiring;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

/**
 * This class contains low level implementations for methods to evaluate certain properties of a sparse COO
 * {@link org.flag4j.algebraic_structures.semirings.Semiring} matrix. For example, if the matrix is symmetric.
 */
public final class CooSemiringMatrixProperties {

    private CooSemiringMatrixProperties() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(getClass()));
    }


    /**
     * Checks if a complex sparse COO matrix is the identity matrix.
     * @param shape Shape of the matrix.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     * @return True if the {@code src} matrix is the identity matrix. Otherwise, returns false.
     */
    public static <T extends Semiring<T>> boolean isIdentity(
            Shape shape, Semiring<T>[] entries, int[] rowIndices, int[] colIndices) {
        // Ensure the matrix is square and there are at least the same number of non-zero data as data on the diagonal.
        if(shape.get(0) != shape.get(1) || entries.length<shape.get(0)) return false;

        for(int i=0, size=entries.length; i<size; i++) {
            // Ensure value is 1 and on the diagonal.
            if(rowIndices[i] != i && rowIndices[i] != i && !entries[i].isOne()) {
                return false;
            } else if((rowIndices[i] != i || rowIndices[i] != i) && !entries[i].isZero()) {
                return false;
            }
        }

        return true; // If we make it to this point the matrix must be an identity matrix.
    }


    /**
     * Checks if a sparse matrix is symmetric.
     * @param shape Shape of the matrix.
     * @param entries Non-zero data of the matrix.
     * @param rowIndices Non-zero row indices of the matrix.
     * @param colIndices Non-zero column indices of the matrix.
     * @return True if the {@code src} matrix is hermitian. False otherwise.
     */
    public static <T extends Semiring<T>> boolean isSymmetric(
            Shape shape, Semiring<T>[] entries, int[] rowIndices, int[] colIndices) {
        if (shape.get(0) != shape.get(1)) return false; // Quick return for non-square matrix.

        List<Semiring<T>> entriesList = Arrays.asList(entries);
        List<Integer> rowIndicesList = IntStream.of(rowIndices).boxed().collect(Collectors.toList());
        List<Integer> colIndicesList = IntStream.of(colIndices).boxed().collect(Collectors.toList());

        boolean result = true;

        while(result && entriesList.size() > 0) {
            // Extract value of interest.
            Semiring<T> value = entriesList.remove(0);
            int row = rowIndicesList.remove(0);
            int col = colIndicesList.remove(0);

            // Find indices of first and last value whose row index matched the value of interests column index.
            int rowStart = rowIndicesList.indexOf(col);
            int rowEnd = rowIndicesList.lastIndexOf(col);

            if(rowStart == -1) {
                // Then no non-zero value was found.
                result = value.equals(0);
            } else {
                // At least one entry has a row-index matching the specified column index.
                List<Integer> colIdxRange = colIndicesList.subList(rowStart, rowEnd + 1);

                // Search for element whose column index matches the specified row index
                int idx = colIdxRange.indexOf(row);

                if(idx == -1) {
                    // Then no non-zero value was found.
                    result = value.equals(0);
                } else {
                    // Check that value with opposite row/column indices is equal.
                    result = value.equals(entriesList.get(idx + rowStart));

                    // Remove the value and the indices.
                    entriesList.remove(idx + rowStart);
                    rowIndicesList.remove(idx + rowStart);
                    colIndicesList.remove(idx + rowStart);
                }
            }
        }

        return result;
    }
}
