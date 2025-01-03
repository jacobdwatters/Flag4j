/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.linalg.ops.sparse.coo;


import org.flag4j.arrays.Shape;
import org.flag4j.util.ArrayUtils;

import java.util.Arrays;
import java.util.List;

/**
 * Utility class for computing certain properties of COO matrices. The methods in this class are agnostic to the type of COO matrix.
 */
public final class CooProperties {

    private CooProperties() {
        // Hide default constructor for utility class.
    }


    /**
     * Checks if a sparse COO matrix is symmetric.
     * @param src Matrix to check if it is the hermitian matrix.
     * @return True if the {@code src} matrix is hermitian. False otherwise.
     */
    public static <T> boolean isSymmetric(Shape shape, T[] data, int[] rowIndices, int[] colIndices) {
        // Check that the matrix is square.
        boolean result = shape.get(0) == shape.get(1);

        List<T> dataList = Arrays.asList(data);
        List<Integer> rowIdxList = ArrayUtils.toArrayList(rowIndices);
        List<Integer> colIdxList = ArrayUtils.toArrayList(colIndices);;

        T value;
        int row;
        int col;

        while(result && dataList.size() > 0) {
            // Extract value of interest.
            value = dataList.remove(0);
            row = rowIdxList.remove(0);
            col = colIdxList.remove(0);

            // Find indices of first and last value whose row index matched the value of interests column index.
            int rowStart = rowIdxList.indexOf(col);
            int rowEnd = rowIdxList.lastIndexOf(col);

            if(rowStart == -1) {
                // Then no non-zero value was found.
                result = value.equals(0);
            } else {
                // At least one entry has a row-index matching the specified column index.
                List<Integer> colIdxRange = colIdxList.subList(rowStart, rowEnd + 1);

                // Search for element whose column index matches the specified row index
                int idx = colIdxRange.indexOf(row);

                if(idx == -1) {
                    // Then no non-zero value was found.
                    result = value.equals(0);
                } else {
                    // Check that value with opposite row/column indices is equal.
                    result = value.equals(dataList.get(idx + rowStart));

                    // Remove the value and the indices.
                    dataList.remove(idx + rowStart);
                    rowIdxList.remove(idx + rowStart);
                    colIdxList.remove(idx + rowStart);
                }
            }
        }

        return result;
    }
}
