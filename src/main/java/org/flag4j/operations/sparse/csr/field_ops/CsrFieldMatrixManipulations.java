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

package org.flag4j.operations.sparse.csr.field_ops;

import org.flag4j.core_temp.arrays.sparse.CsrFieldMatrixBase;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * This utility class contains methods for manipulating CSR {@link org.flag4j.core_temp.structures.fields.Field} matrices.
 */
public final class CsrFieldMatrixManipulations {


    private CsrFieldMatrixManipulations() {
        // Hide default constructor for utility method.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Swaps two rows in a sparse CSR matrix. This is done in place.
     * @param src The matrix to swap rows within, done in place.
     * @param rowIdx1 Index of the first row to swap.
     * @param rowIdx2 Index of the second row to swap.
     * @throws IndexOutOfBoundsException If either {@code rowIdx1} or {@code rowIdx2} is out of bounds of the rows of this matrix.
     */
    public static void swapRows(CsrFieldMatrixBase src, int rowIdx1, int rowIdx2) {
        if(rowIdx1 == rowIdx2) return;
        else if(rowIdx1 > rowIdx2) {
            // ensure the second index is larger than the first.
            int temp = rowIdx1;
            rowIdx1 = rowIdx2;
            rowIdx2 = temp;
        }

        // Get range for values in the given rows.
        int start1 = src.rowPointers[rowIdx1];
        int end1 = src.rowPointers[rowIdx1+1];
        int nnz1 = end1-start1;  // Number of non-zero entries in the first row to swap.

        int start2 = src.rowPointers[rowIdx2];
        int end2 = src.rowPointers[rowIdx2+1];
        int nnz2 = end2-start2;  // Number of non-zero entries in the second row to swap.

        int destPos = 0;

        double[] updatedEntries = new double[end2-start1];
        int[] updatedColIndices = new int[end2-start1];

        // Copy entries from second row in swap.
        System.arraycopy(src.entries, start2, updatedEntries, destPos, nnz2);
        System.arraycopy(src.colIndices, start2, updatedColIndices, destPos, nnz2);

        // Copy entries between rows.
        destPos += nnz2;
        System.arraycopy(src.entries, end1, updatedEntries, destPos, start2-end1);
        System.arraycopy(src.colIndices, end1, updatedColIndices, destPos, start2-end1);

        // Copy entries from first row in swap.
        destPos += (start2-end1);
        System.arraycopy(src.entries, start1, updatedEntries, destPos, nnz1);
        System.arraycopy(src.colIndices, start1, updatedColIndices, destPos, nnz1);

        // Update the row pointers.
        int diff = nnz2-nnz1;  // Difference in number of entries.
        for(int i=rowIdx1+1; i<=rowIdx2; i++)
            src.rowPointers[i] += diff;

        // Copy updated arrays_old to this tensors' storage.
        System.arraycopy(updatedEntries, 0, src.entries, start1, updatedEntries.length);
        System.arraycopy(updatedColIndices, 0, src.colIndices, start1, updatedEntries.length);
    }


    /**
     * Swaps two columns in a sparse CSR matrix. This is done in place.
     * @param src The matrix to swap rows within, done in place.
     * @param colIdx1 Index of the first column to swap.
     * @param colIdx2 Index of the second column to swap.
     * @throws IndexOutOfBoundsException If either {@code colIndex1} or {@code colIndex2} is out of bounds of the columns of this
     * matrix.
     */
    public static <T extends Field<T>> void swapCols(CsrFieldMatrixBase<?, ?, ?, T> src, int colIdx1, int colIdx2) {
        if(colIdx1 == colIdx2) return;

        // Ensure colIndex1 < colIndex2 for simplicity
        if(colIdx1 > colIdx2) {
            int temp = colIdx1;
            colIdx1 = colIdx2;
            colIdx2 = temp;
        }

        // Traverse each row to find and swap the columns.
        for(int i=0; i<src.numRows; i++) {
            int rowStart = src.rowPointers[i];
            int rowEnd = src.rowPointers[i+1];

            int pos1;
            int pos2;

            pos1 = Arrays.binarySearch(src.colIndices, rowStart, rowEnd, colIdx1);
            pos2 = Arrays.binarySearch(src.colIndices, rowStart, rowEnd, colIdx2);

            if(pos1 >= 0 && pos2 >= 0) {
                // Both columns contain a non-zero value in this row.
                ArrayUtils.swap(src.entries, pos1, pos2);
            } else if(pos1 >= 0) {
                // Only first column contains non-zero value.
                int newPos = -pos2-1;
                if(newPos==rowEnd) newPos--;
                moveAndShiftLeft(src, colIdx2, pos1, newPos);
            } else if(pos2 >= 0) {
                // Only second column contains non-zero value.
                int newPos = -pos1-1;
                if(newPos==rowEnd) newPos--;
                moveAndShiftRight(src, colIdx1, pos2, newPos);
            }
        }
    }


    /**
     * Moves a non-zero value in a row of a CSR matrix to a new column to the left of its current column. The new column is assumed
     * to contain a zero. To accommodate this move, all entries between the columns are shifted right within the non-zero entries array.
     * @param src Source matrix to perform the move and shift within (modified).
     * @param newColIdx New column for the value to be moved to within the row.
     * @param currPos Current index of the value within the non-zero entries of {@code src} (assumed to be in the same row as {@code
     * newPos}).
     * @param newPos New index for the value to be moved to within the non-zero entries of {@code src} (assumed to be in the same
     * row as {@code currPos}).
     */
    private static <T extends Field<T>> void moveAndShiftRight(CsrFieldMatrixBase<?, ?, ?, T> src,
                                                               int newColIdx, int currPos, int newPos) {
        T value = src.entries[currPos];  // Extract the non-zero value.

        // Shift entries in row to right.
        for(int j=currPos; j>newPos; j--) {
            src.entries[j] = src.entries[j-1];
            src.colIndices[j] = src.colIndices[j-1];
        }

        src.entries[newPos] = value;  // Move non-zero value to new location.
        src.colIndices[newPos] = newColIdx;  // Update column index for the value.
    }


    /**
     * Moves a non-zero value in a row of a CSR matrix to a new column to the right of its current column. The new column is assumed
     * to contain a zero. To accommodate this move, all entries between the columns are shifted left within the non-zero entries array.
     * @param src Source matrix to perform the move and shift within (modified).
     * @param newColIdx New column for the value to be moved to within the row.
     * @param currPos Current index of the value within the non-zero entries of {@code src} (assumed to be in the same row as {@code
     * newPos}).
     * @param newPos New index for the value to be moved to within the non-zero entries of {@code src} (assumed to be in the same
     * row as {@code currPos}).
     */
    private static <T extends Field<T>> void moveAndShiftLeft(CsrFieldMatrixBase<?, ?, ?, T> src, int newColIdx, int currPos,
                                                       int newPos) {
        T value = src.entries[currPos];  // Extract the non-zero value.

        // Shift entries in row to left.
        for(int j=currPos; j<newPos; j++) {
            src.entries[j] = src.entries[j+1];
            src.colIndices[j] = src.colIndices[j+1];
        }

        src.entries[newPos] = value;  // Move non-zero value to new location.
        src.colIndices[newPos] = newColIdx;  // Update column index for the value.
    }
}
