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

package org.flag4j.linalg.operations.sparse.coo.real;

import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.util.ErrorMessages;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * This class contains low level implementations for methods to evaluate certain properties of a real sparse matrix.
 * (i.e. if the matrix is symmetric).
 */
public final class RealSparseMatrixProperties {

    private RealSparseMatrixProperties() {
        // Hide public constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if a real sparse matrix is the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return True if the {@code src} matrix is the identity matrix. Otherwise, returns false.
     */
    public static boolean isIdentity(CooMatrix src) {
        // Ensure the matrix is square and there are at least the same number of non-zero entries as entries on the diagonal.
        if(!src.isSquare() || src.entries.length<src.numRows) return false;

        for(int i=0, size=src.entries.length; i<size; i++) {
            // Ensure value is 1 and on the diagonal.
            if(src.rowIndices[i] != i && src.rowIndices[i] != i && src.entries[i] != 1) {
                return false;
            } else if((src.rowIndices[i] != i || src.rowIndices[i] != i) && src.entries[i] != 0) {
                return false;
            }
        }

        return true; // If we make it to this point the matrix must be an identity matrix.
    }


    /**
     * Checks if a real sparse matrix is close to the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return True if the {@code src} matrix is the identity matrix. Otherwise, returns false.
     */
    public static boolean isCloseToIdentity(CooMatrix src) {
        // Ensure the matrix is square and there are the same number of non-zero entries as entries on the diagonal.
        boolean result = src.isSquare() && src.entries.length==src.numRows;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.001E-5;
        double nonDiagTol = 1e-08;

        for(int i=0; i<src.entries.length; i++) {
            if(src.rowIndices[i] == i && src.colIndices[i] == i && Math.abs(src.entries[i]-1) > diagTol ) {
                return false; // Diagonal value is not close to one.
            } else if((src.rowIndices[i] != i && src.colIndices[i] != i) && Math.abs(src.entries[i]) > nonDiagTol) {
                return false; // Non-diagonal value is not close to zero.
            }
        }


        return true;
    }


    /**
     * Checks if a real sparse matrix is symmetric.
     * @param src Matrix to check if it is the symmetric matrix.
     * @return True if the {@code src} matrix is symmetric. False otherwise.
     */
    public static boolean isSymmetric(CooMatrix src) {
        boolean result = src.isSquare();

        List<Double> entries = DoubleStream.of(src.entries).boxed().collect(Collectors.toList());
        List<Integer> rowIndices = IntStream.of(src.rowIndices).boxed().collect(Collectors.toList());
        List<Integer> colIndices = IntStream.of(src.colIndices).boxed().collect(Collectors.toList());

        double value;
        int row;
        int col;

        while(result && !entries.isEmpty()) {
            // Extract value of interest.
            value = entries.remove(0);
            row = rowIndices.remove(0);
            col = colIndices.remove(0);

            // Find indices of first and last value whose row index matched the value of interests column index.
            int rowStart = rowIndices.indexOf(col);
            int rowEnd = rowIndices.lastIndexOf(col);

            if(rowStart == -1) {
                // Then no non-zero value was found.
                result = value == 0;
            } else {
                // At least one entry has a row-index matching the specified column index.
                List<Integer> colIdxRange = colIndices.subList(rowStart, rowEnd + 1);

                // Search for element whose column index matches the specified row index
                int idx = colIdxRange.indexOf(row);

                if(idx == -1) {
                    // Then no non-zero value was found.
                    result = value == 0;
                } else {
                    // Check that value with opposite row/column indices is equal.
                    result = value == entries.get(idx + rowStart);

                    // Remove the value and the indices.
                    entries.remove(idx + rowStart);
                    rowIndices.remove(idx + rowStart);
                    colIndices.remove(idx + rowStart);
                }
            }
        }

        return result;
    }


    /**
     * Checks if a real sparse matrix is anti-symmetric.
     * @param src Matrix to check if it is the anti-symmetric matrix.
     * @return True if the {@code src} matrix is anti-symmetric. False otherwise.
     */
    public static boolean isAntiSymmetric(CooMatrix src) {
        boolean result = src.isSquare();

        List<Double> entries = DoubleStream.of(src.entries).boxed().collect(Collectors.toList());
        List<Integer> rowIndices = IntStream.of(src.rowIndices).boxed().collect(Collectors.toList());
        List<Integer> colIndices = IntStream.of(src.colIndices).boxed().collect(Collectors.toList());

        double value;
        int row;
        int col;

        while(result && !entries.isEmpty()) {
            // Extract value of interest.
            value = entries.remove(0);
            row = rowIndices.remove(0);
            col = colIndices.remove(0);

            // Find indices of first and last value whose row index matched the value of interests column index.
            int rowStart = rowIndices.indexOf(col);
            int rowEnd = rowIndices.lastIndexOf(col);

            if(rowStart == -1) {
                // Then no non-zero value was found.
                result = value == 0;
            } else {
                // At least one entry has a row-index matching the specified column index.
                List<Integer> colIdxRange = colIndices.subList(rowStart, rowEnd + 1);

                // Search for element whose column index matches the specified row index
                int idx = colIdxRange.indexOf(row);

                if(idx == -1) {
                    // Then no non-zero value was found.
                    result = value == 0;
                } else {
                    // Check that value with opposite row/column indices is equal.
                    result = value == -entries.get(idx + rowStart);

                    // Remove the value and the indices.
                    entries.remove(idx + rowStart);
                    rowIndices.remove(idx + rowStart);
                    colIndices.remove(idx + rowStart);
                }
            }
        }

        return result;
    }
}







