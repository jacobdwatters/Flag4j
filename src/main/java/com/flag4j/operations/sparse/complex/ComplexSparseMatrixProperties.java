package com.flag4j.operations.sparse.complex;


import com.flag4j.SparseCMatrix;
import com.flag4j.SparseMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.util.ErrorMessages;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * This class contains low level implementations for methods to evaluate certain properties of a complex sparse matrix.
 * (i.e. if the matrix is symmetric).
 */
public class ComplexSparseMatrixProperties {


    private ComplexSparseMatrixProperties() {
        // Hide public constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if a complex sparse matrix is the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return True if the {@code src} matrix is the identity matrix. Otherwise, returns false.
     */
    public static boolean isIdentity(SparseCMatrix src) {
        // Ensure the matrix is square and there are the same number of non-zero entries as entries on the diagonal.
        boolean result = src.isSquare() && src.entries.length==src.numRows;

        if(result) {
            for(int i=0; i<src.entries.length; i++) {
                // Ensure value is 1 and on the diagonal.
                if(src.entries[i].equals(1) || src.rowIndices[i] != i || src.colIndices[i] != i) {
                    result = false;
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if a complex sparse matrix is hermation.
     * @param src Matrix to check if it is the hermation matrix.
     * @return True if the {@code src} matrix is hermation. False otherwise.
     */
    public static boolean isHermation(SparseCMatrix src) {
        boolean result = src.isSquare();

        List<CNumber> entries = Arrays.asList(src.entries);
        List<Integer> rowIndices = IntStream.of(src.rowIndices).boxed().collect(Collectors.toList());
        List<Integer> colIndices = IntStream.of(src.colIndices).boxed().collect(Collectors.toList());

        CNumber value;
        int row;
        int col;

        while(result && entries.size() > 0) {
            // Extract value of interest.
            value = entries.remove(0);
            row = rowIndices.remove(0);
            col = colIndices.remove(0);

            // Find indices of first and last value whose row index matched the value of interests column index.
            int rowStart = rowIndices.indexOf(col);
            int rowEnd = rowIndices.lastIndexOf(col);

            if(rowStart == -1) {
                // Then no non-zero value was found.
                result = value.equals(CNumber.ZERO);
            } else {
                // At least one entry has a row-index matching the specified column index.
                List<Integer> colIdxRange = colIndices.subList(rowStart, rowEnd + 1);

                // Search for element whose column index matches the specified row index
                int idx = colIdxRange.indexOf(row);

                if(idx == -1) {
                    // Then no non-zero value was found.
                    result = value.equals(CNumber.ZERO);
                } else {
                    // Check that value with opposite row/column indices is equal.
                    result = value.equals(entries.get(idx + rowStart).conj());

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
     * Checks if a real sparse matrix is anti-hermation.
     * @param src Matrix to check if it is the anti-hermation matrix.
     * @return True if the {@code src} matrix is anti-hermation. False otherwise.
     */
    public static boolean isAntiHermation(SparseCMatrix src) {
        boolean result = src.isSquare();

        List<CNumber> entries = Arrays.asList(src.entries);
        List<Integer> rowIndices = IntStream.of(src.rowIndices).boxed().collect(Collectors.toList());
        List<Integer> colIndices = IntStream.of(src.colIndices).boxed().collect(Collectors.toList());

        CNumber value;
        int row;
        int col;

        while(result && entries.size() > 0) {
            // Extract value of interest.
            value = entries.remove(0);
            row = rowIndices.remove(0);
            col = colIndices.remove(0);

            // Find indices of first and last value whose row index matched the value of interests column index.
            int rowStart = rowIndices.indexOf(col);
            int rowEnd = rowIndices.lastIndexOf(col);

            if(rowStart == -1) {
                // Then no non-zero value was found.
                result = value.equals(0);
            } else {
                // At least one entry has a row-index matching the specified column index.
                List<Integer> colIdxRange = colIndices.subList(rowStart, rowEnd + 1);

                // Search for element whose column index matches the specified row index
                int idx = colIdxRange.indexOf(row);

                if(idx == -1) {
                    // Then no non-zero value was found.
                    result = value.equals(0);
                } else {
                    // Check that value with opposite row/column indices is equal.
                    result = value.equals(entries.get(idx + rowStart).addInv().conj());

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
