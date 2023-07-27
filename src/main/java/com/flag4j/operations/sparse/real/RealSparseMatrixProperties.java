package com.flag4j.operations.sparse.real;

import com.flag4j.SparseMatrix;
import com.flag4j.util.ErrorMessages;

import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

/**
 * This class contains low level implementations for methods to evaluate certain properties of a real sparse matrix.
 * (i.e. if the matrix is symmetric).
 */
public class RealSparseMatrixProperties {

    private RealSparseMatrixProperties() {
        // Hide public constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if a real sparse matrix is the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return True if the {@code src} matrix is the identity matrix. Otherwise, returns false.
     */
    public static boolean isIdentity(SparseMatrix src) {
        // Ensure the matrix is square and there are the same number of non-zero entries as entries on the diagonal.
        boolean result = src.isSquare() && src.entries.length==src.numRows;

        if(result) {
            for(int i=0; i<src.entries.length; i++) {
                // Ensure value is 1 and on the diagonal.
                if(src.entries[i] != 1 || src.rowIndices[i] != i || src.colIndices[i] != i) {
                    result = false;
                    break;
                }
            }
        }

        return result;
    }


    /**
     * Checks if a real sparse matrix is symmetric.
     * @param src Matrix to check if it is the symmetric matrix.
     * @return True if the {@code src} matrix is symmetric. False otherwise.
     */
    public static boolean isSymmetric(SparseMatrix src) {
        boolean result = src.isSquare();

        List<Double> entries = DoubleStream.of(src.entries).boxed().collect(Collectors.toList());
        List<Integer> rowIndices = IntStream.of(src.rowIndices).boxed().collect(Collectors.toList());
        List<Integer> colIndices = IntStream.of(src.colIndices).boxed().collect(Collectors.toList());

        double value;
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
    public static boolean isAntiSymmetric(SparseMatrix src) {
        boolean result = src.isSquare();

        List<Double> entries = DoubleStream.of(src.entries).boxed().collect(Collectors.toList());
        List<Integer> rowIndices = IntStream.of(src.rowIndices).boxed().collect(Collectors.toList());
        List<Integer> colIndices = IntStream.of(src.colIndices).boxed().collect(Collectors.toList());

        double value;
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







