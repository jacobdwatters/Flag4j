package com.flag4j.operations.sparse.complex;

import com.flag4j.Shape;
import com.flag4j.SparseCMatrix;
import com.flag4j.SparseMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.operations.sparse.real.RealSparseElementSearch;
import com.flag4j.util.ArrayUtils;
import com.flag4j.util.ErrorMessages;

/**
 * This class contains implementations for complex sparse matrix manipulations.
 */
public class ComplexSparseMatrixManipulations {

    private ComplexSparseMatrixManipulations() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Removes a specified row from a sparse matrix.
     * @param src Source matrix to remove row of.
     * @param rowIdx Row to remove from the {@code src} matrix.
     * @return A sparse matrix which has one less row than the {@code src} matrix with the specified row removed.
     */
    public static SparseCMatrix removeRow(SparseCMatrix src, int rowIdx) {
        Shape shape = new Shape(src.numRows-1, src.numCols);

        // Find the start and end index within the entries array which have the given row index.
        int[] startEnd = ComplexSparseElementSearch.matrixFindRowStartEnd(src, rowIdx);
        int size = src.entries.length - (startEnd[1]-startEnd[0]);

        CNumber[] entries = new CNumber[size];
        int[] rowIndices = new int[size];
        int[] colIndices = new int[size];

        if(startEnd[0] > 0) {
            ArrayUtils.arraycopy(src.entries, 0, entries, 0, startEnd[0]);
            ArrayUtils.arraycopy(src.entries, startEnd[1], entries, startEnd[0], entries.length - startEnd[0]);

            System.arraycopy(src.rowIndices, 0, rowIndices, 0, startEnd[0]);
            System.arraycopy(src.rowIndices, startEnd[1], rowIndices, startEnd[0], entries.length - startEnd[0]);

            System.arraycopy(src.colIndices, 0, colIndices, 0, startEnd[0]);
            System.arraycopy(src.colIndices, startEnd[1], colIndices, startEnd[0], entries.length - startEnd[0]);
        } else {
            ArrayUtils.arraycopy(src.entries, 0, entries, 0, entries.length);
            System.arraycopy(src.rowIndices, 0, rowIndices, 0, rowIndices.length);
            System.arraycopy(src.colIndices, 0, colIndices, 0, colIndices.length);
        }

        return new SparseCMatrix(shape, entries, rowIndices, colIndices);
    }
}
