package com.flag4j.operations.sparse.real;


import com.flag4j.SparseMatrix;
import com.flag4j.util.ErrorMessages;

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
}
