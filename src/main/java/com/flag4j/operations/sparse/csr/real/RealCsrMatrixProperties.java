package com.flag4j.operations.sparse.csr.real;

import com.flag4j.CooMatrix;
import com.flag4j.CsrMatrix;
import com.flag4j.Matrix;
import com.flag4j.Shape;
import com.flag4j.rng.RandomTensor;
import com.flag4j.util.ErrorMessages;

/**
 * This class contains low-level implementations for determining certain properties of real sparse CSR matrices.
 */
public final class RealCsrMatrixProperties {

    private RealCsrMatrixProperties() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg());
    }


    /**
     * Checks if the {@code src} matrix is the identity matrix.
     * @param src The matrix to check if it is the identity matrix.
     * @return True if the {@code src} matrix is the identity matrix. False otherwise.
     */
    public static boolean isIdentity(CsrMatrix src) {
        int diagCount = 0;

        if(src.isSquare() && src.colIndices.length >= src.numCols) {
            for(int i=0; i<src.rowPointers.length-1; i++) {
                for(int j=src.rowPointers[i]; j<src.rowPointers[i+1]; j++) {
                    if(src.entries[j] == 1) {
                        if(src.colIndices[j] != i) {
                            return false;
                        }

                        diagCount++;
                    } else if(src.entries[j] != 0) {
                        return false;
                    }
                }
            }

        } else {
            return false;
        }

        return diagCount == src.numCols;
    }
}
