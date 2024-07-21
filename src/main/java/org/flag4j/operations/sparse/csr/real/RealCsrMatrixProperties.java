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

package org.flag4j.operations.sparse.csr.real;

import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.util.ErrorMessages;

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


    /**
     * Checks if the {@code src} matrix is symmetric.
     * @param src Source matrix to check symmetry of.
     * @return True if {@code src} is symmetric. Otherwise, returns false.
     */
    public static boolean isSymmetric(CsrMatrix src) {
        // Check for early returns.
        if(!src.isSquare()) return false;
        if(src == null) return false;
        if(src.entries.length == 0) return true;

        return src.T().equals(src);
    }


    /**
     * Checks if the {@code src} matrix is anti-symmetric.
     * @param src Source matrix to check symmetry of.
     * @return True if {@code src} is symmetric. Otherwise, returns false.
     */
    public static boolean isAntiSymmetric(CsrMatrix src) {
        // Check for early returns.
        if(!src.isSquare()) return false;
        if(src == null) return false;
        if(src.entries.length == 0) return true;

        return src.T().mult(-1).equals(src);
    }
}
