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

package org.flag4j.linalg.operations.sparse.csr.field_ops;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.CsrFieldMatrixBase;
import org.flag4j.util.ErrorMessages;

/**
 * This utility class contains methods usefully for determining properties of a sparse CSR
 * {@link Field} matrix.
 */
public final class CsrFieldMatrixProperties {

    private CsrFieldMatrixProperties() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if the {@code src} matrix is the identity matrix.
     * @param src The matrix to check if it is the identity matrix.
     * @return True if the {@code src} matrix is the identity matrix. False otherwise.
     */
    public static <T extends Field<T>> boolean isIdentity(CsrFieldMatrixBase<?, ?, ?, ?, T> src) {
        if(src.isSquare() && src.colIndices.length >= src.numCols) {
            int diagCount = 0;

            for(int i=0; i<src.rowPointers.length-1; i++) {
                for(int j=src.rowPointers[i]; j<src.rowPointers[i+1]; j++) {
                    if(src.entries[j].isOne()) {
                        if(src.colIndices[j] != i) return false;
                        diagCount++;
                    } else if(!src.entries[j].isZero()) {
                        return false;
                    }
                }
            }

            return diagCount == src.numCols;
        } else {
            return false;
        }
    }


    /**
     * Checks if the {@code src} matrix is close to the identity matrix.
     * @param src The matrix to check.
     * @return True if the {@code src} matrix is close to identity matrix. False otherwise.
     */
    public static <T extends Field<T>> boolean isCloseToIdentity(CsrFieldMatrixBase<?, ?, ?, ?, T> src) {
        if(src.isSquare() && src.colIndices.length >= src.numCols) {
            // Tolerances corresponds to the allClose(...) methods.
            double diagTol = 1.001E-5;
            double nonDiagTol = 1e-08;
            int diagCount = 0;

            final T ONE = src.nnz > 0 ? src.entries[0].getOne() : null;

            for(int i=0; i<src.rowPointers.length-1; i++) {
                for(int j=src.rowPointers[i]; j<src.rowPointers[i+1]; j++) {
                    if(src.entries[j].sub(ONE).abs() > diagTol) {
                        if(src.colIndices[j] != i) return false; // Diagonal value not close to one.
                        diagCount++;
                    } else if(src.entries[i].abs() > nonDiagTol) {
                        return false; // Non-diagonal value is not close to one.
                    }
                }
            }

            return diagCount == src.numCols;
        } else {
            return false;
        }
    }


    /**
     * Checks if the {@code src} matrix is symmetric.
     * @param src Source matrix to check symmetry of.
     * @return True if {@code src} is symmetric. Otherwise, returns false.
     */
    public static <T extends Field<T>> boolean isSymmetric(CsrFieldMatrixBase<?, ?, ?, ?, T> src) {
        // Check for early returns.
        if(!src.isSquare()) return false;
        if(src.entries.length == 0) return true;

        return src.T().equals(src);
    }


    /**
     * Checks if the {@code src} matrix is anti-symmetric.
     * @param src Source matrix to check symmetry of.
     * @return True if {@code src} is symmetric. Otherwise, returns false.
     */
    public static <T extends Field<T>> boolean isAntiSymmetric(CsrFieldMatrixBase<?, ?, ?, ?, T> src) {
        // Check for early returns.
        if(!src.isSquare()) return false;
        if(src.entries.length == 0) return true;

        return src.T().mult(-1).equals(src);
    }


    /**
     * Checks if the {@code src} matrix is Hermitian.
     * @param src Source matrix to check.
     * @return True if {@code src} is Hermitian. Otherwise, returns false.
     */
    public static <T extends Field<T>> boolean isHermitian(CsrFieldMatrixBase<?, ?, ?, ?, T> src) {
        // Check for early returns.
        if(!src.isSquare()) return false;
        if(src.entries.length == 0) return true;

        return src.H().equals(src);
    }


    /**
     * Checks if the {@code src} matrix is anti-Hermitian.
     * @param src Source matrix to check.
     * @return True if {@code src} is Hermitian. Otherwise, returns false.
     */
    public static <T extends Field<T>> boolean isAntiHermitian(CsrFieldMatrixBase<?, ?, ?, ?, T> src) {
        // Check for early returns.
        if(!src.isSquare()) return false;
        if(src.entries.length == 0) return true;

        return src.H().mult(-1).equals(src);
    }
}
