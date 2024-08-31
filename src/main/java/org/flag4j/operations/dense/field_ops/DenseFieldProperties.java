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

package org.flag4j.operations.dense.field_ops;


import org.flag4j.core.TensorBase;
import org.flag4j.core_temp.arrays.dense.DenseFieldMatrixBase;
import org.flag4j.core_temp.structures.fields.Field;
import org.flag4j.util.ErrorMessages;

/**
 * This utility class contains methods useful for verifying properties of a {@link org.flag4j.core_temp.arrays.dense.FieldTensor}.
 */
public final class DenseFieldProperties {

    private DenseFieldProperties() {
        // Hide constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Checks if a matrix is the identity matrix approximately. Specifically, if the diagonal entries are no farther than
     * 1.001E-5 in absolute value from 1.0 and the non-diagonal entries are no larger than 1e-08 in absolute value.
     * These tolerances are derived from the {@link TensorBase#allClose(Object)} method.
     * @param src MatrixOld of interest to check if it is the identity matrix.
     * @return True if the {@code src} matrix is close the identity matrix or if the matrix has zero entries.
     */
    public static <T extends Field<T>> boolean isCloseToIdentity(DenseFieldMatrixBase<?, ?, ?, ?, T> src) {
        if(src == null || src.numRows!=src.numCols) return false;
        if(src.entries.length == 0) return true;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.001E-5;
        double nonDiagTol = 1e-08;

        final T ONE = src.entries[0].getOne();

        int rows = src.numRows;
        int cols = src.numCols;
        int pos = 0;
        for(int i=0; i<rows; i++) {
            for(int j=0; j<cols; j++) {
                if((i==j && src.entries[pos].sub(ONE).mag() > diagTol)
                        || (i!=j && src.entries[pos].mag() > nonDiagTol)) {
                    return false;
                }

                pos++;
            }
        }

        return true; // If we make it here, the matrix is "close" to the identity matrix.
    }
}
