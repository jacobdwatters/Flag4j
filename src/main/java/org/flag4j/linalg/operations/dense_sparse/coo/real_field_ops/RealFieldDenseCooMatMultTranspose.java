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

package org.flag4j.linalg.operations.dense_sparse.coo.real_field_ops;


import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.Shape;
import org.flag4j.util.ErrorMessages;

import java.util.Arrays;

/**
 * <p>This class contains several low level methods for computing matrix-matrix multiplications with a transpose for
 * a real/field and sparse/dense matrix.</p>
 * 
 * <p><b>Warning:</b> These methods do not perform any sanity checks.</p>
 */
public final class RealFieldDenseCooMatMultTranspose {

    private RealFieldDenseCooMatMultTranspose() {
        // Hide default constructor.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Multiplies a real dense matrix to the transpose of a sparse field matrix.
     * @param dSrc Entries of dense matrix.
     * @param dShape Shape of dense matrix.
     * @param spSrc Non-zero data of sparse matrix.
     * @param rowIndices Row indices of non-zero data in the sparse matrix.
     * @param colIndices Column indices of non-zero data in the sparse matrix.
     * @param spShape Shape of the sparse matrix.
     * @return The data of the matrix resulting from multiplying the first matrix by the transpose of the second matrix.
     */
    public static <T extends Field<T>> Field<T>[] multTranspose(
            double[] dSrc, Shape dShape, 
            Field<T>[] spSrc, int[] rowIndices, int[] colIndices, Shape spShape) {
        int rows1 = dShape.get(0);
        int rows2 = spShape.get(0);
        int cols2 = spShape.get(1);

        Field<T>[] dest = new Field[rows1*rows2]; // Since second matrix is transposed, its columns will become rows.
        Arrays.fill(dest, (spSrc.length > 0) ? spSrc[0].getZero() : null);

        for(int i=0; i<rows1; i++) {
            int destStart = i*rows2;
            int dSrcStart = i*cols2;

            // Loop over non-zero data of sparse matrix.
            for(int j=0, len = spSrc.length; j<len; j++) {
                int row = colIndices[j];
                int col = rowIndices[j];

                dest[destStart + col] = dest[destStart + col].add(spSrc[j].mult(dSrc[dSrcStart + row]));
            }
        }

        return dest;
    }


    /**
     * Multiplies a dense field matrix to the transpose of a real sparse matrix.
     * @param dSrc Entries of dense matrix.
     * @param dShape Shape of dense matrix.
     * @param spSrc Non-zero data of sparse matrix.
     * @param rowIndices Row indices of non-zero data in the sparse matrix.
     * @param colIndices Column indices of non-zero data in the sparse matrix.
     * @param spShape Shape of the sparse matrix.
     * @return The data of the matrix resulting from multiplying the first matrix by the transpose of the second matrix.
     */
    public static <T extends Field<T>> Field<T>[] multTranspose(
            Field<T>[] dSrc, Shape dShape,
            double[] spSrc, int[] rowIndices, int[] colIndices, Shape spShape) {
        int rows1 = dShape.get(0);
        int rows2 = spShape.get(0);
        int cols2 = spShape.get(1);

        Field<T>[] dest = new Field[rows1*rows2]; // Since second matrix is transposed, its columns will become rows.
        Arrays.fill(dest, (dSrc.length > 0) ? dSrc[0].getZero() : null);

        for(int i=0; i<rows1; i++) {
            int destStart = i*rows2;
            int dSrcStart = i*cols2;

            // Loop over non-zero data of sparse matrix.
            for(int j=0; j<spSrc.length; j++) {
                int row = colIndices[j];
                int col = rowIndices[j];

                dest[destStart + col] = dest[destStart + col].add(dSrc[dSrcStart + row].mult(spSrc[j]));
            }
        }

        return dest;
    }
}
