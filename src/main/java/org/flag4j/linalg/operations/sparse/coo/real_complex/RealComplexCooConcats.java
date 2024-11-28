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

package org.flag4j.linalg.operations.sparse.coo.real_complex;


import org.flag4j.algebraic_structures.fields.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCMatrix;
import org.flag4j.arrays.sparse.CooMatrix;
import org.flag4j.arrays.sparse.CooVector;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;

/**
 * This utility class contains implementations for operations which concatenate sparse COO matrices and vectors.
 */
public final class RealComplexCooConcats {

    private RealComplexCooConcats() {
        // Hide default constructor for utility class.
        throw new UnsupportedOperationException(ErrorMessages.getUtilityClassErrMsg(getClass()));
    }


    /**
     * Augments two matrices. That is, combines the columns of the two matrices.
     * @param a The first matrix in the augmentation.
     * @param b The second matrix in the augmentation.
     * @return The result of augmenting the two matrices {@code a} and {@code b}.
     * @throws IllegalArgumentException If {@code a.numRows != b.numRows}
     */
    public static CooCMatrix augment(CooCMatrix a, CooMatrix b) {
        ValidateParameters.ensureEquals(a.numRows, b.numRows);

        Shape destShape = new Shape(a.numRows, a.numCols + b.numCols);
        Complex128[] destEntries = new Complex128[a.data.length + b.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy non-zero values.
        System.arraycopy(a.data, 0, destEntries, 0, a.data.length);
        ArrayUtils.arraycopy(b.data, 0, destEntries, a.data.length, b.data.length);

        // Copy row indices.
        System.arraycopy(a.rowIndices, 0, destRowIndices, 0, a.rowIndices.length);
        System.arraycopy(b.rowIndices, 0, destRowIndices, a.rowIndices.length, b.rowIndices.length);

        // Copy column indices (with shifts if appropriate).
        int[] shifted = b.colIndices.clone();
        System.arraycopy(a.colIndices, 0, destColIndices, 0, a.colIndices.length);
        System.arraycopy(ArrayUtils.shift(a.numCols, shifted), 0,
                destColIndices, a.colIndices.length, b.colIndices.length);

        CooCMatrix dest = new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
        dest.sortIndices(); // Ensure indices are sorted properly.

        return dest;
    }



    /**
     * Augments a real sparse COO vector to a complex sparse COO matrix.
     * @param a COO matrix in the augmentation operation.
     * @param b COO vector in the augmentation operation.
     * @return The result of augmenting the vector {@code b} to the right hand side of the matrix {@code a}.
     */
    public static CooCMatrix augment(CooCMatrix a, CooVector b) {
        ValidateParameters.ensureEquals(a.numCols, b.size);

        Shape destShape = new Shape(a.numRows + 1, a.numCols);
        Complex128[] destEntries = new Complex128[a.data.length + b.data.length];
        int[] destRowIndices = new int[destEntries.length];
        int[] destColIndices = new int[destEntries.length];

        // Copy values and indices from this matrix.
        System.arraycopy(a.data, 0, destEntries, 0, a.data.length);
        System.arraycopy(a.rowIndices, 0, destRowIndices, 0, b.data.length);
        System.arraycopy(a.colIndices, 0, destColIndices, 0, b.data.length);

        // Copy values and indices from vector.
        ArrayUtils.arraycopy(b.data, 0, destEntries, a.data.length, b.data.length);
        Arrays.fill(destRowIndices, a.data.length, destRowIndices.length, a.numRows);
        System.arraycopy(b.indices, 0, destColIndices, a.data.length, b.data.length);

        return new CooCMatrix(destShape, destEntries, destRowIndices, destColIndices);
    }
}
