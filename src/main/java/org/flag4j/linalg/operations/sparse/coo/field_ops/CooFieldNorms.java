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

package org.flag4j.linalg.operations.sparse.coo.field_ops;

import org.flag4j.algebraic_structures.fields.Field;
import org.flag4j.arrays.backend.CooFieldMatrixBase;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.ErrorMessages;
import org.flag4j.util.ValidateParameters;

import java.util.HashMap;


/**
 * This utility class contains low level implementations of norms for sparse field coo tensors, matrices and vectors.
 */
public final class CooFieldNorms {

    private CooFieldNorms() {
        // Hide default constructor for utility class.
        throw new IllegalStateException(ErrorMessages.getUtilityClassErrMsg(this.getClass()));
    }


    /**
     * Computes the L<sub>2</sub> norm of a matrix.
     * @param src Source matrix to compute norm of.
     * @return The L<sub>2</sub> of the {@code src} matrix.
     */
    public static <T extends Field<T>> double matrixNormL2(CooFieldMatrixBase<?, ?, ?, ?, T> src) {
        double norm = 0;
        double[] colSums = new double[ArrayUtils.numUnique(src.colIndices)];

        // Create a mapping from the unique column indices to a unique position in the colSums array.
        HashMap<Integer, Integer> columnMap = ArrayUtils.createUniqueMapping(src.colIndices);

        // Compute the column sums.
        for(int i=0; i<src.entries.length; i++) {
            double mag = src.entries[i].mag();
            colSums[columnMap.get(src.colIndices[i])] += mag*mag;
        }

        // Compute the norm from the column sums.
        for(double colSum : colSums)
            norm += Math.sqrt(colSum);

        return norm;
    }


    /**
     * Computes the L<sub>p</sub> norm of a matrix.
     * @param src Source matrix to compute norm of.
     * @param p Parameter for L<sub>p</sub> norm
     * @return The L<sub>p</sub> of the {@code src} matrix.
     */
    public static <T extends Field<T>> double matrixNormLp(CooFieldMatrixBase<?, ?, ?, ?, T> src, double p) {
        ValidateParameters.ensureGreaterEq(1, p);

        double norm = 0;
        double[] colSums = new double[ArrayUtils.numUnique(src.colIndices)];

        // Create a mapping from the unique column indices to a unique position in the colSums array.
        HashMap<Integer, Integer> columnMap = ArrayUtils.createUniqueMapping(src.colIndices);

        // Compute the column sums.
        for(int i=0; i<src.entries.length; i++)
            colSums[columnMap.get(src.colIndices[i])] += Math.pow(src.entries[i].mag(), p);

        // Compute the norm from the column sums.
        for(double colSum : colSums)
            norm += Math.pow(colSum, 1.0/p);

        return norm;
    }


    /**
     * Computes the L<sub>p, q</sub> norm of a matrix.
     * @param src Source matrix to compute norm of.
     * @param p First parameter for L<sub>p, q</sub> norm
     * @return The L<sub>p, q</sub> of the {@code src} matrix.
     */
    public static <T extends Field<T>> double matrixNormLpq(CooFieldMatrixBase<?, ?, ?, ?, T> src, double p, double q) {
        ValidateParameters.ensureGreaterEq(1, p, q);

        double norm = 0;
        double[] colSums = new double[ArrayUtils.numUnique(src.colIndices)];

        // Create a mapping from the unique column indices to a unique position in the colSums array.
        HashMap<Integer, Integer> columnMap = ArrayUtils.createUniqueMapping(src.colIndices);

        // Compute the column sums.
        for(int i=0; i<src.entries.length; i++)
            colSums[columnMap.get(src.colIndices[i])] += Math.pow(src.entries[i].mag(), p);

        // Compute the norm from the column sums.
        for(double colSum : colSums)
            norm += Math.pow(colSum, p/q);

        return Math.pow(norm, 1.0/q);
    }
}
