/*
 * MIT License
 *
 * Copyright (c) 2024-2025. Jacob Watters
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

package org.flag4j.linalg.ops.sparse.coo.ring_ops;

import org.flag4j.algebraic_structures.Ring;
import org.flag4j.arrays.Pair;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.SparseMatrixData;
import org.flag4j.arrays.backend.ring_arrays.AbstractCooRingMatrix;
import org.flag4j.util.ValidateParameters;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Utility class for computing ops on sparse COO {@link Ring} matrices.
 */
public final class CooRingMatrixOps {

    private CooRingMatrixOps() {
        // Hide default constructor for utility class.
    }


    /**
     * Computes the element-wise difference of two sparse matrices. This method assumes that the indices of the two matrices are
     * sorted
     * lexicographically.
     * @param shape1 Shape of the first matrix.
     * @param src1Entries Non-zero data of the first matrix.
     * @param src1RowIndices Non-zero row indices of the first matrix.
     * @param src1ColIndices Non-zero column indices of the first matrix.
     * @param shape2 Shape of the second matrix.
     * @param src2Entries Non-zero data of the second matrix.
     * @param src2RowIndices Non-zero row indices of the second matrix.
     * @param src2ColIndices Non-zero column indices of the second matrix.
     * @return The element-wise difference of the two matrices.
     * @throws IllegalArgumentException If the two matrices do not have the same shape.
     */
    public static <V extends Ring<V>> SparseMatrixData<V> sub(
            Shape shape1, V[] src1Entries, int[] src1RowIndices, int[] src1ColIndices,
            Shape shape2, V[] src2Entries, int[] src2RowIndices, int[] src2ColIndices) {
        ValidateParameters.ensureEqualShape(shape1, shape2);

        int initCapacity = Math.max(src1Entries.length, src2Entries.length);

        List<V> diff = new ArrayList<>(initCapacity);
        List<Integer> rowIndices = new ArrayList<>(initCapacity);
        List<Integer> colIndices = new ArrayList<>(initCapacity);

        int src1Counter = 0;
        int src2Counter = 0;

        // Flags which indicate if a value should be added from the corresponding matrix
        boolean add1;
        boolean add2;

        while(src1Counter < src1Entries.length || src2Counter < src2Entries.length) {

            if(src1Counter >= src1Entries.length || src2Counter >= src2Entries.length) {
                add1 = src2Counter >= src2Entries.length;
                add2 = !add1;
            } else if(src1RowIndices[src1Counter] == src2RowIndices[src2Counter]
                    && src1ColIndices[src1Counter] == src2ColIndices[src2Counter]) {
                // Found matching indices.
                add1 = true;
                add2 = true;
            } else if(src1RowIndices[src1Counter] == src2RowIndices[src2Counter]) {
                // Matching row indices.
                add1 = src1ColIndices[src1Counter] < src2ColIndices[src2Counter];
                add2 = !add1;
            } else {
                add1 = src1RowIndices[src1Counter] < src2RowIndices[src2Counter];
                add2 = !add1;
            }

            if(add1 && add2) {
                diff.add(src1Entries[src1Counter].sub(src2Entries[src2Counter]));
                rowIndices.add(src1RowIndices[src1Counter]);
                colIndices.add(src1ColIndices[src1Counter]);
                src1Counter++;
                src2Counter++;
            } else if(add1) {
                diff.add(src1Entries[src1Counter]);
                rowIndices.add(src1RowIndices[src1Counter]);
                colIndices.add(src1ColIndices[src1Counter]);
                src1Counter++;
            } else {
                diff.add(src2Entries[src2Counter].addInv());
                rowIndices.add(src2RowIndices[src2Counter]);
                colIndices.add(src2ColIndices[src2Counter]);
                src2Counter++;
            }
        }

        return new SparseMatrixData<V>(shape1, diff, rowIndices, colIndices);
    }


    /**
     * Checks if a real sparse matrix is close to the identity matrix.
     * @param src Matrix to check if it is the identity matrix.
     * @return {@code true} if the {@code src} matrix is the identity matrix; {@code false} otherwise.
     */
    public static <T extends Ring<T>> boolean isCloseToIdentity(AbstractCooRingMatrix<?, ?, ?, T> src) {
        // Ensure the matrix is square and there are the same number of non-zero data as data on the diagonal.
        boolean result = src.isSquare() && src.data.length==src.numRows;

        // Tolerances corresponds to the allClose(...) methods.
        double diagTol = 1.E-5;
        double nonDiagTol = 1e-08;

        final T ONE = src.data.length > 0 ? src.data[0].getOne() : null;

        for(int i = 0; i<src.data.length; i++) {
            if(src.rowIndices[i] == i && src.colIndices[i] == i && src.data[i].sub(ONE).abs() > diagTol ) {
                return false; // Diagonal value is not close to one.
            } else if((src.rowIndices[i] != i && src.colIndices[i] != i) && src.data[i].mag() > nonDiagTol) {
                return false; // Non-diagonal value is not close to zero.
            }
        }

        return true;
    }


    /**
     * Checks if a sparse COO matrix is hermitian. That is, the matrix is equal to its conjugate transpose.
     * @param shape Shape of the COO matrix.
     * @param entries Non-zero data of the COO matrix.
     * @param rowIndices Non-zero row indices of the COO matrix.
     * @param colIndices Non-zero column indices of the COO matrix.
     * @return {@code true} if the {@code src} matrix is hermitian. {@code false} otherwise.
     */
    public static <T extends Ring<T>> boolean isHermitian(Shape shape, T[] entries, int[] rowIndices, int[] colIndices) {
        // Check if the matrix is square.
        if (shape.get(0) != shape.get(1))
            return false;

        // Build a map from (row, col) to value for quick access.
        Map<Pair<Integer, Integer>, T> matrixMap = new HashMap<>();
        int nnz = entries.length; // Number of non-zero data.

        for (int i = 0; i < nnz; i++) {
            int row = rowIndices[i];
            int col = colIndices[i];
            T value = entries[i];

            matrixMap.put(new Pair<>(row, col), value);
        }

        // Iterate over the data to check for Hermitian property.
        for (Map.Entry<Pair<Integer, Integer>, T> entry : matrixMap.entrySet()) {
            int row = entry.getKey().first();
            int col = entry.getKey().second();
            T value = entry.getValue();

            // Skip data where row > col to avoid redundant checks.
            if (row > col) continue;

            if (row == col) {
                // Diagonal data must be real: value == value.conj()
                if (!value.equals(value.conj())) return false;

            } else {
                // Get the symmetric value at (col, row).
                T symValue = matrixMap.get(new Pair<>(col, row));

                if (symValue == null) // Missing symmetric entry implies zero.
                    symValue = value.getZero();

                // Check if value equals the conjugate of the symmetric value.
                if (!value.equals(symValue.conj())) return false;
            }
        }

        // If all checks pass, the matrix is Hermitian.
        return true;
    }
}