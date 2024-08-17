/*
 * MIT License
 *
 * Copyright (c) 2023-2024. Jacob Watters
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

package org.flag4j.linalg.decompositions.unitary;

import org.flag4j.core.MatrixMixin;
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.util.ParameterChecks;


/**
 * This class is the base class for all decompositions which proceed by using unitary transformations
 * (specifically Householder reflectors) to bring a matrix into an upper triangular matrix or an upper Hessenburg matrix.
 *
 * @param <T> Type of the matrix to be decomposed.
 * @param <U> Internal storage datatype of the matrix.
 */
public abstract class UnitaryDecomposition<
        T extends MatrixMixin<T, ?, ?, ?, ?, ?, ?, ?>, U> implements Decomposition<T> {

    /**
     * <p>
     * Storage for the upper triangular/Hessenburg matrix and the vectors of the Householder reflectors used in the decomposition.
     * </p>
     *
     * <p>
     * The upper triangular/Hessenburg will have all zeros below either the diagonal or the first sub-diagonal and will be stored
     * in the top corner above that diagonal. For instance, if the quasi-triangular matrix is truly upper triangular, it will be
     * stored at and above the principle diagonal. If the quasi-triangular matrix is upper Hessenburg, it will be stored at and
     * above the first sub-diagonal.
     * </p>
     *
     * <p>
     * The Householder reflectors used to bring the original matrix to the upper triangular/Hessenburg
     * form will be stored as the columns below the last non-zero sub-diagonal of the quasi-triangular matrix. The first value of
     * each reflector is not stored but is assumed to be 1.
     * </p>
     *
     * <p>This provides compact storage for decompositions which proceed by unitary transformations. Further, the full computation
     * of the unitary matrix can be deferred until it is needed.</p>
     */
    protected T transformMatrix;
    /**
     * Pointer to the internal data array of {@link #transformMatrix}.
     */
    protected U transformData;
    /**
     * Number of rows in {@link #transformMatrix}.
     */
    protected int numRows;
    /**
     * Number of columns in {@link #transformMatrix}.
     */
    protected int numCols;
    /**
     * Storage of the scalar factors for the Householder reflectors used in the decomposition.
     */
    protected U qFactors;
    /**
     * For storing a Householder vectors.
     */
    protected U householderVector;
    /**
     * For temporarily storage when applying Householder vectors. This is useful for
     * avoiding unneeded garbage collection and for improving cache performance when traversing columns.
     */
    protected U workArray;
    /**
     * The minimum of rows and columns in the matrix to be decomposed.
     */
    protected int minAxisSize;
    /**
     * Sub-diagonal of the upper quasi-triangular matrix. That is, the sub0diagonal for which all entries below
     *                    will be zero in the final upper quasi-triangular matrix. Must be zero or one.
     *                    If zero, it will be upper triangular. If one, it will be upper Hessenburg.
     */
    protected final int subDiagonal;
    /**
     * Flag indicating if a Householder reflector was needed for the current column meaning the {@link #transformMatrix} should
     * be updated.
     */
    protected boolean applyUpdate;
    /**
     * Flag indicating if {@code Q} should be computed in the decomposition.
     */
    protected boolean storeReflectors;


    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper quasi-triangular matrix (Either truly upper or
     * upper Hessenburg).
     * @param subDiagonal Sub-diagonal of the upper quasi-triangular matrix. Must be zero or one. If zero, it will be upper triangular.
     *                   If one, it will be upper Hessenburg.
     * @throws IllegalArgumentException If {@code 1 < subDiagonal < 0}.
     */
    public UnitaryDecomposition(int subDiagonal, boolean storeReflectors) {
        ParameterChecks.assertInRange(subDiagonal, 0, 1, "subDiagonal");
        this.subDiagonal = subDiagonal;
        this.storeReflectors = storeReflectors;
    }


    /**
     * Applies the unitary decomposition to the matrix. Note, the full computation of Q is deferred until {@link #getQ()} is
     * explicitly called.
     * @param src The source matrix to decompose.
     */
    public void decomposeBase(T src) {
        setUp(src); // Initialize datastructures and storage for the decomposition.
        int offSet = (subDiagonal == 0) ? 0 : subDiagonal + 1;

        for(int j=0; j<minAxisSize-offSet; j++) {
            computeHouseholder(j + subDiagonal); // Compute the householder reflector.
            if(applyUpdate) updateData(j + subDiagonal); // Update the upper-triangular matrix and store the reflectors.
        }
    }


    /**
     * Gets the unitary {@code Q} matrix from the decomposition. This is the accumulation of all unitary transformation applied
     * to bring the matrix to an upper triangular or Hessenburg form.
     * @return The {@code Q} matrix from the decomposition. I.e. the accumulation of all unitary transformation applied during the
     * decomposition.
     */
    public abstract T getQ();


    /**
     * Creates and initializes Q to the appropriately sized identity matrix.
     * @return An identity matrix with the appropriate size.
     */
    protected abstract T initQ();


    /**
     * Gets the upper triangular/Hessenburg matrix from the last decomposition.
     * @param U Storage for upper triangular/Hessenburg matrix. Assumed to be the zero matrix of an appropriate size.
     * @return The upper triangular/Hessenburg matrix from the last decomposition.
     */
    protected abstract T getUpper(T U);


    /**
     * Gets the upper triangular/Hessenburg matrix from the last decomposition.
     * @return The upper triangular/Hessenburg matrix from the last decomposition.
     */
    public abstract T getUpper();


    /**
     * Initializes storage and other parameters for the decomposition.
     * @param src Source matrix to be decomposed.
     */
    protected void setUp(T src) {
        transformMatrix = src.copy(); // Initialize QR as the matrix to be decomposed.
        numRows = transformMatrix.numRows();
        numCols = transformMatrix.numCols();
        minAxisSize = Math.min(numRows, numCols);

        int maxAxisSize = Math.max(numRows, numCols);
        initWorkArrays(maxAxisSize);
    }


    /**
     * Initialized any work arrays_old to be used in computing the decomposition with the proper size.
     * @param maxAxisSize Length of the largest axis in the matrix to be decomposed. That is, {@code max(numRows, numCols)}
     */
    protected abstract void initWorkArrays(int maxAxisSize);


    /**
     * Computes the Householder vector for the first column of the sub-matrix with upper left corner at {@code (j, j)}.
     * @param j Index of the upper left corner of the sub-matrix for which to compute the Householder vector for the first column.
     *          That is, a Householder vector will be computed for the portion of column {@code j} below row {@code j}.
     */
    protected abstract void computeHouseholder(int j);


    /**
     * Updates the {@link #transformMatrix} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param j Index of sub-matrix for which the Householder reflector was computed for.
     */
    protected abstract void updateData(int j);


    /**
     * Computes the norm of column {@code j} at and below the {@code j}th row of the matrix to be decomposed. The norm will have the
     * same parity as the first entry in the sub-column.
     * @param j Column to compute norm of below the {@code j}th row.
     * @param maxAbs Maximum absolute value in the column. Used for scaling norm to minimize potential overflow issues.
     */
    protected abstract void computePhasedNorm(int j, double maxAbs);


    /**
     * Finds the maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row. This method also initializes
     * the first {@code numRows-j} entries of the storage array {@link #householderVector} to the entries of this column.
     * @param j Index of column (and starting row) to compute max of.
     * @return The maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row.
     */
    protected abstract double findMaxAndInit(int j);
}
