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

package org.flag4j.linalg.decompositions.hess;

import org.flag4j.algebraic_structures.Complex128;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.transformations.Householder;
import org.flag4j.util.ValidateParameters;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * Computes the Hessenburg decomposition of a complex dense Hermitian matrix. That is, for a square, Hermitian matrix
 * A, computes the decomposition A=QHQ<sup>H</sup> where Q is an unitary matrix and
 * H is a Hermitian matrix in tri-diagonal form (special case of Hessenburg form) which is similar to A
 * (i.e. has the same eigenvalues).
 * <p>
 * A matrix H is in tri-diagonal form if it is nearly diagonal except for possibly the first sub/super-diagonal.
 * Specifically, if H has all zeros below the first sub-diagonal and above the first super-diagonal.
 * <p>
 * For example, the following matrix is in Hermitian tri-diagonal form where each 'x' may hold a different value (provided
 * the matrix is Hermitian):
 * <pre>
 *     [[ x x 0 0 0 ]
 *      [ x x x 0 0 ]
 *      [ 0 x x x 0 ]
 *      [ 0 0 x x x ]
 *      [ 0 0 0 x x ]]</pre>
 *
 */
public class HermHess extends ComplexHess {

    /**
     * Flag indicating if an explicit check should be made that the matrix to be decomposed is Hermitian.
     */
    protected boolean enforceHermitian;

    /**
     * Constructs a Hessenberg decomposer for Hermitian matrices. By default, the Householder vectors used in the decomposition will be
     * stored so that the full unitary {@code Q} matrix can be formed by calling {@link #getQ()}.
     */
    public HermHess() {
        super();
    }


    /**
     * Constructs a Hessenberg decomposer for Hermitian matrices.
     * @param computeQ Flag indicating if the unitary {@code Q} matrix from the Hessenberg decomposition should be explicitly computed.
     * If true, then the {@code Q} matrix will be computed explicitly.
     */
    public HermHess(boolean computeQ) {
        super(computeQ);
    }


    /**
     * Constructs a Hessenberg decomposer for Hermitian matrices.
     * @param computeQ Flag indicating if the unitary {@code Q} matrix from the Hessenberg decomposition should be explicitly computed.
     * If true, then the {@code Q} matrix will be computed explicitly.
     * @param enforceHermitian Flag indicating if an explicit check should be made to ensure any matrix passed to
     * {@link #decompose(CMatrix)} is truly Hermitian. If true, an exception will be thrown if the matrix is not Hermitian. If false,
     * the decomposition will proceed under the assumption that the matrix is Hermitian whether it actually is or not. If the
     * matrix is not Hermitian, then the values in the upper triangular portion of the matrix are taken to be the values.
     */
    public HermHess(boolean computeQ, boolean enforceHermitian) {
        super(computeQ);
        this.enforceHermitian = enforceHermitian;
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose. (Assumed to be a Hermitian matrix).
     * @return A reference to this decomposer.
     */
    @Override
    public HermHess decompose(CMatrix src) {
        super.decompose(src);
        return this;
    }


    /**
     * Gets the Hessenberg matrix from the decomposition. The matrix will be Hermitian tri-diagonal.
     * @return The Hermitian tri-diagonal (Hessenberg) matrix from this decomposition.
     */
    @Override
    public CMatrix getH() {
        CMatrix H = new CMatrix(numRows);

        H.data[0] = transformMatrix.data[0];
        int idx1;
        int idx0;
        int rowOffset = numRows;

        for(int i=1; i<numRows; i++) {
            idx1 = rowOffset + i;
            idx0 = idx1 - numRows;

            H.data[idx1] = transformMatrix.data[idx1]; // extract diagonal value.

            // extract off-diagonal values.
            Complex128 a = transformMatrix.data[idx0];
            H.data[idx0] = a;
            H.data[idx1 - 1] = a;

            // Update row index.
            rowOffset += numRows;
        }

        if(numRows > 1) {
            int rowColBase = numRows*numRows - 1;
            H.data[rowColBase] = transformMatrix.data[rowColBase];
            H.data[rowColBase - 1] = transformMatrix.data[rowColBase - numRows];
        }

        return H;
    }


    /**
     * Finds the maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row. This method also initializes
     * the first {@code numRows-j} data of the storage array {@link #householderVector} to the data of this column.
     * @param j Index of column (and starting row) to compute max of.
     * @return The maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row.
     */
    @Override
    protected double findMaxAndInit(int j) {
        double maxAbs = 0;

        // Compute max-abs value in row. (Equivalent to max value in column since matrix is Hermitian.)
        int rowU = (j-1)*numRows;
        for(int i=j; i<numRows; i++) {
            Complex128 d = householderVector[i] = transformMatrix.data[rowU + i];
            maxAbs = Math.max(d.abs(), maxAbs);
        }

        return maxAbs;
    }


    /**
     * Performs basic setup for the decomposition.
     * @param src The matrix to be decomposed.
     * @throws LinearAlgebraException If the matrix is not Hermitian and {@link #enforceHermitian} is true or if the matrix is not
     * square regardless of the value of {@link #enforceHermitian}.
     */
    @Override
    protected void setUp(CMatrix src) {
        if(enforceHermitian && !src.isHermitian()) // If requested, check the matrix is Hermitian.
            throw new LinearAlgebraException(getClass().getSimpleName() + " only supports Hermitian matrices.");
        else
            ValidateParameters.ensureSquareMatrix(src.shape); // Otherwise, Just ensure the matrix is square.

        numRows = numCols = minAxisSize = src.numRows;
        copyUpperTri(src);  // Initializes transform matrix.
        initWorkArrays(numRows);
    }


    /**
     * Copies the upper triangular portion of a matrix to the working matrix {@link #transformMatrix}.
     * @param src The source matrix to decompose of.
     */
    private void copyUpperTri(CMatrix src) {
        transformMatrix = new CMatrix(numRows);

        // Copy upper triangular portion.
        for(int i=0; i<numRows; i++) {
            int pos = i*numRows + i;
            System.arraycopy(src.data, pos, transformMatrix.data, pos, numRows - i);
        }
    }


    /**
     * Updates the {@link #transformMatrix} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param j Index of sub-matrix for which the Householder reflector was computed for.
     */
    @Override
    protected void updateData(int j) {
        Householder.hermLeftRightMultReflector(transformMatrix, householderVector, currentFactor, j, workArray);

        if(j < numRows) transformMatrix.data[(j-1)*numRows + j] = norm.addInv();
        if(storeReflectors) {
            // Store the Q matrix in the lower portion of the transformation data matrix.
            int col = j-1;
            for(int i=j+1; i<numRows; i++) {
                transformMatrix.data[i*numRows + col] = householderVector[i];
            }
        }
    }
}
