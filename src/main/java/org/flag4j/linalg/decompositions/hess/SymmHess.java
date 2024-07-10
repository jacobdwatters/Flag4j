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

package org.flag4j.linalg.decompositions.hess;


import org.flag4j.dense.Matrix;
import org.flag4j.linalg.transformations.Householder;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;

/**
 * <p>Computes the Hessenburg decomposition of a real dense symmetric matrix. That is, for a square, symmetric matrix
 * {@code A}, computes the decomposition {@code A=QHQ}<sup>T</sup> where {@code Q} is an orthogonal matrix and
 * {@code H} is a symmetric matrix in tri-diagonal form (special case of Hessenburg form) which is similar to {@code A}
 * (i.e. has the same eigenvalues).</p>
 *
 * <p>A matrix {@code H} is in tri-diagonal form if it is nearly diagonal except for possibly the first sub/super-diagonal.
 * Specifically, if {@code H} has all zeros below the first sub-diagonal and above the first super-diagonal.</p>
 *
 * <p>For example, the following matrix is in symmetric tri-diagonal form where each {@code x} may hold a different value (provided
 * the
 * matrix is symmetric):
 * <pre>
 *     [[ x x 0 0 0 ]
 *      [ x x x 0 0 ]
 *      [ 0 x x x 0 ]
 *      [ 0 0 x x x ]
 *      [ 0 0 0 x x ]]</pre>
 * </p>
 */
public class SymmHess extends RealHess {

    /**
     * Flag indicating if an explicit check should be made that the matrix to be decomposed is symmetric.
     */
    protected boolean enforceSymmetric;

    /**
     * Constructs a Hessenberg decomposer for symmetric matrices. By default, the Householder vectors used in the decomposition will be
     * stored so that the full orthogonal {@code Q} matrix can be formed by calling {@link #getQ()}.
     */
    public SymmHess() {
        super();
    }


    /**
     * Constructs a Hessenberg decomposer for symmetric matrices.
     * @param computeQ Flag indicating if the orthogonal {@code Q} matrix from the Hessenberg decomposition should be explicitly computed.
     * If true, then the {@code Q} matrix will be computed explicitly.
     */
    public SymmHess(boolean computeQ) {
        super(computeQ);
    }


    /**
     * Constructs a Hessenberg decomposer for symmetric matrices.
     * @param computeQ Flag indicating if the orthogonal {@code Q} matrix from the Hessenberg decomposition should be explicitly computed.
     * If true, then the {@code Q} matrix will be computed explicitly.
     * @param enforceSymmetric Flag indicating if an explicit check should be made to ensure any matrix passed to
     * {@link #decompose(Matrix)} is truly symmetric. If true, an exception will be thrown if the matrix is not symmetric. If false,
     * the decomposition will proceed under the assumption that the matrix is symmetric whether it actually is or not. If the
     * matrix is not symmetric, then the values in the upper triangular portion of the matrix are taken to be the values.
     */
    public SymmHess(boolean computeQ, boolean enforceSymmetric) {
        super(computeQ);
        this.enforceSymmetric = enforceSymmetric;
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose. (Assumed to be a symmetric matrix).
     * @return A reference to this decomposer.
     */
    @Override
    public SymmHess decompose(Matrix src) {
        super.decompose(src);
        return this;
    }


    /**
     * Gets the Hessenberg matrix from the decomposition. The matrix will be symmetric tri-diagonal.
     * @return The symmetric tri-diagonal (Hessenberg) matrix from this decomposition.
     */
    @Override
    public Matrix getH() {
        Matrix H = new Matrix(numRows);
        H.entries[0] = transformMatrix.entries[0];

        int idx1;
        int idx0;
        int rowOffset = numRows;

        for(int i=1; i<numRows; i++) {
            idx1 = rowOffset + i;
            idx0 = idx1 - numRows;

            H.entries[idx1] = transformMatrix.entries[idx1]; // extract diagonal value.

            // extract off-diagonal values.
            double a = transformMatrix.entries[idx0];
            H.entries[idx0] = a;
            H.entries[idx1 - 1] = a;

            // Update row index.
            rowOffset += numRows;
        }

        if(numRows > 1) {
            int rowColBase = numRows*numRows - 1;
            H.entries[rowColBase] = transformMatrix.entries[rowColBase];
            H.entries[rowColBase - 1] = transformMatrix.entries[rowColBase - numRows];
        }

        return H;
    }


    /**
     * Finds the maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row. This method also initializes
     * the first {@code numRows-j} entries of the storage array {@link #householderVector} to the entries of this column.
     * @param j Index of column (and starting row) to compute max of.
     * @return The maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row.
     */
    @Override
    protected double findMaxAndInit(int j) {
        double maxAbs = 0;

        // Compute max-abs value in row. (Equivalent to max value in column since matrix is symmetric.)
        int rowU = (j-1)*numRows;
        for(int i=j; i<numRows; i++) {
            double d = householderVector[i] = transformMatrix.entries[rowU + i];
            maxAbs = Math.max(Math.abs(d), maxAbs);
        }

        return maxAbs;
    }


    /**
     * Performs basic setup for the decomposition.
     * @param src The matrix to be decomposed.
     * @throws LinearAlgebraException If the matrix is not symmetric and {@link #enforceSymmetric} is true or if the matrix is not
     * square regardless of the value of {@link #enforceSymmetric}.
     */
    @Override
    protected void setUp(Matrix src) {
        if(enforceSymmetric && !src.isSymmetric()) // If requested, check the matrix is symmetric.
            throw new LinearAlgebraException("Decomposition only supports symmetric matrices.");
        else
            ParameterChecks.assertSquareMatrix(src.shape); // Otherwise, Just ensure the matrix is square.

        numRows = numCols = minAxisSize = src.numRows;
        copyUpperTri(src);  // Initializes transform matrix.
        initWorkArrays(numRows);
    }


    /**
     * Copies the upper triangular portion of a matrix to the working matrix {@link #transformMatrix}.
     * @param src The source matrix to decompose of.
     */
    private void copyUpperTri(Matrix src) {
        transformMatrix = new Matrix(numRows);

        // Copy upper triangular portion.
        for(int i=0; i<numRows; i++) {
            int pos = i*numRows + i;
            System.arraycopy(src.entries, pos, transformMatrix.entries, pos, numRows - i);
        }
    }


    /**
     * Updates the {@link #transformMatrix} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param j Index of sub-matrix for which the Householder reflector was computed for.
     */
    @Override
    protected void updateData(int j) {
        Householder.symmLeftRightMultReflector(transformMatrix, householderVector, currentFactor, j, workArray);

        if(j < numRows) transformMatrix.entries[(j-1)*numRows + j] = -norm;
        if(storeReflectors) {
            // Store the Q matrix in the lower portion of the transformation data matrix.
            int col = j-1;
            for(int i=j+1; i<numRows; i++) {
                transformMatrix.entries[i*numRows + col] = householderVector[i];
            }
        }
    }
}
