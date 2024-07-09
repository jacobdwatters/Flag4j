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
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.linalg.transformations.Householder;
import org.flag4j.util.Flag4jConstants;
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
public class SymmHess implements Decomposition<Matrix> {

    /**
     * Flag indicating if an explicit check should be made that the matrix to be decomposed is symmetric.
     */
    protected boolean enforceSymmetric;

    /**
     * Storage for the symmetric tri-diagonal matrix and if requested, the Householder vectors used to bring the original matrix
     * into upper Hessenberg form. The symmetric tri-diagonal matrix will be stored in the principle diagonal and the first
     * super-diagonal (Since the matrix is symmetric there is no need to store the first sub-diagonal). The rows of the strictly
     * lower-triangular portion of the matrix will be used to store the Householder vectors used to transform the source matrix
     * to upper Hessenburg form if it is requested via {@link #computeQ}. These can be used to compute the full orthogonal matrix
     * {@code Q} of the Hessenberg decomposition.
     */
    protected Matrix transformMatrix;
    /**
     * For storing norms of columns in A when computing Householder reflectors.
     */
    double norm;
    /**
     * Size of the symmetric matrix to be decomposed. That is, the number of rows and columns.
     */
    protected int size;
    /**
     * Flag indicating if the orthogonal transformation matrix from the Hessenburg decomposition should be explicitly computed.
     */
    protected boolean computeQ;
    /**
     * Stores the scalar factor for the current Householder reflector.
     */
    double currentFactor;
    /**
     * Storage of the scalar factors for the Householder reflectors used in the decomposition.
     */
    protected double[] qFactors;
    /**
     * For storing a Householder vectors.
     */
    protected double[] householderVector;
    /**
     * For temporarily storage when applying Householder vectors. This is useful for
     * avoiding unneeded garbage collection and for improving cache performance when traversing columns.
     */
    protected double[] workArray;
    /**
     * Flag indicating if a Householder reflector was needed for the current column meaning an update needs to be applied.
     */
    protected boolean applyUpdate;
    /**
     * Stores the shifted value of the first entry in a Householder vector.
     */
    private double shift;


    /**
     * Constructs a Hessenberg decomposer for symmetric matrices. To compute the
     */
    public SymmHess() {
        computeQ = false;
        enforceSymmetric = false;
    }


    /**
     * Constructs a Hessenberg decomposer for symmetric matrices.
     * @param computeQ Flag indicating if the orthogonal {@code Q} matrix from the Hessenberg decomposition should be explicitly computed.
     * If true, then the {@code Q} matrix will be computed explicitly.
     */
    public SymmHess(boolean computeQ) {
        enforceSymmetric = false;
        this.computeQ = computeQ;
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
        this.enforceSymmetric = enforceSymmetric;
        this.computeQ = computeQ;
    }


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose. (Assumed to be a symmetric matrix).
     * @return A reference to this decomposer.
     */
    @Override
    public SymmHess decompose(Matrix src) {
        setUp(src);
        int stop = size-2;

        for(int k=0; k<stop; k++) {
            computeHouseholder(k+1);
            if(applyUpdate) updateData(k+1);
        }

        return this;
    }


    /**
     * Gets the Hessenberg matrix from the decomposition. The matrix will be symmetric tri-diagonal.
     * @return The symmetric tri-diagonal (Hessenberg) matrix from this decomposition.
     */
    public Matrix getH() {
        Matrix H = new Matrix(size);
        H.entries[0] = transformMatrix.entries[0];

        int idx1;
        int idx0;
        int rowOffset = H.numRows;

        for(int i=1; i<size; i++) {
            idx1 = rowOffset + i;
            idx0 = idx1 - H.numRows;

            H.entries[idx1] = transformMatrix.entries[idx1]; // extract diagonal value.

            // extract off-diagonal values.
            double a = transformMatrix.entries[idx0];
            H.entries[idx0] = a;
            H.entries[idx1 - 1] = a;

            // Update row index.
            rowOffset += H.numRows;
        }

        if(size > 1) {
            int rowColBase = size*size - 1;
            H.entries[rowColBase] = transformMatrix.entries[rowColBase];
            H.entries[rowColBase - 1] = transformMatrix.entries[rowColBase - size];
        }

        return H;
    }


    /**
     * <p>Gets the unitary {@code Q} matrix from the Hessenberg decomposition.</p>
     *
     * <p>Note, if the reflectors for this decomposition were not saved, then {@code Q} can not be computed and this method will be
     * null.</p>
     *
     * @return The {@code Q} matrix from the {@code QR} decomposition. Note, if the reflectors for this decomposition were not saved,
     * then {@code Q} can not be computed and this method will return {@code null}.
     */
    public Matrix getQ() {
        if(!computeQ)
            return null;

        Matrix Q = Matrix.I(size);

        for(int j=size - 1; j>=1; j--) {
            householderVector[j] = 1.0; // Ensure first value of reflector is 1.

            for(int i=j + 1; i<size; i++) {
                householderVector[i] = transformMatrix.entries[i*size + j - 1]; // Extract column containing reflector vector.
            }

            if(qFactors[j]!=0) { // Otherwise, no reflector to apply.
                Householder.leftMultReflector(Q, householderVector, qFactors[j], j, j, size, workArray);
            }
        }

        return Q;
    }


    /**
     * Computes the Householder vector for the first column of the sub-matrix with upper left corner at {@code (j, j)}.
     *
     * @param j Index of the upper left corner of the sub-matrix for which to compute the Householder vector for the first column.
     *          That is, a Householder vector will be computed for the portion of column {@code j} below row {@code j}.
     */
    protected void computeHouseholder(int j) {
        // Initialize storage array for Householder vector and compute maximum absolute value in jth column at or below jth row.
        double maxAbs = findMaxAndInit(j);
        norm = 0; // Ensure norm is reset.

        applyUpdate = maxAbs >= Flag4jConstants.EPS_F64;

        if(!applyUpdate) {
            currentFactor = 0;
        } else {
            computePhasedNorm(j, maxAbs);

            householderVector[j] = 1.0; // Ensure first value in Householder vector is one.
            for(int i=j+1; i<size; i++) {
                householderVector[i] /= shift; // Scale all but first entry of the Householder vector.
            }
        }

        qFactors[j] = currentFactor; // Store the factor for the Householder vector.
    }


    /**
     * Computes the norm of column {@code j} below the {@code j}th row of the matrix to be decomposed. The norm will have the same
     * parity as the first entry in the sub-column.
     * @param j Column to compute norm of below the {@code j}th row.
     * @param maxAbs Maximum absolute value in the column. Used for scaling norm to minimize potential overflow issues.
     */
    protected void computePhasedNorm(int j, double maxAbs) {
        // Computes the 2-norm of the column.
        for(int i=j; i<size; i++) {
            householderVector[i] /= maxAbs; // Scale entries of the householder vector to help reduce potential overflow.
            double scaledValue = householderVector[i];
            norm += scaledValue*scaledValue;
        }
        norm = Math.sqrt(norm); // Finish 2-norm computation for the column.

        // Change sign of norm depending on first entry in column for stability purposes in Householder vector.
        if(householderVector[j] < 0) norm = -norm;

        shift = householderVector[j] + norm;
        currentFactor = shift/norm;
        norm *= maxAbs; // Rescale norm.
    }


    /**
     * Finds the maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row. This method also initializes
     * the first {@code numRows-j} entries of the storage array {@link #householderVector} to the entries of this column.
     * @param j Index of column (and starting row) to compute max of.
     * @return The maximum value in {@link #transformMatrix} at column {@code j} at or below the {@code j}th row.
     */
    protected double findMaxAndInit(int j) {
        double maxAbs = 0;

        // Compute max-abs value in row. (Equivalent to max value in column since matrix is symmetric.)
        int rowU = (j-1)*size;
        for(int i=j; i<size; i++) {
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
    protected void setUp(Matrix src) {
        if(enforceSymmetric && !src.isSymmetric()) // If requested, check the matrix is symmetric.
            throw new LinearAlgebraException("Decomposition only supports symmetric matrices.");
        else
            ParameterChecks.assertSquareMatrix(src.shape); // Otherwise, Just ensure the matrix is square.

        size = src.numRows;
        householderVector = new double[size];
        qFactors = new double[size];
        workArray = new double[size];
        copyUpperTri(src);
    }


    /**
     * Copies the upper triangular portion of a matrix to the working matrix {@link #transformMatrix}.
     * @param src The source matrix to decompose of.
     */
    protected void copyUpperTri(Matrix src) {
        transformMatrix = new Matrix(size);

        // Copy upper triangular portion.
        for(int i=0; i<size; i++) {
            int pos = i*size + i;
            System.arraycopy(src.entries, pos, transformMatrix.entries, pos, size - i);
        }
    }


    /**
     * Updates the {@link #transformMatrix} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param j Index of sub-matrix for which the Householder reflector was computed for.
     */
    protected void updateData(int j) {
        Householder.symmLeftRightMultReflector(transformMatrix, householderVector, currentFactor, j, workArray);

        if(j < size) transformMatrix.entries[(j-1)*size + j] = -norm;
        if(computeQ) {
            // Store the Q matrix in the lower portion of the transformation data matrix.
            int col = j-1;
            for(int i=j+1; i<size; i++) {
                transformMatrix.entries[i*size + col] = householderVector[i];
            }
        }
    }
}
