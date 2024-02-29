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
import org.flag4j.sparse.SymmTriDiagonal;
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
     * For holding a working copy of the source matrix to be decomposed.
     */
    protected Matrix A;
    /**
     * Stores the symmetric tri-diagonal matrix from the symmetric Hessenburg Decomposition.
     */
    protected SymmTriDiagonal H;
    /**
     * For storing the diagonal entries of the symmetric tri-diagonal matrix.
     */
    protected double[] diag;
    /**
     * For storing the off-diagonal entries of the symmetric tri-diagonal matrix.
     */
    protected double[] offDiag;
    /**
     * Stores the orthogonal Q matrix from the Hessenburg decomposition {@code A=QHQ}<sup>T</sup>
     */
    protected Matrix Q;
    /**
     * For storing norms of columns in A when computing Householder reflectors.
     */
    double norm;
    /**
     * Size of the symmetric matrix to be decomposed. That is, the number of rows and columns.
     */
    protected int size;
    /**
     * Flag indicating if an explicit check should be made that the matrix to be decomposed is symmetric.
     */
    protected final boolean enforceSymmetric = true; // TODO: Make this configurable via constructors.
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
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose. (Assumed to be a symmetric matrix).
     * @return A reference to this decomposer.
     */
    @Override
    public SymmHess decompose(Matrix src) {
        setUp(src);

        int stop = size-1;

        for(int k=0; k<stop; k++) {
            computeHouseholder(k+1);
            if(applyUpdate) {
                Householder.symmLeftRightMultReflector(A, householderVector, currentFactor, k+1, new double[size]);
                // TODO: Apply reflector to symmetric matrix.
            }
        }

        return this;
    }


    public Matrix getT() {
        Matrix T = new Matrix(size);

        System.out.println("A_HESS:\n" + A);

        T.entries[0] = A.entries[0];

        for(int i=1; i<size; i++) {
            T.set(A.get(i, i), i, i);
            double a = A.get(i - 1, i);
            T.set(a, i - 1, i);
            T.set(a, i, i - 1);
        }

        if(size > 1) {
            T.entries[(size - 1)*size + size - 1] = A.entries[(size - 1)*size + size - 1];
            T.entries[(size - 1)*size + size - 2] = A.entries[(size - 2)*size + size - 1];
        }

        return T;
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
     * Finds the maximum value in {@link #A} at column {@code j} at or below the {@code j}th row. This method also initializes
     * the first {@code numRows-j} entries of the storage array {@link #householderVector} to the entries of this column.
     * @param j Index of column (and starting row) to compute max of.
     * @return The maximum value in {@link #A} at column {@code j} at or below the {@code j}th row.
     */
    protected double findMaxAndInit(int j) {
        double maxAbs = 0;
        System.out.println("A:\n" + A);

        // Compute max-abs value in column. (Equivalent to max value in row since matrix is symmetric.)
        int rowU = (j-1)*size;
        for(int i=j; i<size; i++) {
            double d = householderVector[i] = A.entries[rowU + i];
            maxAbs = Math.max(Math.abs(d), maxAbs);
        }

        System.out.println("Max abs: " + maxAbs + "\n\n");

//        // Max-abs value in column
//        for(int i=j; i<size; i++) {
//            double d = householderVector[i] = A.entries[idx];
//            idx += size; // Move index to next row.
//            maxAbs = Math.max(Math.abs(d), maxAbs);
//        }

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
//        copyLowerTri(src);
        copyUpperTri(src);
//        A = src.copy();
        diag = new double[size];
        offDiag = new double[size-1];
    }


    /**
     * Copies the lower triangular portion of a matrix to the working matrix {@link #A}.
     * @param src The source matrix to decompose of.
     */
    protected void copyLowerTri(Matrix src) {
        A = new Matrix(size);

        // Copy lower triangular portion.
        for(int i=0; i<size; i++) {
            System.arraycopy(src.entries, i*size, A.entries, i*size, i+1);
        }
    }


    /**
     * Copies the upper triangular portion of a matrix to the working matrix {@link #A}.
     * @param src The source matrix to decompose of.
     */
    protected void copyUpperTri(Matrix src) {
        A = new Matrix(size);

        // Copy lower triangular portion.
        for(int i=0; i<size; i++) {
            int pos = i*size + i;
            System.arraycopy(src.entries, pos, A.entries, pos, size - i);
        }
    }


    public static void main(String[] args) {
        Matrix A = new Matrix(new double[][]{
                {1, 2, 3},
                {2, 5, 6},
                {3, 6, 9}
        });

        Matrix B = new Matrix(new double[][]{
                {1,  -4,   2.5, 15, 0   },
                {-4,  2,   8.1, 4,  1   },
                {2.5, 8.1, 4,  -9,  8.25},
                {15,  4,  -9, 10.3, 6   },
                {0,   1,  8.25, 6, -18.5},
        });

        Matrix C = new Matrix(new double[][]{
                {1.4, -0.002, 14.51},
                {-0.002, 4.501, -9.14},
                {14.51, -9.14, 16.5}
        });

        SymmHess symmHess = new SymmHess();

//        symmHess.decompose(A);
//        System.out.println("A:\n" + A + "\n");
//        System.out.println("T:\n" + symmHess.getT() + "\n\n");
//
//        symmHess.decompose(B);
//        System.out.println("B:\n" + B + "\n");
//        System.out.println("T:\n" + symmHess.getT());

        symmHess.decompose(C);
        System.out.println("C:\n" + C + "\n");
        System.out.println("T:\n" + symmHess.getT());
    }
}
