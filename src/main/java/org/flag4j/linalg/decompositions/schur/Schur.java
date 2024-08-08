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

package org.flag4j.linalg.decompositions.schur;

import org.flag4j.core.MatrixMixin;
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.linalg.decompositions.unitary.UnitaryDecomposition;
import org.flag4j.rng.RandomCNumber;
import org.flag4j.util.ParameterChecks;
import org.flag4j.util.exceptions.LinearAlgebraException;


/**
 * <p>The base class for Schur decompositions.</p>
 *
 * <p>The Schur decomposition decompose a square matrix {@code A} into {@code A=UTU<sup>H</sup>} where {@code U} is a unitary
 * matrix and {@code T} is a quasi-upper triangular matrix called the Schur form of {@code A}. {@code T} is upper triangular
 * except for possibly 2x2 blocks along the diagonal. {@code T} is similar to {@code A} meaning they share the same eigenvalues.
 * </p>
 *
 * @param <T> The type of matrix to be decomposed.
 * @param <U> The type for the internal storage datastructure of the matrix to be decomposed.
 */
public abstract class Schur<
        T extends MatrixMixin<T, ?, ?, ?, ?, ?, ?, ?>,
        U> implements Decomposition<T> {

    /**
     * Random number generator to be used when computing a random exceptional shift.
     */
    protected final RandomCNumber rng;
    /**
     * Default number of iterations to apply before doing an exceptional shift.
     */
    protected final int DEFAULT_EXCEPTIONAL_ITERS = 20;
    /**
     * Default factor for computing the maximum number of iterations to perform.
     */
    protected final int DEFAULT_MAX_ITERS_FACTOR = 30;
    /**
     *For storing the (possibly block) upper triangular matrix {@code T} in the Schur decomposition.
     */
    protected T T;
    /**
     *For storing the unitary {@code U} matrix in the Schur decomposition.
     */
    protected T U;
    /**
     *Decomposer to compute the Hessenburg decomposition as a setup step for the implicit double step QR algorithm.
     */
    protected UnitaryDecomposition<T, U> hess;
    /**
     *Stores the number of rows in the matrix being decomposed.
     */
    protected int numRows;
    /**
     * Stores the vector {@code v} in the Householder reflector {@code P = I - }&alpha{@code vv}<sup>T</sup>.
     */
    protected U householderVector;
    /**
     * Stores the non-zero entries of the first column of the shifted matrix {@code (A-}&rho<sub>1</sub>{@code I)(A-}&rho<sub>2</sub
     * >{@code I)}
     * where
     * &rho<sub>1</sub> and &rho<sub>2</sub> are the two shifts.
     */
    protected U shiftCol;
    /**
     * An array for storing temporary values along the colum of a matrix when applying Householder reflectors.
     * This can help improve cache performance when applying the reflector.
     */
    protected U workArray;
    /**
     * Array for holding temporary values when computing the shifts.
     */
    protected U temp;
    /**
     * Factor for computing the maximum number of iterations to run the {@code QR} algorithm for.
     */
    protected int maxIterationsFactor;
    /**
     * Maximum number of iterations to run QR algorithm for.
     */
    protected int maxIterations;
    /**
     * Number of iterations to run without deflation before an exceptional shift is done.
     */
    protected int exceptionalThreshold;
    /**
     * The number of iterations run in the QR algorithm without deflating or performing an exceptional shift.
     */
    protected int sinceLastExceptional;
    /**
     * The total number of exceptional shifts that have been used during the decomposition.
     */
    protected int numExceptional;
    /**
     * Flag indicating if a check should be made during the decomposition that the working matrix contains only finite values.
     * If true, an explicit check will be made and an exception will be thrown if {@link Double#isFinite(double) non-finite} values are
     * found. If false, no check will be made and the floating point arithmetic will carry on with {@link Double#POSITIVE_INFINITY
     * infinities},  {@link Double#NEGATIVE_INFINITY negative-infinities}, and {@link Double#NaN NaNs} present.
     */
    protected boolean checkFinite;  // TODO: Make use of this field.
    /**
     * Flag indicating if the orthogonal matrix {@code U} in the Schur decomposition should be computed. If false, {@code U} will
     * not be computed. This may provide performance improvements for large matrices when {@code U} is not required (for instance:
     * in eigenvalue computations where eigenvectors are not needed).
     */
    protected final boolean computeU;


    /**
     * <p>Creates a decomposer to compute the Schur decomposition for a real dense matrix.</p>
     *
     * <p>If the {@code U} matrix is not needed, passing {@code computeU = false} may provide a performance improvement.</p>
     *
     * @param computeU Flag indicating if the unitary {@code U} matrix should be computed for the Schur decomposition. If true,
     * {@code U} will be computed. If false, {@code U} will not be computed.
     * @param rng Random number generator to use when performing random exceptional shifts.
     * @param hess Decomposer to compute the Hessenburg decomposition as a setup step for the {@code QR} algorithm.
     */
    public Schur(boolean computeU, RandomCNumber rng, UnitaryDecomposition<T, U> hess) {
        maxIterationsFactor = DEFAULT_MAX_ITERS_FACTOR;
        exceptionalThreshold = DEFAULT_EXCEPTIONAL_ITERS;
        this.rng = rng;
        this.computeU = computeU;
        this.hess = hess;
    }


    /**
     * <p>Sets the number of iterations of the {@code QR} algorithm to perform without deflation before performing a random shift.</p>
     *
     * <p>That is, if {@code exceptionalThreshold = 10}, then at most 10 iterations {@code QR} algorithm iterations will be performed.
     * If, by the 10th iteration, no convergence has been detected which allows for deflation, then a {@code QR} algorithm iteration
     * will be performed with a random (i.e. exceptional) shift.</p>
     *
     * <p>By default, the threshold is set to {@link #DEFAULT_EXCEPTIONAL_ITERS}</p>
     *
     * @param exceptionalThreshold The new exceptional shift threshold. i.e. the number of iterations to perform without deflation
     *                             before performing an iteration with random shifts.
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If {@code exceptionalThreshold} is not positive.
     */
    public Schur<T, U> setExceptionalThreshold(int exceptionalThreshold) {
        ParameterChecks.assertPositive(exceptionalThreshold);
        this.exceptionalThreshold = exceptionalThreshold;
        return this;
    }


    /**
     * <p>Specify maximum iteration factor for computing the total number of iterations to run the {@code QR} algorithm
     * for when computing the decomposition. The maximum number of iterations is computed as
     * <pre>
     *     {@code maxIteration = maxIterationFactor * src.numRows;} </pre>
     * If the algorithm does not converge within this limit, an error will be thrown.</p>
     *
     * <p>By default, this is computed as
     * <pre>
     *     {@code maxIterations = }{@link #DEFAULT_MAX_ITERS_FACTOR}{@code * src.numRows;}</pre>
     *
     * where {@code src} is the matrix
     * being decomposed.</p>
     *
     * @param maxIterationFactor maximum iteration factor for use in computing the total maximum number of iterations to run the
     * {@code QR} algorithm for.
     * @throws IllegalArgumentException If {@code maxIterationFactor} is not positive.
     */
    public Schur<T, U> setMaxIterationFactor(int maxIterationFactor) {
        ParameterChecks.assertPositive(maxIterationFactor);
        this.maxIterationsFactor = maxIterationFactor;
        return this;
    }


    /**
     * Gets the upper, or possibly block-upper, triangular Schur matrix {@code T} from the Schur decomposition
     * @return The T matrix from the Schur decomposition {@code A=UTU}<sup>H</sup>
     */
    public T getT() {
        return T;
    }


    /**
     * Gets the unitary matrix {@code U} from the Schur decomposition containing the Schur vectors as its columns.
     * @return {@code A=UTU}<sup>H</sup>
     */
    public T getU() {
        return U;
    }


    /**
     * <p>Computes the Schur decomposition of the input matrix.</p>
     *
     * @implNote The Schur decomposition is computed using Francis implicit double shifted {@code QR} algorithm.
     * There are known cases where this variant of the {@code QR} algorithm fail to converge. Random shifting is employed when the
     * matrix is not converging which greatly minimizes this issue. It is unlikely that a general matrix will fail to converge with
     * these random shifts however, no guarantees of convergence can be made.
     * @param src The source matrix to decompose.
     * @throws LinearAlgebraException If the decomposition does not converge within the specified number of max iterations. See
     * {@link }
     */
    public void decomposeBase(T src) {
        setUp(src);

        int workingSize = numRows-1;
        int iters = 0;

        while(workingSize >= 2 && iters < maxIterations) {
            if(sinceLastExceptional >= exceptionalThreshold) {
                // Perform an exceptional shift iteration.
                sinceLastExceptional = 0; // Reset number of iterations completed.
                performExceptionalShift(workingSize);
            } else {
                // Perform a normal double shift iteration.
                sinceLastExceptional++; // Increase number of iterations performed without an exceptional shift.
                numExceptional++;
                performDoubleShift(workingSize);
            }

            // Check for convergence and deflate as needed.
            int deflate = checkConvergence(workingSize);
            if(deflate > 0) {
                sinceLastExceptional = 0; // Reset the number of iterations since the last exceptional shift.
                workingSize -= deflate; // Reduce working size.
            }

            iters++;
        }

        if(iters == maxIterations) {
            throw new LinearAlgebraException("Schur decomposition failed to converge in " + maxIterations + " iterations.");
        }
    }


    /**
     * Performs basic setup and initializes data structures to be used in the decomposition.
     * @param src The matrix to be decomposed.
     */
    protected void setUp(T src) {
        ParameterChecks.assertSquare(src.shape());

        sinceLastExceptional = 0;
        numExceptional = 0;
        numRows = src.numRows();
        maxIterations = numRows*maxIterationsFactor;

        setUpArrays();

        hess.decompose(src);
        T = hess.getUpper(); // Reduce matrix to upper Hessenburg form.
        // Initialize U as the product of transformations used in Hessenburg decomposition if requested.
        U = computeU ? hess.getQ() : null; // Hessenburg decomposition computes U lazily only when getQ() is called.
    }


    /**
     * Initializes temporary work arrays to be used in the decomposition.
     */
    protected abstract void setUpArrays();


    /**
     * Performs a full iteration of the single shifted {@code QR} algorithm (this includes the bulge chase) where the shift is
     * chosen to be a random value with the same magnitude as the lower right element of the working matrix. This can help the
     * {@code QR} converge for certain pathological cases where the double shift algorithm oscillates or fails to converge for
     * repeated eigenvalues.
     * @param workingSize The current working size for the decomposition. I.e. all entries below this row have converged to an upper
     *                   or possible 2x2 block upper triangular form.
     */
    protected abstract void performExceptionalShift(int workingSize);


    /**
     * Performs a full iteration of the Francis implicit double shifted {@code QR} algorithm (this includes the bulge chase).
     * @param workingSize The current working size for the decomposition. I.e. all entries below this row have converged to an upper
     *                   or possible 2x2 block upper triangular form.
     */
    protected abstract void performDoubleShift(int workingSize);


    /**
     * Checks for convergence of lower 2x2 sub-matrix within working matrix to upper triangular or block upper triangular form. If
     * convergence is found, this will also zero out the values which have converged to near zero.
     * @param workingSize Size of current working matrix.
     * @return Returns the amount the working matrix size should be deflated. Will be zero if no convergence is detected, one if
     * convergence to upper triangular form is detected and two if convergence to block upper triangular form is detected.
     */
    protected abstract int checkConvergence(int workingSize);
}
