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

import org.flag4j.arrays_old.dense.CMatrixOld;
import org.flag4j.arrays_old.dense.CVectorOld;
import org.flag4j.arrays_old.dense.MatrixOld;
import org.flag4j.complex_numbers.CNumber;
import org.flag4j.linalg.Eigen;
import org.flag4j.linalg.decompositions.hess.RealHessOld;
import org.flag4j.linalg.transformations.Givens;
import org.flag4j.linalg.transformations.HouseholderOld;
import org.flag4j.operations_old.common.real.AggregateReal;
import org.flag4j.rng.RandomCNumber;

import static org.flag4j.util.Flag4jConstants.EPS_F64;

/**
 * <p>This class computes the Schur decomposition of a real dense square matrix.</p>
 *
 * <p>That is, decompose a square matrix {@code A} into {@code A=UTU}<sup>T</sup> where {@code U} is an orthogonal
 * matrix and {@code T} is a block-upper triangular matrix called the real-Schur form of {@code A}. {@code T} is upper triangular
 * except for possibly 2x2 blocks along the diagonal. {@code T} is similar to {@code A}.
 * </p>
 *
 * <p>This code was adapted from the code found in the <a href="http://ejml.org/wiki/index.php?title=Main_Page">EJML</a>
 * library and the description of
 * the Francis implicit double shifted QR algorithm given in
 * <a href="https://www.math.wsu.edu/faculty/watkins/books.html">Fundamentals of MatrixOld
 * Computations 3rd Edition by David S. Watkins</a>.</p>
 */
public class RealSchur extends Schur<MatrixOld, double[]> {

    /**
     * For computing the norm of a column for use when computing HouseholderOld reflectors.
     */
    protected double norm;
    /**
     * Stores the scalar factor &alpha for use in computation of the HouseholderOld reflector {@code P = I - }&alpha{@code vv}<sup>T
     * </sup>.
     */
    protected double currentFactor;


    /**
     * <p>Creates a decomposer to compute the Schur decomposition for a real dense matrix.</p>
     *
     * <p>Note: This decomposer <i><b>may</b></i> use random numbers during the decomposition. If reproducible results are needed,
     * set the seed for the pseudo-random number generator using {@link #RealSchur(long)}</p>
     */
    public RealSchur() {
        super(true, new RandomCNumber(), new RealHessOld());
    }


    /**
     * <p>Creates a decomposer to compute the Schur decomposition for a real dense matrix where the {@code U} matrix may or may not
     * be computed.</p>
     *
     * <p>If the {@code U} matrix is not needed, passing {@code computeU = false} may provide a performance improvement.</p>
     *
     * <p>By default if a constructor with no {@code computeU} parameter is called, {@code U} <b>WILL</b> be computed.</p>
     *
     * <p>Note: This decomposer <i><b>may</b></i> use random numbers during the decomposition. If reproducible results are needed,
     * set the seed for the pseudo-random number generator using {@link #RealSchur(boolean, long)}</p>
     *
     * @param computeU Flag indicating if the unitary {@code U} matrix should be computed for the Schur decomposition. If true,
     * {@code U} will be computed. If false, {@code U} will not be computed.
     */
    public RealSchur(boolean computeU) {
        super(computeU, new RandomCNumber(), new RealHessOld(computeU));
    }


    /**
     * Creates a decomposer to compute the Schur decomposition for a real dense matrix.
     * @param seed Seed to use for pseudo-random number generator when computing exceptional shifts during the {@code QR} algorithm.
     */
    public RealSchur(long seed) {
        super(true, new RandomCNumber(seed), new RealHessOld());
    }


    /**
     * Creates a decomposer to compute the Schur decomposition for a real dense matrix.
     * @param seed Seed to use for pseudo-random number generator when computing exceptional shifts during the {@code QR} algorithm.
     */
    public RealSchur(boolean computeU, long seed) {
        super(computeU, new RandomCNumber(seed), new RealHessOld(computeU));
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
    public RealSchur setExceptionalThreshold(int exceptionalThreshold) {
        // Provided so that calls like the following can be made:
        //    RealSchur schur = new RealSchur().setExceptionalThreshold(x).decompose()
        return (RealSchur) super.setExceptionalThreshold(exceptionalThreshold);
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
     * @return A reference to this decomposer.
     * @throws IllegalArgumentException If {@code maxIterationFactor} is not positive.
     */
    public RealSchur setMaxIterationFactor(int maxIterationFactor) {
        // Provided so calls like the following can be made:
        //    RealSchur schur = new RealSchur().setMaxIterationFactor(x).decompose()
        return (RealSchur) super.setMaxIterationFactor(maxIterationFactor);
    }


    /**
     * <p>Computes the Schur decomposition of the input matrix.</p>
     *
     * @implNote The Schur decomposition is computed using the Francis implicit double shifted {@code QR} algorithm.
     * There are known cases where this variant of the {@code QR} algorithm fail to converge. Random shifting is employed when the
     * matrix is not converging which greatly minimizes this issue. It is unlikely that a general matrix will fail to converge with
     * these random shifts however, no guarantees of convergence can be made.
     * @param src The source matrix to decompose.
     */
    @Override
    public RealSchur decompose(MatrixOld src) {
        decomposeBase(src);
        return this;
    }


    /**
     * Initializes temporary work arrays_old to be used in the decomposition.
     */
    @Override
    protected void setUpArrays() {
        householderVector = new double[numRows];
        workArray = new double[numRows]; // TODO: If givens are used in the single step, the work array will need length 2*numRows.
        shiftCol = new double[3]; // For storing non-zero entries (the first 2 or 3) of the first column of the double/single shift.
        temp = new double[9]; // For storing temporary values when computing shifts.
    }


    /**
     * Performs a full iteration of the single shifted {@code QR} algorithm (this includes the bulge chase) where the shift is
     * chosen to be a random value with the same magnitude as the lower right element of the working matrix. This can help the
     * {@code QR} converge for certain pathological cases where the double shift algorithm oscillates or fails to converge for
     * repeated eigenvalues.
     *
     * @param workingSize The current working size for the decomposition. I.e. all entries below this row have converged to an upper
     *                    or possible 2x2 block upper triangular form.
     */
    @Override
    protected void performExceptionalShift(int workingSize) {
        performSingleShift(workingSize, computeExceptionalShift(workingSize));
    }


    /**
     * Computes a random shift to help the {@code QR} algorithm converge if it gets stuck.
     * @param k The current size of the working matrix. Specifically, the index of the lower right value in the working matrix is
     *          {@code (k, k)}.
     * @return A shift in a random direction which has the same magnitude as the elements in the matrix.
     */
    protected double computeExceptionalShift(int k) {
        double value = T.entries[k*numRows + k];
        value = (value==0) ? 1 : Math.abs(value); // Ensure shift is not zero.

        double p = 1 - Math.pow(0.1, numExceptional);
        value *= p + 2*(1 - p)*(rng.nextDouble() - 0.5);

        // Choose a sign randomly.
        if(rng.nextBoolean()) {
            value = -value;
        }

        return value;
    }


    /**
     * Computes the non-zero entries of the first column for the single shifted {@code QR} algorithm.
     * @param k Size of current working matrix.
     * @param shift The shift to use.
     */
    protected void computeImplicitSingleShift(int k, double shift) {
        int leftIdx = k-1;
        shiftCol[0] = T.entries[leftIdx*numRows + leftIdx] - shift;
        shiftCol[1] = T.entries[k*numRows + leftIdx];
    }


    /**
     * Performs a full iteration of the implicit single shifted {@code QR} algorithm (this includes the bulge chase).
     * @param workingSize The current working size for the decomposition. I.e. all entries below this row have converged to an upper
     *                   or possible 2x2 block upper triangular form.
     * @param shift The shift to use in the implicit single shifted {@code QR} algorithm.
     */
    protected void performSingleShift(int workingSize, double shift) {
        // Compute the non-zero entries of first column for shifted matrix.
        computeImplicitSingleShift(workingSize, shift);

        // Extract non-zero values from first column in shifted matrix.
        double p1 = shiftCol[0];
        double p2 = shiftCol[1];

        for(int i=0; i<=workingSize-1; i++) {
            if(makeReflector(i, p1, p2)) // Construct reflector.
                applySingleShiftReflector(i, i>0); // Apply the reflector if needed.

            // Set values to be used in computing the next bulge chasing reflector.
            p1 = T.entries[(i + 1)*numRows + i];
            if(i < workingSize-1) p2 = T.entries[(i + 2)*numRows + i];
        }
    }


    /**
     * Applies reflector for the double shift. This method can be used to apply either be the reflector constructed for the first
     * column of the shifted matrix, or a reflector being used in the bulge chase of size 2 which arises from the first case.
     * @param i The starting row the reflector is being applied to.
     */
    protected void applySingleShiftReflector(int i, boolean set) {
        applyReflector(i, 1);  // Apply reflector for shift size of 1.

        // TODO: Since we only need to zero a single element, consider using a givens rotator here instead.
        //  There should be negligible stability difference between the two for a 2x2 rotator, but the givens rotator is more
        //  simple to calculate. However, this seems to be incorrect. Figure out what this should be or if the left/right
        //  multiplication methods are incorrect.
//        MatrixOld G = Givens.get2x2Rotator(T.entries[i*numRows + i - 1], T.entries[(i+1)*numRows + i - 1]);
//        Givens.leftMult2x2Rotator(T, G, i+1, givensWorkArray);
//        Givens.rightMult2x2Rotator(T, G, i+1, givensWorkArray);

        if(set) {
            // Explicitly zeros out values which should be zeroed by the reflector.
            T.entries[i*numRows + i - 1] = -norm;
            T.entries[(i+1)*numRows + i - 1] = 0.0;
        }
    }


    /**
     * Performs a full iteration of the Francis implicit double shifted {@code QR} algorithm (this includes the bulge chase).
     * @param workingSize The current working size for the decomposition. I.e. all entries below this row have converged to an upper
     *                   or possible 2x2 block upper triangular form.
     */
    protected void performDoubleShift(int workingSize) {
        // Compute the non-zero entries (first three) of the first column of the double shifted matrix.
        computeImplicitDoubleShift(workingSize);

        // Extract non-zero values in first column of the double shifted matrix.
        double p1 = shiftCol[0];
        double p2 = shiftCol[1];
        double p3 = shiftCol[2];

        // Apply shift and chase bulge.
        for(int i=0; i<=workingSize-2; i++) {
            if(makeReflector(i, p1, p2, p3)) // Construct HouseholderOld reflector.
                applyDoubleShiftReflector(i, i>0); // Apply the reflector if needed.

            // Set values to be used in computing the next bulge chasing reflector.
            p1 = T.entries[(i + 1)*numRows + i];
            p2 = T.entries[(i + 2)*numRows + i];
            if(i < workingSize-2) p3 = T.entries[(i + 3)*numRows + i];
        }

        // The last reflector in the bulge chase only acts on last two rows of the working matrix.
        if(makeReflector(workingSize-1, p1, p2)) // Construct HouseholderOld reflector.
            applySingleShiftReflector(workingSize-1, true); // Apply the reflector if needed.
    }


    /**
     * Computes the shifts for a Francis double shift iteration. Specifically, the shifts are the generalized Rayleigh quotients of
     * degree two.
     * @param workingSize Size of current working matrix.
     */
    protected void computeImplicitDoubleShift(int workingSize) {
        // The shift computed here, p, represent the double shift
        //  p = (T - rho1*I)(T - rho2*I)*e1 where I is the identity matrix, e1 is the first column of I, and (rho1, rho2)
        //  are taken to be the eigenvalues of the lower 2x2 sub-matrix within the working matrix.
        //  Note: (rho1, rho2) are either both real, or both complex (and specifically complex conjugates). In either case, has
        //  only three non-zero entries (the first three entries) all of which are real. Hence, all arithmetic may be carried out in
        //  real arithmetic. As such, eigenvalues are not explicitly computed as that would require complex arithmetic.

        // Extract values from lower right 2x2 sub-matrix within the working size.
        int leftIdx = workingSize-1;
        double x11 = T.entries[leftIdx*numRows + leftIdx];
        double x12 = T.entries[leftIdx*numRows + workingSize];
        double x21 = T.entries[workingSize*numRows + leftIdx];
        double x22 = T.entries[workingSize*numRows + workingSize];

        // Extract top right entries of T for use in computing the shift p.
        double a11 = T.entries[0];
        double a12 = T.entries[1];
        double a21 = T.entries[numRows];
        double a22 = T.entries[numRows + 1];
        double a32 = T.entries[2*numRows + 1];

        // Scale values to improve stability and help avoid possible over(under)flow issues.
        temp[0] = a11; temp[1] = a21; temp[2] = a12; temp[3] = a22; temp[4] = a32;
        temp[5] = x11; temp[6] = x22; temp[7] = x12; temp[8] = x21;
        double maxAbs = AggregateReal.maxAbs(temp);

        a11 /= maxAbs; a12 /= maxAbs; a21 /= maxAbs; a22 /= maxAbs; a32 /= maxAbs;
        x11 /= maxAbs; x12 /= maxAbs; x21 /= maxAbs; x22 /= maxAbs;

        double trace = x11 + x22; // Compute trace (useful in computing the shift p).
        double det = x11*x22 - x12*x21; // Compute determinant (useful in computing shift p).

        // Compute first three non-zero entries of the shift p.
        shiftCol[0] = a11*a11 + a12*a21 - a11*trace + det;
        shiftCol[1] = a21*(a11 + a22 - trace);
        shiftCol[2] = a32*a21;
    }


    /**
     * Applies reflector for the double shift. This method can be used to apply either be the reflector constructed for the first
     * column of the shifted matrix, or a reflector being used in the bulge chase of size 2 which arises from the first case.
     * @param i The starting row the reflector is being applied to.
     */
    protected void applyDoubleShiftReflector(int i, boolean set) {
        applyReflector(i, 2); // Apply reflector for shift size of 2.

        if(set) {
            // Explicitly zeros out values which should be zeroed by the reflector.
            T.entries[i*numRows + i - 1] = -norm;
            T.entries[(i+1)*numRows + i - 1] = 0.0;
            T.entries[(i+2)*numRows + i - 1] = 0.0;
        }
    }


    /**
     * Applies the constructed HouseholderOld reflector which has been constructed for the given shift size.
     * @param i The stating row the reflector is being applied to.
     * @param shiftSize The size of the shift the reflector was constructed for.
     */
    protected void applyReflector(int i, int shiftSize) {
        int endRow = i + shiftSize + 1;

        // Apply reflector to left (Assumes T is upper hessenburg except for possibly a bulge of size shiftSize).
        HouseholderOld.leftMultReflector(T, householderVector, currentFactor, i, i, endRow, workArray);
        // Apply reflector to right (Assumes T is upper hessenburg except for possibly a bulge of size shiftSize).
        HouseholderOld.rightMultReflector(T, householderVector, currentFactor, 0, i, endRow);

        if(computeU) {
            // Accumulate the reflector in U if it is being computed.
            HouseholderOld.rightMultReflector(U, householderVector, currentFactor, 0, i, endRow);
        }
    }


    /**
     * Constructs a householder reflector given specified values for a column to apply the reflector to. This reflector is stored in
     * indices {@code i}, {@code i+1}, and {@code i+2} of {@link #householderVector}.
     * @param i Row of working matrix to construct reflector for.
     * @param p1 First entry to in column to apply reflector to.
     * @param p2 Second entry in column to apply reflector to.
     * @param p3 Third entry in column to apply reflector to.
     * @return True if a reflector needs to be constructed to return matrix to upper Hessenburg form. False if column is
     * already in the correct form.
     */
    protected boolean makeReflector(int i, double p1, double p2, double p3) {
        // Scale components for stability and overflow purposes.
        double maxAbs = Math.max(Math.abs(p1), Math.max(Math.abs(p2), Math.abs(p3)));

        if(maxAbs <= EPS_F64*Math.abs(T.entries[i*numRows + i])) {
            return false; // No reflector needs to be constructed or applied.
        }

        p1 /= maxAbs;
        p2 /= maxAbs;
        p3 /= maxAbs;

        norm = Math.sqrt(p1*p1 + p2*p2 + p3*p3); // Compute scaled 2-norm.

        // Change sign of norm depending on first entry in column for stability purposes in HouseholderOld vector.
        if(p1 < 0) norm = -norm;

        double div = p1 + norm;
        currentFactor = div/norm;
        norm *= maxAbs; // Rescale norm to be proper magnitude.

        householderVector[i] = 1.0;
        householderVector[i+1] = p2 / div;
        householderVector[i+2] = p3 / div;

        return true;
    }


    /**
     * Constructs a householder reflector given specified values for a column to apply the reflector to. This reflector is stored in
     * indices {@code i} and {@code i+1} of {@link #householderVector}.
     * @param i Row of working matrix to construct reflector for.
     * @param p1 First entry to in column to apply reflector to.
     * @param p2 Second entry in column to apply reflector to.
     * @return True if a reflector needs to be constructed to return matrix to upper Hessenburg form. False if column is
     * already in the correct form.
     */
    protected boolean makeReflector(int i, double p1, double p2) {
        double maxAbs = Math.max(Math.abs(p1), Math.abs(p2));
        if(maxAbs <= EPS_F64*Math.abs(T.entries[i*numRows + i])) {
            return false; // No reflector needs to be constructed or applied.
        }

        // Scale components for stability and over(under)flow purposes.
        p1 /= maxAbs;
        p2 /= maxAbs;

        norm = Math.sqrt(p1*p1 + p2*p2); // Compute scaled norm.
        // Change sign of norm depending on first entry in column for stability purposes in HouseholderOld vector.
        if(p1 < 0) norm = -norm;

        double div = p1 + norm;
        currentFactor = div/norm;
        norm *= maxAbs; // Rescale norm to be proper magnitude.

        householderVector[i] = 1.0; // Ensure first value of reflector is 1.
        householderVector[i+1] = p2 / div;

        return true; // Reflector has been constructed and must be applied.
    }


    /**
     * Checks for convergence of lower 2x2 sub-matrix within working matrix to upper triangular or block upper triangular form. If
     * convergence is found, this will also zero out the values which have converged to near zero.
     * @param workingSize Size of current working matrix.
     * @return Returns the amount the working matrix size should be deflated. Will be zero if no convergence is detected, one if
     * convergence to upper triangular form is detected and two if convergence to block upper triangular form is detected.
     */
    protected int checkConvergence(int workingSize) {
        int leftRow = (workingSize-1)*numRows;

        double a11 = T.entries[(workingSize-2)*numRows + workingSize - 2];
        double a21 = T.entries[leftRow + workingSize - 2];
        double a22 = T.entries[leftRow + workingSize - 1];
        double a23 = T.entries[leftRow + workingSize];
        double a32 = T.entries[workingSize*numRows + workingSize - 1];
        double a33 = T.entries[workingSize*numRows + workingSize];

        // Uses deflation criteria proposed by Wilkinson: |A[k, k-1]| < eps*(|A[k, k]| + |A[k-1, k-1]|)
        // AND the deflation criteria proposed by Ahues and Tisseur:
        //     |A[k, k-1]| *|A[k-1, k]| <= eps * |A[k, k]| * |A[k, k] - A[k-1, k-1]|

        if(Math.abs(a32) < EPS_F64*( Math.abs(a33) + Math.abs(a22) )
                && Math.abs(a32)*Math.abs(a23) <= EPS_F64*Math.abs(a33)*Math.abs(a33 - a22)) {
            T.entries[workingSize*numRows + workingSize - 1] = 0; // Zero out converged value.
            return 1; // Deflate by 1.
        } else if(Math.abs(a21) < EPS_F64*( Math.abs(a11) + Math.abs(a22) )) {
            T.entries[leftRow + workingSize - 2] = 0; // Zero out converged value.
            return 2; // Deflate by 2.
        }

        return 0; // No convergence detected. Do not deflate.
    }


    /**
     * <p>Converts the real schur form computed in the last decomposition to the complex Schur form.</p>
     *
     * <p>That is, converts the real block
     * upper triangular Schur matrix to a complex valued properly upper triangular matrix. If the unitary transformation matrix
     * {@code U} was computed, the transformations will also be updated accordingly.</p>
     *
     * <p>This method was adapted from the code given by
     * <a href="https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.rsf2csf.html">scipy.linalg.rsf2csf</a> (v1.12.0).</p>
     *
     * @return An array of length 2 containing the complex Schur matrix {@code T} from the last decomposition, and if computed, the
     * complex unitary transformation matrix {@code U} from the decomposition. If {@code U} was not computed, then the arrays_old second
     * value will be null.
     */
    public CMatrixOld[] real2ComplexSchur() {
        // Convert matrices to complex matrices.
        CMatrixOld tComplex = T.toComplex();
        CMatrixOld uComplex = computeU ? U.toComplex() : null;
        CNumber[] givensWorkComplex = new CNumber[2*numRows];

        for(int m=numRows-1; m>0; m--) {
            CNumber a11 = tComplex.entries[(m - 1)*numRows + m - 1];
            CNumber a12 = tComplex.entries[(m - 1)*numRows + m];
            CNumber a21 = tComplex.entries[m*numRows + m - 1];
            CNumber a22 = tComplex.entries[m*numRows + m];

            if(a21.mag() > EPS_F64*(a11.mag() + a22.mag())) {
                // non-converged 2x2 block found.
                CNumber[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                mu[0] = mu[0].sub(a22); // Shift eigenvalue.

                // Construct a givens rotator to bring matrix into properly upper triangular form.
                CMatrixOld G = Givens.get2x2Rotator(new CVectorOld(mu[0], a21));
                // Apply rotation to T matrix to bring it into upper triangular form.
                Givens.leftMult2x2Rotator(tComplex, G, m, givensWorkComplex);
                // Apply hermitian transpose to keep transformation similar.
                Givens.rightMult2x2Rotator(tComplex, G, m, givensWorkComplex);

                if(uComplex != null) {
                    // Accumulate similarity transforms in the U matrix.
                    Givens.rightMult2x2Rotator(uComplex, G, m, givensWorkComplex);
                }

                tComplex.set(0, m, m-1);
            }
        }

        return new CMatrixOld[]{tComplex, uComplex};
    }
}
