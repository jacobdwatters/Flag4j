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

package com.flag4j.linalg.decompositions.schur;

import com.flag4j.complex_numbers.CNumber;
import com.flag4j.dense.CMatrix;
import com.flag4j.dense.CVector;
import com.flag4j.dense.Matrix;
import com.flag4j.linalg.Eigen;
import com.flag4j.linalg.decompositions.Decomposition;
import com.flag4j.linalg.decompositions.hess.RealHessenburgDecomposition;
import com.flag4j.linalg.transformations.Givens;
import com.flag4j.linalg.transformations.Householder;
import com.flag4j.operations.common.real.AggregateReal;
import com.flag4j.rng.RandomTensor;
import com.flag4j.util.ParameterChecks;

import static com.flag4j.linalg.decompositions.schur.SchurTestHelpers.printBulge;
import static com.flag4j.util.Flag4jConstants.EPS_F64;


/**
 * <p>This class computes the Schur decomposition of a square matrix.</p>
 *
 * <p>That is, decompose a square matrix {@code A} into {@code A=UTU<sup>T</sup>} where {@code U} is an orthogonal
 * matrix and {@code T} is a block-upper triangular matrix called the real-Schur form of {@code A}. {@code T} is upper triangular
 * except for possibly 2x2 blocks along the diagonal. {@code T} is similar to {@code A}.
 * </p>
 *
 * <p>This code was adapted from the <a href="http://ejml.org/wiki/index.php?title=Main_Page">EJML</a> library, the description of
 * the Francis implicit double shifted QR algorithm given in
 * <a href="https://www.math.wsu.edu/faculty/watkins/books.html">Fundamentals of Matrix
 * Computations 3rd Edition by David S. Watkins</a>,
 * and the <a href="https://www.netlib.org/lapack/">LAPACK</a> library</p>
 */
public class RealSchurFrancisOptimized implements Decomposition<Matrix> {

    /**
     * Default number of iterations to apply before doing an exceptional shift.
     */
    private final int DEFAULT_EXCEPTIONAL_ITERS = 20;
    /**
     * Default factor for computing the maximum number of iterations to perform.
     */
    private final int DEFAULT_MAX_ITERS_FACTOR = 30;

    /**
    *For storing the (possibly block) upper triangular matrix {@code T} in the Schur decomposition.
     */
    protected Matrix T;
    /**
    *For storing the unitary {@code U} matrix in the Schur decomposition.
     */
    protected Matrix U;
    /**
    *Decomposer to compute the Hessenburg decomposition as a setup step for the implicit double step QR algorithm.
     */
    protected RealHessenburgDecomposition hess = new RealHessenburgDecomposition();
    /**
    *Stores the number of rows in the matrix being decomposed.
     */
    protected int numRows;
    /**
     * For computing the norm of a column for use when computing Householder reflectors.
     */
    protected double norm;
    /**
     * Stores the scalar factor &alpha for use in computation of the Householder reflector {@code P = I - }&alpha{@code vv}<sup>T
     * </sup>.
     */
    protected double currentFactor;
    /**
     * Stores the vector {@code v} in the Householder reflector {@code P = I - }&alpha{@code vv}<sup>T</sup>.
     */
    protected double[] householderVector;
    /**
     * Stores the non-zero entries of the first column of the shifted matrix {@code (A-}&rho<sub>1</sub>{@code I)(A-}&rho<sub>2</sub
     * >{@code I)}
     * where
     * &rho<sub>1</sub> and &rho<sub>2</sub> are the two shifts.
     */
    protected double[] shiftCol;
    /**
     * An array for storing temporary values along the colum of a matrix when applying Householder reflectors.
     * This can help improve cache performance when applying the reflector.
     */
    protected double[] workArray;
    /**
     * Array for holding temporary values when computing the shifts.
     */
    protected double[] temp;
    /**
     * Maximum number of iterations to run QR algorithm for.
     */
    protected int maxIterations;
    /**
     * Number of iterations to run without deflation before an exceptional shift is done.
     */
    protected int exceptionalIters;
    /**
     * Flag indicating if a check should be made during the decomposition that the working matrix contains only finite values.
     * If true, an explicit check will be made and an exception will be thrown if {@link Double#isFinite(double) non-finite} values are
     * found. If false, no check will be made and the floating point arithmetic will carry on with {@link Double#POSITIVE_INFINITY
     * infinities},  {@link Double#NEGATIVE_INFINITY negative-infinities}, and {@link Double#NaN NaNs} present.
     */
    protected boolean checkFinite;


    // TODO: Just for debugging. Needs to be removed. -------------
    private final boolean debug;
    private final String infoStr = "\u001B[33m[Info]\u001B[0m";
    // TODO: END --------------------------------------------------

    /**
     * Creates a decomposer to compute the Schur decomposition for a real dense matrix.
     */
    public RealSchurFrancisOptimized() {
        debug = false;
        exceptionalIters = DEFAULT_EXCEPTIONAL_ITERS;
    }

    // TODO: Remove. Just for debugging.
    public RealSchurFrancisOptimized(boolean debug) {
        this.debug = debug;
        exceptionalIters = DEFAULT_EXCEPTIONAL_ITERS;
    }

    /**
     * Specify the maximum number of total iterations to run the {@code QR} algorithm for when computing the decomposition.
     * @param maxIterations maximum number of total iterations to run the {@code QR} algorithm for when computing the decomposition.
     */
    public void setMaxIterations(int maxIterations) {
        this.maxIterations = maxIterations;
    }


    /**
     * Computes the Schur decomposition using an unsifted, explicit QR algorithm. This is the most simplified and bare implementation
     * of the QR algorithm and may fail to converge for many inputs. Further, each iteration is extremely inefficient AND
     * convergence may be slow without shifting so many iterations may need to be performed.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public RealSchurFrancisOptimized decompose(Matrix src) {
        setUp(src);

        int workingSize = numRows-1;
        double p1, p2, p3;

        int iters = 0;
        final int MAX_ITERS = 5000;

        while(workingSize >= 2 && iters < MAX_ITERS) {
            // Compute the non-zero entries (first three) of the first column of the double shifted matrix.
            computeDoubleShift(workingSize);
            p1 = shiftCol[0]; p2 = shiftCol[1]; p3 = shiftCol[2]; // Extract non-zero values.

            // Apply shift and chase bulge.
            for(int i=0; i<=workingSize-2; i++) {
                if(debug) System.out.println(infoStr + " Performing double shift iteration...");
                makeReflector(i, p1, p2, p3); // Construct Householder reflector.
                applyDoubleShiftReflector(i, i>0); // Apply the reflector.

                // Set values to be used in computing the next bulge chasing reflector.
                p1 = T.entries[(i + 1)*T.numCols + i];
                p2 = T.entries[(i + 2)*T.numCols + i];
                if(i < workingSize-2) p3 = T.entries[(i + 3)*T.numCols + i];
            }
            if(debug) System.out.println(infoStr + " Last transformation in bulge chase must be a single shift iteration...");

            // The last reflector in the bulge chase only acts on last two rows of the working matrix.
            makeReflector(workingSize-1, p1, p2); // Construct Householder reflector.
            applySingleReflector(workingSize-1, true); // Apply the reflector.

            // Check for convergence and deflate if needed.
            int deflate = checkConvergence(workingSize);
            if(deflate > 0) workingSize -= deflate;

            iters++;
            if(debug) System.out.println("-".repeat(150) + "\n");
        }

        if(debug) System.out.println(infoStr + " Completed in " + iters + " iterations.");

        return this;
    }


    /**
     * Computes the shifts for a Francis double shift iteration. Specifically, the shifts are the generalized Rayleigh quotients of
     * degree two.
     * @param k Size of current working matrix.
     */
    protected void computeDoubleShift(int k) {
        // The shift computed here, p, represent the double shift
        //  p = (T - rho1*I)(T - rho2*I)*e1 where I is the identity matrix, e1 is the first column of I, and (rho1, rho2)
        //  are taken to be the eigenvalues of the lower 2x2 sub-matrix within the working matrix.
        //  Note: (rho1, rho2) are either both real, or both complex (and specifically complex conjugates). In either case, has
        //  only three non-zero entries (the first three entries) all of which are real. Hence, all arithmetic may be carried out in
        //  real arithmetic. As such, eigenvalues are not explicitly computed as that would require complex arithmetic.

        // Extract values from lower right 2x2 sub-matrix within the working size.
        int leftIdx = k-1;
        double x11 = T.entries[leftIdx*numRows + leftIdx];
        double x12 = T.entries[leftIdx*numRows + k];
        double x21 = T.entries[k*numRows + leftIdx];
        double x22 = T.entries[k*numRows + k];

        // Extract top right entries of T for use in computing the shift p.
        double a11 = T.entries[0];
        double a12 = T.entries[1];
        double a21 = T.entries[numRows];
        double a22 = T.entries[numRows + 1];
        double a32 = T.entries[2*numRows + 1];

        // Scale values to improve stability and help avoid possible overflow issues.
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
     * Constructs a householder reflector given specified values for a column to apply the reflector to. This reflector is stored in
     * indices {@code i}, {@code i+1}, and {@code i+2} of {@link #householderVector}.
     * @param i Row of working matrix to construct reflector for.
     * @param p1 First entry to in column to apply reflector to.
     * @param p2 Second entry in column to apply reflector to.
     * @param p3 Third entry in column to apply reflector to.
     */
    protected void makeReflector(int i, double p1, double p2, double p3) {
        if(debug) System.out.printf(infoStr + " Values for reflector [%f, %f, %f]\n", p1, p2, p3);

        // Scale components for stability and overflow purposes.
        double maxAbs = Math.max(Math.abs(p1), Math.max(Math.abs(p2), Math.abs(p3)));
        if(debug) System.out.println(infoStr + " maxAbs: " + maxAbs);

        p1 /= maxAbs;
        p2 /= maxAbs;
        p3 /= maxAbs;

        norm = Math.sqrt(p1*p1 + p2*p2 + p3*p3); // Compute scaled 2-norm.
        if(debug) System.out.println(infoStr + " scaled norm: " + norm);

        // Change sign of norm depending on first entry in column for stability purposes in Householder vector.
        if(p1 < 0) norm = -norm;

        double div = p1 + norm;
        currentFactor = div/norm;
        norm *= maxAbs; // Rescale norm to be proper magnitude.

        if(debug) System.out.println(infoStr + " shift: " + div);

        householderVector[i] = 1.0;
        householderVector[i+1] = p2 / div;
        householderVector[i+2] = p3 / div;

        if(debug)
            System.out.printf(infoStr + " Computed reflector: [%f, %f, %f]\n",
                    householderVector[i], householderVector[i+1], householderVector[i+2]);
    }


    /**
     * Constructs a householder reflector given specified values for a column to apply the reflector to. This reflector is stored in
     * indices {@code i} and {@code i+1} of {@link #householderVector}.
     * @param i Row of working matrix to construct reflector for.
     * @param p1 First entry to in column to apply reflector to.
     * @param p2 Second entry in column to apply reflector to.
     */
    protected void makeReflector(int i, double p1, double p2) {
        // Scale components for stability and overflow purposes.
        double maxAbs = Math.max(Math.abs(p1), Math.abs(p2));
        p1 /= maxAbs;
        p2 /= maxAbs;

        norm = Math.sqrt(p1*p1 + p2*p2); // Compute scaled norm.
        // Change sign of norm depending on first entry in column for stability purposes in Householder vector.
        if(p1 < 0) norm = -norm;

        double div = p1 + norm;
        currentFactor = div/norm;
        norm *= maxAbs; // Rescale norm to be proper magnitude.

        householderVector[i] = 1.0;
        householderVector[i+1] = p2 / div;

        if(debug) System.out.printf(infoStr + " Computed reflector: [%f, %f]\n", householderVector[i], householderVector[i+1]);
    }


    /**
     * Applies reflector for the double shift. This method can be used to apply either be the reflector constructed for the first
     * column of the shifted matrix, or a reflector being used in the bulge chase of size 2 which arises from the first case.
     * @param i The row the reflector is being applied to.
     */
    protected void applyDoubleShiftReflector(int i, boolean set) {
        // Apply reflector to left (Assumes T is upper hessenburg except for possibly a bulge of size 2).
        Householder.leftMultReflector(T, householderVector, currentFactor, 0, i, i+3, workArray);

        // Apply reflector to right (Assumes T is upper hessenburg except for possibly a bulge of size 2).
        Householder.rightMultReflector(T, householderVector, currentFactor, 0, i, i+3);

        if(set) {
            // Explicitly zeros out values which should be zeroed by the reflector.
            T.entries[i*numRows + i - 1] = -norm;
            T.entries[(i+1)*numRows + i - 1] = 0.0;
            T.entries[(i+2)*numRows + i - 1] = 0.0;

            if(debug) System.out.printf(infoStr + " Setting (%d, %d) and (%d, %d) to zero.\n", i+1, i-1, i+2, i-1);
        }

        if(debug) {
            if(i==0) System.out.println(infoStr + " Applied first reflector. Should be bulged now.");
            else System.out.println(infoStr + " Chasing bulge...");
            System.out.println("working T:\n" + T);
            printBulge(T);
        }
    }


    /**
     * Applies reflector for the double shift. This method can be used to apply either be the reflector constructed for the first
     * column of the shifted matrix, or a reflector being used in the bulge chase of size 2 which arises from the first case.
     * @param i The row the reflector is being applied to.
     */
    protected void applySingleReflector(int i, boolean set) {
        // TODO: Since we only need to zero a single element, consider using a givens rotator here instead.
        //  There should be negligible stability difference between the two for a 2x2 rotator, but the givens rotator is more
        //  simple to calculate.

        // Apply reflector to left (Assumes T is upper hessenburg except for possibly a bulge of size 1).
        Householder.leftMultReflector(T, householderVector, currentFactor, 0, i, i+2, workArray);

        // Apply reflector to right (Assumes T is upper hessenburg except for possibly a bulge of size 1).
        Householder.rightMultReflector(T, householderVector, currentFactor, 0, i, i+2);

        if(set) {
            // Explicitly zeros out values which should be zeroed by the reflector.
            T.entries[i*numRows + i - 1] = -norm;
            T.entries[(i+1)*numRows + i - 1] = 0.0;

            if(debug) System.out.printf(infoStr + " Setting (%d, %d) to zero.\n", i+1, i-1);
        }

        if(debug) {
            if(i==0) System.out.println(infoStr + " Applied first reflector. Should be bulged now.");
            else System.out.println(infoStr + " Chasing bulge (off edge)...");
            System.out.println("working T:\n" + T);
            printBulge(T);
        }
    }


    /**
     * Checks for convergence of lower 2x2 sub-matrix within working matrix to upper triangular or block upper triangular form. If
     * convergence is found, this will also zero out the values which have converged to near zero.
     * @param k Size of current working matrix.
     * @return Returns the amount the working matrix size should be deflated. Will be zero if no convergence is detected, one if
     * convergence to upper triangular form is detected and two if convergence to block upper triangular form is detected.
     */
    protected int checkConvergence(int k) {
        double a11 = T.entries[(k-2)*numRows + k - 2];
        double a21 = T.entries[(k-1)*numRows + k - 2];
        double a22 = T.entries[(k-1)*numRows + k - 1];
        double a32 = T.entries[k*numRows + k - 1];
        double a33 = T.entries[k*numRows + k];

        if(Math.abs(a32) < 0.5*EPS_F64*( Math.abs(a33) + Math.abs(a22) )) {
            if(debug) {
                System.out.println(infoStr + " Deflating by 1.");
                System.out.printf(infoStr + " Value: %f - Threshold: %f\n", Math.abs(a32), EPS_F64*(Math.abs(a33) + Math.abs(a22)));
            }

            T.entries[k*numRows + k - 1] = 0; // Zero out converged value.
            return 1; // Deflate by 1.
        }
        else if(Math.abs(a21) < 0.5*EPS_F64*( Math.abs(a11) + Math.abs(a22) )) {
            if(debug) {
                System.out.println(infoStr + " Deflating by 2.");
                System.out.printf(infoStr + " Value: %f - Threshold: %f\n", Math.abs(a21), EPS_F64*(Math.abs(a11) + Math.abs(a22)));
            }

            T.entries[(k-1)*numRows + k - 2] = 0; // Zero out converged value.
            return 2; // Deflate by 2.
        }

        return 0; // No convergence detected. Do not deflate.
    }


    /**
     * Performs basic setup and initializes data structures to be used in the decomposition.
     * @param src The matrix to be decomposed.
     */
    protected void setUp(Matrix src) {
        ParameterChecks.assertSquare(src.shape);

        numRows = src.numRows;
        maxIterations = src.numRows*DEFAULT_MAX_ITERS_FACTOR;
        householderVector = new double[numRows];
        workArray = new double[numRows];
        shiftCol = new double[3]; // For storing the non-zero entries (i.e. first three) of the first column of the double shift.
        temp = new double[9]; // For storing temporary values when computing shifts.

        T = hess.decompose(src).getH(); // Reduce matrix to upper Hessenburg form.
        U = hess.getQ(); // Initialize U as the product of transformations used in Hessenburg decomposition.

        if(debug) {
            System.out.println(infoStr + " Beginning Decomposition...\n");
            System.out.println("Matrix to decompose:\n" + src + "\n");
            System.out.println("Hess:\n" + T + "\n");
        }
    }


    /**
     * Gets the upper, or possibly block-upper, triangular Schur matrix {@code T} from the Schur decomposition
     * @return {@code A=UTU<sup>T</sup>}
     */
    public Matrix getT() {
        return T;
    }


    public Matrix getU() {
        return U;
    }


    /**
     * <p>Converts the real schur form computed in the last decomposition to the complex Schur form.</p>
     *
     * <p>That is, converts the real block
     * upper triangular Schur matrix to a complex valued properly upper triangular matrix. If the unitary transformation matrix
     * {@code U} was computed, the transformations will also be updated accordingly.</p>
     * @return An array of length 2 containing the complex Schur matrix {@code T} from the last decomposition, and if computed, the
     * complex unitary transformation matrix {@code U} from the decomposition. If {@code U} was not computed, then the arrays second
     * value will be null.
     */
    public CMatrix[] real2ComplexSchur() {
        // TODO: Write method for efficiently applying givens rotators or use Householder vectors instead.
        // Convert matrices to complex matrices.
        CMatrix tComplex = T.toComplex();
        CMatrix uComplex = (U != null) ? U.toComplex() : null;

        for(int m=numRows-1; m>0; m--) {
            CNumber a11 = tComplex.entries[(m - 1)*numRows + m - 1];
            CNumber a12 = tComplex.entries[(m - 1)*numRows + m];
            CNumber a21 = tComplex.entries[m*numRows + m - 1];
            CNumber a22 = tComplex.entries[m*numRows + m];

            if(a21.mag() > EPS_F64*(a11.mag() + a22.mag())) {
                // non-converged 2x2 block found.

                CNumber[] mu = Eigen.get2x2EigenValues(a11, a12, a21, a22);
                mu[0].subEq(a22); // Shift eigenvalue.

                // Construct a givens rotator to bring matrix into properly upper triangular form.
                CMatrix G = Givens.get2x2Rotator(new CVector(mu[0], a21));

                // Apply rotation to T matrix to bring it into upper triangular form
                tComplex.setSlice(
                        G.mult(tComplex.getSlice(m-1, m+1, m-1, numRows)),
                        m-1, m-1
                );
                tComplex.setSlice(
                        tComplex.getSlice(0, m+1, m-1, m+1).mult(G.H()),
                        0, m-1
                );

                if(uComplex != null) {
                    // Accumulate similarity transforms in the U matrix.
                    uComplex.setSlice(
                            uComplex.getSlice(0, numRows, m-1, m+1).mult(G.H()),
                            0, m-1
                    );
                }

                tComplex.set(0, m, m-1);
            }
        }

        return new CMatrix[]{tComplex, uComplex};
    }


    public static void main(String[] args) {
        int randomSize = 10;
        char matrixChoice = 'D';

        RandomTensor rtg = new RandomTensor(0xBEEF);
        Matrix A = new Matrix(new double[][]{{1, 8, 9}, {-6, 1, 4}, {2, 0, -3}}); // Has complex eigenvalues.
        Matrix B = new Matrix(new double[][]{{-2, -4, 2}, {-2, 1, 2}, {4, 2, 5}}); // Has real eigenvalues.
        Matrix C = new Matrix(new double[][]{
                {0, 0, 0, 1},
                {1, 0, 0, 0},
                {0, 1, 0, 0},
                {0, 0, 1, 0}}); // Difficult to compute eigenvalues of (Standard un-shifted QR algorithm fails on this).
        Matrix D = new Matrix(new double[][]{
                {0, 0, 0, 0, 0,  720},
                {1, 0, 0, 0, 0, -684},
                {0, 1, 0, 0, 0, -212},
                {0, 0, 1, 0, 0,  149},
                {0, 0, 0, 1, 0,  33},
                {0, 0, 0, 0, 1, -5}}); // Eigenvalues = (-6, -4, -3, 1, 2, 5)
        Matrix E = rtg.randomMatrix(randomSize, randomSize, -10, 10);
        Matrix[] matrices = {A, B, C, D, E};

        Matrix mat = matrices[matrixChoice - 65];
        RealSchurFrancisOptimized schur = new RealSchurFrancisOptimized(true).decompose(mat);

        System.out.printf("Original Matrix %s:\n%s\n\n", matrixChoice, mat);
        System.out.println("-".repeat(150) + "\n\n");

        System.out.printf("T:\n%s\n\n", schur.getT());
        System.out.printf("U:\n%s\n\n", schur.getU());

        CMatrix[] complexMatrices = schur.real2ComplexSchur();

        System.out.println("Complex T:\n" + complexMatrices[0].isReal());
    }
}
