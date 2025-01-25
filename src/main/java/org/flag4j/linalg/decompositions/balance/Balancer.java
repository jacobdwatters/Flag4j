/*
 * MIT License
 *
 * Copyright (c) 2025. Jacob Watters
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

package org.flag4j.linalg.decompositions.balance;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.dense.Matrix;
import org.flag4j.arrays.sparse.PermutationMatrix;
import org.flag4j.linalg.decompositions.Decomposition;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.Flag4jConstants;
import org.flag4j.util.ValidateParameters;

import java.util.Arrays;


/**
 * <p>Base class for all decompositions which implement matrix balancing. Balancing a matrix involves computing a
 * diagonal similarity transformation to "balance" the rows and columns of the matrix. This balancing is achieved
 * by attempting to scale the entries of the matrix by similarity transformations such that the 1-norms of corresponding
 * rows and columns have the similar 1-norms. Rows and columns may also be permuted during balancing if requested.
 *
 * <p>Balancing is often used as a preprocessing step to improve the conditioning of eigenvalue problems. Because the
 * balancing transformation is a similarity transformation, the eigenvalues are preserved. Further, when permutations are
 * done during balancing it is possible to isolate decoupled eigenvalues.
 *
 * <p>The similarity transformation of a square matrix A into the balanced matrix B can be described as:
 * <pre>
 *     B = T<sup>-1</sup> A T
 *       = D<sup>-1</sup> P<sup>-1</sup> A P D.</pre>
 * Solving for A, balancing may be viewed as the following decomposition:
 * <pre>
 *     A = T B T<sup>-1</sup>
 *       = P D B D<sup>-1</sup> P<sup>-1</sup>.</pre>
 * Where P is a permutation matrix, and D is a diagonal scaling matrix.
 *
 * <p>When permutations are used during balancing we obtain a specific form. First,
 * <pre>
 *             [ T<sub>1</sub>  X   Y  ]
 *   P<sup>-1</sup> A P = [  0  B<sub>1</sub>  Z  ]
 *             [  0  0   T<sub>2</sub> ]</pre>
 * Where T<sub>1</sub> and T<sub>2</sub> are upper triangular matrices whose eigenvalues lie along the diagonal. These are also
 * eigenvalues of A. Then, if scaling is applied we obtain:
 * <pre>
 *                  [ T<sub>1</sub>     X*D<sub>1</sub>       Y    ]
 *   D<sup>-1</sup> P<sup>-1</sup> A P D = [  0  D<sub>1</sub><sup>-1</sup>*B*<sub>1</sub>D<sub>1</sub>  D<sub>1</sub><sup>-1</sup>*Z  ]
 *                  [  0      0         T<sub>2</sub>   ]</pre>
 * Where D<sub>1</sub> is a diagonal matrix such that,
 * <pre>
 *         [ I<sub>1</sub> 0  0  ]
 *     D = [ 0  D<sub>1</sub> 0  ]
 *         [ 0  0  I<sub>2</sub> ]</pre>
 * Where I<sub>1</sub> and I<sub>2</sub> are identity matrices with equivalent shapes to T<sub>1</sub> and T<sub>2</sub>.
 *
 * <p>Once balancing has been applied, one need only compute the eigenvalues of B<sub>1</sub> and combine them with the diagonal
 * entries of T<sub>1</sub> and T<sub>2</sub> to obtain all eigenvalues of A.
 *
 * <p>The code in this class if heavily based on LAPACK's reference implementations of
 * <a href=https://www.netlib.org/lapack/explore-html/df/df3/group__gebal.html>xGEBAL</a> (v 3.12.1).
 *
 * @param <T> The type of matrix being balanced.
 *
 * @see #getB()
 * @see #getBSubMatrix()
 * @see #getD(boolean)
 * @see #getD()
 * @see #getP()
 * @see #getT()
 */
public abstract class Balancer<T extends MatrixMixin<T, ?, ?, ?>> implements Decomposition<T> {

    /**
     * Simple scaling factor used to help ensure safe scaling without over/underflow.
     */
    private static final double FACTOR = 0.95;

    // Scaling factor to keep values as powers of two.
    private static final double BASE_SCALE = 2.0;

    // Some constants which specify "safe" maximum and minimum values to avoid under/overflow.
    private static final double SAFE_MIN_1 = Flag4jConstants.SAFE_MIN_F64 / (Flag4jConstants.EPS_F64*2.0);
    private static final double SAFE_MAX_1 = 1.0 / SAFE_MIN_1;
    private static final double SAFE_MIN_2 = SAFE_MIN_1*BASE_SCALE;
    private static final double SAFE_MAX_2 = 1.0 / SAFE_MIN_2;

    /**
     * <p>Stores both the scaling and permutation information for the balanced matrix.
     *
     * <p>Let {@code perm[j]} be the index of the row and column swapped with row and column {@code j} and
     * {@code scale[j]} be the scaling factor applied to row and column {@code j}. Then,
     * <ul>
     *     <li>{@code scalePerm[j] = perm[j]} for {@code j = 0, ..., iLow-1}.</li>
     *     <li>{@code scalePerm[j] = scale[j]} for {@code j = iLow, ..., iHigh-1}.</li>
     *     <li>{@code scalePerm[j] = perm[j]} for {@code j = iHigh, ..., size-1}.</li>
     * </ul>
     *
     * The order which row and column swaps are made is {@code size-1} to {@code iHigh}, then from {@code 0} to {@code iLow}.
     */
    protected double[] scalePerm;
    /**
     * Stores the balanced matrix.
     */
    protected T balancedMatrix;
    /**
     * This size of the matrix to be balanced.
     */
    protected int size;
    /**
     * Tracks the ending row/column index of the un-permuted submatrix to be balanced (exclusive).
     */
    protected int iHigh;
    /**
     * Tracks the starting row/column index of the un-permuted submatrix to be balanced (inclusive).
     */
    protected int iLow;
    /**
     * Flag indicating if scaling should be done during balancing.
     * <ul>
     *     <li>If {@code true}, then scaling will be performed during balancing.</li>
     *     <li>If {@code false}, the no scaling will be done during balancing.</li>
     * </ul>
     */
    protected boolean doScaling;
    /**
     * Flag indicating if permutations should be done during balancing.
     * <ul>
     *     <li>If {@code true}: Then row/column permutations will be performed during balancing.</li>
     *     <li>If {@code false}: Then row/column permutations will be performed during balancing.</li>
     * </ul>
     */
    protected boolean doPermutations;
    /**
     * Flag indicating if the balancing should be done in-place or if a copy should be made.
     * <ul>
     *     <li>If {@code true}, the balancing will be done in-place and the matrix to be balanced will be overwritten.</li>
     *     <li>If {@code false}, a copy will be made of the matrix before balancing is applied and the original matrix will remain
     *     unmodified.</li>
     * </ul>
     */
    public final boolean inPlace;


    /**
     * @param doPermutations Flag indicating if row/column permutations should be used when balancing the matrix.
     * <ul>
     *     <li>If {@code true}, permutations will be used and P will be computed.</li>
     *     <li>If {@code false}, permutations will <i>not</i> be used and the row and column positions will not be affected.</li>
     * </ul>
     * @param doScaling Flag indicating if row/column scaling should be done when balancing the matrix.
     * <ul>
     *     <li>If {@code true}, scaling will be used and D will be computed.</li>
     *     <li>If {@code false}, scaling will <i>not</i> be used.</li>
     * </ul>
     * @param inPlace Flag indicating if the balancing should be done in-place or if a copy should be made.
     * <ul>
     *     <li>If {@code true}, the balancing will be done in-place and the matrix to be balanced will be overwritten.</li>
     *     <li>If {@code false}, a copy will be made of the matrix before balancing is applied and the original matrix will remain
     *     unmodified.</li>
     * </ul>
     */
    protected Balancer(boolean doPermutations, boolean doScaling, boolean inPlace) {
        this.inPlace = inPlace;
        this.doPermutations = doPermutations;
        this.doScaling = doScaling;
    }


    /**
     * Swaps two rows, over a specified range, within the {@link #balancedMatrix} matrix.
     * @param rowIdx1 Index of the first row to swap.
     * @param rowIdx2 Index of the second row to swap.
     * @param start Index of the column specifying the start of the range for the row swap (inclusive).
     * @param stop Index of the column specifying the end of the range for the row swap (exclusive).
     */
    protected abstract void swapRows(int rowIdx1, int rowIdx2, int start, int stop);


    /**
     * Swaps two columns, over a specified range, within the {@link #balancedMatrix} matrix.
     * @param colIdx1 Index of the first column to swap.
     * @param colIdx2 Index of the second column to swap.
     * @param start Index of the row specifying the start of the range for the column swap (inclusive).
     * @param stop Index of the row specifying the end of the range for the column swap (exclusive).
     */
    protected abstract void swapCols(int colIdx1, int colIdx2, int start, int stop);


    /**
     * Checks if a value within {@link #balancedMatrix} is zero.
     * @param idx Index of value within {@link #balancedMatrix}'s 1D data array to check if it is zero.
     */
    protected abstract boolean isZero(int idx);


    /**
     * Computes the &ell;<sup>2</sup> norm of a vector with {@code n} elements from {@link #balancedMatrix}'s 1D data array
     * starting at index {@code start} and spaced by {@code stride}.
     * @param start Starting index within {@link #balancedMatrix}'s 1D data array to compute norm of.
     * @param n The number of elements in the vector to compute norm of.
     * @param stride The spacing between each element within {@link #balancedMatrix}'s 1D data array to norm of.
     * @return The norm of the vector containing the specified elements from {@link #balancedMatrix}'s 1D data array.
     */
    protected abstract double vectorNorm(int start, int n, int stride);


    /**
     * Computes the maximum absolute value of a vector with {@code n} elements from {@link #balancedMatrix}'s 1D data array
     * starting at index {@code start} and spaced by {@code stride}.
     * @param start Starting index within {@link #balancedMatrix}'s 1D data array to compute maximum absolute value of.
     * @param n The number of elements in the vector to compute maximum absolute value of.
     * @param stride The spacing between each element within {@link #balancedMatrix}'s 1D data array to compute maximum absolute
     * value of.
     * @return The maximum absolute value of the vector containing the specified elements from {@link #balancedMatrix}'s 1D data
     * array.
     */
    protected abstract double vectorMaxAbs(int start, int n, int stride);


    /**
     * Scales a vector with {@code n} elements from {@link #balancedMatrix}'s 1D data array
     * starting at index {@code start} and spaced by {@code stride}. This operation must be done in-place.
     *
     * @param factor Factor to scale elements by.
     * @param start Starting index within {@link #balancedMatrix}'s 1D data array begin scaling.
     * @param n The number of elements to scale.
     * @param stride The spacing between each element within {@link #balancedMatrix}'s 1D data array to scale.
     */
    protected abstract void vectorScale(double factor, int start, int n, int stride);


    /**
     * <p>Performs basic setup for balancing.
     * <p>Specifically, copies the matrix to be balanced if an out-of-place computation was requested, initializes the matrix size,
     * {@link #iLow}, {@link #iHigh}, and {@link #scalePerm}.
     * @param src The matrix to balance.
     */
    private void setUp(T src) {
        ValidateParameters.ensureSquare(src.getShape());
        balancedMatrix = inPlace ? src : src.copy();
        size = balancedMatrix.numRows();

        iLow = 0;
        iHigh = size;

        scalePerm = new double[size];
    }


    /**
     * Balances a matrix so that the rows and columns have roughly similar sized norms.
     * @param src Matrix to balance. Must be square. If {@link #inPlace == true} then {@code src} will be modified.
     * Otherwise, {@code src} will <i>not</i> be modified.
     * @return A reference to this balancer object.
     * @throws org.flag4j.util.exceptions.TensorShapeException If {@code src} is not a square matrix.
     */
    @Override
    public Balancer<T> decompose(T src) {
        setUp(src);

        if (doPermutations) doIterativePermutations();

        // Initialize scaling factors for remaining un-permuted rows and columns.
        for(int i = iLow; i<iHigh; i++)
            scalePerm[i] = 1.0;

        if (doScaling && iLow != iHigh) doIterativeScaling();

        return this;
    }


    /**
     * <p>Performs the permutation step of matrix balancing.
     *
     * <p>This is, identifies rows and columns which are decoupled from the rest of the matrix and hence isolate an
     * eigenvalue. Rows which isolate an eigenvalue are pushed to the bottom of the matrix. Similarly, columns which isolate an
     * eigenvalue are pushed to the left of the matrix. To ensure that the row/column swaps are similarity transforms, if any two
     * rows are swapped the same columns are swapped.
     *
     * <p>Such row and column permutations transform the original matrix {@code A} into the following form:
     * <pre>
     *             [ T1  X  Y  ]
     *   P<sup>-1</sup> A P = [  0  B  Z  ]
     *             [  0  0  T2 ]</pre>
     * <p>Where {@code T1} and {@code T2} are upper-triangular matrices whose eigenvalues are the diagonal elements of the matrix.
     * {@code P} is the permutation matrix representing the row and column swaps performed within this method.
     *
     * <p>{@link #iLow} and {@link #iHigh} Specify the starting (inclusive) and ending (exclusive) row/column index of the submatrix
     * {@code B}.
     */
    protected void doIterativePermutations() {
        boolean notConverged = true;

        // Find rows isolating eigenvalues and push to the bottom of the matrix.
        while (notConverged) {
            notConverged = false;

            for(int i=iHigh-1; i>=0; i--) {
                int rowOffset = i*size;
                boolean canSwap = true;

                for(int j=0; j<iHigh; j++) {
                    if(i != j && !isZero(rowOffset + j)) {
                        canSwap = false;
                        break;
                    }
                }

                if(canSwap) {
                    scalePerm[iHigh-1] = i;

                    if(i != iHigh-1) {
                        swapCols(i, iHigh-1, 0, iHigh);
                        swapRows(i, iHigh-1, 0, size);
                    }

                    notConverged = true; // A swap was made. Must do another check for more rows to be swapped.

                    if(iHigh == 0) {
                        // Then we have permuted all rows. Nothing left to do.
                        iLow = 0;
                        return;
                    }

                    iHigh--;
                }
            }
        }

        notConverged = true;
        while (notConverged) {
            notConverged = false;

            // Find zero columns and permute to left.
            for(int j=iLow; j<iHigh; j++) {
                boolean canSwap = true;

                for(int i=iLow; i<iHigh; i++) {
                    if(i != j && !isZero(i*size + j)) {
                        canSwap = false;
                        break;
                    }
                }

                if(canSwap) {
                    scalePerm[iLow] = j;

                    if(j != iLow) {
                        swapCols(j, iLow, 0, iHigh);
                        swapRows(j, iLow, iLow, size);
                    }

                    notConverged = true; // A swap was made. Must do another check for more columns to be swapped.
                    iLow++;
                }
            }
        }
    }


    /**
     * <p>Performs the scaling step of matrix balancing.
     *
     * <p>That is, computes scaling factors such that when a column is scaled by such value and the row is scaled by the reciprocal
     * of that value, there &ell;<sup>1</sup> norms are "close". Scaling need only be done for rows/column of the matrix which do not
     * isolate eigenvalues; rows between {@link #iLow} (inclusive) to {@link #iHigh} (exclusive).
     *
     * <p>D<sub>1</sub> is the diagonal matrix describing such scaling and is the diagonal matrix computed by this method. \
     * The diagonal values of D<sub>1</sub> are stored in {@link #scalePerm} between indices {@link #iLow} (inclusive) to
     * {@link #iHigh} (exclusive).
     */
    protected void doIterativeScaling() {
        int n = iHigh - iLow;
        int bRowOffset = iLow*size;

        boolean notConverged = true;

        while (notConverged) {
            notConverged = false; // Set true if any scaling is applied.

            // Process each column/row i in the sub-block.
            for (int i = iLow; i < iHigh; i++) {
                int rowStart = i*size + iLow;
                int colStart = i + bRowOffset;

                // Compute the row/column l2 norms.
                double colNorm = vectorNorm(colStart, n, size);
                double rowNorm = vectorNorm(rowStart, n, 1);

                // Compute the row/column maximum absolute values.
                double colMaxAbs = vectorMaxAbs(i, iHigh - 1, size);
                double rowMaxAbs = vectorMaxAbs(rowStart, size - iLow, 1);

                // Avoid division by zero.
                if (colNorm == 0.0 || rowNorm == 0.0)
                    continue;

                // Report if any NaN value is encountered.
                if (Double.isNaN(colNorm + colMaxAbs + rowNorm + rowMaxAbs))
                    throw new IllegalArgumentException("NaN encountered in balancing step.");

                double g = rowNorm / BASE_SCALE;
                double f = 1.0;
                double s = colNorm + rowNorm;

                // Scale up colNorm and down rowNorm and avoid under/overflow.
                while (colNorm < g
                        && Math.max(Math.max(f, colNorm), colMaxAbs) < SAFE_MAX_2
                        && Math.min(Math.min(rowNorm, g), rowMaxAbs) > SAFE_MIN_2) {

                    f  *= BASE_SCALE;  // multiply f by 2
                    colNorm  *= BASE_SCALE;
                    colMaxAbs *= BASE_SCALE;
                    rowNorm  /= BASE_SCALE;
                    g  /= BASE_SCALE;
                    rowMaxAbs /= BASE_SCALE;
                }

                // Now consider if colNorm should be scaled down and rowNorm up while avoiding under/overflow.
                g = colNorm / BASE_SCALE;
                while (g >= rowNorm
                        && Math.max(rowNorm, rowMaxAbs) < SAFE_MAX_2
                        && Math.min(Math.min(f, colNorm), Math.min(g, colMaxAbs)) > SAFE_MIN_2) {

                    f  /= BASE_SCALE;  // divide f by 2
                    colNorm  /= BASE_SCALE;
                    g  /= BASE_SCALE;
                    colMaxAbs /= BASE_SCALE;
                    rowNorm  *= BASE_SCALE;
                    rowMaxAbs *= BASE_SCALE;
                }

                // Now we check if this scaling factor f actually improves (colNorm + rowNorm)
                // enough relative to factor*s. If not, skip.
                if ( (colNorm + rowNorm) >= FACTOR*s )
                    continue;

                // Ensure we don't underflow or overflow scalePerm[i]
                if (f < 1.0 && scalePerm[i] < 1.0) {
                    if (f * scalePerm[i] <= SAFE_MIN_1)
                        continue; // scalePerm[i] would underflow
                }
                if (f > 1.0 && scalePerm[i] > 1.0) {
                    if (scalePerm[i] >= SAFE_MAX_1/ f)
                        continue; // scalePerm[i] would overflow.
                }

                // All checks have passed, apply the scaling
                double fInv = 1.0 / f;
                scalePerm[i] *= f;
                notConverged = true; // Indicate another pass should be performed.

                // Scale row i by fInv and column by f (the diagonal entries of inv(D) and D respectively).
                vectorScale(fInv, rowStart, size - iLow, 1);
                vectorScale(f, i, iHigh, size);
            }
        }
    }


    /**
     * Ensures that {@link #decompose(MatrixMixin)} has been called on this instance.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not been called on this instance.
     */
    private void ensureHasBalanced() {
        // If balancedMatrix has not been instantiated, then balance(...) has not been called.
        if(balancedMatrix == null)
            throw new IllegalStateException("No matrix has been balanced by this balancer. Must call balance(...) first.");
    }


    /**
     * Gets the starting index (inclusive) for the sub-matrix B<sub>1</sub> of the balanced matrix which did not isolate eigenvalues.
     * @return The starting index (inclusive) for the sub-matrix of the balanced matrix which did not isolate eigenvalues.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public int getILow() {
        ensureHasBalanced();
        return iLow;
    }


    /**
     * Gets the starting index (exclusive) for the sub-matrix B<sub>1</sub> of the balanced matrix which did not isolate eigenvalues.
     * @return The starting index (exclusive) for the sub-matrix of the balanced matrix which did not isolate eigenvalues.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public int getIHigh() {
        ensureHasBalanced();
        return iHigh;
    }


    /**
     * Gets the full balanced matrix, B, for the last matrix balanced by this balancer.
     * @return The full balanced matrix for the last matrix balanced by this balancer.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public T getB() {
        ensureHasBalanced();
        return balancedMatrix;
    }


    /**
     * Gets the sub-matrix B<sub>1</sub> of the full balanced matrix which did not isolate eigenvalues.
     * @return The sub-matrix of the full balanced matrix which did not isolate eigenvalues.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public T getBSubMatrix() {
        ensureHasBalanced();
        return balancedMatrix.getSlice(iLow, iHigh, iLow, iHigh);
    }


    /**
     * <p>Gets the raw scaling factors and permutation data stored in a single array.
     *
     * <p>Let {@code perm[j]} be the index of the row and column swapped with row and column {@code j} and
     * {@code scale[j]} be the scaling factor applied to row and column {@code j}. Then,
     * <ul>
     *     <li>{@code scalePerm[j] = perm[j]} for {@code j = 0, ..., iLow-1}.</li>
     *     <li>{@code scalePerm[j] = scale[j]} for {@code j = iLow, ..., iHigh-1}.</li>
     *     <li>{@code scalePerm[j] = perm[j]} for {@code j = iHigh, ..., size-1}.</li>
     * </ul>
     *
     * The order which row and column swaps are made is {@code size-1} to {@code iHigh}, then from {@code 0} to {@code iLow}.
     * @return The raw scaling factors and permutation data stored in a single array.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public double[] getScalePerm() {
        ensureHasBalanced();
        return scalePerm;
    }


    /**
     * Gets the diagonal scaling matrix for the last matrix balanced by this balancer.
     * @param full Flag indicating if the full diagonal scaling matrix should be constructed or if only the scaling factors should
     * be returned. If the last matrix balanced had shape n-by-n then,
     * <ul>
     *     <li>If {@code true}: The full n-by-n diagonal scaling matrix will be created.</li>
     *     <li>If {@code false}: A matrix of shape 1-by-n containing only the scaling factors
     *     (i.e. the diagonal entries of the full scaling matrix).
     *     </li>
     * </ul>
     * @return If {@code full == true} then the full n-by-n scaling matrix is returned. Otherwise if {@code full == false}
     * a matrix of shape 1-by-n containing only the diagonal scaling factors is returned.
     * @see #getD()
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public Matrix getD(boolean full) {
        ensureHasBalanced();

        if (full) {
            Matrix D = Matrix.I(size);

            for(int i=iLow; i<iHigh; i++)
                D.data[i*size + i] = scalePerm[i];

            return D;
        } else {
            double[] data = new double[size];

            for(int i=0; i<iLow; i++)
                data[i] = 1.0;

            for(int i=iLow; i<iHigh; i++)
                data[i] = scalePerm[i];

            for(int i=iHigh; i<size; i++)
                data[i] = 1.0;

            return new Matrix(1, size, data);
        }
    }


    /**
     * <p>Gets the diagonal scaling factors for the last matrix balanced by this balancer.
     *
     * <p> Note, this method will <i>not</i> construct the full diagonal scaling matrix. If the full matrix is desired, use
     * {@link #getD(boolean)}.
     *
     * @return A 1-by-n matrix containing the diagonal elements of the full n-by-n diagonal scaling matrix.
     * @see #getD(boolean)
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public Matrix getD() {
        return getD(false);
    }


    /**
     * <p>Gets the permutation matrix for the last matrix balanced by this balancer.
     * @return The permutation matrix for the last matrix balanced by this balancer.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public PermutationMatrix getP() {
        ensureHasBalanced();
        int[] swapPointers = ArrayUtils.intRange(0, size);

        int temp;
        int value;
        int count = 1;
        for(int i=size-1; i>=iHigh; i--) {
            value = (int) scalePerm[i];

            if(size - count != value) {
                temp = swapPointers[size - count];
                swapPointers[size - count] = swapPointers[value];
                swapPointers[value] = temp;
            }

            count++;
        }

        for(int i=0; i<iLow; i++) {
            value = (int) scalePerm[i];

            if(size != value) {
                temp = swapPointers[i];
                swapPointers[i] = swapPointers[value];
                swapPointers[value] = temp;
            }
        }

        return new PermutationMatrix(swapPointers);
    }


    /**
     * Get the combined permutation and diagonal scaling matrix, T, from the last matrix balanced.
     * This is equivalent to {@code getP().leftMult(getD(true))}.
     * @return The combined permutation and diagonal scaling matrix from the last matrix balanced.
     * @throws IllegalStateException If {@link #decompose(MatrixMixin)} has not yet been called on this instance.
     */
    public Matrix getT() {
        return getP().leftMult(getD(true));
    }


    public static void main(String[] args) {
        Shape aShape = new Shape(8, 8);
        double[] aData = new double[]{3.0, 1.0, 0.0, 1.0, 0.0, 5.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                500000000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 1.0, 0.0, 1000145.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                999.0, 1499514.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 23456.0, 0.0, 0.0, 0.0, 2000000.0, 0.0, 0.0,
                -1000000.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0};
        Matrix a = new Matrix(aShape, aData);

        Balancer<Matrix> balancer = new RealBalancer(true, true).decompose(a);
        Matrix B = balancer.getB();
        PermutationMatrix P = balancer.getP();

        System.out.println("A:\n" + a + "\n");
        System.out.println("B:\n" + B + "\n");
        System.out.println("P:\n" + P.inv().toDense() + "\n");
        System.out.println("perm:\n" + Arrays.toString(P.getPermutation()) + "\n");

        System.out.println("lo: " + balancer.getILow());
        System.out.println("hi: " + balancer.getIHigh());
        System.out.println("ps: " + Arrays.toString(balancer.getScalePerm()));
    }
}
