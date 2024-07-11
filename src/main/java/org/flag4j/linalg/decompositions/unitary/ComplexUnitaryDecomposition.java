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

import org.flag4j.complex_numbers.CNumber;
import org.flag4j.dense.CMatrix;
import org.flag4j.linalg.transformations.Householder;
import org.flag4j.util.ArrayUtils;
import org.flag4j.util.Flag4jConstants;

/**
 * This class is the base class for complex matrix decompositions which proceed by using unitary transformations
 * (specifically Householder reflectors) to bring a matrix into an upper triangular/Hessenburg matrix. Specifically, the {@code QR} and
 * Hessenburg decompositions.
 */
public abstract class ComplexUnitaryDecomposition extends UnitaryDecomposition<CMatrix, CNumber[]> {

    /**
     * To store norms of columns in {@link #transformMatrix}. Will be real.
     */
    protected double normRe;
    /**
     * For storing the scaled, phase-adjusted norm. This may be complex.
     */
    protected CNumber norm;
    /**
     * Scalar factor of the currently computed Householder reflector.
     */
    private CNumber currentFactor;
    /**
     * Stores the shifted value of the first entry in a Householder vector.
     */
    private CNumber shift;
    /**
     * The complex number equal to zero.
     */
    private static final CNumber ZERO = CNumber.zero();


    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper triangular/Hessenburg matrix which is has zeros below
     * the specified sub-diagonal.
     * @param subDiagonal Sub-diagonal of the upper triangular/Hessenburg matrix. That is, the sub-diagonal for which all entries
     *                    below will be zero in the final upper quasi-triangular matrix. Must be Zero or one. If zero, it will be
     *                    upper triangular. If one, it will be upper Hessenburg.
     */
    public ComplexUnitaryDecomposition(int subDiagonal) {
        super(subDiagonal, true);
    }


    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper triangular/Hessenburg matrix which is has zeros below
     * the specified sub-diagonal.
     *
     * <p>Allows for specification if the reflectors used to bring matrix to upper triangular/Hessenburg form to be stored or not.</p>
     *
     * <p>If the {@code Q} matrix is need, then {@code storeReflectors} must be true. If {@code Q} is <b>NOT</b> needed, then
     * not storing the reflectors <i>may</i> improve performance slightly by avoiding unneeded copies.</p>
     *
     * <p>It should be noted that if performance is improved, it will be a very slight improvement compared
     * to the total time to compute the decomposition. This is because the computation of {@code Q} is only
     * evaluated lazily once {@link #getQ()} is called, so this will only save on copy operations.</p>
     *
     * @param subDiagonal Sub-diagonal of the upper triangular/Hessenburg matrix. That is, the sub-diagonal for which all entries
     *                    below will be zero in the final upper quasi-triangular matrix. Must be Zero or one. If zero, it will be
     *                    upper triangular. If one, it will be upper Hessenburg.
     * @param storeReflectors Flag indicating if the reflectors used to bring the matrix to upper triangular/Hessenburg form
     *                        should be stored.
     */
    public ComplexUnitaryDecomposition(int subDiagonal, boolean storeReflectors) {
        super(subDiagonal, storeReflectors);
    }


    /**
     * <p>Gets the unitary {@code Q} matrix from the {@code QR} decomposition.</p>
     *
     * <p>Note, if the reflectors for this decomposition were not saved, then {@code Q} can not be computed and this method will be
     * null.</p>
     *
     * @return The {@code Q} matrix from the {@code QR} decomposition. Note, if the reflectors for this decomposition were not saved,
     * then {@code Q} can not be computed and this method will return {@code null}.
     */
    @Override
    public CMatrix getQ() {
        if(!storeReflectors)
            return null;

        CMatrix Q = initQ();

        for(int j=minAxisSize - 1; j>=subDiagonal; j--) {
            householderVector[j] = CNumber.one(); // Ensure first value of reflector is 1.

            for(int i=j + 1; i<numRows; i++) {
                householderVector[i] = transformData[i*numCols + j - subDiagonal]; // Extract column containing reflector vector.
            }

            if(!(qFactors[j]==null || qFactors[j].equals(ZERO))) { // Otherwise, no reflector to apply.
                Householder.leftMultReflector(Q, householderVector, qFactors[j], j, j, numRows, workArray);
            }
        }

        return Q;
    }


    /**
     * Gets the upper triangular/Hessenburg matrix from the last decomposition.
     * @return The upper triangular/Hessenburg matrix from the last decomposition.
     */
    @Override
    protected CMatrix getUpper(CMatrix H) {
        // Copy top rows.
        for(int i=0; i<subDiagonal; i++) {
            int rowOffset = i*numCols;
            ArrayUtils.arraycopy(transformData, rowOffset, H.entries, rowOffset, numCols);
        }

        // Copy rest of the rows.
        for(int i=subDiagonal; i<minAxisSize; i++) {
            int rowOffset = i*numCols;

            int length = numCols - (i-subDiagonal);
            ArrayUtils.arraycopy(transformData, rowOffset + i - subDiagonal,
                    H.entries, rowOffset + i - subDiagonal, length);
        }

        return H;
    }


    /**
     * Initialized any work arrays to be used in computing the decomposition with the proper size.
     *
     * @param maxAxisSize Length of the largest axis in the matrix to be decomposed. That is, {@code max(numRows, numCols)}
     */
    @Override
    protected void initWorkArrays(int maxAxisSize) {
        transformData = transformMatrix.entries;
        qFactors = new CNumber[minAxisSize]; // Stores scaler factors for the Householder vectors.
        householderVector = new CNumber[maxAxisSize];
        workArray = new CNumber[maxAxisSize];
    }


    /**
     * Computes the Householder vector for the first column of the sub-matrix with upper left corner at {@code (j, j)}.
     *
     * @param j Index of the upper left corner of the sub-matrix for which to compute the Householder vector for the first column.
     *          That is, a Householder vector will be computed for the portion of column {@code j} below row {@code j}.
     */
    @Override
    protected void computeHouseholder(int j) {
        // Initialize storage array for Householder vector and compute maximum absolute value in jth column at or below jth row.
        double maxAbs = findMaxAndInit(j);
        normRe = 0; // Ensure norm is reset.

        applyUpdate = maxAbs >= Flag4jConstants.EPS_F64;

        if(!applyUpdate) {
            currentFactor = CNumber.zero();
        } else {
            computePhasedNorm(j, maxAbs);

            householderVector[j] = CNumber.one(); // Ensure first value in Householder vector is one.
            for(int i=j+1; i<numRows; i++) {
                householderVector[i].divEq(shift); // Scale all but first entry of the Householder vector.
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
    @Override
    protected void computePhasedNorm(int j, double maxAbs) {
        // Computes the 2-norm of the column.
        for(int i=j; i<numRows; i++) {
            householderVector[i].divEq(maxAbs); // Scale entries of the householder vector to help reduce potential overflow.
            CNumber scaledValue = householderVector[i];
            normRe += scaledValue.magSquared();
        }
        normRe = Math.sqrt(normRe); // Finish 2-norm computation for the column.

        // Change phase of the norm depending on first entry in column for stability purposes in Householder vector.
        norm = householderVector[j].equals(ZERO) ? new CNumber(normRe) : CNumber.sgn(householderVector[j]).mult(normRe);

        shift = householderVector[j].add(norm);
        currentFactor = shift.div(norm);
        norm.multEq(maxAbs); // Rescale norm.
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
        int idx = j*numCols + j - subDiagonal;

        for(int i=j; i<numRows; i++) {
            CNumber d = householderVector[i] = transformData[idx].copy();
            idx += numCols; // Move index to next row.
            maxAbs = Math.max(d.mag(), maxAbs);
        }

        return maxAbs;
    }


    /**
     * Updates the {@link #transformMatrix} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param j Index of sub-matrix for which the Householder reflector was computed for.
     */
    @Override
    protected void updateData(int j) {
        if(subDiagonal >= 0) // Right multiply transform matrix to reflector. (i.e. left multiply reflector to matrix).
            Householder.leftMultReflector(transformMatrix, householderVector, qFactors[j], j, j, numRows, workArray);

        if(subDiagonal == 1) // Left multiply transform matrix to reflector. (i.e. right multiply reflector to matrix).
            Householder.rightMultReflector(transformMatrix, householderVector, qFactors[j], 0, j, numRows);

        if(j < numCols) transformData[j*numCols + j - subDiagonal] = norm.addInv();


        if(storeReflectors) {
            // Store the Q matrix in the lower portion of the transformation data matrix.
            for(int i=j+1; i<numRows; i++) {
                transformData[i*numCols + j - subDiagonal] = householderVector[i];
            }
        }
    }
}
