package com.flag4j.linalg.decompositions.unitary;

import com.flag4j.Matrix;
import com.flag4j.linalg.decompositions.HouseholderUtils;
import com.flag4j.linalg.decompositions.hess.RealHessenburgDecomposition;
import com.flag4j.linalg.decompositions.qr.RealQRDecompTest;

/**
 * This class is the base class for real matrix decompositions which proceed by using unitary/orthogonal transformations
 * (specifically Householder reflectors) to bring a matrix into an upper triangular/Hessenburg matrix. Specifically, the
 * {@link RealQRDecompTest QR} and {@link RealHessenburgDecomposition Hessenburg} decompositions.
 */
public abstract class RealUnitaryDecomposition extends UnitaryDecomposition<Matrix, double[]> {

    /**
     * To store norms of columns in {@link #transformMatrix}.
     */
    protected double norm;
    /**
     * Scalar factor of the currently computed Householder reflector.
     */
    private double currentFactor;
    /**
     * Stores the shifted value of the first entry in a Householder vector.
     */
    private double shift;


    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper triangular/Hessenburg matrix which is has zeros below
     * the specified sub-diagonal.
     * @param subDiagonal Sub-diagonal of the upper triangular/Hessenburg matrix. That is, the sub-diagonal for which all entries
     *                    below will be zero in the final upper quasi-triangular matrix. Must be Zero or one. If zero, it will be
     *                    upper triangular. If one, it will be upper Hessenburg.
     */
    public RealUnitaryDecomposition(int subDiagonal) {
        super(subDiagonal);
    }


    /**
     * Gets the unitary {@code Q} matrix from the {@code QR} decomposition.
     *
     * @return The {@code Q} matrix from the {@code QR} decomposition.
     */
    @Override
    public Matrix getQ() {
        Matrix Q = initQ();

        for(int j=minAxisSize - 1; j>=subDiagonal; j--) {
            householderVector[j] = 1; // Ensure first value of reflector is 1.

            for(int i=j + 1; i<numRows; i++) {
                householderVector[i] = transformData[i*numCols + j - subDiagonal]; // Extract column containing reflector vector.
            }

            if(qFactors[j]!=0) { // Otherwise, no reflector to apply.
                HouseholderUtils.leftMultReflector(Q, householderVector, qFactors[j], j, j, numRows, workArray);
            }
        }

        return Q;
    }


    /**
     * Gets the upper triangular/Hessenburg matrix from the last decomposition.
     * @return The upper triangular/Hessenburg matrix from the last decomposition.
     */
    @Override
    protected Matrix getUpper(Matrix H) {
        // Copy top rows.
        for(int i=0; i<subDiagonal; i++) {
            int rowOffset = i*numCols;
            System.arraycopy(transformData, rowOffset, H.entries, rowOffset, numCols);
        }

        // Copy rest of the rows.
        for(int i=subDiagonal; i<minAxisSize; i++) {
            int rowOffset = i*numCols;

            int length = numCols - (i-subDiagonal);
            System.arraycopy(transformData, rowOffset + i - subDiagonal,
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
        qFactors = new double[minAxisSize]; // Stores scaler factors for the Householder vectors.
        householderVector = new double[maxAxisSize];
        workArray = new double[maxAxisSize];
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

        applyUpdate = maxAbs >= Math.ulp(1.0);

        if(!applyUpdate) {
            currentFactor = 0;
        } else {
            computePhasedNorm(j, maxAbs);

            householderVector[j] = 1.0; // Ensure first value in Householder vector is one.
            for(int i=j+1; i<numRows; i++) {
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
        for(int i=j; i<numRows; i++) {
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
        int idx = j*numCols + j - subDiagonal;

        for(int i=j; i<numRows; i++) {
            double d = householderVector[i] = transformData[idx];
            idx += numCols; // Move index to next row.
            maxAbs = Math.max(Math.abs(d), maxAbs);
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
            HouseholderUtils.leftMultReflector(transformMatrix, householderVector, qFactors[j], j, j, numRows, workArray);

        if(subDiagonal == 1) // Left multiply transform matrix to reflector. (i.e. right multiply reflector to matrix).
            HouseholderUtils.rightMultReflector(transformMatrix, householderVector, qFactors[j], 0, j, numRows);

        if(j < numCols) transformData[j*numCols + j - subDiagonal] = -norm;

        // Store the Q matrix in the lower portion of the transformation data matrix.
        for(int i=j+1; i<numRows; i++) {
            transformData[i*numCols + j - subDiagonal] = householderVector[i];
        }
    }
}
