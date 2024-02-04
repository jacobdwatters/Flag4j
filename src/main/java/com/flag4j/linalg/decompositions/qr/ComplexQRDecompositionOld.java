package com.flag4j.linalg.decompositions.qr;


import com.flag4j.CMatrix;
import com.flag4j.complex_numbers.CNumber;
import com.flag4j.linalg.decompositions.HouseholderUtils;

/**
 * <p>Instances of this class compute the {@code QR} decomposition of a complex dense matrix.</p>
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into a unitary matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}.</p>
 *
 * <p>Much of this code has been adapted from the EJML library.</p>
 */
@Deprecated
public class ComplexQRDecompositionOld extends QRDecompositionOld<CMatrix, CNumber[]> {


    /**
     * Scalar factor of the currently computed Householder reflector.
     */
    private CNumber currentFactor;
    /**
     * Stores the shifted value of the first entry in a Householder vector.
     */
    private CNumber shift;
    /**
     * For storing the scaled norm. This may be complex.
     */
    private CNumber phaseAdjustedNorm;
    /**
     * The complex number equal to zero.
     */
    static final CNumber ZERO = CNumber.zero();

    /**
     * Creates a {@code QR} decomposer. This decomposer will compute the reduced {@code QR} decomposition.
     * @see #ComplexQRDecompositionOld(boolean)
     */
    public ComplexQRDecompositionOld() {
        super(true);
    }


    /**
     * Creates a {@code QR} decomposer to compute either the full or reduced {@code QR} decomposition.
     *
     * @param reduced Flag indicating if this decomposer should compute the full or reduced {@code QR} decomposition.
     */
    public ComplexQRDecompositionOld(boolean reduced) {
        super(reduced);
    }


    /**
     * Gets the unitary {@code Q} matrix from the {@code QR} decomposition.
     *
     * @return The {@code Q} matrix from the {@code QR} decomposition.
     */
    @Override
    public CMatrix getQ() {
        int qCols = reduced ? minAxisSize : numRows;
        CMatrix Q = CMatrix.I(numRows, qCols);

        for(int j=minAxisSize-1; j>=0; j--) {
            householderVector[j] = CNumber.one();
            for(int i=j + 1; i<numRows; i++) {
                householderVector[i] = qrData[i*numCols + j];
            }

            // Apply the reflector to the entries.
            if(qFactors[j]!=null && !qFactors[j].equals(ZERO))
                HouseholderUtils.leftMultReflector(Q, householderVector, qFactors[j], j, j, numRows, workArray);
        }

        return Q;
    }


    /**
     * Gets the upper triangular matrix {@code R} from the {@code QR} decomposition.
     *
     * @return The upper triangular matrix {@code R} from the {@code QR} decomposition.
     */
    @Override
    public CMatrix getR() {
        int rRows = reduced ? minAxisSize : numRows;
        CMatrix R = new CMatrix(rRows, numCols); // Get R in reduced form.

        for(int i=0; i<minAxisSize; i++) {
            int idx = i*numCols + i;

            for(int j=i; j<numCols; j++) {
                R.entries[idx] = QR.entries[idx++];
            }
        }

        return R;
    }


    /**
     * Initialized any work arrays to be used in computing the decomposition with the proper size.
     *
     * @param maxAxisSize Length of the largest axis in the matrix to be decomposed. That is, {@code max(numRows, numCols)}
     */
    @Override
    protected void initWorkArrays(int maxAxisSize) {
        qrData = QR.entries; // Create reference to the internal data array of the QR matrix.
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
        norm = 0; // Ensure norm is reset.

        if(maxAbs < Math.ulp(1.0)) {
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
     * Updates the {@link #QR} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param j Index of sub-matrix for which the Householder reflector was computed for.
     */
    protected void updateData(int j) {
        HouseholderUtils.leftMultReflector(QR, householderVector, qFactors[j], j, j, numRows, workArray);

        if(j < numCols) qrData[j + j*numCols] = phaseAdjustedNorm.addInv();

        // Store the Q matrix in the lower portion of QR.
        for(int i=j+1; i<numRows; i++) {
            qrData[j + i*numCols] = householderVector[i];
        }
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
            householderVector[i].divEq(maxAbs); // Scale entries of the householder vector to help reduce potential overflow.
            CNumber scaledValue = householderVector[i];
            norm += scaledValue.magSquared();
        }
        norm = Math.sqrt(norm); // Finish 2-norm computation for the column.

        // Change phase of the norm depending on first entry in column for stability purposes in Householder vector.
        phaseAdjustedNorm = householderVector[j].equals(ZERO) ? new CNumber(norm) : CNumber.sgn(householderVector[j]).mult(norm);

        shift = householderVector[j].add(phaseAdjustedNorm);
        currentFactor = shift.div(phaseAdjustedNorm);
        phaseAdjustedNorm.multEq(maxAbs); // Rescale norm.
    }


    /**
     * Finds the maximum value in {@link #QR} at column {@code j} at or below the {@code j}th row. This method also initializes
     * the first {@code numRows-j} entries of the storage array {@link #householderVector} to the entries of this column.
     * @param j Index of column (and starting row) to compute max of.
     * @return The maximum value in {@link #QR} at column {@code j} at or below the {@code j}th row.
     */
    protected double findMaxAndInit(int j) {
        double maxAbs = 0;
        int idx = j*numCols + j;

        for(int i=j; i<numRows; i++) {
            CNumber d = householderVector[i] = qrData[idx].copy();
            idx += numCols; // Move index to next row.
            maxAbs = Math.max(d.mag(), maxAbs);
        }

        return maxAbs;
    }
}
