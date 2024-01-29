package com.flag4j.linalg.decompositions.qr;

import com.flag4j.core.MatrixMixin;
import com.flag4j.linalg.decompositions.Decomposition;


/**
 * <p>This abstract class specifies methods for computing the {@code QR} decomposition of a matrix.</p>
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into a unitary matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}. There exists a {@code QR} decomposition for every rectangular
 * matrix.</p>
 *
 * @param <T> The type of the matrix to be decomposed.
*  @param <V> The internal storage type for the matrix to be decomposed.
 */
public abstract class QRDecomposition<
        T extends MatrixMixin<T, ?, ?, ?, ?, ?, ?>, V>
        implements Decomposition<T> {

    /**
     * <p>
     * Storage for the R matrix and the vectors of the Householder reflectors used in the decomposition.
     * </p>
     *
     * <p>
     * The R matrix in the upper triangular portion of this matrix and the Householder vectors
     * are stored as the columns in the lower triangular portion (not including the diagonal) of this matrix.<br>
     * </p>
     */
    protected T QR;
    /**
     * Pointer to the internal data array of {@link #QR}.
     */
    protected V qrData;
    /**
     * Number of rows in {@link #QR}.
     */
    protected int numRows;
    /**
     * Number of columns in {@link #QR}.
     */
    protected int numCols;
    /**
     * Storage of the scalar factors for the Householder reflectors used in the decomposition.
     */
    protected V qFactors;
    /**
     * For storing a Householder vectors.
     */
    protected V householderVector;
    /**
     * For temporarily storage when applying Householder vectors. This is useful for
     * avoiding unneeded garbage collection.
     */
    protected V workArray;
    /**
     * To store norms of columns in {@link #QR}.
     */
    protected double norm;
    /**
     * The minimum of rows and columns in the matrix to be decomposed.
     */
    protected int minAxisSize;
    /**
     * Flag indicating if this decomposer should return the reduced (true) or full (false) QR decomposition.
     */
    protected final boolean reduced;


    /**
     * Creates a {@code QR} decomposer.
     * @param reduced Flag indicating if this decomposer should compute the full or reduced {@code QR} decomposition.
     */
    protected QRDecomposition(boolean reduced) {
        this.reduced = reduced;
    }


    /**
     * Gets the unitary {@code Q} matrix from the {@code QR} decomposition.
     * @return The {@code Q} matrix from the {@code QR} decomposition.
     */
    public abstract T getQ();


    /**
     * Gets the upper triangular matrix {@code R} from the {@code QR} decomposition.
     * @return The upper triangular matrix {@code R} from the {@code QR} decomposition.
     */
    public abstract T getR();


    /**
     * Applies decomposition to the source matrix.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public QRDecomposition<T, V> decompose(T src) {
        setUp(src); // Initialize datastructures and storage for the decomposition.

        for(int j=0; j<minAxisSize; j++) {
            computeHouseholder(j); // Compute the householder reflector.
            if(norm!=0) updateData(j); // Update the upper-triangular matrix and store the reflectors.
        }

        return this;
    }


    /**
     * Initializes storage and other parameters for the decomposition.
     * @param src Source matrix to be decomposed.
     */
    private void setUp(T src) {
        QR = src.copy(); // Initialize QR as the matrix to be decomposed.
        numRows = QR.numRows();
        numCols = QR.numCols();
        minAxisSize = Math.min(numRows, numCols);

        int maxAxisSize = Math.max(numRows, numCols);
        initWorkArrays(maxAxisSize);
    }


    /**
     * Initialized any work arrays to be used in computing the decomposition with the proper size.
     * @param maxAxisSize Length of the largest axis in the matrix to be decomposed. That is, {@code max(numRows, numCols)}
     */
    protected abstract void initWorkArrays(int maxAxisSize);


    /**
     * Computes the Householder vector for the first column of the sub-matrix with upper left corner at {@code (j, j)}.
     * @param j Index of the upper left corner of the sub-matrix for which to compute the Householder vector for the first column.
     *          That is, a Householder vector will be computed for the portion of column {@code j} below row {@code j}.
     */
    protected abstract void computeHouseholder(int j);


    /**
     * Updates the {@link #QR} matrix using the computed Householder vector from {@link #computeHouseholder(int)}.
     * @param w Index of sub-matrix for which the Householder reflector was computed for.
     */
    protected abstract void updateData(int w);
}
