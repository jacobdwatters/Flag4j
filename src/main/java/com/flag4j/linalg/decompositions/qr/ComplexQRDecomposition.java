package com.flag4j.linalg.decompositions.qr;

import com.flag4j.CMatrix;
import com.flag4j.linalg.decompositions.unitary.ComplexUnitaryDecomposition;


/**
 * <p>Instances of this class compute the {@code QR} decomposition of a complex dense matrix.</p>
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into a unitary matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}.</p>
 *
 * <p>Much of this code has been adapted from the EJML library.</p>
 */
public class ComplexQRDecomposition extends ComplexUnitaryDecomposition {

    /**
     * Flag indicating if the reduced (true) or full (false) {@code QR} decomposition should be computed.
     */
    protected final boolean reduced;

    /**
     * Creates a {@code QR} decomposer. This decomposer will compute the reduced {@code QR} decomposition.
     * @see #ComplexQRDecomposition(boolean)
     */
    public ComplexQRDecomposition() {
        super(0);
        this.reduced = true;
    }


    /**
     * Creates a {@code QR} decomposer to compute either the full or reduced {@code QR} decomposition.
     *
     * @param reduced Flag indicating if this decomposer should compute the full or reduced {@code QR} decomposition.
     */
    public ComplexQRDecomposition(boolean reduced) {
        super(0);
        this.reduced = reduced;
    }


    /**
     * Computes the {@code QR} decomposition of a real dense matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public ComplexQRDecomposition decompose(CMatrix src) {
        decomposeBase(src);
        return this;
    }


    /**
     * Creates and initializes Q to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected CMatrix initQ() {
        int qCols = reduced ? minAxisSize : numRows; // Get Q in reduced form or not.
        return CMatrix.I(numRows, qCols);
    }


    /**
     * Gets the upper triangular matrix {@code R} from the last decomposition.
     *
     * @return The upper triangular matrix {@code R} from the last decomposition.
     */
    @Override
    protected CMatrix getUpper() {
        return getR();
    }


    /**
     * Gets the upper triangular matrix {@code R} from the {@code QR} decomposition.
     * @return The upper triangular matrix {@code R} from the {@code QR} decomposition.
     */
    public CMatrix getR() {
        int rRows = reduced ? minAxisSize : numRows; // Get R in reduced form or not.
        return getUpper(new CMatrix(rRows, numCols));
    }
}
