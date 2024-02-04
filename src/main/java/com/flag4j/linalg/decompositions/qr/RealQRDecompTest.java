package com.flag4j.linalg.decompositions.qr;

import com.flag4j.Matrix;
import com.flag4j.linalg.decompositions.unitary.RealUnitaryDecomposition;


/**
 * <p>Instances of this class compute the {@code QR} decomposition of a real dense matrix.</p>
 * <p>The {@code QR} decomposition, decomposes a matrix {@code A} into an orthogonal matrix {@code Q}
 * and an upper triangular matrix {@code R} such that {@code A=QR}.</p>
 *
 * <p>Much of this code has been adapted from the EJML library.</p>
 */
public class RealQRDecompTest extends RealUnitaryDecomposition {

    /**
     * Flag indicating if the reduced (true) or full (false) {@code QR} decomposition should be computed.
     */
    protected final boolean reduced;


    /**
     * Creates a {@code QR} decomposer. This decomposer will compute the reduced {@code QR} decomposition.
     * @see #RealQRDecompTest(boolean)
     */
    public RealQRDecompTest() {
        super(0);
        this.reduced = true;
    }


    /**
     * Creates a {@code QR} decomposer to compute either the full or reduced {@code QR} decomposition.
     *
     * @param reduced Flag indicating if this decomposer should compute the full or reduced {@code QR} decomposition.
     */
    public RealQRDecompTest(boolean reduced) {
        super(0);
        this.reduced = reduced;
    }


    /**
     * Computes the {@code QR} decomposition of a real dense matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public RealQRDecompTest decompose(Matrix src) {
        decomposeBase(src);
        return this;
    }


    /**
     * Creates and initializes Q to the appropriately sized identity matrix.
     *
     * @return An identity matrix with the appropriate size.
     */
    @Override
    protected Matrix initQ() {
        int qCols = reduced ? minAxisSize : numRows; // Get Q in reduced form or not.
        return Matrix.I(numRows, qCols);
    }


    /**
     * Gets the upper triangular matrix {@code R} from the last decomposition. Same as {@link #getR()}.
     *
     * @return The upper triangular matrix from the last decomposition.
     */
    @Override
    protected Matrix getUpper() {
        return getR();
    }


    /**
     * Gets the upper triangular matrix {@code R} from the {@code QR} decomposition.
     * @return The upper triangular matrix {@code R} from the {@code QR} decomposition.
     */
    public Matrix getR() {
        int rRows = reduced ? minAxisSize : numRows; // Get R in reduced form or not.
        return getUpper(new Matrix(rRows, numCols));
    }
}
