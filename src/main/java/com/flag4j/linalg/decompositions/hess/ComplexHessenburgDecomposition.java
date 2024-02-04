package com.flag4j.linalg.decompositions.hess;

import com.flag4j.CMatrix;
import com.flag4j.linalg.decompositions.unitary.ComplexUnitaryDecomposition;
import com.flag4j.util.ParameterChecks;


/**
 * <p>Computes the Hessenburg decomposition of a complex dense square matrix. That is, for a square matrix
 * {@code A}, computes the decomposition {@code A=QHQ}<sup>H</sup> where {@code Q} is an orthogonal matrix and
 * {@code H} is a matrix in upper Hessenburg form which is similar to {@code A} (i.e. has the same eigenvalues/vectors).</p>
 *
 * <p>A matrix {@code H} is in upper Hessenburg form if it is nearly upper triangular. Specifically, if {@code H} has
 * all zeros below the first sub-diagonal.</p>
 *
 * <p>For example, the following matrix is in upper Hessenburg form where each {@code x} is a placeholder which may hold a different
 * value:
 * <pre>
 *     [[ x x x x x ]
 *      [ x x x x x ]
 *      [ 0 x x x x ]
 *      [ 0 0 x x x ]
 *      [ 0 0 0 x x ]]</pre>
 * </p>
 */
public class ComplexHessenburgDecomposition extends ComplexUnitaryDecomposition {


    /**
     * Creates a Hessenburg decomposer. This decomposer will compute the Hessenburg decomposition for complex dense matrices.
     */
    public ComplexHessenburgDecomposition() {
        super(1);
    }


    /**
     * Computes the {@code QR} decomposition of a real dense matrix.
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     */
    @Override
    public ComplexHessenburgDecomposition decompose(CMatrix src) {
        ParameterChecks.assertSquare(src.shape);
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
        return CMatrix.I(numRows);
    }


    /**
     * Gets the upper Hessenburg matrix from the last decomposition. Same as {@link #getH()}
     *
     * @return The upper Hessenburg matrix from the last decomposition.
     */
    @Override
    protected CMatrix getUpper() {
        return getH();
    }


    /**
     * Gets the upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     * @return The upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     */
    public CMatrix getH() {
        return getUpper(new CMatrix(numRows));
    }
}
