package com.flag4j.linalg.decompositions.hess;

import com.flag4j.Matrix;
import com.flag4j.linalg.decompositions.unitary.RealUnitaryDecomposition;
import com.flag4j.util.ParameterChecks;


/**
 * <p>Computes the Hessenburg decomposition of a real dense square matrix. That is, for a square matrix
 * {@code A}, computes the decomposition {@code A=QHQ}<sup>T</sup> where {@code Q} is an orthogonal matrix and
 * {@code H} is a matrix in upper Hessenburg form which is similar to {@code A} (i.e. has the same eigenvalues/vectors).</p>
 *
 * <p>A matrix {@code H} is in upper Hessenburg form if it is nearly upper triangular. Specifically, if {@code H} has
 * all zeros below the first sub-diagonal.</p>
 *
 * <p>For example, the following matrix is in upper Hessenburg form where each {@code x} may hold a different value:
 * <pre>
 *     [[ x x x x x ]
 *      [ x x x x x ]
 *      [ 0 x x x x ]
 *      [ 0 0 x x x ]
 *      [ 0 0 0 x x ]]</pre>
 * </p>
 */
public class RealHessenburgDecomposition extends RealUnitaryDecomposition {

    /**
     * Creates a real unitary decomposer which will reduce the matrix to an upper quasi-triangular matrix which is has zeros below
     * the specified sub-diagonal.
     */
    public RealHessenburgDecomposition() {
        super(1);
    }


    /**
     * Applies decomposition to the source matrix. Note, the computation of the unitary matrix {@code Q} in the decomposition is
     * deferred until {@link #getQ()} is explicitly called. This allows for efficient decompositions when {@code Q} is not needed.
     *
     * @param src The source matrix to decompose.
     * @return A reference to this decomposer.
     * @throws com.flag4j.exceptions.LinearAlgebraException If {@code src} is not a square matrix.
     */
    @Override
    public RealHessenburgDecomposition decompose(Matrix src) {
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
    protected Matrix initQ() {
        return Matrix.I(numRows);
    }


    /**
     * Gets the upper Hessenburg matrix from the last decomposition. Same as {@link #getH()}
     *
     * @return The upper Hessenburg matrix from the last decomposition.
     */
    @Override
    protected Matrix getUpper() {
        return getH();
    }


    /**
     * Gets the upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     * @return The upper Hessenburg matrix {@code H} from the Hessenburg decomposition.
     */
    public Matrix getH() {
        return getUpper(new Matrix(numRows));
    }
}
