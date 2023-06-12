package com.flag4j.linalg.solvers;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.linalg.decompositions.ComplexQRDecomposition;
import com.flag4j.linalg.decompositions.QRDecomposition;

/**
 * This class solves a linear system of equations {@code Ax=b} in a least-squares sense. That is,
 * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations
 * {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
 * This is done using a {@link QRDecomposition}.
 */
public class ComplexLstsqSolver extends LstsqSolver<CMatrix, CVector> {

    /**
     * Backwards solver for solving the system of equations formed from the {@code QR} decomposition,
     * {@code Rx=Q<sup>T</sup>b} which is an equivalent system to {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
     */
    private final ComplexBackSolver backSolver;


    /**
     * Constructs a least-squares solver to solve a system {@code Ax=b} in a least square sense. That is,
     * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations
     * {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
     */
    public ComplexLstsqSolver() {
        super(new ComplexQRDecomposition(false));
        backSolver = new ComplexBackSolver();
    }


    /**
     * Solves the linear system given by {@code Ax=b} in the least-squares sense.
     *
     * @param A Coefficient matrix in the linear system.
     * @param b Vector of constants in the linear system.
     * @return The least squares solution to {@code x} in the linear system {@code Ax=b}.
     */
    @Override
    public CVector solve(CMatrix A, CVector b) {
        decompose(A); // Compute the reduced QR decomposition of A.
        return backSolver.solve(R, Q.H().mult(b));
    }


    /**
     * Solves the set of linear system of equations given by {@code A*X=B} for the matrix {@code X} where
     * {@code A}, {@code B}, and {@code X} are matrices.
     *
     * @param A Coefficient matrix in the linear system.
     * @param B Matrix of constants in the linear system.
     * @return The solution to {@code X} in the linear system {@code A*X=B}.
     */
    @Override
    public CMatrix solve(CMatrix A, CMatrix B) {
        decompose(A); // Compute the reduced QR decomposition of A.
        return backSolver.solve(R, Q.H().mult(B));
    }
}
