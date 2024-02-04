package com.flag4j.linalg.solvers.lstsq;

import com.flag4j.CMatrix;
import com.flag4j.CVector;
import com.flag4j.linalg.decompositions.qr.ComplexQRDecompositionOld;
import com.flag4j.linalg.decompositions.qr.QRDecompositionOld;
import com.flag4j.linalg.solvers.exact.triangular.ComplexBackSolver;

/**
 * This class solves a linear system of equations {@code Ax=b} in a least-squares sense. That is,
 * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations
 * {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
 * This is done using a {@link QRDecompositionOld}.
 */
public class ComplexLstsqSolver extends LstsqSolver<CMatrix, CVector> {


    /**
     * Constructs a least-squares solver to solve a system {@code Ax=b} in a least square sense. That is,
     * minimizes {@code ||Ax-b||<sub>2</sub>} which is equivalent to solving the normal equations
     * {@code A<sup>T</sup>Ax=A<sup>T</sup>b}.
     */
    public ComplexLstsqSolver() {
        super(new ComplexQRDecompositionOld(), new ComplexBackSolver());
    }
}
