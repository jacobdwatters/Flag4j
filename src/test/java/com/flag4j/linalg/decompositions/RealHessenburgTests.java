package com.flag4j.linalg.decompositions;

import com.flag4j.Matrix;
import com.flag4j.linalg.decompositions.hess.RealHessenburgDecompositionOld;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class RealHessenburgTests {

    double[][] aEntries;
    Matrix A, Q, H, A_hat;

    RealHessenburgDecompositionOld hess;

    @Test
    void hessDecompTestCase() {
        // ----------------------- Sub-case 1 -----------------------
        aEntries = new double[][]{
                {0, 0, 0, 1},
                {0, 0, -1, 0},
                {0, 1, 0, 0},
                {-1, 0, 0, 0}};
        A = new Matrix(aEntries);
        hess = new RealHessenburgDecompositionOld();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        assertEquals(new Matrix(A.shape.copy()).round(), A.sub(A_hat).round());

        // ----------------------- Sub-case 1.1 -----------------------
        hess = new RealHessenburgDecompositionOld(false);
        hess.decompose(A);

        assertEquals(H, hess.getH());

        // ----------------------- Sub-case 2 -----------------------
        aEntries = new double[][]{
                {1.44, 5.26, -35, 1.9},
                {0.00024, 16.7, 0, 13.56},
                {1.35345, -2.0525, 18056.2, 1.5},
                {1.56, 1.6, 1.656, 0.1}};
        A = new Matrix(aEntries);
        hess = new RealHessenburgDecompositionOld();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        assertEquals(new Matrix(A.shape.copy()), A.sub(A_hat).roundToZero(1.0e-10));

        // ----------------------- Sub-case 2.1 -----------------------
        hess = new RealHessenburgDecompositionOld(false);
        hess.decompose(A);

        assertEquals(H, hess.getH());
    }
}
