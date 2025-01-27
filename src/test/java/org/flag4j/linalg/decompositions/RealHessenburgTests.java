package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.hess.RealHess;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

class RealHessenburgTests {

    double[][] aEntries;
    Matrix A, Q, H, A_hat;

    RealHess hess;

    @Test
    void hessDecompTestCase() {
        // ----------------------- sub-case 1 -----------------------
        aEntries = new double[][]{
                {0, 0, 0, 1},
                {0, 0, -1, 0},
                {0, 1, 0, 0},
                {-1, 0, 0, 0}};
        A = new Matrix(aEntries);
        hess = new RealHess();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        Assertions.assertEquals(new Matrix(A.shape).round(), A.sub(A_hat).round());

        // ----------------------- sub-case 2 -----------------------
        aEntries = new double[][]{
                {1.44, 5.26, -35, 1.9},
                {0.00024, 16.7, 0, 13.56},
                {1.35345, -2.0525, 18056.2, 1.5},
                {1.56, 1.6, 1.656, 0.1}};
        A = new Matrix(aEntries);
        hess = new RealHess();
        hess.decompose(A);

        H = hess.getH();
        Q = hess.getQ();
        A_hat = Q.mult(H).multTranspose(Q);

        Assertions.assertEquals(new Matrix(A.shape), A.sub(A_hat).roundToZero(1.0e-10));

        // ----------------------- sub-case 2.1 -----------------------
        hess = new RealHess();
        hess.decompose(A);

        Assertions.assertEquals(H, hess.getH());
    }
}
