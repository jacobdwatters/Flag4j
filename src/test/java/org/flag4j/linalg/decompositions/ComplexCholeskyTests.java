package org.flag4j.linalg.decompositions;

import org.flag4j.linalg.decompositions.chol.ComplexCholesky;
import org.junit.jupiter.api.Test;

class ComplexCholeskyTests {

    static String[][] aEntries;
    static CMatrix A, L, A_hat;

    static void setMatrices() {
        A = new CMatrix(aEntries);
    }


    @Test
    void choleskyTestCase() {
        ComplexCholesky cholesky = new ComplexCholesky();

        // --------------------- Sub-case 1 ---------------------
        aEntries = new String[][]{
                {"1", "-2i"},
                {"2i", "5"}};
        setMatrices();

        L = cholesky.decompose(A).getL();
        A_hat = L.mult(L.H());

        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));

        // --------------------- Sub-case 2 ---------------------
        aEntries = new String[][]{
                {"2", "i"},
                {"-i", "2"}};
        setMatrices();

        L = cholesky.decompose(A).getL();
        A_hat = L.mult(L.H());
        
        assertEquals(new CMatrix(A.shape), A.sub(A_hat).roundToZero(1.0e-12));
    }
}
